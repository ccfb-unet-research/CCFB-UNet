"""DDCNet: Dual Domain Collaborative Network for Polyp Segmentation.

Faithful PyTorch re-implementation of Tong et al. (IEEE JBHI 2025):
ResNet34 encoder + FCEM (frequency) + CSFM (spatial cross-level fusion).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..builder import HEADS
from .decode_head import BaseDecodeHead


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class FCEM(nn.Module):
    """Frequency Context Enhancement Module."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.low_refine = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False), nn.Sigmoid())
        self.high_refine = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False), nn.Sigmoid())
        self.fuse = ConvBNReLU(channels, channels, 1, 1, 0)
        # spatial boundary enhance
        self.edge = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        fft = torch.fft.rfft2(x.float(), norm='ortho')
        amp = torch.abs(fft)
        phase = torch.angle(fft)
        low_amp = F.avg_pool2d(amp, kernel_size=3, stride=1, padding=1)
        high_amp = amp - low_amp
        low_fft = torch.complex(low_amp * torch.cos(phase), low_amp * torch.sin(phase))
        high_fft = torch.complex(high_amp * torch.cos(phase), high_amp * torch.sin(phase))
        low_map = torch.fft.irfft2(low_fft, s=x.shape[-2:], norm='ortho').type_as(x)
        high_map = torch.fft.irfft2(high_fft, s=x.shape[-2:], norm='ortho').type_as(x)
        out = x * (1.0 + self.low_refine(low_map)) + x * self.high_refine(high_map)
        out = out * (1.0 + self.edge(out))
        return self.fuse(out)


class CSFM(nn.Module):
    """Cross-level Shift-recalibrated Fusion Module."""

    def __init__(self, low_ch, high_ch, out_ch):
        super().__init__()
        self.low_proj = ConvBNReLU(low_ch, out_ch, 1, 1, 0)
        self.high_proj = ConvBNReLU(high_ch, out_ch, 1, 1, 0)
        self.shift = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.recal = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch * 2, max(out_ch // 4, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_ch // 4, 4), out_ch * 2, 1),
            nn.Sigmoid())
        self.out = nn.Sequential(
            ConvBNReLU(out_ch, out_ch),
            ConvBNReLU(out_ch, out_ch))

    def forward(self, low_feat, high_feat):
        h = self.high_proj(high_feat)
        h = F.interpolate(h, size=low_feat.shape[-2:], mode='bilinear', align_corners=False)
        l = self.low_proj(low_feat)
        h_shift = self.shift(h)
        w = self.recal(torch.cat([l, h_shift], dim=1))
        w_l, w_h = torch.chunk(w, 2, dim=1)
        return self.out(l * w_l + h_shift * w_h)


@HEADS.register_module()
class DDCNet(BaseDecodeHead):
    """Dual Domain Collaborative Network with ResNet34 encoder."""

    def __init__(self, base_channels=64, pretrained_backbone=True, **kwargs):
        super(DDCNet, self).__init__(**kwargs)
        # ResNet34 encoder
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            backbone = models.resnet34(weights=weights)
        except Exception:
            backbone = models.resnet34(pretrained=pretrained_backbone)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 64
        self.layer2 = backbone.layer2  # 128
        self.layer3 = backbone.layer3  # 256
        self.layer4 = backbone.layer4  # 512

        c = base_channels
        self.reduce4 = ConvBNReLU(512, c * 4, 1, 1, 0)
        self.reduce3 = ConvBNReLU(256, c * 2, 1, 1, 0)
        self.reduce2 = ConvBNReLU(128, c, 1, 1, 0)
        self.reduce1 = ConvBNReLU(64, c // 2, 1, 1, 0)
        self.reduce0 = ConvBNReLU(64, c // 2, 1, 1, 0)

        self.fcem4 = FCEM(c * 4)
        self.fcem3 = FCEM(c * 2)
        self.fcem2 = FCEM(c)

        self.csfm3 = CSFM(c * 2, c * 4, c * 2)
        self.csfm2 = CSFM(c, c * 2, c)
        self.csfm1 = CSFM(c // 2, c, c // 2)
        self.csfm0 = CSFM(c // 2, c // 2, c // 2)

        self.cls = nn.Sequential(
            ConvBNReLU(c // 2, c // 2),
            nn.Conv2d(c // 2, self.num_classes, kernel_size=1))

    def forward(self, inputs):
        x0 = self.stem(inputs)          # /2, 64
        x = self.maxpool(x0)            # /4
        e1 = self.layer1(x)             # /4, 64
        e2 = self.layer2(e1)            # /8, 128
        e3 = self.layer3(e2)            # /16, 256
        e4 = self.layer4(e3)            # /32, 512

        f4 = self.fcem4(self.reduce4(e4))
        f3 = self.fcem3(self.reduce3(e3))
        f2 = self.fcem2(self.reduce2(e2))
        f1 = self.reduce1(e1)
        f0 = self.reduce0(x0)

        d3 = self.csfm3(f3, f4)
        d2 = self.csfm2(f2, d3)
        d1 = self.csfm1(f1, d2)
        d0 = self.csfm0(f0, d1)
        out = self.cls(d0)
        if out.shape[-2:] != inputs.shape[-2:]:
            out = F.interpolate(out, size=inputs.shape[-2:], mode='bilinear',
                                align_corners=False)
        return out
