"""UltraSeg-130K decode head, adapted from https://github.com/AI-thpremed/ultraseg."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead

try:
    from timm.models.layers import trunc_normal_
except Exception:  # pragma: no cover
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        with torch.no_grad():
            return tensor.normal_(mean, std).clamp_(a, b)


class EnhancedDilatedBlock(nn.Module):
    def __init__(self, channels, key=3):
        super().__init__()
        g = max(channels // key, 1)
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels, channels // 3, 3, padding=1, dilation=1, groups=g),
            nn.Conv2d(channels, channels // 3, 3, padding=2, dilation=2, groups=g),
            nn.Conv2d(channels, channels // 3, 3, padding=3, dilation=3, groups=g),
        ])
        self.fusion = nn.Conv2d(channels, channels, 1)
        self.res_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        features = [conv(x) for conv in self.dilated_convs]
        fused = torch.cat(features, dim=1)
        return x + self.fusion(fused) * self.res_weight


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_first'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                                self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Down(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(self.bn(x))


class Image_Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        gt_pre = self.conv(x)
        x = x + x * torch.sigmoid(gt_pre)
        return x, gt_pre


class Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        boundary = torch.sigmoid(self.conv1(x))
        gt_pre = self.conv2(x)
        return (x + x * boundary + x * torch.sigmoid(gt_pre)), gt_pre, boundary


class Group_shuffle_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        if dim_in % 4 != 0:
            dim_in = (dim_in // 4) * 4
        c_dim = dim_in // 4
        self.share_space1 = nn.Parameter(torch.ones(1, c_dim, 8, 8))
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(), nn.Conv2d(c_dim, c_dim, 1))
        self.share_space2 = nn.Parameter(torch.ones(1, c_dim, 8, 8))
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(), nn.Conv2d(c_dim, c_dim, 1))
        self.share_space3 = nn.Parameter(torch.ones(1, c_dim, 8, 8))
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(), nn.Conv2d(c_dim, c_dim, 1))
        self.share_space4 = nn.Parameter(torch.ones(1, c_dim, 8, 8))
        self.conv4 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(), nn.Conv2d(c_dim, c_dim, 1))
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(), nn.Conv2d(dim_in, dim_out, 1))

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        size = x1.shape[2:4]
        x1 = x1 * self.conv1(
            F.interpolate(self.share_space1, size=size, mode='bilinear',
                          align_corners=True))
        x2 = x2 * self.conv2(
            F.interpolate(self.share_space2, size=size, mode='bilinear',
                          align_corners=True))
        x3 = x3 * self.conv3(
            F.interpolate(self.share_space3, size=size, mode='bilinear',
                          align_corners=True))
        x4 = x4 * self.conv4(
            F.interpolate(self.share_space4, size=size, mode='bilinear',
                          align_corners=True))
        x = torch.cat([x2, x4, x1, x3], dim=1)
        x = self.norm2(x)
        return self.ldw(x)


class AdaptiveMerge(nn.Module):
    def __init__(self, dim_in, use_boundary=True):
        super().__init__()
        self.use_boundary = use_boundary
        self.region_weight = nn.Parameter(torch.tensor(0.3))
        if self.use_boundary:
            self.boundary_weight = nn.Parameter(torch.tensor(0.1))
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(dim_in, max(dim_in // 8, 4), 1),
            nn.ReLU(),
            nn.Conv2d(max(dim_in // 8, 4), dim_in, 1),
        ) if dim_in > 16 else nn.Identity()

    def forward(self, x1, x2, gt_pre, boundary_pre=None):
        x1_adapted = self.feature_adapter(x1)
        if gt_pre.size(1) == 1:
            gt_pre = gt_pre.expand(-1, x2.size(1), -1, -1)
        base_fusion = x1_adapted + x2
        region_contrib = torch.sigmoid(gt_pre) * x2 * self.region_weight
        boundary_contrib = 0
        if self.use_boundary and boundary_pre is not None:
            if boundary_pre.size(1) == 1:
                boundary_pre = boundary_pre.expand(-1, x2.size(1), -1, -1)
            boundary_contrib = boundary_pre * x2 * self.boundary_weight
        return base_fusion + region_contrib + boundary_contrib


class AttentionGuidedFusion(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.output_ch = channels_list[-1]
        self.transforms = nn.ModuleList()
        for ch in channels_list:
            if ch != self.output_ch:
                self.transforms.append(nn.Conv2d(ch, self.output_ch, 1))
            else:
                self.transforms.append(None)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(len(channels_list) * self.output_ch, self.output_ch // 4,
                      3, padding=1),
            nn.BatchNorm2d(self.output_ch // 4),
            nn.GELU(),
            nn.Conv2d(self.output_ch // 4, len(channels_list), 1),
            nn.Softmax(dim=1))

    def forward(self, features):
        transformed = []
        target_size = features[0].shape[2:]
        for feat, trans in zip(features, self.transforms):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear',
                                     align_corners=True)
            if trans is not None:
                feat = trans(feat)
            transformed.append(feat)
        concat_feat = torch.cat(transformed, dim=1)
        attention_weights = self.spatial_attention(concat_feat)
        return sum(attn * feat for attn, feat in
                   zip(attention_weights.split(1, dim=1), transformed))


class SimpleSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(torch.mean(x, dim=1, keepdim=True))


class UltraSeg130(nn.Module):
    def __init__(self, out_ch=2, in_ch=3, key=3, width_mult=1):
        super().__init__()
        # official 130K uses [8,16,48,64,96]; width_mult>1 for stronger cmp runs
        base = [8, 16, 48, 64, 96]
        c_list = [max(4, int(round(c * width_mult / 4.0)) * 4) for c in base]
        self.attention_alpha = nn.Parameter(torch.tensor(0.3))
        self.encoder1 = nn.Sequential(nn.Conv2d(in_ch, c_list[0], 3, stride=1, padding=1))
        self.encoder2 = nn.Sequential(nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1))
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, padding=1),
            EnhancedDilatedBlock(c_list[2], key),
            EnhancedDilatedBlock(c_list[2], key))
        self.encoder4 = nn.Sequential(Group_shuffle_block(c_list[2], c_list[3]))
        self.multiscale_fusion = AttentionGuidedFusion([c_list[2], c_list[3]])
        self.simple_attention = SimpleSpatialAttention()
        self.bottleneck = nn.Conv2d(c_list[3], c_list[4], 3, padding=1)
        self.Down1 = Down(c_list[0])
        self.Down2 = Down(c_list[1])
        self.Down3 = Down(c_list[2])
        self.merge1 = AdaptiveMerge(c_list[0], use_boundary=True)
        self.merge2 = AdaptiveMerge(c_list[1], use_boundary=True)
        self.merge3 = AdaptiveMerge(c_list[2], use_boundary=True)
        self.merge4 = AdaptiveMerge(c_list[3], use_boundary=False)
        self.decoder1 = nn.Sequential(Group_shuffle_block(c_list[4], c_list[3]))
        self.decoder2 = nn.Sequential(Group_shuffle_block(c_list[3], c_list[2]))
        self.decoder3 = nn.Sequential(Group_shuffle_block(c_list[2], c_list[1]))
        self.decoder4 = nn.Sequential(nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1))
        self.pred1 = Image_Prediction_Generator(c_list[3])
        self.gate1 = Prediction_Generator(c_list[2])
        self.gate2 = Prediction_Generator(c_list[1])
        self.gate3 = Prediction_Generator(c_list[0])
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.dbn1 = nn.GroupNorm(4, c_list[3])
        self.dbn2 = nn.GroupNorm(4, c_list[2])
        self.dbn3 = nn.GroupNorm(4, c_list[1])
        self.dbn4 = nn.GroupNorm(4, c_list[0])
        self.final = nn.Sequential(nn.Conv2d(c_list[0], out_ch, kernel_size=1))
        self.apply(self._init_weights)

    def forward(self, x):
        out = self.encoder1(x)
        out = F.gelu(self.Down1(self.ebn1(out)))
        t1 = out
        out = self.encoder2(out)
        out = F.gelu(self.Down2(self.ebn2(out)))
        t2 = out
        out = self.encoder3(out)
        out = F.gelu(self.Down3(self.ebn3(out)))
        t3 = out
        out = self.encoder4(out)
        out = F.gelu(F.max_pool2d(self.ebn4(out), 2))
        t4 = out

        x3_up = F.interpolate(t3, size=t4.shape[2:], mode='bilinear', align_corners=True)
        fused_features = self.multiscale_fusion([x3_up, t4])
        bottleneck_out = self.bottleneck(fused_features)
        attended_features = self.simple_attention(bottleneck_out)
        x = (1 - self.attention_alpha) * bottleneck_out + self.attention_alpha * attended_features

        out = F.gelu(self.dbn1(self.decoder1(x)))
        out, gt_pre4 = self.pred1(out)
        out = self.merge4(out, t4, gt_pre4)

        out = self.decoder2(out)
        out = F.gelu(F.interpolate(self.dbn2(out), scale_factor=2, mode='bilinear',
                                   align_corners=True))
        out, gt_pre3, weight1 = self.gate1(out)
        out = self.merge3(out, t3, gt_pre3, weight1)

        out = self.decoder3(out)
        out = F.gelu(F.interpolate(self.dbn3(out), scale_factor=2, mode='bilinear',
                                   align_corners=True))
        out, gt_pre2, weight2 = self.gate2(out)
        out = self.merge2(out, t2, gt_pre2, weight2)

        out = self.decoder4(out)
        out = F.gelu(F.interpolate(self.dbn4(out), scale_factor=2, mode='bilinear',
                                   align_corners=True))
        out, gt_pre1, weight3 = self.gate3(out)
        out = self.merge1(out, t1, gt_pre1, weight3)

        out = self.final(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        # deep-supervision logits (before sigmoid), upsampled to input scale
        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)
        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        weight1 = F.interpolate(weight1, scale_factor=8, mode='bilinear', align_corners=True)
        weight2 = F.interpolate(weight2, scale_factor=4, mode='bilinear', align_corners=True)
        weight3 = F.interpolate(weight3, scale_factor=2, mode='bilinear', align_corners=True)
        return out, (gt_pre4, gt_pre3, gt_pre2, gt_pre1), (weight1, weight2, weight3)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


@HEADS.register_module()
class UltraSeg130K(BaseDecodeHead):
    """UltraSeg-130K full network used as decode head (EncoderDecoderFull)."""

    def __init__(self, in_channels=3, key=3, use_deep_supervision=True,
                 width_mult=1, **kwargs):
        super(UltraSeg130K, self).__init__(in_channels=in_channels, **kwargs)
        self.use_deep_supervision = use_deep_supervision
        self.net = UltraSeg130(
            out_ch=self.num_classes, in_ch=in_channels, key=key,
            width_mult=width_mult)

    def forward(self, inputs):
        out, _, _ = self.net(inputs)
        return out

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        out, gt_pres, boundaries = self.net(inputs)
        losses = self.losses(out, gt_semantic_seg)
        if self.use_deep_supervision:
            # official-style deep supervision: BCE+Dice on side outputs
            label = gt_semantic_seg.squeeze(1).float()
            fg = (label == 1).float()
            # ignore 255
            valid = (label != 255).float()
            weights = (0.1, 0.2, 0.3, 0.4)
            ds_loss = 0.0
            for w, gp in zip(weights, gt_pres):
                # gp: [B,1,H,W] logits
                if gp.shape[-2:] != fg.shape[-2:]:
                    gp = F.interpolate(gp, size=fg.shape[-2:], mode='bilinear',
                                       align_corners=True)
                prob = torch.sigmoid(gp.squeeze(1))
                bce = F.binary_cross_entropy(prob, fg, reduction='none')
                bce = (bce * valid).sum() / (valid.sum() + 1e-6)
                inter = (prob * fg * valid).sum()
                dice = 1 - (2 * inter + 1) / (
                    (prob * valid).sum() + (fg * valid).sum() + 1)
                ds_loss = ds_loss + w * (0.5 * bce + dice)
            # light boundary consistency toward GT edges
            if boundaries is not None:
                # approximate GT boundary via max-pool difference
                fg4 = fg.unsqueeze(1)
                edge = (F.max_pool2d(fg4, 3, 1, 1) - fg4).abs().clamp(0, 1)
                bw = (0.1, 0.2, 0.3)
                for w, bd in zip(bw, boundaries):
                    if bd.shape[-2:] != edge.shape[-2:]:
                        bd = F.interpolate(bd, size=edge.shape[-2:], mode='bilinear',
                                           align_corners=True)
                    ds_loss = ds_loss + w * F.binary_cross_entropy(
                        bd.clamp(0, 1), edge)
            losses['loss_ds'] = ds_loss
        return losses
