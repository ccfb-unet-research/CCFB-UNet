"""EMCAD: Efficient Multi-scale Convolutional Attention Decoding (CVPR 2024).

Reference: https://github.com/SLDGroup/EMCAD
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_tf_

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .pvt import pvt_v2_b0, pvt_v2_b2

PVT_ENCODER_CFG = {
    'pvt_v2_b0': dict(
        factory=pvt_v2_b0,
        channels=(256, 160, 64, 32),
    ),
    'pvt_v2_b2': dict(
        factory=pvt_v2_b2,
        channels=(512, 320, 128, 64),
    ),
}


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(
                module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    if act == 'relu6':
        return nn.ReLU6(inplace)
    if act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    if act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    if act == 'gelu':
        return nn.GELU()
    if act == 'hswish':
        return nn.Hardswish(inplace)
    raise NotImplementedError(f'activation layer [{act}] is not found')


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)


class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6',
                 dw_parallel=True):
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels, kernel_size, stride,
                    kernel_size // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True))
            for kernel_size in kernel_sizes
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, stride,
                 kernel_sizes=(1, 3, 5), expansion_factor=2,
                 dw_parallel=True, add=True, activation='relu6'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_skip_connection = stride == 1
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.ex_channels = int(in_channels * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(activation, inplace=True))
        self.msdc = MSDC(
            self.ex_channels, kernel_sizes, stride, activation,
            dw_parallel=dw_parallel)
        self.combined_channels = (
            self.ex_channels if add else self.ex_channels * self.n_scales)
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels))
        if self.use_skip_connection and in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add:
            dout = sum(msdc_outs)
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(
            dout, gcd(self.combined_channels, self.pconv2[0].out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        return out


def MSCBLayer(in_channels, out_channels, n=1, stride=1,
              kernel_sizes=(1, 3, 5), expansion_factor=2,
              dw_parallel=True, add=True, activation='relu6'):
    convs = [MSCB(
        in_channels, out_channels, stride, kernel_sizes=kernel_sizes,
        expansion_factor=expansion_factor, dw_parallel=dw_parallel,
        add=add, activation=activation)]
    for _ in range(1, n):
        convs.append(MSCB(
            out_channels, out_channels, 1, kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor, dw_parallel=dw_parallel,
            add=add, activation=activation))
    return nn.Sequential(*convs)


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 activation='relu'):
        super().__init__()
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2,
                groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            act_layer(activation, inplace=True))
        self.pwc = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.up_dwc[1].in_channels)
        return self.pwc(x)


class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1,
                 activation='relu'):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g, F_int, kernel_size=kernel_size, stride=1,
                padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l, F_int, kernel_size=kernel_size, stride=1,
                padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        psi = self.psi(self.activation(self.W_g(g) + self.W_x(x)))
        return x * psi


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16,
                 activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        ratio = min(ratio, in_channels)
        reduced_channels = in_channels // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(reduced_channels, self.out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7, 11)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class EMCAD(nn.Module):
    def __init__(self, channels=(512, 320, 128, 64),
                 kernel_sizes=(1, 3, 5), expansion_factor=2,
                 dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super().__init__()
        eucb_ks = 3
        self.mscb4 = MSCBLayer(
            channels[0], channels[0], n=1, stride=1,
            kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
            dw_parallel=dw_parallel, add=add, activation=activation)
        self.eucb3 = EUCB(
            in_channels=channels[0], out_channels=channels[1],
            kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag3 = LGAG(
            F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2,
            kernel_size=lgag_ks, groups=channels[1] // 2)
        self.mscb3 = MSCBLayer(
            channels[1], channels[1], n=1, stride=1,
            kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
            dw_parallel=dw_parallel, add=add, activation=activation)
        self.eucb2 = EUCB(
            in_channels=channels[1], out_channels=channels[2],
            kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag2 = LGAG(
            F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2,
            kernel_size=lgag_ks, groups=channels[2] // 2)
        self.mscb2 = MSCBLayer(
            channels[2], channels[2], n=1, stride=1,
            kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
            dw_parallel=dw_parallel, add=add, activation=activation)
        self.eucb1 = EUCB(
            in_channels=channels[2], out_channels=channels[3],
            kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag1 = LGAG(
            F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2,
            kernel_size=lgag_ks, groups=channels[3] // 2)
        self.mscb1 = MSCBLayer(
            channels[3], channels[3], n=1, stride=1,
            kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
            dw_parallel=dw_parallel, add=add, activation=activation)
        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])
        self.sab = SAB()

    def forward(self, x, skips):
        d4 = self.cab4(x) * x
        d4 = self.sab(d4) * d4
        d4 = self.mscb4(d4)

        d3 = self.eucb3(d4)
        d3 = d3 + self.lgag3(g=d3, x=skips[0])
        d3 = self.cab3(d3) * d3
        d3 = self.sab(d3) * d3
        d3 = self.mscb3(d3)

        d2 = self.eucb2(d3)
        d2 = d2 + self.lgag2(g=d2, x=skips[1])
        d2 = self.cab2(d2) * d2
        d2 = self.sab(d2) * d2
        d2 = self.mscb2(d2)

        d1 = self.eucb1(d2)
        d1 = d1 + self.lgag1(g=d1, x=skips[2])
        d1 = self.cab1(d1) * d1
        d1 = self.sab(d1) * d1
        d1 = self.mscb1(d1)
        return [d4, d3, d2, d1]


class EMCADBackbone(nn.Module):
    """PVT encoder + EMCAD decoder."""

    def __init__(self, num_classes, encoder='pvt_v2_b2', pretrained_pvt=None,
                 kernel_sizes=(1, 3, 5), expansion_factor=2,
                 dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super().__init__()
        if encoder not in PVT_ENCODER_CFG:
            encoder = 'pvt_v2_b2'
        enc_cfg = PVT_ENCODER_CFG[encoder]
        self.backbone = enc_cfg['factory']()
        channels = enc_cfg['channels']
        if pretrained_pvt is not None:
            state = torch.load(pretrained_pvt, map_location='cpu')
            model_dict = self.backbone.state_dict()
            state = {k: v for k, v in state.items() if k in model_dict}
            model_dict.update(state)
            self.backbone.load_state_dict(model_dict)
        self.decoder = EMCAD(
            channels=channels,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            lgag_ks=lgag_ks,
            activation=activation)
        self.out_head = nn.Conv2d(channels[-1], num_classes, 1)
        self.gray_conv = nn.Sequential(
            nn.Conv2d(1, 3, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))

    def forward(self, x):
        input_size = x.shape[2:]
        if x.size(1) == 1:
            x = self.gray_conv(x)
        feats = self.backbone(x)
        x1, x2, x3, x4 = feats[0], feats[1], feats[2], feats[3]
        dec_outs = self.decoder(x4, [x3, x2, x1])
        out = self.out_head(dec_outs[-1])
        return F.interpolate(
            out, size=input_size, mode='bilinear', align_corners=False)


@HEADS.register_module()
class EMCADHead(BaseDecodeHead):
    """EMCAD with PVT-v2 backbone (default pvt_v2_b2)."""

    def __init__(self,
                 encoder='pvt_v2_b2',
                 kernel_sizes=(1, 3, 5),
                 expansion_factor=2,
                 dw_parallel=True,
                 add=True,
                 lgag_ks=3,
                 activation='relu6',
                 pretrained_pvt=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.net = EMCADBackbone(
            self.num_classes,
            encoder=encoder,
            pretrained_pvt=pretrained_pvt,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            lgag_ks=lgag_ks,
            activation=activation)

    def forward(self, inputs):
        return self.net(inputs)
