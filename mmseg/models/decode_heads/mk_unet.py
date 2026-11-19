"""MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation (ICCV 2025).

Official implementation: https://github.com/SLDGroup/MK-UNet
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


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
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
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes or in_planes
        if self.in_planes < ratio:
            ratio = self.in_planes
        reduced_channels = self.in_planes // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(in_planes, reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(reduced_channels, self.out_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
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


class GroupedAttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int, kernel_size=1, groups=1, activation='relu'):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        psi = self.activation(self.w_g(g) + self.w_x(x))
        return x * self.psi(psi)


class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, stride, k // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True))
            for k in kernel_sizes
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


class MultiKernelInvertedResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True,
                 add=True, kernel_sizes=(1, 3, 5), activation='relu6'):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip_connection = stride == 1
        self.ex_c = int(in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True))
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(
            self.ex_c, kernel_sizes, stride, activation, dw_parallel=dw_parallel)
        combined_channels = self.ex_c if add else self.ex_c * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv2d(combined_channels, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c))
        if self.use_skip_connection and in_c != out_c:
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add:
            dout = sum(dwconv_outs)
        else:
            dout = torch.cat(dwconv_outs, dim=1)
        dout = channel_shuffle(dout, gcd(dout.shape[1], self.pconv2[0].out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if x.shape[1] != out.shape[1]:
                x = self.conv1x1(x)
            return x + out
        return out


def mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2, dw_parallel=True,
                      add=True, kernel_sizes=(1, 3, 5), activation='relu6'):
    convs = [MultiKernelInvertedResidualBlock(
        in_c, out_c, s, expansion_factor=expansion_factor, dw_parallel=dw_parallel,
        add=add, kernel_sizes=kernel_sizes, activation=activation)]
    for _ in range(1, n):
        convs.append(MultiKernelInvertedResidualBlock(
            out_c, out_c, 1, expansion_factor=expansion_factor, dw_parallel=dw_parallel,
            add=add, kernel_sizes=kernel_sizes, activation=activation))
    return nn.Sequential(*convs)


class MKUNetBackbone(nn.Module):
    """MK-UNet default variant (channels=[16,32,64,96,160])."""

    def __init__(self, num_classes=1, in_channels=3,
                 channels=(16, 32, 64, 96, 160), depths=(1, 1, 1, 1, 1),
                 kernel_sizes=(1, 3, 5), expansion_factor=2, gag_kernel=3, **kwargs):
        super().__init__()
        self.encoder1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], 1,
                                          expansion_factor=expansion_factor,
                                          kernel_sizes=kernel_sizes)
        self.encoder2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1,
                                          expansion_factor=expansion_factor,
                                          kernel_sizes=kernel_sizes)
        self.encoder3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1,
                                          expansion_factor=expansion_factor,
                                          kernel_sizes=kernel_sizes)
        self.encoder4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1,
                                          expansion_factor=expansion_factor,
                                          kernel_sizes=kernel_sizes)
        self.encoder5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], 1,
                                          expansion_factor=expansion_factor,
                                          kernel_sizes=kernel_sizes)

        self.ag1 = GroupedAttentionGate(channels[3], channels[3], channels[3] // 2,
                                        kernel_size=gag_kernel, groups=channels[3] // 2)
        self.ag2 = GroupedAttentionGate(channels[2], channels[2], channels[2] // 2,
                                        kernel_size=gag_kernel, groups=channels[2] // 2)
        self.ag3 = GroupedAttentionGate(channels[1], channels[1], channels[1] // 2,
                                        kernel_size=gag_kernel, groups=channels[1] // 2)
        self.ag4 = GroupedAttentionGate(channels[0], channels[0], channels[0] // 2,
                                        kernel_size=gag_kernel, groups=channels[0] // 2)

        self.decoder1 = mk_irb_bottleneck(channels[4], channels[3], 1, 1,
                                          expansion_factor=expansion_factor,
                                          kernel_sizes=kernel_sizes)
        self.decoder2 = mk_irb_bottleneck(channels[3], channels[2], 1, 1,
                                          expansion_factor=expansion_factor,
                                          kernel_sizes=kernel_sizes)
        self.decoder3 = mk_irb_bottleneck(channels[2], channels[1], 1, 1,
                                          expansion_factor=expansion_factor,
                                          kernel_sizes=kernel_sizes)
        self.decoder4 = mk_irb_bottleneck(channels[1], channels[0], 1, 1,
                                          expansion_factor=expansion_factor,
                                          kernel_sizes=kernel_sizes)
        self.decoder5 = mk_irb_bottleneck(channels[0], channels[0], 1, 1,
                                          expansion_factor=expansion_factor,
                                          kernel_sizes=kernel_sizes)

        self.ca1 = ChannelAttention(channels[4], ratio=16)
        self.ca2 = ChannelAttention(channels[3], ratio=16)
        self.ca3 = ChannelAttention(channels[2], ratio=16)
        self.ca4 = ChannelAttention(channels[1], ratio=8)
        self.ca5 = ChannelAttention(channels[0], ratio=4)
        self.sa = SpatialAttention()
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        out = F.max_pool2d(self.encoder1(x), 2, 2)
        t1 = out
        out = F.max_pool2d(self.encoder2(out), 2, 2)
        t2 = out
        out = F.max_pool2d(self.encoder3(out), 2, 2)
        t3 = out
        out = F.max_pool2d(self.encoder4(out), 2, 2)
        t4 = out
        out = F.max_pool2d(self.encoder5(out), 2, 2)

        out = self.ca1(out) * out
        out = self.sa(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        t4 = self.ag1(g=out, x=t4)
        out = torch.add(out, t4)

        out = self.ca2(out) * out
        out = self.sa(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        t3 = self.ag2(g=out, x=t3)
        out = torch.add(out, t3)

        out = self.ca3(out) * out
        out = self.sa(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        t2 = self.ag3(g=out, x=t2)
        out = torch.add(out, t2)

        out = self.ca4(out) * out
        out = self.sa(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        t1 = self.ag4(g=out, x=t1)
        out = torch.add(out, t1)

        out = self.ca5(out) * out
        out = self.sa(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        return self.out4(out)


@HEADS.register_module()
class MKUNet(BaseDecodeHead):
    """MK-UNet full segmentation model registered as mmseg decode head."""

    def __init__(self, variant='default', in_channels=3, **kwargs):
        super().__init__(**kwargs)
        if variant == 'small':
            channels = (8, 16, 32, 48, 80)
        elif variant == 'tiny':
            channels = (4, 8, 16, 24, 32)
        else:
            channels = (16, 32, 64, 96, 160)
        self.net = MKUNetBackbone(
            num_classes=self.num_classes,
            in_channels=in_channels,
            channels=channels)

    def forward(self, inputs):
        out = self.net(inputs)
        if out.shape[2:] != inputs.shape[2:]:
            out = F.interpolate(
                out, size=inputs.shape[2:], mode='bilinear', align_corners=False)
        return out
