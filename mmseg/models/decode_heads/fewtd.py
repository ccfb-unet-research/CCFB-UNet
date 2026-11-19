"""FEWTD: Frequency-Enhanced Wavelet Transformer Decoder (plug-and-play decoder).

Reference: https://github.com/yongchangxu/FEWTD
"""
import numbers

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Function

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


# ---------------------------------------------------------------------------
# Wavelet transform (IWT_DWT)
# ---------------------------------------------------------------------------
class DWTFunction(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        h, w = x.shape[2], x.shape[3]
        pad_h = 1 if h % 2 != 0 else 0
        pad_w = 1 if w % 2 != 0 else 0
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        ctx.pad = (pad_h, pad_w)
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            b, c, h, w = ctx.shape
            dx = dx.view(b, 4, -1, h // 2, w // 2).transpose(1, 2).reshape(b, -1, h // 2, w // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(c, 1, 1, 1)
            dx = F.conv_transpose2d(dx, filters, stride=2, groups=c)
            pad_h, pad_w = ctx.pad
            dx = dx[:, :, :h - pad_h, :w - pad_w]
            return dx, None, None, None, None
        return None, None, None, None, None


class DWT2D(nn.Module):
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        return DWTFunction.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class IDWTFunction(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        b, _, h, w = x.shape
        # C is per-subband channels; must read before reshape (4*C total in x).
        x = x.view(b, 4, -1, h, w).transpose(1, 2)
        c = x.shape[1]
        x = x.reshape(b, -1, h, w)
        filters = filters.repeat(c, 1, 1, 1)
        return F.conv_transpose2d(x, filters, stride=2, groups=c)

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            b, c, h, w = ctx.shape
            c = c // 4
            dx = dx.contiguous()
            x_ll = F.conv2d(dx, w_ll.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            x_lh = F.conv2d(dx, w_lh.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            x_hl = F.conv2d(dx, w_hl.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            x_hh = F.conv2d(dx, w_hh.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1), None
        return None, None


class IDWT2D(nn.Module):
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
        rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
        filters = torch.cat([
            w_ll.unsqueeze(0).unsqueeze(0),
            w_lh.unsqueeze(0).unsqueeze(0),
            w_hl.unsqueeze(0).unsqueeze(0),
            w_hh.unsqueeze(0).unsqueeze(0),
        ], dim=0)
        self.register_buffer('filters', filters)

    def forward(self, x):
        return IDWTFunction.apply(x, self.filters)


# ---------------------------------------------------------------------------
# FSA
# ---------------------------------------------------------------------------
class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        return self.fusion(torch.cat([x1, x2], dim=1))


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.sigmoid(self.fc2(self.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)))))
        return x * y


def _align_spatial(x, target_h, target_w):
    """Crop or resize feature map to target spatial size."""
    h, w = x.shape[2], x.shape[3]
    if h == target_h and w == target_w:
        return x
    if h >= target_h and w >= target_w:
        return x[:, :, :target_h, :target_w]
    return F.interpolate(
        x, size=(target_h, target_w), mode='bilinear', align_corners=False)


class FSA(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels * 2, out_channels * 2, 1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.gelu = nn.GELU()
        self.se_real = SEBlock(in_channels)
        self.se_imag = SEBlock(in_channels)
        self.sigma = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ret_ffted = ffted
        ffted = self.gelu(self.bn(self.conv_layer(ffted)))
        ffted = ret_ffted + ffted
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        real = self.se_real(ffted[..., 0])
        imag = self.se_imag(ffted[..., 1])
        real = torch.unsqueeze(real, dim=-1)
        imag = torch.unsqueeze(imag, dim=-1)
        ffted = torch.cat((real, imag), dim=-1)
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        output = _align_spatial(output, h, w)
        return x + output * self.sigma


# ---------------------------------------------------------------------------
# Cross-attention & MDWA
# ---------------------------------------------------------------------------
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.26, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, a, b):
        _, c, h, w = a.shape
        q = self.qkv_dwconv(self.qkv(a))[:, :c]
        k = self.qkv_dwconv(self.qkv(b))[:, c:2 * c]
        v = self.qkv_dwconv(self.qkv(b))[:, 2 * c:]
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.26, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = CrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, a, b):
        a = a + self.attn(self.norm1(a), self.norm1(b))
        return a + self.ffn(self.norm2(a))


class ThreeFeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3):
        return self.fusion(torch.cat([x1, x2, x3], dim=1))


class MSDSA(nn.Module):
    def __init__(self, in_channels, n_heads=2, offset_range_factor=2.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = in_channels // n_heads
        self.scale = self.head_dim ** -0.5
        self.offset_range_factor = offset_range_factor
        self.offset_weights = nn.Parameter(torch.ones(3))
        self.offset_conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                      groups=in_channels, bias=False)
        self.offset_conv5 = nn.Conv2d(in_channels, in_channels, 5, padding=2,
                                      groups=in_channels, bias=False)
        self.offset_conv7 = nn.Conv2d(in_channels, in_channels, 7, padding=3,
                                      groups=in_channels, bias=False)
        self.offset_proj = nn.Conv2d(in_channels, 2, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.proj_q = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.proj_k = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.proj_v = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.beta = nn.Parameter(torch.tensor(0.5))
        self._dist_cache = {}

    def _get_manhattan_dist(self, h, w, device):
        key = (h, w, device)
        if key not in self._dist_cache:
            y = torch.arange(h, device=device)
            x = torch.arange(w, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            grid = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)
            dist = torch.abs(grid.unsqueeze(1) - grid.unsqueeze(0)).sum(-1)
            self._dist_cache[key] = dist
        return self._dist_cache[key]

    def _get_ref_points(self, h, w, b, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, device=device),
            torch.linspace(0.5, w - 0.5, w, device=device),
            indexing='ij')
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(w - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(h - 1.0).mul_(2.0).sub_(1.0)
        return ref[None, ...].expand(b, -1, -1, -1)

    def forward(self, x):
        b, c, h, w = x.size()
        device = x.device
        off_feat = (self.offset_conv3(x) * self.offset_weights[0]
                    + self.offset_conv5(x) * self.offset_weights[1]
                    + self.offset_conv7(x) * self.offset_weights[2])
        offset = rearrange(self.offset_proj(self.relu(off_feat)), 'b c h w -> b h w c')
        ref_points = self._get_ref_points(h, w, b, device)
        sampling_grid = (ref_points + offset.tanh() * self.offset_range_factor).clamp(-1, 1)
        x_sampled = F.grid_sample(
            x, sampling_grid[..., [1, 0]], mode='bilinear', align_corners=True)
        dist_matrix = self._get_manhattan_dist(h, w, device)
        decay_matrix = torch.exp(-dist_matrix / (self.beta.abs() + 1e-6))
        q = self.proj_q(x).view(b * self.n_heads, self.head_dim, h * w)
        k = self.proj_k(x_sampled).view(b * self.n_heads, self.head_dim, h * w)
        v = self.proj_v(x_sampled).view(b * self.n_heads, self.head_dim, h * w)
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = (attn + decay_matrix.unsqueeze(0)).softmax(dim=-1)
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = rearrange(out, '(b nh) d (height width) -> b (nh d) height width',
                        b=b, nh=self.n_heads, height=h, width=w)
        return self.proj_out(out)


class MDWA(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super().__init__()
        self.msdsa = MSDSA(in_channels * 4, n_heads=num_heads)
        self.cross_attention = AttentionBlock(
            dim=in_channels, num_heads=num_heads, bias=False)
        self.fusion_high = ThreeFeatureFusion(in_channels)
        self.alphas = nn.Parameter(torch.ones(2) * 0.5)

    def forward(self, ll, lh, hl, hh):
        combined_in = torch.cat([ll, lh, hl, hh], dim=1)
        feat_msdsa = self.msdsa(combined_in)
        out_lh = self.cross_attention(lh, ll)
        out_hl = self.cross_attention(hl, ll)
        out_hh = self.cross_attention(hh, ll)
        out_ll = self.cross_attention(ll, self.fusion_high(lh, hl, hh))
        feat_dsca = torch.cat([out_ll, out_lh, out_hl, out_hh], dim=1)
        fused = self.alphas[0] * feat_msdsa + self.alphas[1] * feat_dsca
        return torch.chunk(fused, 4, dim=1)


class FEWTDModule(nn.Module):
    def __init__(self, in_channels, wavelet_type='haar', levels=1, num_heads=2):
        super().__init__()
        self.levels = levels
        self.dwt = DWT2D(wavelet_type)
        self.idwt = IDWT2D(wavelet_type)
        self.fsa = FSA(in_channels, in_channels)
        self.mdwa_l1 = MDWA(in_channels, num_heads=num_heads)
        if levels >= 2:
            self.mdwa_l2 = MDWA(in_channels, num_heads=num_heads)
        else:
            self.mdwa_l2 = None
        self.smooth_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        identity = x
        _, c, h, w = x.shape
        x_l1 = self.dwt(x)
        ll1, lh1, hl1, hh1 = x_l1.split(c, 1)
        ll1, lh1, hl1, hh1 = self.mdwa_l1(ll1, lh1, hl1, hh1)
        if self.levels >= 2 and self.mdwa_l2 is not None:
            x_l2 = self.dwt(ll1)
            ll2, lh2, hl2, hh2 = x_l2.split(c, 1)
            ll2, lh2, hl2, hh2 = self.mdwa_l2(ll2, lh2, hl2, hh2)
            f2 = self.idwt(torch.cat([ll2, lh2, hl2, hh2], dim=1))
            f2 = _align_spatial(f2, ll1.shape[2], ll1.shape[3])
            ll1 = ll1 + f2
        f1 = self.idwt(torch.cat([ll1, lh1, hl1, hh1], dim=1))
        f1 = _align_spatial(f1, h, w)
        f1 = self.fsa(f1)
        f1 = _align_spatial(f1, h, w)
        return self.smooth_conv(identity) + f1


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)


class FEWTDDecoder(nn.Module):
    def __init__(self, channels=(256, 160, 64, 32), wavelet_levels=1, num_heads=2):
        super().__init__()
        c4, c3, c2, c1 = channels
        self.up1 = UpConv(c4, c3)
        self.up2 = UpConv(c3, c2)
        self.up3 = UpConv(c2, c1)
        self.db_conv1 = DoubleConv(c3, c3)
        self.db_conv2 = DoubleConv(c2, c2)
        self.db_conv3 = DoubleConv(c1, c1)
        wt_kw = dict(wavelet_type='haar', levels=wavelet_levels, num_heads=num_heads)
        self.fewtd1 = FEWTDModule(c2, **wt_kw)
        self.fewtd2 = FEWTDModule(c1, **wt_kw)
        self.fewtd3 = FEWTDModule(c3, **wt_kw)
        self.fewtd4 = FEWTDModule(c4, **wt_kw)
        self.channels = channels

    def forward(self, x, skips):
        d4 = self.fewtd4(x)
        d3 = self.up1(d4)
        d3 = self.fewtd3(self.db_conv1(d3 + skips[0]))
        d2 = self.fewtd1(self.db_conv2(self.up2(d3) + skips[1]))
        d1 = self.fewtd2(self.db_conv3(self.up3(d2) + skips[2]))
        return [d4, d3, d2, d1]


class FEWTDBackbone(nn.Module):
    """PVT encoder + FEWTD decoder."""

    def __init__(self, num_classes, encoder='pvt_v2_b0', pretrained_pvt=None,
                 wavelet_levels=1, attn_heads=2):
        super().__init__()
        if encoder not in PVT_ENCODER_CFG:
            encoder = 'pvt_v2_b0'
        enc_cfg = PVT_ENCODER_CFG[encoder]
        self.backbone = enc_cfg['factory']()
        channels = enc_cfg['channels']
        if pretrained_pvt is not None:
            state = torch.load(pretrained_pvt, map_location='cpu')
            model_dict = self.backbone.state_dict()
            state = {k: v for k, v in state.items() if k in model_dict}
            model_dict.update(state)
            self.backbone.load_state_dict(model_dict)
        self.decoder = FEWTDDecoder(
            channels, wavelet_levels=wavelet_levels, num_heads=attn_heads)
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
class FEWTDHead(BaseDecodeHead):
    """FEWTD with lightweight PVT-B0 backbone by default."""

    def __init__(self,
                 encoder='pvt_v2_b0',
                 wavelet_levels=1,
                 attn_heads=2,
                 pretrained_pvt=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.net = FEWTDBackbone(
            self.num_classes,
            encoder=encoder,
            pretrained_pvt=pretrained_pvt,
            wavelet_levels=wavelet_levels,
            attn_heads=attn_heads)

    def forward(self, inputs):
        return self.net(inputs)
