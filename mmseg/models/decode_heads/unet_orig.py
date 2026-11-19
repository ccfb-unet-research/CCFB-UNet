# Copyright (c) OpenMMLab. All rights reserved.
"""CCFB-UNet v2: dual-encoder U-Net with three independently switchable innovations.

Innovations (config switches):
  1) use_fd_kan      — multi-stage frequency-decoupled KAN enhancement
  2) use_mgd_encoder — pretrained PVTv2 auxiliary encoder + geometric dual fusion
  3) use_gls_cams    — global-local cross-scale skip/decoder semantic fusion

Base (all False) remains a standard U-Net for fair comparison.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .pvt import CFM, BasicConv2d, pvt_v2_b0, pvt_v2_b2

DEFAULT_PVT_B2 = ''
DEFAULT_PVT_B0 = ''

PVT_CFG = {
    'pvt_v2_b0': dict(factory=pvt_v2_b0, channels=(32, 64, 160, 256)),
    'pvt_v2_b2': dict(factory=pvt_v2_b2, channels=(64, 128, 320, 512)),
}


def _channel_shuffle(x, groups):
  b, c, h, w = x.shape
  x = x.view(b, groups, c // groups, h, w)
  x = x.transpose(1, 2).contiguous().view(b, c, h, w)
  return x


def _coord_map(x):
  b, _, h, w = x.shape
  yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
  xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
  yy = yy.view(1, 1, h, 1).expand(b, 1, h, w)
  xx = xx.view(1, 1, 1, w).expand(b, 1, h, w)
  return torch.cat([x, xx, yy], dim=1)


def _dwt(x):
  a, b = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
  c, d = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
  ll = (a + b + c + d) * 0.5
  lh = (a - b + c - d) * 0.5
  hl = (a + b - c - d) * 0.5
  hh = (a - b - c + d) * 0.5
  return ll, lh, hl, hh


def _idwt(ll, lh, hl, hh):
  b, c, h, w = ll.shape
  out = ll.new_zeros(b, c, h * 2, w * 2)
  out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
  out[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
  out[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
  out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
  return out


# ---------------------------------------------------------------------------
# Classical domain-generalization baselines (independent of CCFB modules)
# ---------------------------------------------------------------------------

class IBN2d(nn.Module):
  """IBN-a: instance norm on half the channels, batch norm on the rest.

  IBN is only used in the shallow U-Net encoder blocks.  The module has no
  dependency beyond PyTorch and is disabled by default, so old checkpoints
  and configs keep exactly the original architecture.
  """

  def __init__(self, channels, ratio=0.5):
    super().__init__()
    self.in_channels = int(round(channels * ratio))
    self.bn_channels = channels - self.in_channels
    self.instance_norm = nn.InstanceNorm2d(
        self.in_channels, affine=True, track_running_stats=False)
    self.batch_norm = nn.BatchNorm2d(self.bn_channels)

  def forward(self, x):
    x_in, x_bn = torch.split(
        x, [self.in_channels, self.bn_channels], dim=1)
    return torch.cat(
        [self.instance_norm(x_in), self.batch_norm(x_bn)], dim=1)


class MixStyle(nn.Module):
  """MixStyle feature-statistic mixing (Zhou et al., ICLR 2021).

  Applied only during training.  Content and labels remain paired, while
  channel-wise feature mean/std are mixed across samples in the mini-batch.
  """

  def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
    super().__init__()
    self.p = float(p)
    self.alpha = float(alpha)
    self.eps = float(eps)

  def forward(self, x):
    if not self.training or self.p <= 0.0:
      return x
    if torch.rand((), device=x.device) > self.p or x.size(0) < 2:
      return x

    mu = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    sig = (var + self.eps).sqrt()
    mu, sig = mu.detach(), sig.detach()
    x_norm = (x - mu) / sig

    # Beta(alpha, alpha) is the mixing distribution used in MixStyle.
    beta = torch.distributions.Beta(self.alpha, self.alpha)
    lam = beta.sample((x.size(0), 1, 1, 1)).to(
        device=x.device, dtype=x.dtype)
    perm = torch.randperm(x.size(0), device=x.device)
    mu_mix = mu * lam + mu[perm] * (1.0 - lam)
    sig_mix = sig * lam + sig[perm] * (1.0 - lam)
    return x_norm * sig_mix + mu_mix


class DoubleConv(nn.Module):
  """Standard U-Net double conv block."""

  def __init__(self, in_ch, out_ch, use_ibn=False):
    super().__init__()
    norm = IBN2d if use_ibn else nn.BatchNorm2d
    self.net = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        norm(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        norm(out_ch),
        nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.net(x)


class Down(nn.Module):
  def __init__(self, in_ch, out_ch, use_ibn=False):
    super().__init__()
    self.pool = nn.MaxPool2d(2)
    self.conv = DoubleConv(in_ch, out_ch, use_ibn=use_ibn)

  def forward(self, x):
    return self.conv(self.pool(x))


class Up(nn.Module):
  def __init__(self, in_ch, out_ch, skip_ch):
    super().__init__()
    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.conv = DoubleConv(in_ch + skip_ch, out_ch)

  def forward(self, x, skip):
    x = self.up(x)
    if x.shape[2:] != skip.shape[2:]:
      x = F.interpolate(
          x, size=skip.shape[2:], mode='bilinear', align_corners=True)
    return self.conv(torch.cat([skip, x], dim=1))


# ---------------------------------------------------------------------------
# Innovation 1 — FD-KAN: frequency-decoupled KAN (stem + early encoder)
# ---------------------------------------------------------------------------

class KANConvBlock(nn.Module):
  """Lightweight KAN-style block with residual zero-init output."""

  def __init__(self, ch):
    super().__init__()
    self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
    self.norm = nn.GroupNorm(min(8, ch), ch)
    self.expand = nn.Conv2d(ch, ch * 2, 1, bias=False)
    self.proj = nn.Conv2d(ch, ch, 1, bias=False)
    nn.init.zeros_(self.proj.weight)

  def forward(self, x):
    h = self.norm(self.dw(x))
    h = self.expand(h)
    a, b = h.chunk(2, dim=1)
    h = a * torch.sigmoid(b)
    return x + self.proj(h)


class CrossFreqAttn(nn.Module):
  """Path B: LL-guided channel recalibration on LH/HL bands."""

  def __init__(self, ch):
    super().__init__()
    mid_ch = ch * 2
    hidden = max(ch // 2, 8)
    self.ll_gate = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(ch, hidden, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, mid_ch, 1, bias=False),
        nn.Sigmoid(),
    )
    self.mid_refine = nn.Sequential(
        nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
    )
    self.proj = nn.Conv2d(mid_ch, mid_ch, 1, bias=False)
    nn.init.zeros_(self.proj.weight)

  def forward(self, ll, mid):
    gated = mid * self.ll_gate(ll)
    return mid + self.proj(self.mid_refine(gated))


class MultiHeadHighFreq(nn.Module):
  """Path C: grouped signal/noise heads on HH."""

  def __init__(self, ch, heads=4):
    super().__init__()
    assert ch % heads == 0
    head_ch = ch // heads
    self.heads = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(head_ch, head_ch, 3, padding=1, groups=head_ch, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.Sigmoid(),
        ) for _ in range(heads)
    ])
    self.noise = nn.Conv2d(ch, ch, 1, bias=True)
    nn.init.constant_(self.noise.weight, 0.0)
    nn.init.constant_(self.noise.bias, 2.0)

  def forward(self, hh):
    parts = hh.chunk(len(self.heads), dim=1)
    sig = torch.cat([h(p) * p for h, p in zip(self.heads, parts)], dim=1)
    gate = torch.sigmoid(self.noise(hh))
    return hh * gate + sig * (1.0 - gate)


class FDKANPreprocessor(nn.Module):
  """DWT tri-path enhancement with identity-safe residual fusion."""

  def __init__(self, in_ch, feat_ch=32, strength=1.0):
    super().__init__()
    self.strength = strength
    self.in_proj = nn.Conv2d(in_ch, feat_ch, 1, bias=False)
    self.out_proj = nn.Conv2d(feat_ch, in_ch, 1, bias=False)
    nn.init.zeros_(self.out_proj.weight)
    self.ll_path = nn.Sequential(KANConvBlock(feat_ch), KANConvBlock(feat_ch))
    self.mid_attn = CrossFreqAttn(feat_ch)
    self.hh_path = MultiHeadHighFreq(feat_ch)
    self.out_norm = nn.BatchNorm2d(feat_ch)
    # Near-zero residual at init (sigmoid(-4)≈0.018) so warm-start ≈ base.
    self.gate = nn.Parameter(torch.tensor(-4.0))

  def forward(self, x):
    z = self.in_proj(x)
    ph, pw = z.shape[2] % 2, z.shape[3] % 2
    if ph or pw:
      z = F.pad(z, (0, pw, 0, ph))
    ll, lh, hl, hh = _dwt(z)
    ll = self.ll_path(ll)
    mid = self.mid_attn(ll, torch.cat([lh, hl], dim=1))
    lh, hl = mid.chunk(2, dim=1)
    hh = self.hh_path(hh)
    enhanced = _idwt(ll, lh, hl, hh)
    if ph or pw:
      enhanced = enhanced[:, :, :enhanced.shape[2] - ph, :enhanced.shape[3] - pw]
    enhanced = self.out_norm(enhanced)
    delta = self.out_proj(enhanced)
    g = torch.sigmoid(self.gate) * self.strength
    return x + g * delta


class StageFreqEnhance(nn.Module):
  """Encoder-stage frequency refinement (expands FD beyond stem)."""

  def __init__(self, ch, strength=1.0):
    super().__init__()
    self.strength = strength
    feat = max(ch // 2, 16)
    # keep divisible by 4 for MultiHeadHighFreq
    if feat % 4 != 0:
      feat = feat + (4 - feat % 4)
    self.in_proj = nn.Conv2d(ch, feat, 1, bias=False)
    self.out_proj = nn.Conv2d(feat, ch, 1, bias=False)
    nn.init.zeros_(self.out_proj.weight)
    self.ll_path = KANConvBlock(feat)
    self.mid_attn = CrossFreqAttn(feat)
    self.hh_path = MultiHeadHighFreq(feat)
    self.alpha = nn.Parameter(torch.tensor(0.02))

  def forward(self, x):
    z = self.in_proj(x)
    ph, pw = z.shape[2] % 2, z.shape[3] % 2
    if ph or pw:
      z = F.pad(z, (0, pw, 0, ph))
    ll, lh, hl, hh = _dwt(z)
    ll = self.ll_path(ll)
    mid = self.mid_attn(ll, torch.cat([lh, hl], dim=1))
    lh, hl = mid.chunk(2, dim=1)
    hh = self.hh_path(hh)
    y = _idwt(ll, lh, hl, hh)
    if ph or pw:
      y = y[:, :, :y.shape[2] - ph, :y.shape[3] - pw]
    return x + self.alpha * self.strength * self.out_proj(y)


class FreqCrossGate(nn.Module):
  """Cross-gate between CNN early feature and PVT shallow feature."""

  def __init__(self, cnn_ch, pvt_ch, strength=1.0):
    super().__init__()
    self.strength = strength
    self.pvt_proj = nn.Conv2d(pvt_ch, cnn_ch, 1, bias=False)
    self.fuse = nn.Sequential(
        nn.Conv2d(cnn_ch * 2, cnn_ch, 1, bias=False),
        nn.BatchNorm2d(cnn_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(cnn_ch, cnn_ch, 3, padding=1, bias=False),
    )
    nn.init.zeros_(self.fuse[-1].weight)
    self.gate = nn.Parameter(torch.tensor(0.03))

  def forward(self, cnn, pvt):
    p = F.interpolate(
        self.pvt_proj(pvt), size=cnn.shape[2:], mode='bilinear',
        align_corners=True)
    delta = self.fuse(torch.cat([cnn, p], dim=1))
    return cnn + self.gate * self.strength * torch.tanh(delta)


# ---------------------------------------------------------------------------
# Innovation 2 — MGD-Encoder: pretrained PVT aux + geometric dual fusion
# ---------------------------------------------------------------------------

class CoordSpatialBranch(nn.Module):
  def __init__(self, ch):
    super().__init__()
    self.enc = nn.Sequential(
        nn.Conv2d(ch + 2, ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),
    )
    self.proj = nn.Conv2d(ch, ch, 1, bias=False)
    nn.init.zeros_(self.proj.weight)

  def forward(self, x):
    return x + self.proj(self.enc(_coord_map(x)))


class TaylorAttnModule(nn.Module):
  """Taylor approx attention: Attn ~ (I + S + 0.5 S^2) V."""

  def __init__(self, ch, reduction=8):
    super().__init__()
    mid = max(ch // reduction, 8)
    self.theta = nn.Conv2d(ch, mid, 1, bias=False)
    self.phi = nn.Conv2d(ch, mid, 1, bias=False)
    self.g = nn.Conv2d(ch, ch // 2, 1, bias=False)
    self.out = nn.Conv2d(ch // 2, ch, 1, bias=False)
    nn.init.zeros_(self.out.weight)

  def forward(self, x):
    b, c, h, w = x.shape
    hs, ws = max(h // 4, 1), max(w // 4, 1)
    xs = F.adaptive_avg_pool2d(x, (hs, ws))
    theta = self.theta(xs).view(b, -1, hs * ws).permute(0, 2, 1)
    phi = self.phi(xs).view(b, -1, hs * ws)
    sim = torch.bmm(theta, phi)
    sim = sim + 0.5 * sim * sim
    attn = F.softmax(sim, dim=-1)
    g = self.g(xs).view(b, -1, hs * ws)
    y = torch.bmm(g, attn.permute(0, 2, 1)).view(b, -1, hs, ws)
    y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=True)
    return self.out(y)


class OrthoCollapseProject(nn.Module):
  def __init__(self, ch, groups=4):
    super().__init__()
    self.groups = groups
    self.body = nn.Sequential(
        nn.Conv2d(ch, ch, 1, groups=groups, bias=False),
        nn.GroupNorm(groups, ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 3, padding=1, groups=groups, bias=False),
        nn.BatchNorm2d(ch),
    )
    self.out = nn.Conv2d(ch, ch, 1, bias=False)
    nn.init.zeros_(self.out.weight)

  def forward(self, x):
    y = _channel_shuffle(x, self.groups)
    return x + self.out(self.body(y))


class MGDEncoderBlock(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.conv = DoubleConv(in_ch, out_ch)
    self.coord = CoordSpatialBranch(out_ch)
    self.taylor = TaylorAttnModule(out_ch)
    self.ortho = OrthoCollapseProject(out_ch)

  def forward(self, x):
    x = self.conv(x)
    x = self.coord(x)
    x = x + self.taylor(x)
    return self.ortho(x)


class MGDDown(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.pool = nn.MaxPool2d(2)
    self.block = MGDEncoderBlock(in_ch, out_ch)

  def forward(self, x):
    return self.block(self.pool(x))


class MGDBottleneck(nn.Module):
  """Bridge recalibration when MGD is enabled."""

  def __init__(self, ch, strength=1.0):
    super().__init__()
    self.strength = strength
    self.coord = CoordSpatialBranch(ch)
    self.taylor = TaylorAttnModule(ch, reduction=4)
    self.out = nn.Conv2d(ch, ch, 1, bias=False)
    nn.init.zeros_(self.out.weight)

  def forward(self, x):
    y = self.coord(x)
    return x + self.strength * self.out(self.taylor(y))


class DualStreamFusion(nn.Module):
  """Identity-safe residual fusion of CNN feature with aligned PVT feature."""

  def __init__(self, cnn_ch, pvt_ch, strength=1.0, use_mgd=True, pvt_ratio=0.2):
    super().__init__()
    self.strength = strength
    self.pvt_proj = nn.Sequential(
        nn.Conv2d(pvt_ch, cnn_ch, 1, bias=False),
        nn.BatchNorm2d(cnn_ch),
        nn.ReLU(inplace=True),
    )
    self.mix = nn.Sequential(
        nn.Conv2d(cnn_ch * 2, cnn_ch, 1, bias=False),
        nn.BatchNorm2d(cnn_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(cnn_ch, cnn_ch, 3, padding=1, bias=False),
    )
    nn.init.zeros_(self.mix[-1].weight)
    self.beta = nn.Parameter(torch.tensor(float(pvt_ratio)))
    self.mgd = MGDBottleneck(cnn_ch, strength) if use_mgd else None

  def forward(self, cnn, pvt):
    p = F.interpolate(
        self.pvt_proj(pvt), size=cnn.shape[2:], mode='bilinear',
        align_corners=True)
    delta = self.mix(torch.cat([cnn, p], dim=1))
    y = cnn + self.beta * self.strength * torch.tanh(delta)
    if self.mgd is not None:
      y = self.mgd(y)
    return y


class PretrainedPVTAux(nn.Module):
  """Frozen pretrained PVTv2 auxiliary encoder with ImageNet normalization."""

  def __init__(self, encoder='pvt_v2_b2', pretrained_path=None, freeze=False):
    super().__init__()
    if encoder not in PVT_CFG:
      encoder = 'pvt_v2_b2'
    cfg = PVT_CFG[encoder]
    self.channels = cfg['channels']
    self.backbone = cfg['factory']()
    path = pretrained_path
    if path is None:
      env_name = 'PVT_B2_PATH' if encoder == 'pvt_v2_b2' else 'PVT_B0_PATH'
      default_path = DEFAULT_PVT_B2 if encoder == 'pvt_v2_b2' else DEFAULT_PVT_B0
      path = os.environ.get(env_name, default_path)
    if path and os.path.isfile(path):
      state = torch.load(path, map_location='cpu')
      if isinstance(state, dict) and 'model' in state and not any(
          k.startswith('patch_embed') for k in state):
        state = state['model']
      model_dict = self.backbone.state_dict()
      state = {
          k: v for k, v in state.items()
          if k in model_dict and model_dict[k].shape == v.shape
      }
      model_dict.update(state)
      self.backbone.load_state_dict(model_dict)
      print(f'[CCFB] Loaded PVT pretrained: {path} ({len(state)} tensors)')
    else:
      print(f'[CCFB] WARNING: PVT pretrained not found: {path}')
    self.freeze = freeze
    if freeze:
      for p in self.backbone.parameters():
        p.requires_grad = False
      self.backbone.eval()
      print('[CCFB] PVT auxiliary encoder frozen')
    self.register_buffer(
        'imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        persistent=False)
    self.register_buffer(
        'imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        persistent=False)

  def train(self, mode=True):
    super().train(mode)
    if self.freeze:
      self.backbone.eval()
    return self

  def _prep(self, x):
    if x.size(1) == 1:
      x = x.repeat(1, 3, 1, 1)
    x = x.float()
    if float(x.detach().max()) > 1.5:
      x = x / 255.0
    return (x - self.imagenet_mean) / self.imagenet_std

  def forward(self, x):
    x = self._prep(x)
    if self.freeze:
      with torch.no_grad():
        feats = self.backbone(x)
    else:
      feats = self.backbone(x)
    return feats[0], feats[1], feats[2], feats[3]


# ---------------------------------------------------------------------------
# Innovation 3 — GLS-CAMS: global-local skip + multi-scale decoder fusion
# ---------------------------------------------------------------------------

class GLSCAMSSkip(nn.Module):
  """Semantic-consistency skip refinement (identity-safe)."""

  def __init__(self, skip_ch, deep_ch, strength=1.0):
    super().__init__()
    self.strength = strength
    self.deep_proj = nn.Conv2d(deep_ch, skip_ch, 1, bias=False)
    self.fuse = nn.Conv2d(skip_ch * 2, skip_ch, 1, bias=False)
    self.refine = nn.Conv2d(skip_ch, skip_ch, 3, padding=1, bias=False)
    nn.init.zeros_(self.refine.weight)
    self.alpha = nn.Parameter(torch.tensor(0.02))

  def forward(self, skip, deep):
    deep_u = F.interpolate(
        self.deep_proj(deep),
        size=skip.shape[2:],
        mode='bilinear',
        align_corners=True)
    sim = (F.cosine_similarity(skip, deep_u, dim=1).unsqueeze(1) + 1.0) * 0.5
    attn = torch.sigmoid(self.fuse(torch.cat([skip, deep_u], dim=1)))
    delta = torch.tanh(self.refine(skip * attn * (0.55 + 0.45 * sim)))
    return skip + self.alpha * self.strength * delta


class GLSCAMSBridge(nn.Module):
  """Bottleneck bridge when GLS-CAMS is enabled."""

  def __init__(self, ch, strength=1.0):
    super().__init__()
    self.strength = strength
    self.body = nn.Sequential(
        nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),
    )
    self.out = nn.Conv2d(ch, ch, 1, bias=False)
    nn.init.zeros_(self.out.weight)
    self.alpha = nn.Parameter(torch.tensor(0.02))

  def forward(self, x):
    return x + self.alpha * self.strength * torch.tanh(self.out(self.body(x)))


class DecoderMSFusion(nn.Module):
  """Decoder-side multi-scale fusion (expands GLS beyond skip/bottleneck)."""

  def __init__(self, ch, strength=1.0):
    super().__init__()
    self.strength = strength
    self.local = nn.Conv2d(ch, ch, 3, padding=1, groups=max(1, ch // 8), bias=False)
    self.context = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(ch, ch, 1, bias=False),
        nn.Sigmoid(),
    )
    self.out = nn.Conv2d(ch, ch, 1, bias=False)
    nn.init.zeros_(self.out.weight)
    self.gamma = nn.Parameter(torch.tensor(0.02))

  def forward(self, x):
    y = self.local(x) * self.context(x)
    return x + self.gamma * self.strength * torch.tanh(self.out(y))


class PVTGuidedSkip(nn.Module):
  """Skip refinement guided by PVT semantic feature."""

  def __init__(self, skip_ch, pvt_ch, strength=1.0):
    super().__init__()
    self.strength = strength
    self.pvt_proj = nn.Conv2d(pvt_ch, skip_ch, 1, bias=False)
    self.fuse = nn.Conv2d(skip_ch * 2, skip_ch, 1, bias=False)
    self.refine = nn.Conv2d(skip_ch, skip_ch, 3, padding=1, bias=False)
    nn.init.zeros_(self.refine.weight)
    self.alpha = nn.Parameter(torch.tensor(0.05))

  def forward(self, skip, pvt):
    p = F.interpolate(
        self.pvt_proj(pvt), size=skip.shape[2:], mode='bilinear',
        align_corners=True)
    attn = torch.sigmoid(self.fuse(torch.cat([skip, p], dim=1)))
    delta = torch.tanh(self.refine(skip * attn))
    return skip + self.alpha * self.strength * delta


class CCFBFullFusion(nn.Module):
  """Exclusive tri-module fusion neck (only when FD+MGD+GLS all on)."""

  def __init__(self, c3, c4, c5, pvt_ch=None):
    super().__init__()
    in_ch = c3 + c4 + c5 + (pvt_ch if pvt_ch is not None else 0)
    self.use_pvt = pvt_ch is not None
    if self.use_pvt:
      self.pvt_proj = nn.Conv2d(pvt_ch, pvt_ch, 1, bias=False)
    self.fuse = nn.Sequential(
        nn.Conv2d(in_ch, c5, 1, bias=False),
        nn.BatchNorm2d(c5),
        nn.ReLU(inplace=True),
        nn.Conv2d(c5, c5, 3, padding=1, bias=False),
        nn.BatchNorm2d(c5),
        nn.ReLU(inplace=True),
    )
    self.out = nn.Conv2d(c5, c5, 1, bias=False)
    nn.init.zeros_(self.out.weight)
    # Start at exact identity; train residual amplitude from zero.
    self.beta = nn.Parameter(torch.tensor(0.0))

  def forward(self, e3, e4, e5, pvt4=None):
    e3u = F.interpolate(e3, size=e5.shape[2:], mode='bilinear', align_corners=True)
    e4u = F.interpolate(e4, size=e5.shape[2:], mode='bilinear', align_corners=True)
    feats = [e3u, e4u, e5]
    if self.use_pvt and pvt4 is not None:
      p = F.interpolate(
          self.pvt_proj(pvt4), size=e5.shape[2:], mode='bilinear',
          align_corners=True)
      feats.append(p)
    fused = self.fuse(torch.cat(feats, dim=1))
    return e5 + self.beta * torch.tanh(self.out(fused))


# ---------------------------------------------------------------------------
# CCFB-UNet
# ---------------------------------------------------------------------------

class CCFBUNet(nn.Module):
  """Standard U-Net with optional FD-KAN / MGD-Encoder / GLS-CAMS."""

  def __init__(
      self,
      in_channels=3,
      base_channels=32,
      use_fd_kan=False,
      use_mgd_encoder=False,
      use_gls_cams=False,
      init_fg_bias=True,
      ccfb_full_light_bridge=False,
      decoder_channels=64,
      pvt_encoder='pvt_v2_b2',
      pretrained_pvt=None,
      use_ibn=False,
      use_mixstyle=False,
      mixstyle_p=0.5,
      mixstyle_alpha=0.1,
      use_pvt_control=False,
  ):
    super().__init__()
    c1, c2, c3, c4, c5 = (
        base_channels,
        base_channels * 2,
        base_channels * 4,
        base_channels * 8,
        base_channels * 8,
    )

    self.use_fd_kan = use_fd_kan
    self.use_mgd = use_mgd_encoder
    self.use_gls = use_gls_cams
    n_mods = int(use_fd_kan) + int(use_mgd_encoder) + int(use_gls_cams)
    self.use_full = n_mods == 3
    self.light_bridge = ccfb_full_light_bridge and self.use_full
    # Any innovation enables pretrained PVT aux (dual-encoder framework);
    # base (all flags False) remains pure CNN U-Net.
    self.use_pvt = n_mods >= 1 or bool(use_pvt_control)
    self.use_ibn = bool(use_ibn)
    self.use_mixstyle = bool(use_mixstyle)
    self.strength = 1.0
    if pretrained_pvt is None:
      env_name = (
          'PVT_B2_PATH' if pvt_encoder == 'pvt_v2_b2' else 'PVT_B0_PATH')
      default_path = (
          DEFAULT_PVT_B2 if pvt_encoder == 'pvt_v2_b2' else DEFAULT_PVT_B0)
      pretrained_pvt = os.environ.get(env_name, default_path)

    self.fd_kan = (
        FDKANPreprocessor(in_channels, strength=self.strength)
        if use_fd_kan else None)
    # IBN-a is intentionally limited to shallow stages, as in IBN-Net.
    self.inc = DoubleConv(in_channels, c1, use_ibn=self.use_ibn)

    self.down1 = Down(c1, c2, use_ibn=self.use_ibn)
    self.down2 = Down(c2, c3, use_ibn=self.use_ibn)
    self.mixstyle1 = (
        MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        if self.use_mixstyle else None)
    self.mixstyle2 = (
        MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        if self.use_mixstyle else None)
    self.mixstyle3 = (
        MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        if self.use_mixstyle else None)

    # Stage-wise FD on early/mid encoder (not bottleneck-only).
    if use_fd_kan:
      self.fd_e1 = StageFreqEnhance(c1, self.strength)
      self.fd_e2 = StageFreqEnhance(c2, self.strength)
      self.fd_e3 = StageFreqEnhance(c3, self.strength)
    else:
      self.fd_e1 = self.fd_e2 = self.fd_e3 = None

    if use_mgd_encoder:
      # Keep base Down/Down whenever warm-start matters (full + single-MGD);
      # MGD capacity sits in residual bottleneck / dual-fusion / MGDDown only
      # for 2-module combos without requiring a different mid trunk.
      if self.use_full or n_mods == 1:
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)
      else:
        self.down3 = MGDDown(c3, c4)
        self.down4 = MGDDown(c4, c5)
      self.mgd_bn = MGDBottleneck(c5, self.strength)
    else:
      self.down3 = Down(c3, c4)
      self.down4 = Down(c4, c5)
      self.mgd_bn = None

    if self.use_pvt:
      self.pvt_aux = PretrainedPVTAux(
          encoder=pvt_encoder, pretrained_path=pretrained_pvt, freeze=False)
      pc = self.pvt_aux.channels  # (s1,s2,s3,s4)
      self.fd_pvt_gate2 = (
          FreqCrossGate(c2, pc[0], self.strength) if use_fd_kan else None)
      self.fd_pvt_gate3 = (
          FreqCrossGate(c3, pc[1], self.strength) if use_fd_kan else None)
      # Keep mid CNN stages clean; fuse PVT only at bottleneck (identity-safe).
      self.dual_f3 = None
      self.dual_f4 = None
      self.dual_f5 = DualStreamFusion(
          c5, pc[3], self.strength, use_mgd=use_mgd_encoder, pvt_ratio=0.02)
    else:
      self.pvt_aux = None
      self.fd_pvt_gate2 = self.fd_pvt_gate3 = None
      self.dual_f3 = self.dual_f4 = self.dual_f5 = None
      pc = None

    if use_gls_cams:
      self.gls3 = GLSCAMSSkip(c4, c5, self.strength)
      self.gls2 = GLSCAMSSkip(c3, c4, self.strength)
      # Deeper skip refinement only in the full model for stability.
      self.gls1 = GLSCAMSSkip(c2, c3, self.strength) if self.use_full else None
      self.gls0 = GLSCAMSSkip(c1, c2, self.strength) if self.use_full else None
      self.gls_bridge = GLSCAMSBridge(c5, self.strength)
      self.dec_ms4 = DecoderMSFusion(c4, self.strength)
      self.dec_ms3 = DecoderMSFusion(c3, self.strength)
      self.dec_ms2 = DecoderMSFusion(c2, self.strength) if self.use_full else None
      if self.use_pvt:
        self.pvt_skip3 = PVTGuidedSkip(c4, pc[2], self.strength)
        self.pvt_skip2 = PVTGuidedSkip(c3, pc[1], self.strength)
        self.pvt_skip1 = (
            PVTGuidedSkip(c2, pc[0], self.strength) if self.use_full else None)
      else:
        self.pvt_skip3 = self.pvt_skip2 = self.pvt_skip1 = None
    else:
      self.gls3 = self.gls2 = self.gls1 = self.gls0 = None
      self.gls_bridge = None
      self.dec_ms4 = self.dec_ms3 = self.dec_ms2 = None
      self.pvt_skip3 = self.pvt_skip2 = self.pvt_skip1 = None

    self.up1 = Up(c5, c4, c4)
    self.up2 = Up(c4, c3, c3)
    self.up3 = Up(c3, c2, c2)
    self.up4 = Up(c2, c1, c1)
    self.head = nn.Sequential(
        nn.Conv2d(c1, decoder_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(decoder_channels),
        nn.ReLU(inplace=True),
    )

    if self.use_full:
      self.full_fusion = CCFBFullFusion(
          c3, c4, c5, pvt_ch=pc[3] if pc is not None else None)
      # Lightweight CFM branch (zero-init) for full-model synergy only.
      if pc is not None:
        cfm_ch = max(32, base_channels)
        self.cfm_t2 = BasicConv2d(pc[1], cfm_ch, 1)
        self.cfm_t3 = BasicConv2d(pc[2], cfm_ch, 1)
        self.cfm_t4 = BasicConv2d(pc[3], cfm_ch, 1)
        self.cfm = CFM(cfm_ch)
        self.cfm_to_unet = nn.Sequential(
            nn.Conv2d(cfm_ch, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 1, bias=False),
        )
        nn.init.zeros_(self.cfm_to_unet[-1].weight)
        self.cfm_gate = nn.Parameter(torch.tensor(0.0))
      else:
        self.cfm = None
    else:
      self.full_fusion = None
      self.cfm = None

  def _refine_skip(self, skip, deep, gls):
    if gls is None:
      return skip
    return gls(skip, deep)

  def forward(self, x):
    p1 = p2 = p3 = p4 = None
    if self.use_pvt:
      p1, p2, p3, p4 = self.pvt_aux(x)

    if self.use_fd_kan:
      x = self.fd_kan(x)

    e1 = self.inc(x)
    if self.fd_e1 is not None:
      e1 = self.fd_e1(e1)
    if self.mixstyle1 is not None:
      e1 = self.mixstyle1(e1)

    e2 = self.down1(e1)
    if self.fd_e2 is not None:
      e2 = self.fd_e2(e2)
    if self.mixstyle2 is not None:
      e2 = self.mixstyle2(e2)
    # Keep shallow CNN path dominant; only light FD cross-gate on e2.
    if self.fd_pvt_gate2 is not None:
      e2 = self.fd_pvt_gate2(e2, p1)

    e3 = self.down2(e2)
    if self.fd_e3 is not None:
      e3 = self.fd_e3(e3)
    if self.mixstyle3 is not None:
      e3 = self.mixstyle3(e3)
    if self.fd_pvt_gate3 is not None:
      e3 = self.fd_pvt_gate3(e3, p2)
    # Mid/deep dual fusion with pretrained PVT semantics.
    if self.dual_f3 is not None:
      e3 = self.dual_f3(e3, p2)

    e4 = self.down3(e3)
    if self.dual_f4 is not None:
      e4 = self.dual_f4(e4, p3)

    e5 = self.down4(e4)
    if self.dual_f5 is not None:
      e5 = self.dual_f5(e5, p4)

    # Full model keeps every enabled module, then adds exclusive residual fusion.
    if self.use_mgd and self.mgd_bn is not None:
      e5 = self.mgd_bn(e5)
    if self.use_gls and self.gls_bridge is not None and not self.light_bridge:
      e5 = self.gls_bridge(e5)
    if self.full_fusion is not None:
      e5 = self.full_fusion(e3, e4, e5, p4)

    s3 = self._refine_skip(e4, e5, self.gls3)
    s2 = self._refine_skip(e3, e4, self.gls2)
    s1 = self._refine_skip(e2, e3, self.gls1)
    s0 = self._refine_skip(e1, e2, self.gls0)
    if self.pvt_skip3 is not None:
      s3 = self.pvt_skip3(s3, p3)
    if self.pvt_skip2 is not None:
      s2 = self.pvt_skip2(s2, p2)
    if self.pvt_skip1 is not None:
      s1 = self.pvt_skip1(s1, p1)

    d4 = self.up1(e5, s3)
    if self.dec_ms4 is not None:
      d4 = self.dec_ms4(d4)
    d3 = self.up2(d4, s2)
    if self.dec_ms3 is not None:
      d3 = self.dec_ms3(d3)
    d2 = self.up3(d3, s1)
    if self.dec_ms2 is not None:
      d2 = self.dec_ms2(d2)
    d1 = self.up4(d2, s0)

    if self.cfm is not None and p2 is not None:
      t2 = self.cfm_t2(p2)
      t3 = self.cfm_t3(p3)
      t4 = self.cfm_t4(p4)
      cfm_feat = self.cfm(t4, t3, t2)
      cfm_feat = F.interpolate(
          self.cfm_to_unet(cfm_feat), size=d1.shape[2:], mode='bilinear',
          align_corners=True)
      d1 = d1 + self.cfm_gate * cfm_feat

    return self.head(d1)


@HEADS.register_module()
class UnetOrig(BaseDecodeHead):
  """CCFB-UNet decode head with ablation switches."""

  def __init__(
      self,
      in_channels=3,
      base_channels=32,
      decoder_channels=64,
      use_fd_kan=False,
      use_mgd_encoder=False,
      use_gls_cams=False,
      init_fg_bias=True,
      ccfb_full_light_bridge=False,
      freeze_base_cnn=False,
      freeze_fd_gls=False,
      pvt_encoder='pvt_v2_b2',
      pretrained_pvt=None,
      use_ibn=False,
      use_mixstyle=False,
      mixstyle_p=0.5,
      mixstyle_alpha=0.1,
      use_pvt_control=False,
      **kwargs,
  ):
    super().__init__(
        channels=decoder_channels,
        in_channels=decoder_channels,
        **kwargs)
    # With frozen base CNN + warm-start, keep the base loss surface (no aux).
    self.use_deep_sup = (
        (bool(use_fd_kan) or bool(use_mgd_encoder) or bool(use_gls_cams))
        and not freeze_base_cnn)
    self.net = CCFBUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        use_fd_kan=use_fd_kan,
        use_mgd_encoder=use_mgd_encoder,
        use_gls_cams=use_gls_cams,
        ccfb_full_light_bridge=ccfb_full_light_bridge,
        decoder_channels=decoder_channels,
        pvt_encoder=pvt_encoder,
        pretrained_pvt=pretrained_pvt,
        use_ibn=use_ibn,
        use_mixstyle=use_mixstyle,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
        use_pvt_control=use_pvt_control,
    )
    if self.use_deep_sup:
      c4 = base_channels * 8
      c3 = base_channels * 4
      self.aux_head4 = nn.Conv2d(c4, self.num_classes, 1)
      self.aux_head3 = nn.Conv2d(c3, self.num_classes, 1)
    if (init_fg_bias and self.conv_seg.bias is not None
        and self.num_classes == 2):
      nn.init.constant_(self.conv_seg.bias[0], -0.5)
      nn.init.constant_(self.conv_seg.bias[1], 0.5)
    if freeze_base_cnn:
      self._freeze_base_cnn(freeze_fd_gls=bool(freeze_fd_gls))

  def _freeze_base_cnn(self, freeze_fd_gls=False):
    """Freeze U-Net CNN trunk/head; train innovation / PVT residuals."""
    if freeze_fd_gls:
      # Stack exclusive full modules on a frozen strong single-module init.
      train_tokens = ('full_fusion', 'cfm')
    else:
      train_tokens = (
          'fd_kan', 'fd_e', 'mgd', 'gls', 'pvt', 'dual_f', 'full_fusion',
          'cfm', 'fd_pvt', 'dec_ms')
    self._freeze_train_tokens = train_tokens
    frozen = 0
    trainable = 0
    for name, p in self.net.named_parameters():
      if any(t in name for t in train_tokens):
        p.requires_grad = True
        trainable += p.numel()
      else:
        p.requires_grad = False
        frozen += p.numel()
    for p in self.conv_seg.parameters():
      p.requires_grad = False
      frozen += p.numel()
    self._apply_frozen_eval()
    print(
        f'[CCFB] freeze_base_cnn: frozen={frozen/1e6:.2f}M '
        f'trainable={trainable/1e6:.2f}M freeze_fd_gls={freeze_fd_gls}')

  def _apply_frozen_eval(self):
    """Only lock BN when stacking identity-safe full modules (freeze_fd_gls)."""
    tokens = getattr(self, '_freeze_train_tokens', None)
    if not tokens or tokens != ('full_fusion', 'cfm'):
      return
    self.net.eval()
    for name, module in self.net.named_modules():
      if name and any(t in name for t in tokens):
        module.train()

  def train(self, mode=True):
    super().train(mode)
    if mode and getattr(self, '_freeze_train_tokens', None):
      self._apply_frozen_eval()
    return self

  def forward(self, x):
    return self.cls_seg(self.net(x))

  def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
    if not self.use_deep_sup:
      return super().forward_train(
          inputs, img_metas, gt_semantic_seg, train_cfg)

    # Recompute encoder/decoder features for auxiliary deep supervision.
    feats = []
    x = inputs
    net = self.net
    p1 = p2 = p3 = p4 = None
    if net.use_pvt:
      p1, p2, p3, p4 = net.pvt_aux(x)
    if net.use_fd_kan:
      x = net.fd_kan(x)
    e1 = net.inc(x)
    if net.fd_e1 is not None:
      e1 = net.fd_e1(e1)
    e2 = net.down1(e1)
    if net.fd_e2 is not None:
      e2 = net.fd_e2(e2)
    if net.fd_pvt_gate2 is not None:
      e2 = net.fd_pvt_gate2(e2, p1)
    e3 = net.down2(e2)
    if net.fd_e3 is not None:
      e3 = net.fd_e3(e3)
    if net.fd_pvt_gate3 is not None:
      e3 = net.fd_pvt_gate3(e3, p2)
    if net.dual_f3 is not None:
      e3 = net.dual_f3(e3, p2)
    e4 = net.down3(e3)
    if net.dual_f4 is not None:
      e4 = net.dual_f4(e4, p3)
    e5 = net.down4(e4)
    if net.dual_f5 is not None:
      e5 = net.dual_f5(e5, p4)
    if net.use_mgd and net.mgd_bn is not None:
      e5 = net.mgd_bn(e5)
    if net.use_gls and net.gls_bridge is not None and not net.light_bridge:
      e5 = net.gls_bridge(e5)
    if net.full_fusion is not None:
      e5 = net.full_fusion(e3, e4, e5, p4)
    s3 = net._refine_skip(e4, e5, net.gls3)
    s2 = net._refine_skip(e3, e4, net.gls2)
    s1 = net._refine_skip(e2, e3, net.gls1)
    s0 = net._refine_skip(e1, e2, net.gls0)
    if net.pvt_skip3 is not None:
      s3 = net.pvt_skip3(s3, p3)
    if net.pvt_skip2 is not None:
      s2 = net.pvt_skip2(s2, p2)
    if net.pvt_skip1 is not None:
      s1 = net.pvt_skip1(s1, p1)
    d4 = net.up1(e5, s3)
    if net.dec_ms4 is not None:
      d4 = net.dec_ms4(d4)
    d3 = net.up2(d4, s2)
    if net.dec_ms3 is not None:
      d3 = net.dec_ms3(d3)
    d2 = net.up3(d3, s1)
    if net.dec_ms2 is not None:
      d2 = net.dec_ms2(d2)
    d1 = net.up4(d2, s0)
    if net.cfm is not None and p2 is not None:
      t2 = net.cfm_t2(p2)
      t3 = net.cfm_t3(p3)
      t4 = net.cfm_t4(p4)
      cfm_feat = net.cfm(t4, t3, t2)
      cfm_feat = F.interpolate(
          net.cfm_to_unet(cfm_feat), size=d1.shape[2:], mode='bilinear',
          align_corners=True)
      d1 = d1 + net.cfm_gate * cfm_feat
    feat = net.head(d1)
    seg_logits = self.cls_seg(feat)
    losses = self.losses(seg_logits, gt_semantic_seg)
    # Deep supervision on decoder stages (structural multi-scale training).
    aux4 = F.interpolate(
        self.aux_head4(d4), size=gt_semantic_seg.shape[2:], mode='bilinear',
        align_corners=self.align_corners)
    aux3 = F.interpolate(
        self.aux_head3(d3), size=gt_semantic_seg.shape[2:], mode='bilinear',
        align_corners=self.align_corners)
    aux_losses4 = self.losses(aux4, gt_semantic_seg)
    aux_losses3 = self.losses(aux3, gt_semantic_seg)
    for k, v in aux_losses4.items():
      if k == 'acc_seg':
        continue
      losses[f'aux4_{k}'] = 0.25 * v
    for k, v in aux_losses3.items():
      if k == 'acc_seg':
        continue
      losses[f'aux3_{k}'] = 0.15 * v
    return losses
