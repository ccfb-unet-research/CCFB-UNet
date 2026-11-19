"""Reviewer-defence DG baselines for the CCFB-UNet MMSegmentation tree.

The five heads below are framework-native, binary-polyp adaptations of two
classic DGSS ideas (RobustNet and SAN-SAW) and three recent ideas (BlindNet,
Rein and DPMFormer).  They share the same U-Net/PVT graph and training
protocol.  They must be reported as ``-PVT adaptations`` rather than outputs
from the authors' original urban-scene repositories.  The separate official
track records and runs those repositories when their large pretrained models
are available.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .unet_orig import UnetOrig


def _resize_labels(labels, size):
    return F.interpolate(labels.float(), size=size, mode='nearest').long().squeeze(1)


def _covariance(x, eps=1e-5):
    b, c, h, w = x.shape
    flat = x.reshape(b, c, h * w)
    flat = flat - flat.mean(2, keepdim=True)
    flat = flat / (flat.std(2, keepdim=True, unbiased=False) + eps)
    return torch.bmm(flat, flat.transpose(1, 2)) / max(h * w - 1, 1)


def _offdiag(matrix):
    c = matrix.size(-1)
    mask = ~torch.eye(c, dtype=torch.bool, device=matrix.device)
    return matrix[:, mask]


class Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = None

    def forward(self, x):
        self.value = x
        return x


class SemanticAwareNorm(nn.Module):
    """Content-gated IN used by the SAN-SAW adaptation."""

    def __init__(self, channels, strength=0.5, eps=1e-5):
        super().__init__()
        hidden = max(channels // 8, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        self.strength, self.eps = float(strength), eps

    def forward(self, x):
        mu = x.mean((2, 3), keepdim=True)
        std = x.var((2, 3), keepdim=True, unbiased=False).add(self.eps).sqrt()
        normalized = (x - mu) / std
        gate = self.gate(F.adaptive_avg_pool2d(x, 1)) * self.strength
        return x * (1.0 - gate) + normalized * gate


class BlindStyle(nn.Module):
    """Paired style perturbation for covariance and semantic consistency."""

    def __init__(self, magnitude=0.15, eps=1e-5):
        super().__init__()
        self.magnitude, self.eps = float(magnitude), eps
        self.original = self.perturbed = None

    def forward(self, x):
        self.original = x
        if not self.training:
            self.perturbed = x
            return x
        mu = x.mean((2, 3), keepdim=True)
        std = x.var((2, 3), keepdim=True, unbiased=False).add(self.eps).sqrt()
        noisy_mu = mu + torch.randn_like(mu) * self.magnitude * std.detach()
        noisy_std = std * torch.exp(torch.randn_like(std) * self.magnitude)
        self.perturbed = (x - mu) / std * noisy_std + noisy_mu
        return self.perturbed


class ReinTokenRefiner(nn.Module):
    """Small token residual adapter following the core Rein mechanism."""

    def __init__(self, channels, tokens=16):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(tokens, channels) * 0.02)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        b, c, h, w = x.shape
        flat = x.flatten(2).transpose(1, 2)
        token = F.normalize(self.tokens, dim=1)
        attention = torch.softmax(
            torch.matmul(F.normalize(flat, dim=2), token.t()) / math.sqrt(c), dim=2)
        refined = torch.matmul(attention, self.tokens).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(refined)


class DomainPromptFiLM(nn.Module):
    """Image-conditioned domain-property prompt and texture perturbation."""

    def __init__(self, channels, hidden=64, noise=0.08):
        super().__init__()
        hidden = max(16, min(hidden, channels * 2))
        self.prompt = nn.Sequential(
            nn.Linear(channels * 2, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2))
        self.noise = float(noise)
        self.original = self.transformed = self.prompt_vector = None

    def forward(self, x):
        self.original = x
        mu = x.mean((2, 3))
        std = x.var((2, 3), unbiased=False).add(1e-5).sqrt()
        prompt = self.prompt(torch.cat([mu, std], dim=1))
        gamma, beta = prompt.chunk(2, dim=1)
        gamma = 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = 0.1 * torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        out = x * (1.0 + gamma) + beta
        if self.training:
            out = out + torch.randn_like(out) * self.noise * std.unsqueeze(-1).unsqueeze(-1)
        self.prompt_vector, self.transformed = prompt, out
        return out


@HEADS.register_module()
class ReviewerDGUnet(UnetOrig):
    """One controlled DG adaptation on the existing U-Net/PVT implementation."""

    METHODS = {'robustnet', 'san_saw', 'blindnet', 'rein', 'dpmformer'}

    def __init__(self, dg_method, dg_loss_weight=0.1, dg_temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        if dg_method not in self.METHODS:
            raise ValueError('Unknown dg_method %r' % dg_method)
        self.dg_method = dg_method
        self.dg_loss_weight = float(dg_loss_weight)
        self.dg_temperature = float(dg_temperature)
        c1 = self.net.inc.net[0].out_channels
        c2 = self.net.down1.conv.net[0].out_channels
        self.rec1, self.rec2, self.rec3 = Recorder(), Recorder(), Recorder()
        self._trans1 = self._trans2 = None
        if dg_method == 'san_saw':
            self._trans1, self._trans2 = SemanticAwareNorm(c1), SemanticAwareNorm(c2)
        elif dg_method == 'blindnet':
            self._trans2 = BlindStyle()
        elif dg_method == 'rein':
            self._trans1, self._trans2 = ReinTokenRefiner(c1), ReinTokenRefiner(c2)
        elif dg_method == 'dpmformer':
            self._trans2 = DomainPromptFiLM(c2)
        self._hook_handles = [
            self.net.inc.register_forward_hook(self._hook(self.rec1, '_trans1')),
            self.net.down1.register_forward_hook(self._hook(self.rec2, '_trans2')),
            self.net.down2.register_forward_hook(self._hook(self.rec3, None)),
        ]

    def _hook(self, recorder, transform_name):
        def apply(_module, _inputs, output):
            transform = getattr(self, transform_name) if transform_name else None
            if transform is not None:
                output = transform(output)
            return recorder(output)
        return apply

    def _isw_loss(self, feature):
        off = _offdiag(_covariance(feature))
        variability = off.detach().var(0, unbiased=False)
        selected = variability >= variability.median()
        return off[:, selected].abs().mean() if selected.any() else off.abs().mean()

    def _semantic_whitening(self, feature, labels):
        labels = _resize_labels(labels, feature.shape[2:])
        losses = []
        for class_id in range(self.num_classes):
            mask = (labels == class_id).to(feature.dtype).unsqueeze(1)
            count = mask.sum((2, 3), keepdim=True).clamp_min(2.0)
            centre = (feature * mask).sum((2, 3), keepdim=True) / count
            flat = ((feature - centre) * mask).flatten(2)
            cov = torch.bmm(flat, flat.transpose(1, 2)) / (count.flatten(1) - 1).clamp_min(1).unsqueeze(2)
            losses.append(_offdiag(cov).abs().mean())
        return torch.stack(losses).mean()

    def _centroids(self, feature, labels):
        labels = _resize_labels(labels, feature.shape[2:])
        result = []
        for class_id in range(self.num_classes):
            mask = (labels == class_id).to(feature.dtype).unsqueeze(1)
            denom = mask.sum((2, 3)).clamp_min(1.0)
            result.append(F.normalize((feature * mask).sum((2, 3)) / denom, dim=1))
        return torch.stack(result, 1)

    def _blind_loss(self, labels):
        original, perturbed = self._trans2.original, self._trans2.perturbed
        cov = F.smooth_l1_loss(_covariance(original), _covariance(perturbed))
        a, b = self._centroids(original, labels), self._centroids(perturbed, labels)
        positive = 1.0 - (a * b).sum(2).mean()
        negative = (a[:, 0] * b[:, 1]).sum(1).clamp_min(0).mean()
        return cov + positive + 0.25 * negative

    def _dpm_loss(self):
        module = self._trans2
        pooled_a = F.adaptive_avg_pool2d(module.original, 1).flatten(1)
        pooled_b = F.adaptive_avg_pool2d(module.transformed, 1).flatten(1)
        consistency = F.smooth_l1_loss(pooled_b, pooled_a.detach())
        prompt_reg = module.prompt_vector.pow(2).mean()
        return consistency + 0.01 * prompt_reg

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        losses = super().forward_train(inputs, img_metas, gt_semantic_seg, train_cfg)
        if self.dg_method == 'robustnet':
            dg = 0.5 * (self._isw_loss(self.rec2.value) + self._isw_loss(self.rec3.value))
        elif self.dg_method == 'san_saw':
            dg = self._semantic_whitening(self.rec3.value, gt_semantic_seg)
        elif self.dg_method == 'blindnet':
            dg = self._blind_loss(gt_semantic_seg)
        elif self.dg_method == 'rein':
            dg = self.rec1.value.new_zeros(())
        else:
            dg = self._dpm_loss()
        if self.dg_method != 'rein':
            losses['loss_dg_' + self.dg_method] = self.dg_loss_weight * dg
        return losses
