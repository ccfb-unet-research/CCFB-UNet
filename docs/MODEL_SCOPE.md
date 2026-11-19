# Scope of the comparison models

The repository supports two complementary comparison tracks.

## Matched-protocol track

All methods in this track use the same polyp training data, 256 x 256 input protocol, five random seeds, augmentation, losses, AdamW optimizer, batch size, 14,000-iteration budget, final-checkpoint rule, and five test domains.

The additional DG models in `my_config/reviewer/` are PVTv2-B2 segmentation adaptations of their core mechanisms:

- RobustNet-PVT
- SAN-SAW-PVT
- BlindNet-PVT
- Rein-PVT
- DPMFormer-PVT

They must be labeled **our reimplementation/adaptation** in tables and prose. This track answers whether the mechanisms help under a matched segmentation backbone and training budget. It does not claim byte-for-byte reproduction of the original authors' implementations.

IBN-Net and MixStyle are implemented within the common U-Net/PVT control framework. DDCNet, ResUNet++, and UltraSeg are retrained under the same 14k protocol using their repository implementations.

## Official-code track

`official_track/` records upstream repositories and setup helpers for separately running original codebases. Official-code results should be reported in a distinct table because their backbones, pretraining, inputs, schedules, and original task domains may differ.

Never merge official-code and matched-protocol numbers into a single "all else equal" ranking.

## Causal interpretation

Architecture claims are supported by within-protocol contrasts:

- CCFB scratch versus PVT control scratch;
- CCFB pretrained versus PVT control pretrained;
- FD/MGD/GLS factorial ablations under the same initialization condition.

Comparisons across different pretraining states must not be interpreted as pure architecture effects.

