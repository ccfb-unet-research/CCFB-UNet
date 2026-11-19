# Official-code validation track

The primary reviewer suite uses matched PVT adaptations because all methods
must share the same data, backbone, pretraining, optimizer and 14k budget.
Those rows must be named `RobustNet-PVT adapted`, `SAN-SAW-PVT adapted`,
`BlindNet-PVT adapted`, `Rein-PVT adapted`, and `DPMFormer-PVT adapted`.

For an additional official-code audit, run `setup_official_repos.sh`.  The
script clones the authors' repositories and records exact commits.  Do not pool
official Rein/DPMFormer scores with the matched table: their DINOv2/CLIP/EVA
pretraining, decoder, resolution and software environment differ materially.
Report them in a separate “foundation-model track” table if completed.

Official source pages:

- RobustNet: https://github.com/sunghac/RobustNet
- SAN-SAW: https://github.com/leolyj/SAN-SAW
- BlindNet: https://github.com/root0yang/BlindNet
- Rein: https://github.com/w1oves/Rein
- DPMFormer: https://github.com/jone1222/DPMFormer

