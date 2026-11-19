# CCFB-UNet 复现实验代码

本仓库对应论文 **CCFB-UNet: Frequency–Geometry–Semantic Fusion for Cross-Dataset Polyp Segmentation**，包含模型、统一五种子域泛化对比、预训练/不预训练控制、完整消融、五域区域与边界评价、显著性检验、效率统计和结果打包工具。

仓库只发布代码，不包含医学图像数据、PVT 预训练权重、训练 checkpoint、日志或论文未公开的结果。

## 主要目录

- `mmseg/models/decode_heads/unet_orig.py`：CCFB-UNet 与 FD、MGD、GLS 三个可独立开关的模块。
- `mmseg/models/decode_heads/reviewer_dg.py`：统一 PVTv2-B2 条件下的 DG 对比实现。
- `my_config/dg/`：IBN-Net、MixStyle、PVT 控制组与 CCFB-UNet 的公平配置。
- `my_config/reviewer/`：新增 5 个 DG 对比、3 个分割基线和 6 个组合消融配置。
- `tools/dg/`、`tools/reviewer/`：训练、五域测试、边界评价、统计、效率和论文图生成。

新增的五个 DG 对比必须表述为 `*-PVT（本文复现/适配）`，不能表述为原作者官方城市街景或 VFM 代码的直接结果。详细边界见 [docs/MODEL_SCOPE.md](docs/MODEL_SCOPE.md)。

## 环境与路径

论文实验环境为 Python 3.8、PyTorch 1.10.0+cu113、MMCV 1.7.0、MMSegmentation 0.29.1 和单张 A100 40GB。

```bash
pip install -r requirements.txt
pip install -v -e .

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export DATA_ROOT=/你的绝对路径/polyp_datasets
export PVT_B2_PATH=/你的绝对路径/pvt_v2_b2.pth
```

数据目录：

```text
polyp_datasets/
├── train/{images_albu,labels_albu}/
└── test/
    ├── Kvasir/{images,labels}/
    ├── CVC-ClinicDB/{images,labels}/
    ├── CVC-ColonDB/{images,labels}/
    ├── CVC-300/{images,labels}/
    └── ETIS-Larib/{images,labels}/
```

## 预检与训练

```bash
python tools/reviewer/preflight_review_patch.py \
  --data-root "$DATA_ROOT" --pvt-path "$PVT_B2_PATH" --build-models

GPU_ID=0 SEEDS="0 42 1234 2026 3407" \
WORK_ROOT="$PWD/work_dirs_polyp_v2/dg_multiseed" \
bash tools/dg/run_multiseed.sh \
  fair_unet fair_ibn fair_mixstyle \
  fair_pvt_control_scratch fair_pvt_control_pretrained \
  fair_ccfb_scratch fair_ccfb_pretrained
```

新增 DG、原分割基线和组合消融分别使用 `STAGE=dg5`、`STAGE=baselines3`、`STAGE=ablation6` 运行 `tools/reviewer/run_review_training.sh`。完整运行、评价、结果位置和断点续跑方法见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 许可

本工程基于 MMSegmentation 0.29.1 修改，遵循 Apache License 2.0。数据集与预训练权重遵循各自许可，不随本仓库发布。

