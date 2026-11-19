# Reproducibility guide

## 1. Historical environment

```text
Python 3.8
PyTorch 1.10.0+cu113
CUDA 11.3
MMCV 1.7.0
MMSegmentation 0.29.1
GPU: NVIDIA A100-SXM4-40GB
```

Install a PyTorch/CUDA-compatible build of `mmcv-full==1.7.0`, then install this repository in editable mode. Do not combine MMCV 2.x or MMSegmentation 1.x with these 0.x configurations.

## 2. Required external objects

The public repository intentionally excludes datasets, augmented training images, `pvt_v2_b2.pth`, trained checkpoints, generated masks, and logs.

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export DATA_ROOT=/absolute/path/to/polyp_datasets
export PVT_B2_PATH=/absolute/path/to/pvt_v2_b2.pth
```

## 3. Preflight

```bash
python tools/reviewer/preflight_review_patch.py \
  --data-root "$DATA_ROOT" \
  --pvt-path "$PVT_B2_PATH" \
  --build-models
```

This read-only check should validate 14 reviewer configurations.

## 4. Controlled core experiment

```bash
GPU_ID=0 \
SEEDS="0 42 1234 2026 3407" \
WORK_ROOT="$PWD/work_dirs_polyp_v2/dg_multiseed" \
PVT_B2_PATH="$PVT_B2_PATH" \
bash tools/dg/run_multiseed.sh \
  fair_unet fair_ibn fair_mixstyle \
  fair_pvt_control_scratch fair_pvt_control_pretrained \
  fair_ccfb_scratch fair_ccfb_pretrained
```

Expected completion: 35 final checkpoints.

## 5. Reviewer-defense matrix

Run stages sequentially on one GPU. Each stage skips an existing final checkpoint.

```bash
for stage in dg5 baselines3 ablation6; do
  GPU_ID=0 \
  SEEDS="0 42 1234 2026 3407" \
  STAGE="$stage" \
  WORK_ROOT="$PWD/work_dirs_polyp_v2/reviewer_multiseed" \
  PVT_B2_PATH="$PVT_B2_PATH" \
  bash tools/reviewer/run_review_training.sh
done
```

Expected new final checkpoints: 25 DG + 15 segmentation baselines + 30 ablations = 70.

## 6. Data audit

```bash
DATA_ROOT="$DATA_ROOT" RESULT_ROOT="$PWD/reviewer_results" \
bash tools/reviewer/run_data_audit.sh
```

## 7. Five-domain region and boundary evaluation

```bash
GPU_ID=0 \
SEEDS="0 42 1234 2026 3407" \
DATA_ROOT="$DATA_ROOT" \
CORE_WORK_ROOT="$PWD/work_dirs_polyp_v2/dg_multiseed" \
REVIEW_WORK_ROOT="$PWD/work_dirs_polyp_v2/reviewer_multiseed" \
RESULT_ROOT="$PWD/reviewer_results/evaluation" \
STAGE=all SAVE_MASKS=0 \
bash tools/reviewer/run_boundary_evaluation.sh
```

Use `SAVE_MASKS=1` only when qualitative masks are needed; it consumes substantially more storage.

## 8. Efficiency and checkpoint provenance

```bash
CORE_WORK_ROOT="$PWD/work_dirs_polyp_v2/dg_multiseed" \
REVIEW_WORK_ROOT="$PWD/work_dirs_polyp_v2/reviewer_multiseed" \
RESULT_ROOT="$PWD/reviewer_results" \
bash tools/reviewer/run_efficiency_and_manifest.sh
```

## 9. Figures and archive

```bash
python tools/reviewer/make_review_figures.py \
  --summary-dir reviewer_results/evaluation/summary \
  --output-dir reviewer_results/figures

RESULT_ROOT="$PWD/reviewer_results" \
bash tools/reviewer/package_review_results.sh
```

## 10. Provenance checklist

For every reported result retain the resolved configuration, seed, final checkpoint SHA-256, logs, software versions, GPU model, and dataset audit. Numerical claims must be traced to archived result tables and the checkpoint manifest, not reconstructed from console snippets.

