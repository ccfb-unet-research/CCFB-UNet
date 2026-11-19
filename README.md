# CCFB-UNet

Official research code for **CCFB-UNet: Frequency–Geometry–Semantic Fusion for Cross-Dataset Polyp Segmentation**.

CCFB-UNet couples frequency decoupling (FD-KAN), geometric duality (MGD-Encoder), and global-local semantic fusion (GLS-CAMS). This repository contains the model implementation, a controlled five-seed domain-generalization protocol, matched scratch/pretrained controls, classical and recent DG comparison adaptations, factorial ablations, five-domain evaluation, boundary metrics, statistical tests, efficiency measurement, and result packaging.

> This repository contains code only. Datasets, pretrained weights, trained checkpoints, and experiment logs are intentionally excluded.

## What is included

- `mmseg/models/decode_heads/unet_orig.py`: CCFB-UNet and its independently switchable FD, MGD, and GLS components.
- `mmseg/models/decode_heads/reviewer_dg.py`: matched-backbone DG comparison implementations.
- `my_config/dg/`: IBN-Net, MixStyle, scratch/pretrained PVT controls, and CCFB-UNet.
- `my_config/reviewer/`: five additional DG comparisons, three segmentation baselines, and six missing factorial ablations.
- `tools/dg/`: multi-seed training, five-domain testing, and summary generation.
- `tools/reviewer/`: data audit, training, region/boundary evaluation, statistics, efficiency/provenance reporting, figures, and archival.
- `official_track/`: helper for obtaining upstream official DG repositories for a separate official-code track.

The five additional DG comparisons are named `*-PVT (our reimplementation/adaptation)`. They implement the corresponding core mechanisms under the same PVTv2-B2 segmentation protocol; they are **not** claimed to be direct results from the authors' original city-scene/VFM codebases. See [docs/MODEL_SCOPE.md](docs/MODEL_SCOPE.md).

## Environment

The reported experiments used Python 3.8, PyTorch 1.10.0 + CUDA 11.3, MMCV 1.7.0, MMSegmentation 0.29.1 (vendored and modified here), and one NVIDIA A100 40 GB GPU.

```bash
pip install -r requirements.txt
pip install -v -e .
```

MMCV must match the installed PyTorch/CUDA build. The exact historical environment is documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Data layout

Download the public polyp datasets from their official sources and arrange them as follows:

```text
polyp_datasets/
├── train/
│   ├── images_albu/
│   └── labels_albu/
└── test/
    ├── Kvasir/{images,labels}/
    ├── CVC-ClinicDB/{images,labels}/
    ├── CVC-ColonDB/{images,labels}/
    ├── CVC-300/{images,labels}/
    └── ETIS-Larib/{images,labels}/
```

The repository does not redistribute medical images. Confirm each dataset's license and citation requirements before use.

## Pretrained backbone and paths

Obtain the public PVTv2-B2 ImageNet checkpoint (`pvt_v2_b2.pth`) from the PVTv2 project, then export portable paths:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export DATA_ROOT=/absolute/path/to/polyp_datasets
export PVT_B2_PATH=/absolute/path/to/pvt_v2_b2.pth
```

## Quick validation

This checks the dataset structure, checkpoint path, dependencies, and all reviewer configurations without starting training:

```bash
python tools/reviewer/preflight_review_patch.py \
  --data-root "$DATA_ROOT" \
  --pvt-path "$PVT_B2_PATH" \
  --build-models
```

## Controlled five-seed experiment

The controlled protocol uses seeds `0 42 1234 2026 3407`, identical augmentation, losses, optimizer, batch size, and 14,000 iterations.

```bash
GPU_ID=0 \
SEEDS="0 42 1234 2026 3407" \
WORK_ROOT="$PWD/work_dirs_polyp_v2/dg_multiseed" \
bash tools/dg/run_multiseed.sh \
  fair_unet fair_ibn fair_mixstyle \
  fair_pvt_control_scratch fair_pvt_control_pretrained \
  fair_ccfb_scratch fair_ccfb_pretrained
```

Additional DG methods, segmentation baselines, and factorial ablations are resumable and skip completed `iter_14000.pth` checkpoints:

```bash
GPU_ID=0 SEEDS="0 42 1234 2026 3407" STAGE=dg5 \
  WORK_ROOT="$PWD/work_dirs_polyp_v2/reviewer_multiseed" \
  bash tools/reviewer/run_review_training.sh

# Later stages: STAGE=baselines3 and STAGE=ablation6
```

## Evaluation and outputs

```bash
GPU_ID=0 SEEDS="0 42 1234 2026 3407" \
DATA_ROOT="$DATA_ROOT" \
CORE_WORK_ROOT="$PWD/work_dirs_polyp_v2/dg_multiseed" \
REVIEW_WORK_ROOT="$PWD/work_dirs_polyp_v2/reviewer_multiseed" \
RESULT_ROOT="$PWD/reviewer_results/evaluation" \
STAGE=all SAVE_MASKS=0 \
bash tools/reviewer/run_boundary_evaluation.sh
```

Important generated files include `summary_mean_sd.tsv`, `macro_seed.tsv`, `paired_tests_holm.tsv`, `hierarchical_bootstrap.tsv`, `difficulty_summary.tsv`, the efficiency report, checkpoint manifest, and paper figures under `reviewer_results/`.

For the full execution order, resume behavior, and completion checks, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Reproducibility notes

- The 28k boundary-focused study and the 14k controlled multi-seed study are separate protocols. Do not mix checkpoints or transfer metrics between them.
- The final 14k checkpoint is fixed before target-domain testing; target sets are not used for checkpoint selection.
- Pretrained and scratch controls are reported separately to avoid attributing ImageNet initialization gains to architecture.
- Every public result should be accompanied by the configuration, seed, checkpoint SHA-256, and software environment.

## Citation

If this code supports your work, please cite the accompanying paper. Bibliographic metadata can be updated in `CITATION.cff` after the final author list and DOI are public.

## License and acknowledgements

This codebase is derived from MMSegmentation 0.29.1 and retains its Apache License 2.0. See [LICENSE](LICENSE) and [LICENSES.md](LICENSES.md) for component-specific terms. Dataset and pretrained-weight licenses are separate and are not granted by this repository.

