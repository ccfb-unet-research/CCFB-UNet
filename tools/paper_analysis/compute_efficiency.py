#!/usr/bin/env python3
"""Measure parameters, FLOPs and batch-1 CUDA latency for all DG models.

The timing measures the FP32 network forward pass at 256x256 without image
loading or preprocessing. FLOPs are reported by MMCV and may omit unsupported
custom operators; the CSV records failures rather than silently inventing a
number.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor


MODELS = [
    'fair_unet', 'fair_ibn', 'fair_mixstyle',
    'fair_pvt_control_scratch', 'fair_pvt_control_pretrained',
    'fair_ccfb_scratch', 'fair_ccfb_pretrained',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shape', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    return parser.parse_args()


def build_model(project_root, name, checkpoint):
    config_path = project_root / 'my_config' / 'dg' / f'{name}.py'
    cfg = Config.fromfile(str(config_path))
    cfg.model.pretrained = None
    # The final checkpoint fully specifies the weights. Avoid loading the
    # external ImageNet file a second time during model construction.
    if 'decode_head' in cfg.model and 'pretrained_pvt' in cfg.model.decode_head:
        cfg.model.decode_head.pretrained_pvt = ''
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, str(checkpoint), map_location='cpu')
    return model


def measure_one(project_root, work_root, name, seed, shape, warmup, repeats):
    checkpoint = work_root / f'seed_{seed}' / name / 'iter_14000.pth'
    row = {
        'model': name, 'seed': seed, 'input_height': shape[0],
        'input_width': shape[1], 'params_m': '', 'trainable_params_m': '',
        'gflops': '', 'latency_ms_mean': '', 'latency_ms_sd': '',
        'fps_batch1': '', 'peak_gpu_memory_mb': '', 'status': 'ok',
        'notes': 'FP32, batch=1, network forward only; FLOPs may omit unsupported custom ops',
    }
    if not checkpoint.is_file():
        row['status'] = 'missing_checkpoint'
        return row

    try:
        model = build_model(project_root, name, checkpoint)
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        row['params_m'] = round(params / 1e6, 4)
        row['trainable_params_m'] = round(trainable / 1e6, 4)

        original_forward = model.forward
        if not hasattr(model, 'forward_dummy'):
            raise RuntimeError('model has no forward_dummy method')
        model.forward = model.forward_dummy
        try:
            flops, _ = get_model_complexity_info(
                model, (3, shape[0], shape[1]), as_strings=False,
                print_per_layer_stat=False)
            row['gflops'] = round(float(flops) / 1e9, 4)
        except Exception as exc:  # MMCV counters do not cover every custom op.
            row['notes'] += f'; FLOPs unavailable: {type(exc).__name__}: {exc}'
        finally:
            model.forward = original_forward

        if not torch.cuda.is_available():
            row['status'] = 'params_only_no_cuda'
            return row

        model = model.cuda().eval()
        sample = torch.randn(1, 3, shape[0], shape[1], device='cuda')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(warmup):
                model.forward_dummy(sample)
        torch.cuda.synchronize()

        elapsed = []
        with torch.no_grad():
            for _ in range(repeats):
                start = time.perf_counter()
                model.forward_dummy(sample)
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter() - start) * 1000.0)

        mean_ms = sum(elapsed) / len(elapsed)
        variance = sum((x - mean_ms) ** 2 for x in elapsed) / max(1, len(elapsed) - 1)
        row['latency_ms_mean'] = round(mean_ms, 4)
        row['latency_ms_sd'] = round(variance ** 0.5, 4)
        row['fps_batch1'] = round(1000.0 / mean_ms, 4)
        row['peak_gpu_memory_mb'] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
        del model, sample
        torch.cuda.empty_cache()
    except Exception as exc:
        row['status'] = 'error'
        row['notes'] += f'; {type(exc).__name__}: {exc}'
    return row


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    work_root = Path(args.work_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in MODELS:
        print(f'[EFFICIENCY] {name}', flush=True)
        rows.append(measure_one(
            project_root, work_root, name, args.seed, tuple(args.shape),
            args.warmup, args.repeats))

    fields = list(rows[0].keys())
    csv_path = output_dir / 'model_efficiency.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'model_efficiency.json').open('w', encoding='utf-8') as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(f'[EFFICIENCY] wrote {csv_path}')


if __name__ == '__main__':
    main()

