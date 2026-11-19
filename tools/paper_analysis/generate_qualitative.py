#!/usr/bin/env python3
"""Generate reproducible qualitative segmentation comparisons.

Samples are selected as the lowest-Dice cases of the fixed fair_unet baseline
within each requested domain. This transparent rule avoids manually choosing
attractive examples. Raw prediction masks and the selection table are saved so
the figure is fully auditable.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mmcv import Config
from mmseg.apis import inference_segmentor, init_segmentor


ALL_MODELS = [
    'fair_unet', 'fair_ibn', 'fair_mixstyle',
    'fair_pvt_control_scratch', 'fair_pvt_control_pretrained',
    'fair_ccfb_scratch', 'fair_ccfb_pretrained',
]
PANEL_MODELS = [
    'fair_unet', 'fair_ibn', 'fair_mixstyle',
    'fair_pvt_control_pretrained', 'fair_ccfb_pretrained',
]
DISPLAY = {
    'fair_unet': 'U-Net', 'fair_ibn': 'IBN', 'fair_mixstyle': 'MixStyle',
    'fair_pvt_control_scratch': 'PVT control\n(scratch)',
    'fair_pvt_control_pretrained': 'PVT control\n(pretrained)',
    'fair_ccfb_scratch': 'CCFB\n(scratch)',
    'fair_ccfb_pretrained': 'CCFB\n(pretrained)',
}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--datasets', nargs='+', default=['ETIS-Larib', 'CVC-ColonDB'])
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def load_binary(path):
    arr = np.asarray(Image.open(path).convert('L'))
    return (arr > 127).astype(np.uint8)


def find_label(label_dir, image_path):
    exact = label_dir / image_path.name
    if exact.is_file():
        return exact
    candidates = sorted(p for p in label_dir.glob(image_path.stem + '.*')
                        if p.suffix.lower() in IMAGE_EXTS)
    if not candidates:
        raise FileNotFoundError(f'No label for {image_path.name} below {label_dir}')
    return candidates[0]


def dice_score(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    denom = pred.sum() + target.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(pred, target).sum() / denom)


def build(project_root, work_root, model_name, seed, device):
    config_path = project_root / 'my_config' / 'dg' / f'{model_name}.py'
    checkpoint = work_root / f'seed_{seed}' / model_name / 'iter_14000.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = Config.fromfile(str(config_path))
    cfg.model.pretrained = None
    if 'decode_head' in cfg.model and 'pretrained_pvt' in cfg.model.decode_head:
        cfg.model.decode_head.pretrained_pvt = ''
    return init_segmentor(cfg, str(checkpoint), device=device)


def predict(model, image_path):
    result = inference_segmentor(model, str(image_path))[0]
    return (np.asarray(result) > 0).astype(np.uint8)


def save_mask(mask, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def render_panel(dataset, selected, predictions, output_dir):
    columns = ['image', 'ground_truth'] + PANEL_MODELS
    titles = ['Image', 'Ground truth'] + [DISPLAY[x] for x in PANEL_MODELS]
    fig, axes = plt.subplots(
        len(selected), len(columns),
        figsize=(2.05 * len(columns), 1.85 * len(selected)),
        squeeze=False)
    for row_idx, item in enumerate(selected):
        image = np.asarray(Image.open(item['image']).convert('RGB'))
        gt = item['gt']
        panels = [image, gt] + [predictions[m][item['stem']] for m in PANEL_MODELS]
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap='gray', vmin=0, vmax=1)
            ax.set_axis_off()
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=8, pad=4)
            if col_idx == 0:
                ax.text(-0.04, 0.5, item['stem'], transform=ax.transAxes,
                        rotation=90, ha='right', va='center', fontsize=6)
    fig.suptitle(
        f'{dataset}: hardest cases ranked by the fixed U-Net baseline',
        fontsize=10, y=0.997)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.01, top=0.95,
                        wspace=0.025, hspace=0.045)
    base = output_dir / f'qualitative_{dataset}'
    fig.savefig(str(base) + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(str(base) + '.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    work_root = Path(args.work_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = build(project_root, work_root, 'fair_unet', args.seed, args.device)
    selection_rows = []

    for dataset in args.datasets:
        image_dir = data_root / 'test' / dataset / 'images'
        label_dir = data_root / 'test' / dataset / 'labels'
        images = sorted(p for p in image_dir.rglob('*')
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not images:
            raise RuntimeError(f'No images below {image_dir}')

        ranked = []
        baseline_cache = {}
        for index, image_path in enumerate(images):
            label_path = find_label(label_dir, image_path)
            gt = load_binary(label_path)
            pred = predict(baseline, image_path)
            if pred.shape != gt.shape:
                pred = np.asarray(Image.fromarray(pred).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST))
            score = dice_score(pred, gt)
            baseline_cache[image_path.stem] = pred
            ranked.append((score, index, image_path, label_path, gt))
        ranked.sort(key=lambda x: (x[0], x[1]))
        selected = []
        for rank, (score, _, image_path, label_path, gt) in enumerate(
                ranked[:args.samples], start=1):
            selected.append({'image': image_path, 'label': label_path,
                             'gt': gt, 'stem': image_path.stem,
                             'baseline_dice': score})
            selection_rows.append({
                'dataset': dataset, 'rank_by_unet_error': rank,
                'image': image_path.name, 'label': label_path.name,
                'unet_dice': f'{score:.6f}', 'seed': args.seed,
                'selection_rule': 'lowest final-checkpoint fair_unet Dice',
            })

        predictions = {name: {} for name in ALL_MODELS}
        predictions['fair_unet'] = {
            item['stem']: baseline_cache[item['stem']] for item in selected}
        for model_name in ALL_MODELS[1:]:
            print(f'[QUAL] dataset={dataset} model={model_name}', flush=True)
            model = build(project_root, work_root, model_name, args.seed, args.device)
            for item in selected:
                pred = predict(model, item['image'])
                if pred.shape != item['gt'].shape:
                    pred = np.asarray(Image.fromarray(pred).resize(
                        (item['gt'].shape[1], item['gt'].shape[0]), Image.NEAREST))
                predictions[model_name][item['stem']] = pred
            del model

        dataset_out = output_dir / dataset
        for item in selected:
            save_mask(item['gt'], dataset_out / 'masks' / 'ground_truth' /
                      f"{item['stem']}.png")
            for model_name in ALL_MODELS:
                save_mask(predictions[model_name][item['stem']],
                          dataset_out / 'masks' / model_name /
                          f"{item['stem']}.png")
        render_panel(dataset, selected, predictions, dataset_out)

    with (output_dir / 'selected_samples.tsv').open(
            'w', newline='', encoding='utf-8') as handle:
        fields = list(selection_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(selection_rows)
    print(f'[QUAL] wrote {output_dir}')


if __name__ == '__main__':
    main()

