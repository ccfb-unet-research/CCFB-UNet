#!/usr/bin/env python3
"""Evaluate one checkpoint with overlap, boundary and difficulty metadata."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from mmcv import Config
from mmcv.utils import import_modules_from_strings
from mmseg.apis import inference_segmentor, init_segmentor


EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image-dir', required=True)
    p.add_argument('--label-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--boundary-tolerance', type=float, default=2.0)
    p.add_argument('--save-masks', action='store_true')
    return p.parse_args()


def load_mask(path):
    return np.asarray(Image.open(path).convert('L')) > 0


def find_label(label_dir, image):
    exact = label_dir / image.name
    if exact.is_file():
        return exact
    candidates = [p for p in label_dir.glob(image.stem + '.*') if p.suffix.lower() in EXTS]
    if not candidates:
        raise FileNotFoundError('No label for %s' % image)
    return sorted(candidates)[0]


def boundary(mask):
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.logical_xor(mask, ndimage.binary_erosion(mask))


def metrics(pred, gt, tolerance):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    ps, gs = pred.sum(dtype=np.float64), gt.sum(dtype=np.float64)
    union = np.logical_or(pred, gt).sum(dtype=np.float64)
    dice = 1.0 if ps + gs == 0 else 2.0 * tp / max(ps + gs, 1.0)
    iou = 1.0 if union == 0 else tp / union
    precision = 1.0 if ps == 0 and gs == 0 else (tp / ps if ps else 0.0)
    recall = 1.0 if gs == 0 and ps == 0 else (tp / gs if gs else 0.0)
    pb, gb = boundary(pred), boundary(gt)
    if not pb.any() and not gb.any():
        hd95 = assd = 0.0
        bf = 1.0
    elif not pb.any() or not gb.any():
        hd95 = assd = float(np.hypot(*gt.shape))
        bf = 0.0
    else:
        d_to_g = ndimage.distance_transform_edt(~gb)[pb]
        d_to_p = ndimage.distance_transform_edt(~pb)[gb]
        distances = np.concatenate([d_to_g, d_to_p])
        hd95 = float(np.percentile(distances, 95))
        assd = float(distances.mean())
        bp = float((d_to_g <= tolerance).mean())
        br = float((d_to_p <= tolerance).mean())
        bf = 0.0 if bp + br == 0 else 2.0 * bp * br / (bp + br)
    return dict(dice=float(dice), iou=float(iou), precision=float(precision),
                recall=float(recall), hd95=hd95, assd=assd, boundary_f=bf)


def difficulty(image, gt):
    gray = np.asarray(image.convert('L'), dtype=np.float32)
    area = int(gt.sum())
    inside = float(gray[gt].mean()) if area else 0.0
    outside = float(gray[~gt].mean()) if (~gt).any() else inside
    perimeter = int(boundary(gt).sum())
    components = int(ndimage.label(gt)[1])
    return dict(
        lesion_area_px=area,
        lesion_area_ratio=float(area / gt.size),
        contrast_abs=float(abs(inside - outside) / 255.0),
        boundary_complexity=float(perimeter / max(np.sqrt(area), 1.0)),
        gt_components=components)


def main():
    args = parse_args()
    image_dir, label_dir = Path(args.image_dir), Path(args.label_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    if 'decode_head' in cfg.model and 'pretrained_pvt' in cfg.model.decode_head:
        cfg.model.decode_head.pretrained_pvt = ''
    cfg.model.train_cfg = None
    model = init_segmentor(cfg, args.checkpoint, device=args.device)
    images = sorted(p for p in image_dir.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
    rows = []
    mask_dir = output / 'masks'
    for index, image_path in enumerate(images, 1):
        label_path = find_label(label_dir, image_path)
        pil = Image.open(image_path).convert('RGB')
        gt = load_mask(label_path)
        pred = np.asarray(inference_segmentor(model, str(image_path))[0]) > 0
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)) > 0
        row = dict(image=image_path.name, label=label_path.name,
                   image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest())
        row.update(metrics(pred, gt, args.boundary_tolerance))
        row.update(difficulty(pil, gt))
        rows.append(row)
        if args.save_masks:
            mask_dir.mkdir(exist_ok=True)
            Image.fromarray(pred.astype(np.uint8) * 255).save(mask_dir / (image_path.stem + '.png'))
        if index % 50 == 0:
            print('[EVAL] %d/%d' % (index, len(images)), flush=True)
    fields = list(rows[0])
    with (output / 'per_image.tsv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    metric_names = ['dice', 'iou', 'precision', 'recall', 'hd95', 'assd', 'boundary_f']
    summary = {name: float(np.mean([r[name] for r in rows])) for name in metric_names}
    summary.update(n_images=len(rows), config=str(args.config), checkpoint=str(args.checkpoint),
                   boundary_tolerance_px=args.boundary_tolerance,
                   empty_mask_rule='both empty=perfect; one empty=diagonal distance and BF=0')
    with (output / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

