#!/usr/bin/env python3
"""Offline Albumentations augmentation for ISIC2018 train set.

Augmentations follow ISIC Challenge 2018 winning strategies:
  - geometric: horizontal/vertical flip, rotate90, transpose, shift-scale-rotate
  - photometric: brightness/contrast, HSV, CLAHE (dermoscopy-friendly)
  - light blur/noise for imaging variability

Outputs:
  train/images_albu/  (original + augmented images)
  train/labels_albu/  (matching binary masks, values 0/1)
"""
import argparse
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


def build_isic_aug_profiles():
    """13 deterministic profiles (same count as polyp images_albu)."""
    border = dict(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
    return [
        A.Compose([A.HorizontalFlip(p=1.0)]),
        A.Compose([A.VerticalFlip(p=1.0)]),
        A.Compose([A.RandomRotate90(p=1.0)]),
        A.Compose([A.Transpose(p=1.0)]),
        A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.15, rotate_limit=25,
                **border, p=1.0),
        ]),
        A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.03, scale_limit=0.20, rotate_limit=0,
                **border, p=1.0),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.20, contrast_limit=0.20, p=1.0),
        ]),
        A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15,
                p=1.0),
        ]),
        A.Compose([A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0)]),
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        ]),
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.ShiftScaleRotate(
                shift_limit=0.03, scale_limit=0.10, rotate_limit=15,
                **border, p=1.0),
        ]),
        A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.GaussNoise(var_limit=(5.0, 25.0)),
            ], p=1.0),
        ]),
        A.Compose([
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=1.0),
        ]),
    ]


def binarize_mask(mask):
    return (mask > 0).astype(np.uint8)


def process_split(data_root, num_aug, overwrite):
    img_dir = data_root / 'images'
    lbl_dir = data_root / 'labels'
    out_img = data_root / 'images_albu'
    out_lbl = data_root / 'labels_albu'

    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f'Missing images/labels under {data_root}')

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    profiles = build_isic_aug_profiles()[:num_aug]
    img_paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'})

    print(f'Found {len(img_paths)} images in {img_dir}')
    print(f'Generating {num_aug} aug/profile + originals -> {out_img}')

    for img_path in tqdm(img_paths, desc='ISIC2018 albu'):
        stem = img_path.stem
        lbl_path = lbl_dir / f'{stem}.png'
        if not lbl_path.exists():
            for ext in ('.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
                alt = lbl_dir / f'{stem}{ext}'
                if alt.exists():
                    lbl_path = alt
                    break
            else:
                print(f'[skip] no label for {img_path.name}')
                continue

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = binarize_mask(cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE))

        orig_img_out = out_img / img_path.name
        orig_lbl_out = out_lbl / lbl_path.name
        if overwrite or not orig_img_out.exists():
            cv2.imwrite(str(orig_img_out), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(orig_lbl_out), mask)

        for idx, profile in enumerate(profiles):
            aug_img_out = out_img / f'{stem}_aug_{idx}.png'
            aug_lbl_out = out_lbl / f'{stem}_aug_{idx}.png'
            if not overwrite and aug_img_out.exists() and aug_lbl_out.exists():
                continue
            augmented = profile(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = binarize_mask(augmented['mask'])
            cv2.imwrite(
                str(aug_img_out), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(aug_lbl_out), aug_mask)

    total = len(list(out_img.glob('*.png')))
    print(f'Done. {total} images in {out_img}')


def main():
    parser = argparse.ArgumentParser(description='Generate ISIC2018 albu dataset')
    parser.add_argument(
        '--data-root',
        type=str,
        default='./ISIC2018/train',
        help='Path to ISIC2018/train (contains images/ and labels/)')
    parser.add_argument(
        '--num-aug',
        type=int,
        default=13,
        help='Augmented copies per image (default 13, same as polyp)')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Regenerate even if output files already exist')
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    process_split(data_root, args.num_aug, args.overwrite)


if __name__ == '__main__':
    main()
