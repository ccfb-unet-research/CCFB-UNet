# Copyright (c) OpenMMLab. All rights reserved.
"""Polyp segmentation demo for a single image or a directory.

Example (from repo root):
  python demo/image_demo.py
  python demo/image_demo.py --img polyp_datasets/test/Kvasir/images --out demo/outputs/Kvasir
  python demo/image_demo.py --config my_config/unet_gls.py \\
      --checkpoint work_dirs_polyp_v2/unet_gls/best_mdice.pth
"""
import argparse
import os
import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')

import mmcv
from tqdm import tqdm

# repo root on PYTHONPATH
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def collect_images(path):
    if osp.isfile(path):
        return [path]
    files = []
    for name in sorted(os.listdir(path)):
        if osp.splitext(name)[1].lower() in IMG_EXTS:
            files.append(osp.join(path, name))
    return files


def main():
    parser = argparse.ArgumentParser(description='Polyp image demo')
    parser.add_argument(
        '--config',
        default=osp.join(REPO_ROOT, 'my_config/unet_all_eval.py'),
        help='Config file')
    parser.add_argument(
        '--checkpoint',
        default=osp.join(REPO_ROOT, 'work_dirs_polyp_v2/unet_all/best_mdice.pth'),
        help='Checkpoint file')
    parser.add_argument(
        '--img',
        default=osp.join(REPO_ROOT, 'polyp_datasets/test/Kvasir/images'),
        help='Image file or directory')
    parser.add_argument(
        '--out',
        default=osp.join(REPO_ROOT, 'demo/outputs'),
        help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Overlay opacity in (0, 1]')
    parser.add_argument(
        '--mask-only',
        action='store_true',
        help='Save binary mask (white=FG) instead of overlay')
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Only process first N images (0 = all)')
    args = parser.parse_args()

    imgs = collect_images(args.img)
    if not imgs:
        raise FileNotFoundError(f'No images under: {args.img}')
    if args.limit > 0:
        imgs = imgs[:args.limit]

    mmcv.mkdir_or_exist(args.out)
    print(f'config     : {args.config}')
    print(f'checkpoint : {args.checkpoint}')
    print(f'images     : {len(imgs)} from {args.img}')
    print(f'out        : {args.out}')

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # ckpt palette is [[0,0,0],[1,1,1]] — use visible white FG for demo
    palette = [[0, 0, 0], [255, 255, 255]]

    for path in tqdm(imgs, desc='infer'):
        result = inference_segmentor(model, path)
        out_file = osp.join(args.out, osp.basename(path))
        if args.mask_only:
            import numpy as np
            seg = result[0]
            if hasattr(seg, 'cpu'):
                seg = seg.cpu().numpy()
            mask = (np.asarray(seg) > 0).astype('uint8') * 255
            mmcv.imwrite(mask, out_file)
        else:
            show_result_pyplot(
                model,
                path,
                result,
                palette=palette,
                opacity=args.opacity,
                block=False,
                out_file=out_file)

    print(f'done -> {args.out}')


if __name__ == '__main__':
    main()
