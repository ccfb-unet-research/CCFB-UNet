#!/usr/bin/env python3
"""Fail-fast check for dependencies, paths, config merging and model registry."""

import argparse
import gc
import os
from pathlib import Path

import torch
import cv2
import matplotlib
import numpy
import scipy
from PIL import Image
from mmcv import Config
from mmcv.utils import import_modules_from_strings
from mmseg.models import build_segmentor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--pvt-path', required=True)
    p.add_argument('--build-models', action='store_true')
    a = p.parse_args()
    root = Path(a.data_root)
    expected = [root/'train'/'images_albu', root/'train'/'labels_albu']
    expected += [root/'test'/d for d in ['Kvasir','CVC-ClinicDB','CVC-ColonDB','CVC-300','ETIS-Larib']]
    missing = [str(x) for x in expected if not x.exists()]
    if missing:
        raise SystemExit('Missing data paths:\n' + '\n'.join(missing))
    if not Path(a.pvt_path).is_file():
        raise SystemExit('Missing PVT checkpoint: ' + a.pvt_path)
    configs = sorted(Path('my_config/reviewer').glob('*.py'))
    configs = [x for x in configs if not x.name.startswith('_')]
    for path in configs:
        cfg = Config.fromfile(str(path))
        if cfg.get('custom_imports'):
            import_modules_from_strings(**cfg.custom_imports)
        assert cfg.runner.max_iters == 14000
        assert cfg.data.samples_per_gpu == 16
        if a.build_models:
            cfg.model.pretrained = None
            if 'pretrained_pvt' in cfg.model.decode_head:
                cfg.model.decode_head.pretrained_pvt = ''
            model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
            del model; gc.collect()
        print('[OK]', path)
    print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
    print('NumPy:', numpy.__version__, 'SciPy:', scipy.__version__,
          'OpenCV:', cv2.__version__, 'Matplotlib:', matplotlib.__version__)
    print('Validated reviewer configs:', len(configs))


if __name__ == '__main__':
    main()
