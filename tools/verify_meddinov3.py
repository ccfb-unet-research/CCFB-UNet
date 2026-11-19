#!/usr/bin/env python
"""Smoke test for MedDINOv3Head — run in py38 env before dist_train."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--num-classes', type=int, default=2)
    args = parser.parse_args()

    import torch
    from mmseg.models.builder import HEADS
    from mmseg.models.decode_heads.meddinov3 import MedDINOv3Net

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    net = MedDINOv3Net(num_classes=args.num_classes).to(device)
    print('params: %.1fM' % (sum(p.numel() for p in net.parameters()) / 1e6))

    x = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    net.eval()
    with torch.no_grad():
        y = net(x)
    assert y.shape == (args.batch_size, args.num_classes, args.size, args.size), y.shape
    print('forward:', tuple(y.shape))

    net.train()
    net(x).sum().backward()
    print('backward: ok')

    head = HEADS.build(dict(
        type='MedDINOv3Head',
        num_classes=args.num_classes,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.1),
        ],
    )).to(device)
    head.train()
    gt = torch.randint(0, args.num_classes, (args.batch_size, 1, args.size, args.size), device=device)
    losses = head.forward_train(x, [{}] * args.batch_size, gt, {})
    print('losses:', {k: float(v) for k, v in losses.items()})
    print('ALL PASSED')


if __name__ == '__main__':
    main()
