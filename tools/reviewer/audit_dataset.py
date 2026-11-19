#!/usr/bin/env python3
"""Exact and perceptual duplicate audit for the polyp train/test tree."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DOMAINS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-Larib']


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--phash-threshold', type=int, default=4)
    return p.parse_args()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phash(path):
    arr = np.asarray(Image.open(path).convert('L').resize((32, 32), Image.LANCZOS), np.float32)
    coeff = cv2.dct(arr)[:8, :8].reshape(-1)
    return coeff > np.median(coeff[1:])


def label_for(root, domain, image):
    label_dir = root / 'test' / domain / 'labels'
    exact = label_dir / image.name
    if exact.is_file():
        return exact
    candidates = sorted(label_dir.glob(image.stem + '.*'))
    return candidates[0] if candidates else None


def main():
    a = args(); root = Path(a.data_root); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    train_all = sorted((root / 'train' / 'images_albu').glob('*'))
    train_original = [p for p in train_all if p.is_file() and '_aug_' not in p.stem]
    tests = [(d, p) for d in DOMAINS for p in sorted((root / 'test' / d / 'images').glob('*')) if p.is_file()]
    train_sha = {}
    for p in train_all:
        if p.is_file(): train_sha.setdefault(sha(p), []).append(p.name)
    exact_train_test = []
    test_sha = {}
    for d, p in tests:
        h = sha(p); test_sha.setdefault(h, []).append((d, p))
        if h in train_sha:
            exact_train_test.append(dict(domain=d, test=p.name, train='|'.join(train_sha[h]), sha256=h))
    duplicate_test = []
    for h, values in test_sha.items():
        if len(values) <= 1:
            continue
        mask_hashes = []
        for domain, image in values:
            label = label_for(root, domain, image)
            mask_hashes.append(sha(label) if label else 'MISSING')
        duplicate_test.append(dict(
            sha256=h, count=len(values),
            files='|'.join('%s/%s' % (d, p.name) for d, p in values),
            unique_mask_hashes=len(set(mask_hashes)),
            label_inconsistent=int(len(set(mask_hashes)) > 1),
            mask_sha256='|'.join(mask_hashes)))
    train_bits = np.stack([phash(p) for p in train_original])
    near = []
    for d, p in tests:
        bits = phash(p); distance = np.count_nonzero(train_bits != bits, axis=1); minimum = int(distance.min())
        if minimum <= a.phash_threshold:
            for idx in np.where(distance == minimum)[0][:5]:
                near.append(dict(domain=d, test=p.name, train=train_original[int(idx)].name,
                                 phash_distance=minimum))
    def write(name, rows, fields):
        with (out / name).open('w', newline='', encoding='utf-8') as h:
            w = csv.DictWriter(h, fieldnames=fields, delimiter='\t'); w.writeheader(); w.writerows(rows)
    write('exact_train_test.tsv', exact_train_test, ['domain', 'test', 'train', 'sha256'])
    write('near_train_test.tsv', near, ['domain', 'test', 'train', 'phash_distance'])
    write('duplicate_test.tsv', duplicate_test,
          ['sha256', 'count', 'files', 'unique_mask_hashes', 'label_inconsistent', 'mask_sha256'])
    report = dict(train_files=len(train_all), train_originals=len(train_original), test_files=len(tests),
                  exact_train_test=len(exact_train_test), near_train_test=len(near),
                  near_train_test_unique_images=len({(r['domain'], r['test']) for r in near}),
                  duplicate_test_groups=len(duplicate_test), phash_threshold=a.phash_threshold,
                  duplicate_groups_with_inconsistent_labels=sum(r['label_inconsistent'] for r in duplicate_test),
                  warning='pHash candidates require sequence/patient-level manual verification')
    (out / 'audit_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
