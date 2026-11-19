#!/usr/bin/env python3
"""Create manuscript-ready SVG/PNG figures from reviewer summary TSV files."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    'fair_unet': 'U-Net', 'fair_ibn': 'IBN-Net', 'fair_mixstyle': 'MixStyle',
    'fair_pvt_control_scratch': 'PVT control (scratch)',
    'fair_pvt_control_pretrained': 'PVT control (IN-1K)',
    'fair_ccfb_scratch': 'CCFB-UNet (scratch)',
    'fair_ccfb_pretrained': 'CCFB-UNet (IN-1K)',
    'fair_robustnet_pvt_adapted': 'RobustNet-PVT',
    'fair_sansaw_pvt_adapted': 'SAN-SAW-PVT',
    'fair_blindnet_pvt_adapted': 'BlindNet-PVT',
    'fair_rein_pvt_adapted': 'Rein-PVT',
    'fair_dpmformer_pvt_adapted': 'DPMFormer-PVT',
    'ablation_gls_pretrained': 'GLS only',
}
DOMAINS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-Larib']


def read(path):
    with Path(path).open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def save(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / (stem + '.svg'), bbox_inches='tight')
    fig.savefig(out / (stem + '.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary-dir', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    summary, out = Path(a.summary_dir), Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False})

    means = read(summary / 'summary_mean_sd.tsv')
    selected = [m for m in LABELS if any(r['model'] == m for r in means)]
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    x = np.arange(len(DOMAINS)); width = 0.82 / max(len(selected), 1)
    for i, model in enumerate(selected):
        rows = {(r['dataset']): r for r in means if r['protocol'] == 'within_domain_sha256_cluster_weighted'
                and r['metric'] == 'dice' and r['model'] == model}
        y = [float(rows[d]['mean']) if d in rows else np.nan for d in DOMAINS]
        e = [float(rows[d]['sd']) if d in rows else 0 for d in DOMAINS]
        ax.bar(x - .41 + width/2 + i*width, y, width, yerr=e, capsize=1.2,
               label=LABELS[model], linewidth=.3, edgecolor='white')
    ax.set_ylabel('Dice (mean ± SD across seeds)'); ax.set_ylim(0, 1)
    ax.set_xticks(x, DOMAINS, rotation=15, ha='right')
    ax.legend(ncol=3, fontsize=6.5, frameon=False)
    save(fig, out, 'fig_five_domain_dice_errorbars')

    macro = [r for r in read(summary / 'macro_seed.tsv') if r['metric'] == 'dice']
    models = [m for m in selected if any(r['model'] == m for r in macro)]
    values = [[float(r['value']) for r in macro if r['model'] == m] for m in models]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.boxplot(values, showmeans=True, widths=.55)
    rng = np.random.default_rng(20261106)
    for i, vals in enumerate(values, 1):
        ax.scatter(i + rng.uniform(-.07, .07, len(vals)), vals, s=12, color='#1f77b4', zorder=3)
    ax.set_xticks(range(1, len(models)+1), [LABELS[m] for m in models], rotation=28, ha='right')
    ax.set_ylabel('Five-domain macro Dice'); ax.set_ylim(0, 1)
    save(fig, out, 'fig_multiseed_macro_dice')

    pairs = [('fair_pvt_control_scratch', 'fair_pvt_control_pretrained', 'PVT control'),
             ('fair_ccfb_scratch', 'fair_ccfb_pretrained', 'CCFB-UNet')]
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    pos = np.arange(len(pairs)); w = .34
    for j, state in enumerate(['scratch', 'pretrained']):
        ys, es = [], []
        for scratch, pre, _ in pairs:
            vals = [float(r['value']) for r in macro if r['model'] == (scratch if j == 0 else pre)]
            ys.append(np.mean(vals) if vals else np.nan)
            es.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        ax.bar(pos + (j-.5)*w, ys, w, yerr=es, capsize=3, label=state.title())
    ax.set_xticks(pos, [x[2] for x in pairs]); ax.set_ylabel('Five-domain macro Dice')
    ax.set_ylim(0, 1); ax.legend(frameon=False)
    save(fig, out, 'fig_pretraining_control')

    difficulty = read(summary / 'difficulty_summary.tsv')
    focus = ['fair_ccfb_pretrained', 'ablation_gls_pretrained', 'fair_pvt_control_pretrained']
    rows = [r for r in difficulty if r['dataset'] == 'CVC-ColonDB'
            and r['variable'] == 'lesion_area_ratio' and r['metric'] == 'dice'
            and r['model'] in focus]
    fig, ax = plt.subplots(figsize=(5.2, 3.7)); bins = ['low', 'mid', 'high']; x = np.arange(3)
    for model in focus:
        index = {r['bin']: float(r['mean']) for r in rows if r['model'] == model}
        if index:
            ax.plot(x, [index.get(b, np.nan) for b in bins], marker='o', label=LABELS.get(model, model))
    ax.set_xticks(x, ['Small', 'Medium', 'Large']); ax.set_xlabel('Lesion-area tertile')
    ax.set_ylabel('ColonDB Dice'); ax.set_ylim(0, 1); ax.legend(frameon=False)
    save(fig, out, 'fig_colondb_lesion_size_mechanism')
    print('[FIGURES] wrote', out)


if __name__ == '__main__':
    main()
