#!/usr/bin/env python3
"""Create manuscript tables and publication-ready figures from DG results."""

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


MODELS = [
    'fair_unet', 'fair_ibn', 'fair_mixstyle',
    'fair_pvt_control_scratch', 'fair_pvt_control_pretrained',
    'fair_ccfb_scratch', 'fair_ccfb_pretrained',
]
DISPLAY = {
    'fair_unet': 'U-Net', 'fair_ibn': 'IBN', 'fair_mixstyle': 'MixStyle',
    'fair_pvt_control_scratch': 'PVT control (scratch)',
    'fair_pvt_control_pretrained': 'PVT control (pretrained)',
    'fair_ccfb_scratch': 'CCFB (scratch)',
    'fair_ccfb_pretrained': 'CCFB (pretrained)',
}
DOMAINS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300',
           'ETIS-Larib', 'MACRO_5DOMAIN']
DOMAIN_LABELS = ['Kvasir', 'ClinicDB', 'ColonDB', 'CVC-300', 'ETIS',
                 '5-domain macro']
COLORS = ['#7F7F7F', '#E69F00', '#56B4E9', '#009E73', '#0072B2',
          '#CC79A7', '#D55E00']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--efficiency-csv')
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def setup_style():
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.unicode_minus': False,
        'savefig.facecolor': 'white',
    })


def load_records(root):
    records = []
    for path in Path(root).glob('seed_*/*/*/eval_*.json'):
        relative = path.relative_to(root)
        seed_dir, model, dataset = relative.parts[:3]
        try:
            seed = int(seed_dir.replace('seed_', ''))
            with path.open('r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        for metric, value in payload.get('metric', {}).items():
            if metric in ('mDice', 'mIoU') and isinstance(value, (int, float)):
                records.append({
                    'model': model, 'seed': seed, 'dataset': dataset,
                    'metric': metric, 'value': float(value),
                    'source_json': str(path),
                })
    # If interrupted/re-run JSONs exist, retain only the latest file's value.
    dedup = {}
    for record in records:
        key = (record['model'], record['seed'], record['dataset'], record['metric'])
        source = Path(record['source_json'])
        previous = dedup.get(key)
        if previous is None or source.stat().st_mtime > Path(previous['source_json']).stat().st_mtime:
            dedup[key] = record
    records = list(dedup.values())

    grouped = defaultdict(list)
    for record in records:
        grouped[(record['model'], record['seed'], record['metric'])].append(record)
    for (model, seed, metric), group in grouped.items():
        values = {item['dataset']: item['value'] for item in group}
        if all(domain in values for domain in DOMAINS[:-1]):
            records.append({
                'model': model, 'seed': seed, 'dataset': 'MACRO_5DOMAIN',
                'metric': metric,
                'value': statistics.mean(values[d] for d in DOMAINS[:-1]),
                'source_json': 'derived equal-weight five-domain macro',
            })
    return records


def mean_sd(values):
    values = list(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def paired_pvalue(differences):
    differences = list(differences)
    if len(differences) < 2:
        return float('nan'), 'not_testable'
    try:
        from scipy.stats import ttest_1samp
        return float(ttest_1samp(differences, 0.0).pvalue), 'paired_t'
    except ImportError:
        observed = abs(statistics.mean(differences))
        extreme = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
            value = abs(statistics.mean(s * d for s, d in zip(signs, differences)))
            extreme += value >= observed - 1e-15
            total += 1
        return extreme / total, 'exact_sign_flip'


def lookup(records):
    return {(r['model'], r['seed'], r['dataset'], r['metric']): r['value']
            for r in records}


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def create_tables(records, table_dir):
    write_tsv(table_dir / 'source_values_long.tsv', records,
              ['model', 'seed', 'dataset', 'metric', 'value', 'source_json'])
    index = lookup(records)
    seeds = [0, 3407, 2026, 42, 1234]

    for metric in ('mDice', 'mIoU'):
        rows = []
        for model in MODELS:
            row = {'model': DISPLAY[model]}
            for domain, label in zip(DOMAINS, DOMAIN_LABELS):
                values = [index[(model, seed, domain, metric)] for seed in seeds
                          if (model, seed, domain, metric) in index]
                if values:
                    mean, sd = mean_sd(values)
                    row[label] = f'{mean:.4f} +/- {sd:.4f}'
                else:
                    row[label] = 'NA'
            rows.append(row)
        write_tsv(table_dir / f'five_domain_{metric}_mean_sd.tsv', rows,
                  ['model'] + DOMAIN_LABELS)

    raw_rows = []
    for model in MODELS:
        for seed in seeds:
            key = (model, seed, 'MACRO_5DOMAIN', 'mDice')
            if key in index:
                raw_rows.append({'model': DISPLAY[model], 'seed': seed,
                                 'macro_mDice': f'{index[key]:.6f}'})
    write_tsv(table_dir / 'five_seed_macro_mDice.tsv', raw_rows,
              ['model', 'seed', 'macro_mDice'])

    ablation_rows = []
    pairs = [
        ('PVT control', 'fair_pvt_control_scratch', 'fair_pvt_control_pretrained'),
        ('CCFB', 'fair_ccfb_scratch', 'fair_ccfb_pretrained'),
    ]
    for architecture, scratch, pretrained in pairs:
        s_values, p_values, differences = [], [], []
        used_seeds = []
        for seed in seeds:
            sk = (scratch, seed, 'MACRO_5DOMAIN', 'mDice')
            pk = (pretrained, seed, 'MACRO_5DOMAIN', 'mDice')
            if sk in index and pk in index:
                s_values.append(index[sk])
                p_values.append(index[pk])
                differences.append(index[pk] - index[sk])
                used_seeds.append(seed)
        s_mean, s_sd = mean_sd(s_values)
        p_mean, p_sd = mean_sd(p_values)
        pvalue, test = paired_pvalue(differences)
        ablation_rows.append({
            'architecture': architecture,
            'n_paired_seeds': len(used_seeds),
            'seeds': ','.join(map(str, used_seeds)),
            'scratch_macro_mDice': f'{s_mean:.4f} +/- {s_sd:.4f}',
            'pretrained_macro_mDice': f'{p_mean:.4f} +/- {p_sd:.4f}',
            'pretraining_delta_percentage_points': f'{statistics.mean(differences)*100:.3f}',
            'paired_p_value': f'{pvalue:.6g}', 'test': test,
        })
    write_tsv(table_dir / 'pretraining_ablation.tsv', ablation_rows,
              list(ablation_rows[0].keys()))


def save_figure(fig, base):
    fig.savefig(str(base) + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(str(base) + '.tiff', dpi=600, bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(str(base) + '.pdf', bbox_inches='tight')
    fig.savefig(str(base) + '.svg', bbox_inches='tight')
    plt.close(fig)


def plot_domain_errorbars(records, figure_dir):
    index = lookup(records)
    seeds = [0, 3407, 2026, 42, 1234]
    x = np.arange(len(DOMAINS))
    offsets = np.linspace(-0.27, 0.27, len(MODELS))
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    for pos, model in enumerate(MODELS):
        means, sds = [], []
        for domain in DOMAINS:
            values = [index[(model, seed, domain, 'mDice')] for seed in seeds]
            mean, sd = mean_sd(values)
            means.append(mean)
            sds.append(sd)
        ax.errorbar(x + offsets[pos], means, yerr=sds, fmt=MARKERS[pos],
                    color=COLORS[pos], markersize=4.2, capsize=2,
                    linewidth=0.9, elinewidth=0.8, label=DISPLAY[model])
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_LABELS)
    ax.set_ylabel('mDice')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.5, alpha=0.7)
    ax.legend(ncol=2, frameon=False, loc='lower left', fontsize=6.5)
    ax.set_title('Five-domain segmentation performance (mean +/- SD, n=5 seeds)')
    save_figure(fig, figure_dir / 'fig_five_domain_mDice_mean_sd')


def plot_seed_points(records, figure_dir):
    index = lookup(records)
    seeds = [0, 3407, 2026, 42, 1234]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fixed_jitter = np.array([-0.08, -0.04, 0.0, 0.04, 0.08])
    for pos, model in enumerate(MODELS):
        values = [index[(model, seed, 'MACRO_5DOMAIN', 'mDice')] for seed in seeds]
        ax.scatter(pos + fixed_jitter[:len(values)], values, s=24,
                   color=COLORS[pos],
                   marker=MARKERS[pos], alpha=0.85, edgecolor='white', linewidth=0.4)
        mean, sd = mean_sd(values)
        ax.errorbar(pos, mean, yerr=sd, fmt='D', color='black', markersize=4,
                    capsize=3, linewidth=1.0, zorder=5)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([DISPLAY[m].replace(' ', '\n', 1) for m in MODELS],
                       rotation=0, ha='center', fontsize=6.5)
    ax.set_ylabel('Five-domain macro mDice')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.5, alpha=0.7)
    ax.set_title('Seed-level results; black diamonds show mean +/- SD')
    save_figure(fig, figure_dir / 'fig_seed_variability_macro_mDice')


def plot_pretraining(records, figure_dir):
    index = lookup(records)
    seeds = [0, 3407, 2026, 42, 1234]
    pairs = [
        ('PVT control', 'fair_pvt_control_scratch', 'fair_pvt_control_pretrained'),
        ('CCFB', 'fair_ccfb_scratch', 'fair_ccfb_pretrained'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, (title, scratch, pretrained) in zip(axes, pairs):
        for idx, seed in enumerate(seeds):
            y = [index[(scratch, seed, 'MACRO_5DOMAIN', 'mDice')],
                 index[(pretrained, seed, 'MACRO_5DOMAIN', 'mDice')]]
            color = COLORS[idx]
            ax.plot([0, 1], y, marker='o', color=color, linewidth=0.9,
                    markersize=4, alpha=0.9, label=f'seed {seed}')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Scratch', 'ImageNet pretrained'])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.grid(axis='y', color='#D9D9D9', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Five-domain macro mDice')
    axes[1].legend(frameon=False, fontsize=6.5, loc='lower right')
    fig.suptitle('Paired pretraining ablation across identical random seeds', fontsize=9)
    save_figure(fig, figure_dir / 'fig_pretraining_ablation_paired')


def plot_heatmap(records, figure_dir):
    index = lookup(records)
    seeds = [0, 3407, 2026, 42, 1234]
    matrix = np.zeros((len(MODELS), len(DOMAINS)))
    for i, model in enumerate(MODELS):
        for j, domain in enumerate(DOMAINS):
            matrix[i, j] = statistics.mean(
                index[(model, seed, domain, 'mDice')] for seed in seeds)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    image = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels(DOMAIN_LABELS, rotation=25, ha='right')
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([DISPLAY[m] for m in MODELS])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = 'white' if matrix[i, j] < 0.62 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                    fontsize=6.5, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label('Mean mDice')
    ax.set_title('Mean segmentation performance by test domain (n=5 seeds)')
    save_figure(fig, figure_dir / 'fig_domain_mDice_heatmap')


def plot_efficiency(records, efficiency_csv, figure_dir, table_dir):
    if not efficiency_csv or not Path(efficiency_csv).is_file():
        return False
    with Path(efficiency_csv).open('r', encoding='utf-8') as handle:
        efficiency = list(csv.DictReader(handle))
    write_tsv(table_dir / 'model_efficiency.tsv', efficiency, efficiency[0].keys())
    index = lookup(records)
    seeds = [0, 3407, 2026, 42, 1234]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    plotted = 0
    for pos, row in enumerate(efficiency):
        model = row['model']
        if not row.get('latency_ms_mean') or model not in MODELS:
            continue
        latency = float(row['latency_ms_mean'])
        params = float(row['params_m']) if row.get('params_m') else 10.0
        accuracy = statistics.mean(
            index[(model, seed, 'MACRO_5DOMAIN', 'mDice')] for seed in seeds)
        ax.scatter(latency, accuracy, s=25 + 2.2 * math.sqrt(params),
                   color=COLORS[MODELS.index(model)], marker=MARKERS[MODELS.index(model)],
                   edgecolor='black', linewidth=0.35)
        ax.annotate(DISPLAY[model], (latency, accuracy), xytext=(4, 4),
                    textcoords='offset points', fontsize=6.2)
        plotted += 1
    if plotted:
        ax.set_xlabel('Batch-1 FP32 latency at 256x256 (ms)')
        ax.set_ylabel('Five-domain macro mDice')
        ax.set_ylim(0, 1.0)
        ax.grid(color='#D9D9D9', linewidth=0.5, alpha=0.7)
        ax.set_title('Accuracy-efficiency trade-off (marker size reflects parameters)')
        save_figure(fig, figure_dir / 'fig_accuracy_efficiency_tradeoff')
    else:
        plt.close(fig)
    return bool(plotted)


def write_captions(path, has_efficiency):
    text = """Figure captions and statistical definitions

Figure: Five-domain performance. Points show the mean across five independent
random seeds; error bars show sample standard deviation (SD). Every model uses
the fixed final checkpoint at iteration 14,000. The five-domain macro is the
unweighted mean across Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300 and ETIS-Larib.

Figure: Seed variability. Colored symbols are the five individual random-seed
results. Black diamonds and error bars show mean +/- sample SD. Raw points are
shown because n=5 is too small for a reliable box-plot distribution estimate.

Figure: Pretraining ablation. Lines connect scratch and ImageNet-pretrained
models trained with the identical random seed. The paired analysis is reported
in tables/pretraining_ablation.tsv.

Figure: Qualitative comparison. Cases are selected reproducibly as the lowest
Dice cases of the fixed final-checkpoint U-Net baseline, rather than manually
chosen by appearance. The selection rule and filenames are stored in
qualitative/selected_samples.tsv.
"""
    if has_efficiency:
        text += """
Figure: Accuracy-efficiency trade-off. Latency is the FP32 batch-1 CUDA network
forward time at 256x256, excluding disk I/O and preprocessing; markers are sized
by parameter count. FLOPs reported by MMCV may omit unsupported custom ops and
must be interpreted with the status/notes columns in the efficiency table.
"""
    path.write_text(text, encoding='utf-8')


def main():
    args = parse_args()
    setup_style()
    output = Path(args.output_dir).resolve()
    table_dir = output / 'tables'
    figure_dir = output / 'figures'
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.results_root)
    expected = len(MODELS) * 5 * 5 * 2
    if len([r for r in records if r['dataset'] != 'MACRO_5DOMAIN']) < expected:
        raise SystemExit(
            f'Incomplete results: expected at least {expected} mDice/mIoU values; '
            f'found {len([r for r in records if r["dataset"] != "MACRO_5DOMAIN"])}')

    create_tables(records, table_dir)
    plot_domain_errorbars(records, figure_dir)
    plot_seed_points(records, figure_dir)
    plot_pretraining(records, figure_dir)
    plot_heatmap(records, figure_dir)
    has_efficiency = plot_efficiency(
        records, args.efficiency_csv, figure_dir, table_dir)
    write_captions(output / 'FIGURE_CAPTIONS_EN.txt', has_efficiency)
    print(f'[PAPER] tables and figures written to {output}')


if __name__ == '__main__':
    main()

