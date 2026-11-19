#!/usr/bin/env python3
"""Aggregate reviewer experiments with deduplication, uncertainty and Holm tests."""

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

import numpy as np


METRICS = ['dice', 'iou', 'precision', 'recall', 'hd95', 'assd', 'boundary_f']
DOMAINS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-Larib']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--reference', default='fair_ccfb_pretrained')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20261106)
    return p.parse_args()


def read_tsv(path):
    with path.open(encoding='utf-8') as h:
        return list(csv.DictReader(h, delimiter='\t'))


def write_tsv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter='\t'); w.writeheader(); w.writerows(rows)


def load(root):
    rows = []
    for path in Path(root).glob('seed_*/*/*/per_image.tsv'):
        rel = path.relative_to(root).parts
        seed, model, dataset = int(rel[0].replace('seed_', '')), rel[1], rel[2]
        for row in read_tsv(path):
            out = dict(seed=seed, model=model, dataset=dataset, image=row['image'],
                       image_sha256=row['image_sha256'])
            for key in METRICS + ['lesion_area_ratio', 'contrast_abs', 'boundary_complexity']:
                out[key] = float(row[key])
            rows.append(out)
    if not rows:
        raise SystemExit('No per_image.tsv below %s' % root)
    return rows


def grouped_image_means(rows):
    groups = {}
    for r in rows:
        key = (r['seed'], r['model'], r['dataset'], r['image_sha256'])
        groups.setdefault(key, []).append(r)
    result = []
    for (seed, model, dataset, image_hash), values in groups.items():
        row = dict(seed=seed, model=model, dataset=dataset, image_sha256=image_hash,
                   duplicate_count=len(values))
        for metric in METRICS + ['lesion_area_ratio', 'contrast_abs', 'boundary_complexity']:
            row[metric] = statistics.mean(v[metric] for v in values)
        result.append(row)
    return result


def summaries(rows, protocol):
    groups = {}
    for r in rows:
        for metric in METRICS:
            groups.setdefault((r['seed'], r['model'], r['dataset'], metric), []).append(r[metric])
    seed_domain = []
    for (seed, model, dataset, metric), values in sorted(groups.items()):
        seed_domain.append(dict(protocol=protocol, seed=seed, model=model, dataset=dataset,
                                metric=metric, value=statistics.mean(values), n_images=len(values)))
    across = {}
    for r in seed_domain:
        across.setdefault((r['model'], r['dataset'], r['metric']), []).append(r['value'])
    out = []
    for (model, dataset, metric), values in sorted(across.items()):
        out.append(dict(protocol=protocol, model=model, dataset=dataset, metric=metric,
                        n_seeds=len(values), mean=statistics.mean(values),
                        sd=statistics.stdev(values) if len(values) > 1 else float('nan')))
    return seed_domain, out


def macro(seed_domain):
    groups = {}
    for r in seed_domain:
        if r['dataset'] in DOMAINS:
            groups.setdefault((r['seed'], r['model'], r['metric']), []).append(r['value'])
    return [dict(seed=s, model=m, metric=k, value=statistics.mean(v))
            for (s, m, k), v in groups.items() if len(v) == len(DOMAINS)]


def exact_sign_flip(diff):
    observed = abs(statistics.mean(diff)); extreme = total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(diff)):
        value = abs(statistics.mean(s * x for s, x in zip(signs, diff)))
        extreme += value >= observed - 1e-15; total += 1
    return extreme / total


def paired_stats(macro_rows, reference):
    index = {(r['model'], r['metric'], r['seed']): r['value'] for r in macro_rows}
    models = sorted({r['model'] for r in macro_rows if r['model'] != reference})
    metrics = sorted({r['metric'] for r in macro_rows})
    rows = []
    for model in models:
        for metric in metrics:
            pairs = [(index[(reference, metric, s)], index[(model, metric, s)])
                     for s in sorted({r['seed'] for r in macro_rows})
                     if (reference, metric, s) in index and (model, metric, s) in index]
            if len(pairs) < 2: continue
            diff = [a - b for a, b in pairs]
            mean, sd = statistics.mean(diff), statistics.stdev(diff)
            try:
                from scipy.stats import t
                critical = float(t.ppf(0.975, len(diff) - 1))
            except Exception:
                critical = 2.776 if len(diff) == 5 else 1.96
            half = critical * sd / math.sqrt(len(diff))
            rows.append(dict(reference=reference, comparison=model, metric=metric, n=len(diff),
                             mean_difference=mean, ci95_low=mean-half, ci95_high=mean+half,
                             p_raw=exact_sign_flip(diff), effect_cohen_dz=(mean/sd if sd else 0.0)))
    order = sorted(range(len(rows)), key=lambda i: rows[i]['p_raw'])
    adjusted = [1.0] * len(rows); running = 0.0; n = len(rows)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, rows[idx]['p_raw'] * (n - rank)))
        adjusted[idx] = running
    for row, value in zip(rows, adjusted): row['p_holm'] = value
    return rows


def difficulty(rows):
    output = []
    for dataset in sorted({r['dataset'] for r in rows}):
        base = [r for r in rows if r['dataset'] == dataset]
        for variable in ['lesion_area_ratio', 'contrast_abs', 'boundary_complexity']:
            q1, q2 = np.quantile([r[variable] for r in base], [1/3, 2/3])
            for r in base:
                r['_bin'] = 'low' if r[variable] <= q1 else ('mid' if r[variable] <= q2 else 'high')
            groups = {}
            for r in base:
                for metric in ['dice', 'hd95', 'boundary_f']:
                    groups.setdefault((r['model'], r['_bin'], metric), []).append(r[metric])
            for (model, bin_name, metric), values in groups.items():
                output.append(dict(dataset=dataset, variable=variable, bin=bin_name, model=model,
                                   metric=metric, n=len(values), mean=statistics.mean(values)))
    return output


def hierarchical_bootstrap(rows, reference, repeats, rng):
    """Resample seeds, then unique image hashes within each sampled seed."""
    index = {(r['seed'], r['model'], r['dataset'], r['image_sha256']): r for r in rows}
    models = sorted({r['model'] for r in rows if r['model'] != reference})
    output = []
    for model in models:
        for dataset in DOMAINS:
            for metric in ['dice', 'iou', 'hd95', 'assd', 'boundary_f']:
                by_seed = {}
                for key, ref_row in index.items():
                    seed, ref_model, ref_dataset, image_hash = key
                    other_key = (seed, model, dataset, image_hash)
                    if ref_model == reference and ref_dataset == dataset and other_key in index:
                        by_seed.setdefault(seed, []).append(
                            ref_row[metric] - index[other_key][metric])
                if len(by_seed) < 2:
                    continue
                seeds = sorted(by_seed)
                observed = statistics.mean(statistics.mean(by_seed[s]) for s in seeds)
                draws = []
                for _ in range(repeats):
                    sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
                    seed_values = []
                    for seed in sampled_seeds:
                        values = np.asarray(by_seed[int(seed)], dtype=float)
                        seed_values.append(float(rng.choice(values, size=len(values), replace=True).mean()))
                    draws.append(statistics.mean(seed_values))
                low, high = np.quantile(draws, [0.025, 0.975])
                output.append(dict(reference=reference, comparison=model, dataset=dataset,
                                   metric=metric, n_seeds=len(seeds),
                                   observed_difference=observed,
                                   bootstrap_ci95_low=float(low), bootstrap_ci95_high=float(high),
                                   bootstrap_repeats=repeats))
    return output


def main():
    a = parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    raw = load(a.results_root); unique = grouped_image_means(raw)
    raw_seed, raw_summary = summaries(raw, 'official_raw')
    unique_seed, unique_summary = summaries(unique, 'within_domain_sha256_cluster_weighted')
    macro_rows = macro(unique_seed)
    tests = paired_stats(macro_rows, a.reference)
    write_tsv(out/'summary_mean_sd.tsv', raw_summary + unique_summary,
              ['protocol','model','dataset','metric','n_seeds','mean','sd'])
    write_tsv(out/'macro_seed.tsv', macro_rows, ['seed','model','metric','value'])
    write_tsv(out/'paired_tests_holm.tsv', tests,
              ['reference','comparison','metric','n','mean_difference','ci95_low','ci95_high','p_raw','p_holm','effect_cohen_dz'])
    diff = difficulty(unique)
    write_tsv(out/'difficulty_summary.tsv', diff,
              ['dataset','variable','bin','model','metric','n','mean'])
    boot = hierarchical_bootstrap(unique, a.reference, a.bootstrap, np.random.default_rng(a.seed))
    write_tsv(out/'hierarchical_bootstrap.tsv', boot,
              ['reference','comparison','dataset','metric','n_seeds','observed_difference',
               'bootstrap_ci95_low','bootstrap_ci95_high','bootstrap_repeats'])
    manifest = dict(raw_rows=len(raw), unique_image_rows=len(unique), reference=a.reference,
                    domains=DOMAINS, metrics=METRICS,
                    note='Sensitivity analysis gives each SHA256 cluster equal weight within each benchmark domain; raw official scores are retained. Cross-domain overlaps remain visible in the audit.')
    (out/'aggregation_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
