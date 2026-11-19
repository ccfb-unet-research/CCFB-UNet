"""Aggregate five-domain, multi-seed MMSeg evaluation JSON files.

Produces a long-form table, mean +/- sample SD, and paired significance tests
against a chosen reference.  scipy is used for paired t tests when available;
an exact two-sided sign-flip permutation test is the dependency-free fallback.
"""

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--reference', default='fair_ccfb_pretrained')
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def read_records(root):
    records = []
    dataset_dirs = Path(root).glob('seed_*/*/*')
    for dataset_dir in dataset_dirs:
        candidates = list(dataset_dir.glob('eval_*.json'))
        if not candidates:
            continue
        # A re-run may leave several timestamped files; use the newest only.
        path = max(candidates, key=lambda item: item.stat().st_mtime)
        relative = path.relative_to(root)
        seed_dir, model, dataset = relative.parts[:3]
        seed = int(seed_dir.replace('seed_', ''))
        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        for metric, value in payload.get('metric', {}).items():
            if isinstance(value, (int, float)):
                records.append({
                    'model': model,
                    'seed': seed,
                    'dataset': dataset,
                    'metric': metric,
                    'value': float(value),
                })
    # Pre-specified macro average: equal weight for each of the five domains.
    macro_groups = {}
    for record in records:
        key = (record['model'], record['seed'], record['metric'])
        macro_groups.setdefault(key, []).append(record['value'])
    for (model, seed, metric), values in macro_groups.items():
        if len(values) == 5:
            records.append({
                'model': model,
                'seed': seed,
                'dataset': 'MACRO_5DOMAIN',
                'metric': metric,
                'value': statistics.mean(values),
            })
    return records


def write_tsv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def paired_pvalue(differences):
    if len(differences) < 2 or all(abs(x) < 1e-15 for x in differences):
        return 1.0, 'not_testable'
    try:
        from scipy.stats import ttest_rel
        reference = [0.0] * len(differences)
        result = ttest_rel(differences, reference)
        return float(result.pvalue), 'paired_t'
    except ImportError:
        observed = abs(statistics.mean(differences))
        extreme = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
            permuted = abs(statistics.mean(
                sign * value for sign, value in zip(signs, differences)))
            extreme += permuted >= observed - 1e-15
            total += 1
        return extreme / total, 'exact_sign_flip'


def main():
    args = parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_records(root)
    if not records:
        raise SystemExit(f'No eval JSON files found below {root}')

    records.sort(key=lambda x: (x['model'], x['dataset'], x['metric'], x['seed']))
    write_tsv(
        output / 'all_seed_results.tsv', records,
        ['model', 'seed', 'dataset', 'metric', 'value'])

    grouped = {}
    for record in records:
        key = (record['model'], record['dataset'], record['metric'])
        grouped.setdefault(key, []).append(record['value'])
    summary = []
    for (model, dataset, metric), values in sorted(grouped.items()):
        summary.append({
            'model': model,
            'dataset': dataset,
            'metric': metric,
            'n_seeds': len(values),
            'mean': f'{statistics.mean(values):.6f}',
            'sd': f'{statistics.stdev(values):.6f}' if len(values) > 1 else 'NA',
        })
    write_tsv(
        output / 'mean_sd.tsv', summary,
        ['model', 'dataset', 'metric', 'n_seeds', 'mean', 'sd'])

    by_key = {
        (r['model'], r['dataset'], r['metric'], r['seed']): r['value']
        for r in records
    }
    models = sorted({r['model'] for r in records})
    datasets = sorted({r['dataset'] for r in records})
    metrics = sorted({r['metric'] for r in records})
    tests = []
    for model in models:
        if model == args.reference:
            continue
        for dataset in datasets:
            for metric in metrics:
                paired = []
                for seed in sorted({r['seed'] for r in records}):
                    ref_key = (args.reference, dataset, metric, seed)
                    model_key = (model, dataset, metric, seed)
                    if ref_key in by_key and model_key in by_key:
                        paired.append(by_key[ref_key] - by_key[model_key])
                if len(paired) < 2:
                    continue
                pvalue, test_name = paired_pvalue(paired)
                tests.append({
                    'reference': args.reference,
                    'comparison': model,
                    'dataset': dataset,
                    'metric': metric,
                    'n_pairs': len(paired),
                    'mean_reference_minus_comparison':
                        f'{statistics.mean(paired):.6f}',
                    'p_value': f'{pvalue:.6g}',
                    'test': test_name,
                    'significant_0.05': str(pvalue < 0.05),
                })
    write_tsv(
        output / 'paired_tests.tsv', tests,
        ['reference', 'comparison', 'dataset', 'metric', 'n_pairs',
         'mean_reference_minus_comparison', 'p_value', 'test',
         'significant_0.05'])
    print(f'Wrote {len(records)} measurements to {output}')


if __name__ == '__main__':
    main()

