"""Run the adaptive-fusion comparison without retraining any baseline.

Example (from the project folder):
    python experiments.py --root datasets/MVTecAD
Default: screw and transistor, seed 0. Existing checkpoints/split files must be
in results_lightweight/<category>/seed_<seed>/. Results go to results_hybrid_active/. Existing results_hybrid prediction caches are reused.
To run all five existing categories after the pilot:
    python experiments.py --root datasets/MVTecAD --categories bottle carpet leather screw transistor
Use --seeds 0 1 2 for a repeated evaluation; this is not parameter searching.
The old notebook/app use the discontinued student API; do not Run All there.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import detectors as D


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def path_records(paths, root):
    return [(str(Path(p).resolve().relative_to(root)), Path(p).stat().st_size,
             Path(p).stat().st_mtime_ns) for p in paths]


def atomic_npy(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, data)
    tmp.replace(path)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_development(paths, seed, folder):
    """At most 24 validation source images; original+three known synthetic defects.

    Every variant of a source image stays in the same phase. No baseline-fit,
    threshold-calibration or real-test images are used for fitting the gate.
    """
    paths = list(paths)
    if len(paths) < 6:
        raise ValueError('Need at least six saved validation normals to fit/select fusion.')
    rng = np.random.default_rng(seed)
    chosen = [paths[i] for i in rng.permutation(len(paths))[:24]]
    n_select = max(3, len(chosen)//3)
    images, masks, rows = [], [], []
    for i, path in enumerate(chosen):
        image = D.crop_view(path)
        phase = 'gate_select' if i < n_select else 'gate_fit'
        for kind in ('normal', 'scratch', 'texture', 'cutpaste'):
            if kind == 'normal':
                generated, mask = image, np.zeros((D.CROP, D.CROP), dtype=np.uint8)
            else:
                generated, mask = D.synthetic_defect(image, rng, kind)
            images.append(np.asarray(generated, dtype=np.uint8))
            masks.append(mask)
            rows.append({'source': str(path.resolve()), 'phase': phase, 'kind': kind})
    images, masks = np.stack(images), np.stack(masks).astype(bool)
    atomic_npy(folder / 'development_images.npy', images)
    atomic_npy(folder / 'development_masks.npy', masks)
    D.atomic_json(folder / 'development_manifest.json', rows)
    return images, masks, rows


def collect_predictions(model, items, folder, signature, cropped=False, legacy_signature=None):
    """Resume cached expert forward passes. Maps retain float32, not lossy float16."""
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'complete.json'
    score_file, map_file = folder / 'scores.npy', folder / 'maps.npy'
    if marker.is_file() and score_file.is_file() and map_file.is_file():
        saved = json.loads(marker.read_text())
        if saved['signature'] == signature or (legacy_signature is not None and saved['signature'] == legacy_signature):
            scores, maps = np.load(score_file, mmap_mode='r'), np.load(map_file, mmap_mode='r')
            if scores.shape == (len(items),) and maps.shape == (len(items), D.CROP, D.CROP):
                first = items[0]
                x = D.NORMALIZE(Image.fromarray(first)) if cropped else D.image_tensor(first)
                check_score, check_map = model.score_tensor(x.unsqueeze(0))
                agrees = (np.allclose(scores[0], check_score[0], rtol=2e-3, atol=1e-5)
                          and np.allclose(maps[0], check_map[0], rtol=2e-3, atol=1e-5))
                if not agrees:
                    print(f'  Cached {model.name} differs from live inference; recomputing.')
                else:
                    if saved['signature'] != signature:
                        D.atomic_json(marker, {'signature': signature, 'count': len(items), 'migrated_from': saved['signature']})
                    print(f'  Reusing {model.name}: {folder.name}')
                    return scores, maps
    if marker.exists():
        marker.unlink()
    scores = np.lib.format.open_memmap(score_file, mode='w+', dtype=np.float32, shape=(len(items),))
    maps = np.lib.format.open_memmap(map_file, mode='w+', dtype=np.float32,
                                   shape=(len(items), D.CROP, D.CROP))
    for i, item in enumerate(tqdm(items, desc=f'{model.name}: {folder.name}')):
        x = D.NORMALIZE(Image.fromarray(item)) if cropped else D.image_tensor(item)
        score, amap = model.score_tensor(x.unsqueeze(0))
        scores[i], maps[i] = score[0], amap[0]
    scores.flush(); maps.flush()
    D.atomic_json(marker, {'signature': signature, 'count': len(items)})
    return scores, maps


def measure(model, paths):
    result = D.benchmark(model, paths, repeats=3, max_images=20)
    end_to_end = []
    for path in paths[:20]:
        D.synchronize(model.device)
        start = time.perf_counter()
        model.predict(path)
        D.synchronize(model.device)
        end_to_end.append(1000*(time.perf_counter()-start))
    result['end_to_end_mean_ms'] = float(np.mean(end_to_end))
    result['timing_images'] = min(len(paths), 20)
    result['memory_mb'] = model.memory_mb()
    result['parameters_million'] = sum(p.numel() for p in model.net.parameters())/1e6
    return result


def score_metrics(labels, scores, maps, masks, threshold):
    metrics = D.classification_metrics(labels, scores, threshold)
    metrics['pixel_auroc'] = float(roc_auc_score(masks.ravel(), maps.ravel()))
    return metrics


OLD_DETECTOR_HASHES = ['9122027d72dee1456057e1dc31c71851db3330ceca296b7f588f7380ccc14021', 'cba3e575eb8b70faf6bd5e50931da23b7289278b09ae454be93f63708362c349']
OLD_SCRIPT_HASHES = ['4df9f989693c51a6db9a96af076ea0fbe69c2765c0daa1667e809567b46699de', 'd51d25448d6065170792a572947e46bf495d41df5c4d567760d1f3f00ab43e18']


def verified_legacy_metadata(path, current):
    """Reuse old predictions only from the known old code and unchanged sources/weights."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    metadata = data.get('metadata', {})
    if metadata.get('detector_source_hash') not in OLD_DETECTOR_HASHES or metadata.get('script_hash') not in OLD_SCRIPT_HASHES:
        return {}
    for key in ('category', 'seed', 'baseline_hashes', 'development_sources', 'calibration_sources'):
        if json.dumps(metadata.get(key), sort_keys=True) != json.dumps(current.get(key), sort_keys=True):
            return {}
    original_keys = ('category', 'seed', 'baseline_hashes', 'detector_source_hash', 'script_hash',
                     'development_sources', 'calibration_sources', 'fusion_config')
    original = {key: metadata[key] for key in original_keys}
    if fingerprint(original) != metadata.get('signature'):
        return {}
    return metadata


def run_one(args, category, seed):
    root = args.root.resolve()
    old = args.baselines / category / f'seed_{seed}'
    out = args.output / category / f'seed_{seed}'
    cache = args.cache_dir / category / f'seed_{seed}' / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    complete = out / 'run_complete.json'
    if complete.exists():
        complete.unlink()
    split = D.load_saved_split(root, old / 'split.json')
    checkpoints = {name: old / f'{name}.pt' for name in D.BASELINES}
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing trained baseline: {path}. Baselines are not retrained here.')
    print(f'\n=== {category}, seed {seed} ===')
    # Define all development data before touching real test images/labels.
    dev_images, dev_masks, dev_rows = prepare_development(split['validation'], seed, cache)
    hashes = {name: digest_file(path) for name, path in checkpoints.items()}
    source_hash = digest_file(Path(D.__file__))
    fusion_cfg = D.FusionConfig()
    protocol = {'category': category, 'seed': seed, 'baseline_hashes': hashes,
                'detector_source_hash': source_hash, 'script_hash': digest_file(__file__),
                'development_sources': path_records(split['validation'], root),
                'calibration_sources': path_records(split['calibration'], root),
                'fusion_config': D.asdict(fusion_cfg)}
    signature = fingerprint(protocol)
    legacy = verified_legacy_metadata(args.cache_dir / category / f'seed_{seed}' / 'hybrid.json', protocol)
    baseline_code = D.prediction_code_signature()
    def prediction_key(name, kind, inputs):
        return fingerprint({'baseline_code': baseline_code, 'checkpoint': hashes[name],
                            'kind': kind, 'inputs': inputs})
    development_input_key = hashlib.sha256(dev_images.tobytes()).hexdigest()
    dev_s, dev_m, cal_s, cal_m = [], [], [], []
    thresholds = {}
    for name, checkpoint in checkpoints.items():
        model = D.load_detector(checkpoint, args.device)
        if model.cfg.seed != seed or model.name != name:
            raise ValueError(f'Checkpoint category/seed/model mismatch: {checkpoint}')
        if model.cfg.calibration_quantile != fusion_cfg.calibration_quantile:
            raise ValueError('Baseline and fusion threshold calibration policies differ.')
        saved_split = {k: [str(p.relative_to(root)) for p in v] for k, v in split.items()}
        saved_calibration = np.asarray(model.calibration_scores)
        if len(saved_calibration) != len(split['calibration']):
            raise ValueError('Checkpoint calibration count differs from saved split.')
        ds, dm = collect_predictions(model, dev_images, cache / name / 'development',
            prediction_key(name, 'development', development_input_key), cropped=True,
            legacy_signature=legacy.get('signature'))
        cs, cm = collect_predictions(model, split['calibration'], cache / name / 'calibration',
            prediction_key(name, 'calibration', path_records(split['calibration'], root)),
            legacy_signature=legacy.get('signature'))
        # Catch a moved/mismatched split without invalidating old code signatures.
        if not np.allclose(cs, saved_calibration, rtol=2e-3, atol=1e-5):
            raise ValueError(f'{name} normal calibration predictions do not match its saved split/checkpoint.')
        dev_s.append(ds); dev_m.append(dm); cal_s.append(cs); cal_m.append(cm)
        thresholds[name] = float(model.threshold)
        del model; cleanup()
    fusion = D.FusionMath(fusion_cfg).fit(
        np.stack(dev_s, axis=1), np.stack(dev_m, axis=1), dev_masks,
        [row['phase'] for row in dev_rows], [row['source'] for row in dev_rows])
    from copy import deepcopy

    cal_scores = np.stack(cal_s, axis=1)
    cal_maps = np.stack(cal_m, axis=1)
    fusion.calibrate(cal_scores, cal_maps)
    guarded = deepcopy(fusion)

    # Fixed before real-test evaluation. The learned weights still vary per image.
    fusion.image_strength = 0.50
    fusion.map_strength = 1.00
    fusion.image_mode = fusion.map_mode = "active_fixed_strength"
    fusion.mode = "active_dual_hybrid"
    fusion.calibrate(cal_scores, cal_maps)
    del cal_scores, cal_maps

    metadata = {
        **protocol, "signature": signature, "saved_split": saved_split,
        "gate_fit_sources": len({r["source"] for r in dev_rows if r["phase"] == "gate_fit"}),
        "gate_selection_sources": len({r["source"] for r in dev_rows if r["phase"] == "gate_select"}),
        "normalization": "gate-fit normal image q95 / pixel q99",
        "test_labels_used_for_weights": False,
    }
    guarded.save(out / "guarded_hybrid.json", checkpoints, {
        **metadata, "selection_rule": "independent synthetic-validation selection with fallback",
    })
    fusion.save(out / "hybrid.json", checkpoints, {
        **metadata, "selection_rule": "fixed candidate strengths: image=0.50, heatmap=1.00",
    })
    print(f'Image head: {fusion.image_mode}, strength {fusion.image_strength:.2f}')
    print(f'Heatmap head: {fusion.map_mode}, strength {fusion.map_strength:.2f}')
    pd.DataFrame(fusion.selection).assign(category=category, seed=seed).to_csv(out / 'selection.csv', index=False)
    print(pd.DataFrame(fusion.selection).round(4).to_string(index=False))
    del dev_s, dev_m, cal_s, cal_m, dev_images, dev_masks
    cleanup()

    # Fusion parameters/thresholds are now fixed. Real labels enter only evaluation.
    _, test = D.load_category(root, category)
    paths = [path for path, _, _ in test]
    labels = np.array([label for _, label, _ in test])
    kinds = np.array([path.parent.name for path in paths])
    masks = []
    for _, _, mask_path in test:
        if mask_path is None:
            masks.append(np.zeros((D.CROP, D.CROP), dtype=bool))
        else:
            with Image.open(mask_path) as image:
                masks.append(np.asarray(D.MASK_TRANSFORM(image.convert('L'))) > 127)
    masks = np.stack(masks)
    test_signature = fingerprint({'signature': signature, 'images': path_records(paths, root)})
    legacy_test = (fingerprint({'signature': legacy['signature'], 'images': path_records(paths, root)})
                   if legacy else None)
    raw_scores, raw_maps, metrics = [], [], []
    for name, checkpoint in checkpoints.items():
        model = D.load_detector(checkpoint, args.device)
        scores, maps = collect_predictions(model, paths, cache / name / 'test',
            prediction_key(name, 'test', path_records(paths, root)), legacy_signature=legacy_test)
        result = {'category': category, 'seed': seed, 'model': name,
                  **score_metrics(labels, scores, maps, masks, thresholds[name]),
                  **measure(model, paths)}
        metrics.append(result); raw_scores.append(scores); raw_maps.append(maps)
        del model; cleanup()
    raw_scores, raw_maps = np.stack(raw_scores, axis=1), np.stack(raw_maps, axis=1)
    dynamic = fusion.combine(raw_scores, raw_maps)
    equal = fusion.combine(raw_scores, raw_maps, equal=True)
    for name, data, threshold in [('EqualHybrid', equal, fusion.equal_threshold),
                                  ('AdaptiveHybrid', dynamic, fusion.threshold)]:
        result = {'category': category, 'seed': seed, 'model': name,
                  **score_metrics(labels, data['scores'], data['maps'], masks, threshold),
                  'image_mode': 'equal' if name=='EqualHybrid' else fusion.image_mode,
                  'heatmap_mode': 'equal' if name=='EqualHybrid' else fusion.map_mode,
                  'image_strength': 1.0 if name=='EqualHybrid' else fusion.image_strength,
                  'heatmap_strength': 1.0 if name=='EqualHybrid' else fusion.map_strength}
        metrics.append(result)
        atomic_npy(out / f'{name}_heatmaps.npy', data['maps'])
    # Time actual live execution of all three models plus fusion. Never sum cached timings.
    live = D.AdaptiveHybrid.load(out / "hybrid.json", args.device)
    metrics[-1].update(measure(live, paths))
    live.equal, live.name, live.threshold = True, "EqualHybrid", fusion.equal_threshold
    metrics[-2].update(measure(live, paths))

    guarded_data = guarded.combine(raw_scores, raw_maps)
    live.fusion = guarded
    live.equal, live.name, live.threshold = False, "GuardedHybrid", guarded.threshold
    metrics.append({
        "category": category, "seed": seed, "model": "GuardedHybrid",
        **score_metrics(labels, guarded_data["scores"], guarded_data["maps"], masks, guarded.threshold),
        **measure(live, paths),
        "image_mode": guarded.image_mode, "heatmap_mode": guarded.map_mode,
        "image_strength": guarded.image_strength, "heatmap_strength": guarded.map_strength,
    })
    atomic_npy(out / "GuardedHybrid_heatmaps.npy", guarded_data["maps"])
    del live
    cleanup()
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)

    image_rows = []
    for i, path in enumerate(paths):
        row = {'category': category, 'seed': seed, 'path': str(path.relative_to(root)),
               'label': int(labels[i]), 'defect_type': kinds[i], 'fusion_mode': fusion.mode,
               'image_mode': fusion.image_mode, 'heatmap_mode': fusion.map_mode,
               'hybrid_score': float(dynamic['scores'][i]), 'hybrid_threshold': fusion.threshold,
               'hybrid_anomaly': int(dynamic['scores'][i] > fusion.threshold),
               'equal_score': float(equal['scores'][i]), 'equal_threshold': fusion.equal_threshold,
               'equal_anomaly': int(equal['scores'][i] > fusion.equal_threshold)}
        for j, name in enumerate(D.BASELINES):
            row.update({f'{name}_raw_score': float(raw_scores[i,j]),
                        f'{name}_threshold': thresholds[name],
                        f'{name}_anomaly': int(raw_scores[i,j] > thresholds[name]),
                        f'{name}_normalized_score': float(dynamic['normalized_scores'][i,j]),
                        f'{name}_weight': float(dynamic['image_weights'][i,j]),
                        f'{name}_image_weight': float(dynamic['image_weights'][i,j]),
                        f'{name}_heatmap_weight': float(dynamic['map_weights'][i,j]),
                        f'{name}_heatmap_mean_contribution': float(dynamic['map_mean_contributions'][i,j]),
                        f'{name}_heatmap_contribution_share': float(dynamic['map_contribution_shares'][i,j]),
                        f'{name}_contribution': float(dynamic['contributions'][i,j]),
                        f'{name}_contribution_share': float(dynamic['contribution_shares'][i,j])})
        image_rows.append(row)
    pd.DataFrame(image_rows).to_csv(out / 'per_image.csv', index=False)
    atomic_npy(out / 'hybrid_image_weights.npy', dynamic['image_weights'])
    atomic_npy(out / 'hybrid_heatmap_weights.npy', dynamic['map_weights'])
    D.atomic_json(out / 'heatmap_index.json', {
        'paths': [str(path.relative_to(root)) for path in paths], 'crop': D.CROP,
        'baseline_maps': {name: os.path.relpath((cache / name / 'test/maps.npy').resolve(), out.resolve())
                          for name in D.BASELINES},
        'hybrid_maps': {name: f'{name}_heatmaps.npy' for name in ('EqualHybrid', 'AdaptiveHybrid', 'GuardedHybrid')},
        'note': 'Continuous heatmaps, not boundary masks. Baseline maps are raw distances; hybrid maps use normal-reference units.'})

    methods = [(name, raw_scores[:,j], raw_maps[:,j], thresholds[name]) for j,name in enumerate(D.BASELINES)]
    methods += [
        ("EqualHybrid", equal["scores"], equal["maps"], fusion.equal_threshold),
        ("AdaptiveHybrid", dynamic["scores"], dynamic["maps"], fusion.threshold),
        ("GuardedHybrid", guarded_data["scores"], guarded_data["maps"], guarded.threshold),
    ]
    defect_rows = []
    for kind in sorted(set(kinds)-{'good'}):
        subset = (kinds == kind) | (kinds == 'good')
        for name, scores, maps, threshold in methods:
            defect_rows.append({'category': category, 'seed': seed, 'defect_type': kind,
                                'model': name, 'defect_images': int((kinds == kind).sum()),
                                **score_metrics(labels[subset], scores[subset], maps[subset], masks[subset], threshold)})
    pd.DataFrame(defect_rows).to_csv(out / 'per_defect.csv', index=False)
    weights = []
    for head, key, mode in (('image', 'image_weights', fusion.image_mode),
                            ('heatmap', 'map_weights', fusion.map_mode)):
        for j, name in enumerate(D.BASELINES):
            weights.append({'category': category, 'seed': seed, 'model': name, 'head': head,
                            'fusion_mode': mode, 'mean_weight': float(dynamic[key][:,j].mean()),
                            'minimum_weight': float(dynamic[key][:,j].min()),
                            'maximum_weight': float(dynamic[key][:,j].max())})
    pd.DataFrame(weights).to_csv(out / 'weight_summary.csv', index=False)
    D.atomic_json(out / 'run_complete.json', {'signature': signature, 'test_signature': test_signature,
                                            'mode': fusion.mode, 'image_strength': fusion.image_strength, 'map_strength': fusion.map_strength})
    return metrics, defect_rows, weights


def aggregate(output):
    """Aggregate only completed hybrid runs; leave earlier project outputs untouched."""
    files = [p.parent for p in output.glob('*/seed_*/run_complete.json')]
    if not files:
        return
    for filename in ('metrics.csv', 'per_defect.csv', 'weight_summary.csv', 'selection.csv'):
        pd.concat([pd.read_csv(folder / filename) for folder in files], ignore_index=True).to_csv(
            output / filename, index=False)
    metrics = pd.read_csv(output / 'metrics.csv')
    delta = []
    for (category, seed), group in metrics.groupby(['category','seed']):
        table = group.set_index('model'); pc = table.loc['PatchCore']; eq = table.loc['EqualHybrid']
        for name in ('EqualHybrid', 'AdaptiveHybrid', 'GuardedHybrid'):
            row = table.loc[name]
            delta.append({'category': category, 'seed': seed, 'model': name, 'image_mode': row.image_mode, 'heatmap_mode': row.heatmap_mode,
                          'image_strength': row.image_strength, 'heatmap_strength': row.heatmap_strength,
                          'image_gain_vs_PatchCore_pp': 100*(row.image_auroc-pc.image_auroc),
                          'pixel_gain_vs_PatchCore_pp': 100*(row.pixel_auroc-pc.pixel_auroc),
                          'image_gain_vs_equal_pp': 100*(row.image_auroc-eq.image_auroc),
                          'recall_gain_vs_PatchCore_pp': 100*(row.recall-pc.recall),
                          'false_alarm_change_pp': 100*(row.false_alarm_rate-pc.false_alarm_rate),
                          'median_ms': row.benchmark_median_ms})
    pd.DataFrame(delta).to_csv(output / 'hybrid_comparison.csv', index=False)
    print('\nComparison (positive AUROC gain means improvement):')
    print(pd.DataFrame(delta).round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('datasets/MVTecAD'))
    parser.add_argument('--baselines', type=Path, default=Path('results_lightweight'))
    parser.add_argument('--output', type=Path, default=Path('results_hybrid_active'))
    parser.add_argument('--cache-dir', type=Path, default=Path('results_hybrid'))
    parser.add_argument('--categories', nargs='+', default=['screw','transistor'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--device', choices=['cpu','cuda'], default=None)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f'Dataset root does not exist: {args.root}')
    print('Two independent fusion gates: image detection and heatmap localization. No baseline training.')
    print('Real test labels are not used to fit or select weights.')
    for category in args.categories:
        for seed in args.seeds:
            run_one(args, category, seed)
            aggregate(args.output)
    print(f'\nSend metrics.csv, hybrid_comparison.csv and weight_summary.csv from {args.output}.')
    print(f'Weights: {args.output / "weight_summary.csv"}. Heatmaps and hybrid.json are saved per category.')


if __name__ == '__main__':
    main()
