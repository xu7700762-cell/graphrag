"""Compare arrived-only replay against audited P0 cache and frozen models.

This is accelerated numerical verification, not a real-time latency benchmark.
Run from the project: python scripts/verify_online_offline.py --seconds 130
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_agent.config import load_config, contract, digest, file_hash, write_json
from eeg_agent.data import CDTSource
from eeg_agent.models import MODEL_IDS, ONNXPredictor
from eeg_agent.signal import StreamPipeline


def compare_values(left, right, tolerance=1e-10):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(compare_values(left[k], right[k], tolerance) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(compare_values(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return bool(np.isclose(left, right, atol=tolerance, rtol=tolerance))
    return left == right


def verify(cfg, subject=1, seconds=130., blocks=(256, 777, 1024)):
    output = Path(cfg['output_root'])
    cache = output / 'cache'
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['contract_hash'] != digest(contract(cfg)):
        raise ValueError('Cache preprocessing mismatch')
    entry = next(row for row in manifest['subjects'] if row['subject'] == subject)
    cache_file = cache / ('subject_%02d.npz' % subject)
    if file_hash(cache_file) != entry['cache_sha256']:
        raise ValueError('Cache hash mismatch')
    with np.load(cache_file) as archive:
        arrays = {key: archive[key] for key in archive.files}
    starts = arrays.get('window_start', arrays.get('start'))
    ends = arrays.get('window_end', arrays.get('end'))
    path_starts = arrays.get('path_start', arrays.get('path'))
    limit = int(seconds * 1024)
    expected = {(int(starts[i]), int(ends[i]), int(path_starts[i])): i for i in range(len(starts)) if int(ends[i]) <= limit}
    qc_rows = json.loads((cache / ('quality_%02d.json' % subject)).read_text(encoding='utf-8'))
    qc = {int(row['start_sample']): {k: v for k, v in row.items() if k != 'start_sample'} for row in qc_rows}
    if not expected:
        raise ValueError('No completed cached task windows in requested source interval')
    predictors = {model_id: ONNXPredictor(cfg, model_id=model_id) for model_id in MODEL_IDS}
    checks = []
    for block in blocks:
        source = CDTSource(cfg['data_root'], subject)
        if file_hash(source.path) != entry['raw_sha256']:
            raise ValueError('Raw data hash mismatch')
        observed, mismatch = [], []
        maximum = 0.
        quality_matches = 0
        model_errors = {model_id: {'probability_max_abs_error': 0., 'logits_max_abs_error': 0., 'valid_windows': 0} for model_id in MODEL_IDS}

        def on_window(window):
            nonlocal maximum, quality_matches
            key = (window['start_sample'], window['end_sample'], window['path_start_sample'])
            observed.append(key)
            if key not in expected:
                mismatch.append({'kind': 'unexpected_window', 'interval': key})
                return
            index = expected[key]
            actual = window['x'].T.astype(np.float32)
            reference = arrays['x0'][index]
            if actual.shape != (30, 1280) or reference.shape != (30, 1280):
                mismatch.append({'kind': 'shape', 'interval': key})
                return
            maximum = max(maximum, float(np.max(np.abs(actual - reference))))
            same_quality = compare_values(window['quality'], qc[window['start_sample']])
            same_gate = bool(window['quality']['valid']) == bool(arrays['valid'][index])
            if same_quality and same_gate:
                quality_matches += 1
            else:
                mismatch.append({'kind': 'quality', 'interval': key, 'same_gate': same_gate})
            if window['quality']['valid']:
                for model_id, predictor in predictors.items():
                    online = predictor.predict_details(window['x'])
                    offline_probability = predictor.predict_batch(reference[None])[0]
                    offline_logits = predictor.logits_batch(reference[None])[0]
                    error = model_errors[model_id]
                    error['valid_windows'] += 1
                    error['probability_max_abs_error'] = max(error['probability_max_abs_error'], abs(online['probability_high'] - float(offline_probability)))
                    error['logits_max_abs_error'] = max(error['logits_max_abs_error'], float(np.max(np.abs(np.asarray(online['logits']) - offline_logits))))

        pipeline = StreamPipeline(source.meta['channels'], cfg, on_window, lambda _: None)
        for start, raw, events in source.chunks(block, seconds):
            pipeline.feed(start, raw, events)
        pipeline.eof()
        absent = set(expected) - set(observed)
        duplicate_count = len(observed) - len(set(observed))
        passed = (not absent and not mismatch and not duplicate_count and maximum == 0. and
                  all(row['probability_max_abs_error'] <= 1e-12 and row['logits_max_abs_error'] <= 1e-12 for row in model_errors.values()))
        checks.append({'block_samples': block, 'expected_windows': len(expected), 'observed_windows': len(observed),
                       'valid_windows': sum(bool(arrays['valid'][i]) for i in expected.values()), 'quality_matching_windows': quality_matches,
                       'boundary_missing_windows': len(absent), 'duplicate_windows': duplicate_count, 'P0_float32_max_abs_error_uv': maximum,
                       'models': model_errors, 'mismatches': mismatch, 'status': 'passed' if passed else 'failed'})
    report = {'status': 'passed' if all(row['status'] == 'passed' for row in checks) else 'failed',
              'subject': subject, 'source_seconds': seconds, 'source_interval': [0, limit],
              'check_kind': 'accelerated numerical online/offline equivalence; not realtime timing',
              'contract_hash': manifest['contract_hash'], 'cache_sha256': entry['cache_sha256'], 'raw_sha256': entry['raw_sha256'],
              'model_hashes': {key: predictor.model_hash for key, predictor in predictors.items()},
              'inference_contract_hashes': {key: predictor.manifest['inference_contract_hash'] for key, predictor in predictors.items()},
              'checks': checks, 'privacy_note': 'subject reference is local acceptance data; never part of the cloud payload'}
    write_json(output / 'reports/online_offline.json', report)
    if report['status'] != 'passed':
        raise ValueError('Online/offline mismatch; inspect reports/online_offline.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--subject', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=130.)
    args = parser.parse_args()
    report = verify(load_config(args.config), args.subject, args.seconds)
    print(json.dumps(report, ensure_ascii=True, indent=2))
