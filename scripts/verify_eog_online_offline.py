"""Verify causal reference RLS cleaning against its independently saved P1 cache.

This checks arrived-only numerical processing, per-window adaptive summaries,
and frozen model prediction consistency,
not real-time latency or clinical artifact-removal efficacy. Raw EEG and subject
references remain in this local acceptance script and are never sent to a cloud.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_agent.config import cache_path, contract, digest, file_hash, load_config, profile_config, write_json
from eeg_agent.data import CDTSource
from eeg_agent.eog import process_window
from eeg_agent.models import MODEL_IDS, ONNXPredictor
from eeg_agent.signal import StreamPipeline
from verify_online_offline import compare_values


def verify(cfg, subject=1, seconds=130., blocks=(256, 777, 1024), check_models=True,
           checkpoint_id=None):
    cfg = profile_config(cfg, 'P1')
    cache = cache_path(cfg)
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['contract_hash'] != digest(contract(cfg)):
        raise ValueError('P1 cache preprocessing contract mismatch')
    entry = next(row for row in manifest['subjects'] if row['subject'] == subject)
    cache_file = cache / ('subject_%02d.npz' % subject)
    if file_hash(cache_file) != entry['cache_sha256']:
        raise ValueError('P1 cache file hash mismatch')
    with np.load(cache_file) as archive:
        arrays = {key: archive[key] for key in archive.files}
    starts = arrays['window_start']
    ends = arrays['window_end']
    path_starts = arrays['path_start']
    limit = int(seconds * 1024)
    expected = {(int(starts[i]), int(ends[i]), int(path_starts[i])): i
                for i in range(len(starts)) if int(ends[i]) <= limit}
    if not expected:
        raise ValueError('No completed cached task windows in requested interval')
    qc_rows = json.loads((cache / ('quality_%02d.json' % subject)).read_text(encoding='utf-8'))
    qc = {int(row['start_sample']): {key: value for key, value in row.items()
                                   if key not in ('start_sample', 'eog_cleaning')}
          for row in qc_rows}
    eog_rows = json.loads((cache / ('eog_%02d.json' % subject)).read_text(encoding='utf-8'))
    eog = {(int(row['start_sample']), int(row['end_sample'])):
           {key: value for key, value in row.items() if key not in ('start_sample', 'end_sample')}
           for row in eog_rows}
    if len(eog) != len(eog_rows):
        raise ValueError('Duplicate cached EOG window summaries')
    source = CDTSource(cfg['data_root'], subject)
    if file_hash(source.path) != entry['raw_sha256']:
        raise ValueError('P1 raw data hash mismatch')
    predictors = ({model_id: ONNXPredictor(cfg, model_id=model_id, checkpoint_id=checkpoint_id)
                   for model_id in MODEL_IDS} if check_models else {})
    if any(predictor.profiles != ['P1'] for predictor in predictors.values()):
        raise ValueError('P1 verification refuses a P0 checkpoint')
    checks = []
    for block in blocks:
        observed, mismatches = [], []
        maximum = 0.
        quality_matches = 0
        eog_summary_matches = 0
        model_errors = {model_id: {'probability_max_abs_error': 0.,
                                  'logits_max_abs_error': 0., 'valid_windows': 0}
                        for model_id in predictors}

        def on_window(window):
            nonlocal maximum, quality_matches, eog_summary_matches
            processed = process_window(window, cfg, profile='P1')
            key = (processed['start_sample'], processed['end_sample'], processed['path_start_sample'])
            observed.append(key)
            if key not in expected:
                mismatches.append({'kind': 'unexpected_window', 'interval': key})
                return
            index = expected[key]
            actual = processed['x'].T.astype(np.float32)
            reference = arrays['x1'][index]
            if actual.shape != (30, 1280) or reference.shape != (30, 1280):
                mismatches.append({'kind': 'shape', 'interval': key})
                return
            if not np.isfinite(actual).all() or not np.isfinite(reference).all():
                mismatches.append({'kind': 'nonfinite', 'interval': key})
                return
            maximum = max(maximum, float(np.max(np.abs(actual - reference))))
            same_quality = compare_values(processed['quality'], qc[processed['start_sample']])
            expected_eog = eog.get((processed['start_sample'], processed['end_sample']))
            same_eog = expected_eog is not None and compare_values(processed['eog_cleaning'], expected_eog)
            if same_eog:
                eog_summary_matches += 1
            else:
                mismatches.append({'kind': 'eog_window_summary', 'interval': key})
            same_gate = bool(processed['quality']['valid']) == bool(arrays['valid1'][index])
            same_calibration = bool(window['calibration'].available) == bool(arrays['p1'][index])
            if same_quality and same_gate and same_calibration:
                quality_matches += 1
            else:
                mismatches.append({'kind': 'quality_or_calibration', 'interval': key,
                                   'same_quality': same_quality, 'same_gate': same_gate,
                                   'same_calibration': same_calibration})
            if processed['quality']['valid']:
                if processed.get('profile') != 'P1':
                    mismatches.append({'kind': 'profile', 'interval': key})
                    return
                for model_id, predictor in predictors.items():
                    online = predictor.predict_details(processed['x'], profile=processed['profile'])
                    offline_probability = predictor.predict_batch(reference[None])[0]
                    offline_logits = predictor.logits_batch(reference[None])[0]
                    error = model_errors[model_id]
                    error['valid_windows'] += 1
                    error['probability_max_abs_error'] = max(
                        error['probability_max_abs_error'],
                        abs(online['probability_high'] - float(offline_probability)))
                    error['logits_max_abs_error'] = max(
                        error['logits_max_abs_error'],
                        float(np.max(np.abs(np.asarray(online['logits']) - offline_logits))))

        pipeline = StreamPipeline(source.meta['channels'], cfg, on_window, lambda _: None)
        for start, raw, events in source.chunks(block, seconds):
            pipeline.feed(start, raw, events)
        pipeline.eof()
        absent = set(expected) - set(observed)
        duplicates = len(observed) - len(set(observed))
        same_frozen_calibration = compare_values(pipeline.calibration.summary(), entry['calibration'])
        passed = (not absent and not mismatches and not duplicates and maximum == 0.
                  and same_frozen_calibration and all(
                      error['probability_max_abs_error'] <= 1e-12 and
                      error['logits_max_abs_error'] <= 1e-12
                      for error in model_errors.values()))
        checks.append({'block_samples': block, 'expected_windows': len(expected),
                       'observed_windows': len(observed),
                       'valid_P1_windows': sum(bool(arrays['valid1'][i]) for i in expected.values()),
                       'quality_matching_windows': quality_matches,
                       'eog_summary_matching_windows': eog_summary_matches,
                       'boundary_missing_windows': len(absent), 'duplicate_windows': duplicates,
                       'P1_float32_max_abs_error_uv': maximum,
                       'frozen_calibration_matches': same_frozen_calibration,
                       'calibration': pipeline.calibration.summary(),
                       'models': model_errors, 'mismatches': mismatches,
                       'status': 'passed' if passed else 'failed'})
    report = {'status': 'passed' if all(row['status'] == 'passed' for row in checks) else 'failed',
              'subject': subject, 'source_seconds': seconds, 'source_interval': [0, limit],
              'profile': 'P1', 'check_kind': 'accelerated causal-EOG-RLS online/offline numerical equivalence',
              'processing_method': contract(cfg)['eog_cleaning']['method'],
              'processing_version': contract(cfg)['eog_cleaning']['version'],
              'processing_contract': contract(cfg),
              'models_checked': check_models, 'contract_hash': manifest['contract_hash'],
              'cache_sha256': entry['cache_sha256'], 'raw_sha256': entry['raw_sha256'],
              'quality_summary_sha256': file_hash(cache / ('quality_%02d.json' % subject)),
              'eog_summary_sha256': file_hash(cache / ('eog_%02d.json' % subject)),
              'model_hashes': {model_id: predictor.model_hash for model_id, predictor in predictors.items()},
              'checks': checks, 'privacy_note': 'local acceptance only; no cloud payload'}
    destination = Path(cfg['output_root']) / 'reports' / ('eog_online_offline_%02d.json' % subject)
    write_json(destination, report)
    if report['status'] != 'passed':
        raise ValueError('P1 online/offline mismatch; inspect ' + str(destination))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--subject', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=130.)
    parser.add_argument('--skip-models', action='store_true', help='Numerical verification before retraining is ready')
    parser.add_argument('--checkpoint-id')
    args = parser.parse_args()
    result = verify(load_config(args.config), args.subject, args.seconds,
                    check_models=not args.skip_models, checkpoint_id=args.checkpoint_id)
    print(json.dumps(result, ensure_ascii=True, indent=2))
