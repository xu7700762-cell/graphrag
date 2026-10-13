"""Replay local numerical predictions, then score against path labels post hoc.

The prediction stage receives raw CDT/DPO and frozen fold assets only. Labels
are opened after every recording has finished; no cloud API or training runs.
Prefix classes target the end-of-path questionnaire, not instantaneous symptoms.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from eeg_agent.config import load_config
from eeg_agent.data import metadata
from eeg_agent.online.assets import sha
from eeg_agent.online.model import OnlineModel
from eeg_agent.online.stream import StreamingProcessor
from eeg_agent.tools import _spectral


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, score_key):
    usable = [row for row in rows if row.get(score_key) is not None]
    total, count = len(rows), len(usable)
    if not count:
        return {'total': total, 'available': 0, 'coverage': 0., 'failure_rate': 1. if total else None}
    truth = np.asarray([row['target_high'] for row in usable], int)
    scores = np.asarray([row[score_key] for row in usable], float)
    predicted = (scores >= .5).astype(int)
    return {'total': total, 'available': count, 'coverage': count / total,
            'failure_rate': (total - count) / total,
            'accuracy_available': float(accuracy_score(truth, predicted)),
            'accuracy_all_missing_as_incorrect': int(np.count_nonzero(truth == predicted)) / total,
            'balanced_accuracy_available': float(balanced_accuracy_score(truth, predicted)),
            'auroc_available': float(roc_auc_score(truth, scores)) if len(set(truth)) == 2 else None,
            'confusion_low_high': [[int(np.count_nonzero((truth == a) & (predicted == p)))
                                    for p in (0, 1)] for a in (0, 1)]}


def score_predictions(predictions, source_paths, label_path):
    with label_path.open(encoding='utf-8-sig', newline='') as handle:
        candidates = [row for row in csv.DictReader(handle) if row['is_complete'].lower() == 'true'
                      and row['path_score_available'].lower() == 'true' and row['path_score']
                      and float(row['duration_sec']) >= 15]
    alignment, scored_paths, scored_prefixes, excluded = {}, [], [], []
    by_key = {(row['subject'], row['start_sample'], row['end_sample']): row for row in predictions}
    for subject in sorted({int(row['subject_id']) for row in candidates}):
        local = [row for row in candidates if int(row['subject_id']) == subject]
        bounds = {(row['start_sample'], row['end_sample']) for row in source_paths
                  if row['subject'] == subject and row['complete']}
        votes = Counter(a - int(row['start_sample_1024']) for row in local for a, b in bounds
                        if b - a == int(row['end_sample_1024']) - int(row['start_sample_1024']))
        if not votes or (len(votes) > 1 and votes.most_common(2)[0][1] == votes.most_common(2)[1][1]):
            raise ValueError('Ambiguous exact-duration CSV/raw offset')
        offset = votes.most_common(1)[0][0]
        alignment[str(subject)] = {'offset_samples': offset, 'exact_duration_matches': votes[offset]}
        for row in local:
            start, end = int(row['start_sample_1024']) + offset, int(row['end_sample_1024']) + offset
            if (start, end) not in bounds:
                excluded.append({'subject': subject, 'reason': 'no exact complete raw path after fixed offset'})
                continue
            predicted = by_key[(subject, start, end)]
            base = {'subject': subject, 'ordinal': predicted['ordinal'], 'start_sample': start,
                    'end_sample': end, 'target_high': int(float(row['path_score']) >= 30)}
            scored_paths.append({**base, 'accepted_windows': predicted['accepted_windows'],
                                 'total_windows': predicted['total_windows'], 'coverage': predicted['coverage'],
                                 'mil_probability': predicted['mil_probability'],
                                 'numeric_score': predicted['numeric_score'], 'failure_reason': predicted['failure_reason']})
            for prefix in predicted['prefixes']:
                scored_prefixes.append({**base, **prefix})
    return scored_paths, scored_prefixes, alignment, excluded, len(candidates)


def run(data_root, manifest, out):
    result_path = out / 'numeric_evaluation.json'
    if result_path.exists():
        raise FileExistsError('Preserve final evaluation; choose a fresh --out')
    out.mkdir(parents=True, exist_ok=True)
    owner = OnlineModel(manifest, threads=2)
    began = time.perf_counter()
    predictions, source_paths, records = [], [], []
    for file in sorted((data_root / 'data/raw').glob('Acquisition ??.cdt')):
        subject = int(file.stem[-2:])
        meta = metadata(file)
        raw = np.memmap(file, mode='r', dtype='<f4', shape=(meta['actual_samples'], meta['n_channels']))
        stream = StreamingProcessor(meta['channels'])
        active, accumulator = None, None

        def receive(events):
            nonlocal active, accumulator
            for event in events:
                if event['scope'] != 'task':
                    continue
                if event['type'] == 'path_start':
                    active = {'subject': subject, 'ordinal': event['ordinal'], 'start_sample': event['start_sample'],
                              'accepted_windows': 0, 'total_windows': 0, 'prefixes': []}
                    accumulator = owner.start_path(subject) if str(subject) in owner.assets.value['folds'] else None
                elif event['type'] == 'window' and active is not None:
                    active['total_windows'] += 1
                    output = {}
                    if event['valid']:
                        active['accepted_windows'] += 1
                        if accumulator is not None:
                            frequencies, psd, _ = _spectral(event['window'])
                            output = accumulator.add_shared(event['window'], frequencies, psd)
                    elif accumulator is not None:
                        # Runtime displays no VRMS prediction for a rejected current window.
                        output = {}
                    fusion = output.get('numeric_fusion') or {}
                    active['prefixes'].append({'window_index': active['total_windows'],
                        'elapsed_seconds': event['end_seconds'] - active['start_sample'] / 1024,
                        'cutoff_seconds': event['observed_end_sample'] / 1024,
                        'valid': event['valid'], 'accepted_windows': active['accepted_windows'],
                        'mil_probability': output.get('mil_raw_high_probability'),
                        'numeric_score': fusion.get('raw_combined_score')})
                elif event['type'] == 'path_end' and active is not None:
                    output = accumulator.result() if accumulator is not None else {}
                    coverage = active['accepted_windows'] / max(1, active['total_windows'])
                    permitted = bool(event['complete'] and active['accepted_windows'] >= 3
                                     and coverage >= .8 and output.get('available'))
                    fusion = output.get('numeric_fusion') or {}
                    active.update(end_sample=event['end_sample'], complete=event['complete'], coverage=coverage,
                                  mil_probability=output.get('mil_raw_high_probability') if permitted else None,
                                  numeric_score=fusion.get('raw_combined_score') if permitted else None,
                                  failure_reason='' if permitted else ('frozen_fold_unavailable' if accumulator is None
                                      else event.get('reason') or 'completion_window_count_or_coverage_gate'))
                    source_paths.append({key: active[key] for key in ('subject', 'start_sample', 'end_sample', 'complete')})
                    predictions.append(active)
                    active, accumulator = None, None

        for start in range(0, len(raw), 256):
            receive(stream.feed(np.asarray(raw[start:start + 256]), start))
        receive(stream.finish())
        records.append({'subject': subject, 'raw_sha256': sha(file), 'dpo_sha256': sha(Path(str(file) + '.dpo')),
                        'source_seconds': len(raw) / 1024, 'fold_available': str(subject) in owner.assets.value['folds']})
        del raw
        print(json.dumps({'subject': subject, 'stage': 'prediction', 'paths_so_far': len(predictions),
                          'seconds': round(time.perf_counter() - began, 2)}), flush=True)
    # This private prediction audit is written before any label file is opened.
    audit = out / 'numeric_predictions_unlabelled.json'
    audit.write_text(json.dumps({'target_labels_received': 0, 'predictions': predictions, 'records': records},
                               ensure_ascii=False, allow_nan=False), encoding='utf-8')
    label_path = data_root / 'data/labels/task_segments_with_path_scores.csv'
    paths, prefixes, alignment, excluded, candidate_count = score_predictions(predictions, source_paths, label_path)
    if len(paths) != 146 or len({row['subject'] for row in paths}) != 24:
        raise ValueError('Frozen evaluation denominator must remain 146 paths/24 subjects')
    progress = {}
    for fraction in (.25, .5, .75, 1.):
        rows = []
        for path in paths:
            subset = [row for row in prefixes if (row['subject'], row['ordinal']) == (path['subject'], path['ordinal'])]
            cutoff = (path['end_sample'] - path['start_sample']) / 1024 * fraction
            reached = [row for row in subset if row['elapsed_seconds'] <= cutoff]
            rows.append(reached[-1] if reached else {**path, 'mil_probability': None, 'numeric_score': None})
        progress[str(fraction)] = {name: summarize(rows, key) for name, key in
                                  (('mil_preview', 'mil_probability'), ('numeric_fusion', 'numeric_score'))}
    result = {'schema': 'online_numeric_posthoc_v1', 'status': 'passed', 'subject_count': 24,
              'scored_paths': len(paths), 'csv_candidates': candidate_count, 'excluded_candidates': excluded,
              'label_threshold': 30, 'target': 'end_of_path_questionnaire_category',
              'target_labels_received_by_runtime': 0, 'labels_opened_after_all_predictions': True,
              'cloud_calls': 0, 'training_performed': False, 'hardware_tested': False,
              'backend': 'mamba_minimal_cpu_fp32', 'block_samples': 256, 'speed': 'accelerated_validation',
              'prediction_audit_sha256': sha(audit), 'manifest_sha256': sha(manifest), 'label_sha256': sha(label_path),
              'path_end': {name: summarize(paths, key) for name, key in
                           (('mil', 'mil_probability'), ('numeric_fusion', 'numeric_score'))},
              'prefix_all_window_updates': {name: summarize(prefixes, key) for name, key in
                                           (('mil_preview', 'mil_probability'), ('numeric_fusion', 'numeric_score'))},
              'prefix_path_progress': progress, 'mean_path_window_coverage': float(np.mean([row['coverage'] for row in paths])),
              'prefix_scoring_caveat': 'Window updates from one path are correlated; pooled window accuracy is descriptive only. No instantaneous symptom ground truth. Preview MIL can have fewer than three valid windows.',
              'alignment': alignment, 'missing_fold_subjects': [22, 26], 'recordings': records,
              'seconds': time.perf_counter() - began,
              'environment': {'machine': platform.machine(), 'python': platform.python_version(), 'torch': torch.__version__}}
    write_csv(out / 'numeric_evaluation_paths.csv', paths)
    write_csv(out / 'numeric_evaluation_prefixes.csv', prefixes)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'scored_paths': len(paths), 'path_end': result['path_end']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(load_config()['data_root']))
    parser.add_argument('--manifest', type=Path, default=PROJECT / 'outputs/online_assets/seed2026/manifest.json')
    parser.add_argument('--out', type=Path, default=PROJECT / 'outputs/online_implementation/20261011')
    args = parser.parse_args()
    run(args.data_root, args.manifest, args.out)
