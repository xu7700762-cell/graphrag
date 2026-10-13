"""Evaluate frozen CPU/CUDA parity over raw CDT paths; never runs a cloud API.

Only this validation evaluator reads the score CSV, and only its eligibility
and boundary fields. Labels/scores never enter OnlineModel or FrozenNumeric.
Native CUDA modules are imported only here, outside the deployment runtime.
"""
import argparse
from collections import Counter
import csv
import gc
import importlib.util
import json
import math
import platform
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import scipy
import sklearn
import torch

from eeg_agent.config import EEG_CHANNELS
from eeg_agent.data import metadata
from eeg_agent.recordings import preprocess, trigger_events, state_segments_from_events
from eeg_agent.online.assets import sha
from eeg_agent.online.model import OnlineModel
from scripts.prepare_online_assets import local_path


def errors(left, right):
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    difference = np.abs(left - right)
    return {'max_abs': float(difference.max()),
            'rms': float(np.sqrt(np.mean(difference ** 2))),
            'allclose': bool(np.allclose(left, right, atol=1e-4, rtol=1e-4))}


def tool_scores(value):
    return {item['tool']: item['raw_decision_score'] for item in value['actual_tools']}


def tool_classes(value):
    return {item['tool']: item['predicted_class'] for item in value['actual_tools']}


def eligible_boundaries(data_root):
    path = data_root / 'data/labels/task_segments_with_path_scores.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    # Do not construct/read target classes or convert path_score values.
    selected = {(int(row['subject_id']), int(row['start_sample_1024']), int(row['end_sample_1024']))
                for row in rows if row['is_complete'].lower() == 'true'
                and row['path_score_available'].lower() == 'true' and row['path_score']
                and float(row['duration_sec']) >= 15}
    aligned, offsets, excluded = set(), {}, []
    for subject in sorted({row[0] for row in selected}):
        file = data_root / f'data/raw/Acquisition {subject:02d}.cdt'
        meta = metadata(file)
        raw = np.memmap(file, mode='r', dtype='<f4', shape=(meta['actual_samples'], meta['n_channels']))
        tasks, _, _ = state_segments_from_events(
            trigger_events(raw, meta['channels'].index('Trigger')), len(raw), 1024)
        complete = {(row['start_sample'], row['end_sample']) for row in tasks if row['complete']}
        local = [row for row in selected if row[0] == subject]
        votes = Counter(a - start for _, start, end in local for a, b in complete
                        if b - a == end - start)
        if not votes or (len(votes) > 1 and votes.most_common(2)[0][1] == votes.most_common(2)[1][1]):
            raise ValueError('Ambiguous raw/CSV fixed event offset')
        offset = votes.most_common(1)[0][0]
        offsets[str(subject)] = {'offset_samples': offset, 'exact_duration_matches': votes[offset]}
        for _, start, end in local:
            raw_bounds = (start + offset, end + offset)
            if raw_bounds in complete:
                aligned.add((subject, *raw_bounds))
            else:
                excluded.append({'subject': subject, 'csv_start_sample': start,
                                 'csv_end_sample': end, 'raw_bounds': list(raw_bounds),
                                 'reason': 'no exact complete raw task after fixed offset translation'})
        del raw
    return aligned, sha(path), {'csv_candidates': len(selected), 'offsets': offsets,
                               'excluded_csv_candidates': excluded}


def path_windows(processed, start, end):
    total = max(0, (end - start) // 5120)
    accepted = []
    for index in range(total):
        stamp = start + index * 5120
        left = math.ceil(stamp / 4)
        value = processed[:, left:left + 1280]
        if (value.shape == (30, 1280) and np.isfinite(value).all()
                and np.ptp(value, axis=1).max() <= 2000
                and np.count_nonzero(value.std(axis=1) >= .05) >= 28):
            accepted.append(value)
    return np.asarray(accepted, np.float32), total


def run(data_root, source_root, manifest, output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Preserve completed validation reports; select a new --out')
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates, label_file_hash, alignment = eligible_boundaries(data_root)
    began = time.perf_counter()
    owner = OnlineModel(manifest, threads=2)
    sys.path.insert(0, str(source_root))
    source = source_root / 'tmp/github_decisive_rag_20261011/vrms_model/encoder.py'
    spec = importlib.util.spec_from_file_location('all_paths_native_encoder', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if not torch.cuda.is_available():
        raise RuntimeError('Independent reference validation requires the existing CUDA environment')
    native = module.build_encoder(torch.device('cuda')).eval().requires_grad_(False)
    encoder_path, _ = owner.assets.item(None, 'encoder')
    module.load_pretrained_checkpoint(native, encoder_path)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    from vrms_pilot.model_ablation_v3 import PathMIL
    from vrms_cloud.raw_numeric import FrozenNumeric
    # The optional explanation bank does not affect these eight numerical
    # outputs. Suppress only its lookup in this isolated evaluator process.
    import vrms_cloud.raw_case_rag as case_module
    case_module.retrieve_case_context = lambda *args, **kwargs: {
        'available': False, 'reason': 'optional_explanation_omitted_in_numeric_validation'}
    paths, records, matched = [], [], set()
    failures = []
    for fold in range(1, 27):
        file = data_root / f'data/raw/Acquisition {fold:02d}.cdt'
        if not file.exists():
            records.append({'subject': fold, 'status': 'raw_file_missing'})
            continue
        record_start = time.perf_counter()
        if str(fold) not in owner.assets.value['folds']:
            try:
                owner.start_path(fold)
            except ValueError:
                pass
            else:
                raise AssertionError('Missing fold silently used a replacement model')
            records.append({'subject': fold, 'status': 'frozen_fold_unavailable',
                            'replacement_model_used': False, 'raw_sha256': sha(file)})
            print(json.dumps({'fold': fold, 'status': 'frozen_fold_unavailable'}), flush=True)
            continue
        meta = metadata(file)
        raw = np.memmap(file, mode='r', dtype='<f4', shape=(meta['actual_samples'], meta['n_channels']))
        marks = trigger_events(raw, meta['channels'].index('Trigger'))
        segments, _, event_warnings = state_segments_from_events(marks, len(raw), 1024)
        processed, names, compatible = preprocess(raw, meta['channels'], 1024)
        del raw
        if not compatible or names != EEG_CHANNELS:
            raise AssertionError('Raw oracle does not satisfy the frozen input contract')
        bundle = owner.start_path(fold).bundle
        reference_head = PathMIL(525).cuda().eval().requires_grad_(False)
        reference_head.load_state_dict(bundle['head'].state_dict())
        reference_numeric = FrozenNumeric(fold)
        fold_rows = []
        for segment in segments:
            boundary = (fold, segment['start_sample'], segment['end_sample'])
            row = {'subject': fold, 'ordinal': segment['ordinal'],
                   'start_sample': segment['start_sample'], 'end_sample': segment['end_sample'],
                   'complete': segment['complete'], 'eligible_csv_boundary': boundary in candidates}
            if not segment['complete']:
                row.update(status='incomplete_path_not_published')
                paths.append(row); fold_rows.append(row)
                continue
            if boundary in candidates:
                matched.add(boundary)
            windows, total = path_windows(processed, segment['start_sample'], segment['end_sample'])
            row.update(accepted_windows=len(windows), total_windows=total,
                       coverage=len(windows) / max(1, total),
                       publication_eligible=len(windows) >= 3 and len(windows) / max(1, total) >= .8)
            if not len(windows):
                row.update(status='no_valid_windows')
                paths.append(row); fold_rows.append(row)
                continue
            path = owner.start_path(fold)
            for window in windows:
                path.add(window)
            actual = path.result()
            cpu_features = np.asarray(path.features, np.float32)
            gpu_parts, bf16_parts = [], []
            with torch.inference_mode():
                for offset in range(0, len(windows), 16):
                    normalized = ((torch.from_numpy(windows[offset:offset + 16]) - bundle['mean'])
                                  / bundle['scale']).clamp(-20, 20).cuda()
                    gpu_parts.append(native.forward_tokens(normalized).float().mean(1).cpu().numpy())
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        bf16_parts.append(native.forward_tokens(normalized).float().mean(1).cpu().numpy())
                gpu_features = np.concatenate(gpu_parts)
                bf16_features = np.concatenate(bf16_parts)
                gpu_probability = float(reference_head(torch.from_numpy(gpu_features).cuda()).sigmoid().cpu())
                bf16_probability = float(reference_head(torch.from_numpy(bf16_features).cuda()).sigmoid().cpu())
            row.update(embedding_fp32=errors(gpu_features, cpu_features),
                       embedding_bf16_diagnostic=errors(bf16_features, cpu_features),
                       cpu_mil_probability=actual['mil_raw_high_probability'],
                       cuda_fp32_mil_probability=gpu_probability,
                       cuda_bf16_mil_probability=bf16_probability,
                       mil_fp32_abs_error=abs(actual['mil_raw_high_probability'] - gpu_probability),
                       mil_fp32_class_equal=actual['mil_class'] == ('High' if gpu_probability >= .5 else 'Low'),
                       mil_bf16_class_equal=actual['mil_class'] == ('High' if bf16_probability >= .5 else 'Low'),
                       encoding_and_cache_seconds=path.encoding_seconds,
                       cache_extractions=len(path.cached), fresh_encoded_windows=len(path.features))
            passed = row['embedding_fp32']['allclose'] and row['mil_fp32_class_equal'] and row['mil_fp32_abs_error'] <= 1e-4
            if len(windows) >= 3:
                cpu_reference = reference_numeric.predict(windows, cpu_features, actual['mil_raw_high_probability'], baseline=None)
                gpu_reference = reference_numeric.predict(windows, gpu_features, gpu_probability, baseline=None)
                bf16_reference = reference_numeric.predict(windows, bf16_features, bf16_probability, baseline=None)
                actual_fusion = actual['numeric_fusion']
                a, c, g, b = map(tool_scores, (actual_fusion, cpu_reference, gpu_reference, bf16_reference))
                row.update(cached_tool_scores=errors([c[k] for k in a], list(a.values())),
                           cuda_fp32_tool_scores=errors([g[k] for k in a], list(a.values())),
                           cuda_bf16_tool_scores_diagnostic=errors([b[k] for k in a], list(a.values())),
                           cached_tool_classes_equal=tool_classes(actual_fusion) == tool_classes(cpu_reference),
                           cuda_fp32_tool_classes_equal=tool_classes(actual_fusion) == tool_classes(gpu_reference),
                           cuda_bf16_tool_classes_equal=tool_classes(actual_fusion) == tool_classes(bf16_reference),
                           cached_fusion_abs_error=abs(actual_fusion['raw_combined_score'] - cpu_reference['raw_combined_score']),
                           cuda_fp32_fusion_abs_error=abs(actual_fusion['raw_combined_score'] - gpu_reference['raw_combined_score']),
                           cuda_bf16_fusion_abs_error=abs(actual_fusion['raw_combined_score'] - bf16_reference['raw_combined_score']),
                           fusion_fp32_class_equal=actual_fusion['predicted_class'] == gpu_reference['predicted_class'],
                           fusion_bf16_class_equal=actual_fusion['predicted_class'] == bf16_reference['predicted_class'],
                           tool_names=list(a), asset_hashes={k: actual[k] for k in
                               ('encoder_sha256', 'model_sha256', 'normalization_sha256',
                                'numeric_first_sha256', 'numeric_second_sha256', 'numeric_selector_sha256')})
                passed = (passed and row['cached_tool_scores']['allclose'] and row['cuda_fp32_tool_scores']['allclose']
                          and row['cached_tool_classes_equal'] and row['cuda_fp32_tool_classes_equal']
                          and row['cached_fusion_abs_error'] <= 1e-4
                          and row['cuda_fp32_fusion_abs_error'] <= 1e-4 and row['fusion_fp32_class_equal'])
            row['status'] = 'passed' if passed else 'failed'
            if not passed:
                failures.append({'subject': fold, 'ordinal': segment['ordinal']})
            paths.append(row); fold_rows.append(row)
        record = {'subject': fold, 'status': 'passed' if all(row['status'] != 'failed' for row in fold_rows) else 'failed',
                  'raw_sha256': sha(file), 'dpo_sha256': sha(str(file) + '.dpo'),
                  'complete_task_paths': sum(row['complete'] for row in fold_rows),
                  'compared_paths': sum(row['status'] in ('passed', 'failed') for row in fold_rows),
                  'accepted_windows': sum(row.get('accepted_windows', 0) for row in fold_rows),
                  'event_warnings': event_warnings, 'seconds': time.perf_counter() - record_start}
        records.append(record)
        fold_output = output.parent / f'all_paths_fold_{fold:02d}.json'
        if fold_output.exists():
            raise FileExistsError('Preserve existing per-fold validation artifacts')
        fold_output.write_text(json.dumps({'recording': record, 'paths': fold_rows}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'fold': fold, 'status': record['status'], 'paths': record['compared_paths'],
                          'windows': record['accepted_windows'], 'seconds': round(record['seconds'], 2)}), flush=True)
        del processed, reference_head, reference_numeric
        gc.collect()
    compared = [row for row in paths if row['status'] in ('passed', 'failed')]
    scored = [row for row in compared if row['eligible_csv_boundary']]
    numeric = [row for row in compared if 'cuda_fp32_tool_scores' in row]
    missing_candidates = sorted(candidates - matched)
    result = {'status': 'passed' if not failures and len(matched) == 146 and len(scored) == 146 else 'failed',
              'schema': 'all_raw_paths_cpu_native_fp32_parity_v1', 'source_commit': owner.assets.value['mamba_minimal_commit'],
              'manifest_sha256': sha(manifest), 'encoder_tensors': 83, 'encoder_parameters': 28540155,
              'scope': 'All raw complete task paths for each subject with a frozen held-out model',
              'recordings_scanned': len(records), 'available_folds': sorted(map(int, owner.assets.value['folds'])),
              'missing_fold_subjects': [r['subject'] for r in records if r['status'] == 'frozen_fold_unavailable'],
              'csv_candidates': alignment['csv_candidates'], 'csv_alignment': alignment,
              'raw_complete_csv_matches': len(matched),
              'unmatched_csv_boundaries': missing_candidates, 'compared_paths': len(compared),
              'compared_scored_paths': len(scored), 'numeric_compared_paths': len(numeric),
              'accepted_windows': sum(row['accepted_windows'] for row in compared),
              'scored_accepted_windows': sum(row['accepted_windows'] for row in scored),
              'embedding_fp32_max_abs_error': max(row['embedding_fp32']['max_abs'] for row in compared),
              'mil_fp32_max_abs_error': max(row['mil_fp32_abs_error'] for row in compared),
              'cached_tool_score_max_abs_error': max(row['cached_tool_scores']['max_abs'] for row in numeric),
              'cuda_fp32_tool_score_max_abs_error': max(row['cuda_fp32_tool_scores']['max_abs'] for row in numeric),
              'cuda_fp32_fusion_max_abs_error': max(row['cuda_fp32_fusion_abs_error'] for row in numeric),
              'bf16_diagnostic': {'embedding_max_abs_error': max(row['embedding_bf16_diagnostic']['max_abs'] for row in compared),
                                  'mil_class_disagreements': sum(not row['mil_bf16_class_equal'] for row in compared),
                                  'eight_tool_class_disagreements': sum(not row['cuda_bf16_tool_classes_equal'] for row in numeric),
                                  'fusion_class_disagreements': sum(not row['fusion_bf16_class_equal'] for row in numeric)},
              'failures': failures, 'atol': 1e-4, 'rtol': 1e-4, 'cuda_tf32_enabled': False,
              'runtime_target_labels_received': 0, 'csv_used_only_by_evaluator': True, 'csv_sha256': label_file_hash,
              'waveform_copies_written': 0, 'training_performed': False, 'paid_api_calls': 0,
              'pi_hardware_tested': False, 'seconds': time.perf_counter() - began,
              'environment': {'platform': platform.platform(), 'machine': platform.machine(),
                              'python': platform.python_version(), 'torch': torch.__version__,
                              'numpy': np.__version__, 'scipy': scipy.__version__,
                              'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__,
                              'gpu_reference': torch.cuda.get_device_name(0)},
              'recordings': records, 'paths': paths,
              'boundary': 'FP32 migration and cache equivalence, not ACC, instantaneous symptom validation or Raspberry Pi speed'}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key not in ('recordings', 'paths')}, ensure_ascii=False, indent=2), flush=True)
    return result


def align_report(data_root, prior, output):
    """Reuse numerical comparisons while qualifying raw/CSV boundaries exactly."""
    if output.exists():
        raise FileExistsError('Select a fresh aligned report path')
    result = json.loads(prior.read_text(encoding='utf-8'))
    candidates, label_hash, alignment = eligible_boundaries(data_root)
    if label_hash != result['csv_sha256']:
        raise ValueError('Evaluation CSV differs from the completed numerical run')
    for record in result['recordings']:
        if 'raw_sha256' in record:
            file = data_root / f"data/raw/Acquisition {record['subject']:02d}.cdt"
            if sha(file) != record['raw_sha256']:
                raise ValueError('Raw data differ from the completed numerical run')
            if 'dpo_sha256' in record and sha(str(file) + '.dpo') != record['dpo_sha256']:
                raise ValueError('DPO differs from the completed numerical run')
    matched = set()
    for row in result['paths']:
        boundary = (row['subject'], row['start_sample'], row['end_sample'])
        row['eligible_csv_boundary'] = boundary in candidates
        if row['complete'] and boundary in candidates:
            matched.add(boundary)
    compared = [row for row in result['paths'] if row['status'] in ('passed', 'failed')]
    scored = [row for row in compared if row['eligible_csv_boundary']]
    result.update(csv_candidates=alignment['csv_candidates'], csv_alignment=alignment,
                  raw_complete_csv_matches=len(matched), unmatched_csv_boundaries=sorted(candidates - matched),
                  compared_scored_paths=len(scored), scored_accepted_windows=sum(row['accepted_windows'] for row in scored),
                  status='passed' if not result['failures'] and len(matched) == 146 and len(scored) == 146 else 'failed',
                  numerical_comparisons_reused=True, unaligned_report_sha256=sha(prior),
                  alignment_script_sha256=sha(__file__),
                  alignment_rule='same fixed record offset from exact task durations as evaluate_raw_gpt_all; scores never supplied to runtime')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in
        ('status', 'compared_paths', 'compared_scored_paths', 'accepted_windows', 'scored_accepted_windows',
         'embedding_fp32_max_abs_error', 'mil_fp32_max_abs_error', 'cuda_fp32_tool_score_max_abs_error',
         'cuda_fp32_fusion_max_abs_error', 'bf16_diagnostic', 'failures')}, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='D:/桌面/数据集/城市巡航')
    parser.add_argument('--source-root', default='D:/桌面/文献总/EEGAGENT')
    parser.add_argument('--manifest', default='outputs/online_assets/seed2026/manifest.json')
    parser.add_argument('--out', type=Path, default=Path('outputs/online_implementation/20261011/model_validation/all_paths_report.json'))
    parser.add_argument('--align-report', type=Path,
                        help='Requalify existing numerical report without recomputing models')
    args = parser.parse_args()
    if args.align_report is not None:
        result = align_report(local_path(args.data_root), args.align_report, args.out)
        if result['status'] != 'passed':
            raise SystemExit(1)
        return
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always', sklearn.exceptions.InconsistentVersionWarning)
        result = run(local_path(args.data_root), local_path(args.source_root), local_path(args.manifest), args.out)
    # Append the actual version compatibility notices to this run's new report.
    result['compatibility_warnings'] = sorted(set(str(item.message) for item in observed))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
