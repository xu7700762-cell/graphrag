"""Ten fixed old GPT errors: offline preparation, then at most 20 real requests.

Each paid path runs the established single-path probe in a separate process.
No retries, replacements, training, or production configuration changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from scripts import probe_raw_gpt_high_rag as probe
from eeg_agent.recordings import RecordingRepository
import numpy as np

OUT = probe.RESEARCH / 'outputs/vrms_rag_probes/20261011_gpt_high_ten_errors_01'
SEED = 20261011
EXCLUDED = '08:2047409'
WINDOW_ROOT = APP / 'outputs/vrms_v2/brain_recordings/inference'


def directory(row):
    return OUT / 'paths' / row['path_key'].replace(':', '_')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matching_saved_windows(row, reference):
    for path in sorted(WINDOW_ROOT.glob('*/request.json')):
        request = probe.read(path)
        if (request.get('fold') != int(row['subject']) or request.get('baseline_file') is not False
                or request.get('signal_sha256') != reference['signal_sha256']):
            continue
        worker_path = path.parent / 'worker_result.json'
        if not worker_path.exists():
            continue
        worker = probe.read(worker_path)
        actual = next((x for x in worker.get('segments', []) if x['ordinal'] == int(row['ordinal'])), {})
        old = next(x for x in reference['measurements']['segments'] if x['ordinal'] == int(row['ordinal']))
        if (actual.get('mil_raw_high_probability') != old['mil_raw_high_probability'] or
                actual.get('numeric_fusion', {}).get('raw_combined_score') != old['numeric_fusion']['raw_combined_score']):
            continue
        segment = next(x for x in request['segments'] if x['ordinal'] == int(row['ordinal']))
        filename = path.parent / segment['file']
        if filename.is_file():
            return filename
    return None


def numerical_match(old, new):
    """The old query evidence must match; only the new case context is excluded."""
    if isinstance(old, dict):
        old = {k: v for k, v in old.items() if k != 'training_case_context'}
        new = {k: v for k, v in new.items() if k != 'training_case_context'}
        return set(old) == set(new) and all(numerical_match(v, new[k]) for k, v in old.items())
    if isinstance(old, list):
        return len(old) == len(new) and all(numerical_match(a, b) for a, b in zip(old, new))
    if type(old) in (int, float) and type(new) in (int, float):
        return abs(old - new) < 1e-8
    return old == new


def prepare():
    if OUT.exists():
        raise FileExistsError('Preserve the existing fixed selection and paid artifacts')
    source = probe.PREVIOUS / 'path_results.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        errors = [r for r in csv.DictReader(stream)
                  if r['gpt_correct'] == 'False' and r['gpt_generation_status'] == 'cloud_verified']
    rng = random.Random(SEED)
    groups = {c: sorted([r for r in errors if r['true_class'] == c and r['path_key'] != EXCLUDED],
                       key=lambda r: r['path_key']) for c in ('High', 'Low')}
    for values in groups.values():
        rng.shuffle(values)
    selected, subjects = [], set()
    for _ in range(5):
        for cls in ('High', 'Low'):
            row = next(r for r in groups[cls] if r['subject'] not in subjects)
            groups[cls].remove(row)
            selected.append(row)
            subjects.add(row['subject'])
    assert len(selected) == len(subjects) == 10
    OUT.mkdir(parents=True)
    write_csv(OUT / 'all_previous_gpt_errors.csv', errors)
    probe.save(OUT / 'selection.json', selected)
    protected = [APP / '.env', APP / 'cloud_profiles.json', source, probe.PREVIOUS / 'summary.json',
        probe.SOURCE / 'protocol.json', Path(probe.__file__), Path(__file__),
        APP / 'eeg_agent/raw_evidence_teaching.py', APP / 'eeg_agent/raw_tool_agent.py',
        APP / 'eeg_agent/brain_knowledge/vrms/03_nine_evidence_guide.md',
        *[probe.SOURCE / f"predictions/subject_{int(r['subject']):02d}.json" for r in selected]]
    protection = {str(f): probe.sha(f) for f in protected}
    batch_protocol = {'schema': 'ten_previous_gpt_errors_high_rag_v1', 'created_at': probe.stamp(),
        'error_pool': len(errors), 'selected_paths': 10, 'selection_seed': SEED,
        'selection_rule': 'five old false negatives and five old false positives; ten distinct subjects',
        'excluded_previously_probed_path': EXCLUDED, 'maximum_api_requests': 20,
        'maximum_requests_per_path': 2, 'automatic_retries': 0, 'concurrent_path_processes': 2,
        'requested_model': 'gpt-6.1-sol', 'reasoning_effort': 'high',
        'query_truth_hidden': True, 'fused_answer_hidden': True, 'previous_cloud_answer_hidden': True,
        'protected_file_hashes': protection,
        'limitation': 'Selected known errors, not a representative ACC test; both RAG and effort changed.'}
    probe.save(OUT / 'protocol.json', batch_protocol)
    original_protocol = probe.read(probe.SOURCE / 'protocol.json')
    teaching = probe.build_teaching_context()
    cfg = probe.load_config()
    cfg['output_root'] = str(OUT / 'runtime')
    repository = RecordingRepository(cfg)
    for index, row in enumerate(selected, 1):
        fold, ordinal = int(row['subject']), int(row['ordinal'])
        reference = probe.read(probe.SOURCE / f'predictions/subject_{fold:02d}.json')['result']
        prior = next(x for x in reference['measurements']['segments'] if x['ordinal'] == ordinal)
        case_dir = directory(row)
        worker_dir = case_dir / 'local_worker'
        worker_dir.mkdir(parents=True)
        window_path = worker_dir / f'segment-{ordinal:03d}.npy'
        cached = matching_saved_windows(row, reference)
        if cached:
            shutil.copyfile(cached, window_path)
        else:
            evidence = repository.analyze(f'subject-{fold:02d}')
            record = evidence['recording']
            assert record['signal_sha256'] == reference['signal_sha256']
            expected = original_protocol['subjects'][str(fold)]
            assert record['signal_sha256'] == expected['raw_sha256']
            assert probe.sha(Path(cfg['data_root']) / f'data/raw/Acquisition {fold:02d}.cdt.dpo') == expected['dpo_sha256']
            bounds = next(x for x in original_protocol['paths'] if x['path_key'] == row['path_key'])
            segment = next(x for x in record['segments'] if x['ordinal'] == ordinal)
            assert segment['complete'] and segment['start_sample'] == bounds['raw_start_sample']
            assert segment['end_sample'] == bounds['raw_end_sample']
            windows, _ = repository.model_inputs(f'subject-{fold:02d}')[ordinal - 1]
            assert len(windows) == prior['accepted_windows']
            np.save(window_path, windows, allow_pickle=False)
            repository.cache.clear()
            del windows, evidence, record
        assert len(np.load(window_path, mmap_mode='r', allow_pickle=False)) == prior['accepted_windows']
        request = {'fold': fold, 'signal_sha256': reference['signal_sha256'],
                   'segments': [{'ordinal': ordinal, 'file': window_path.name}], 'baseline_file': False}
        probe.save(worker_dir / 'request.json', request)
        command = ['wsl.exe', '-d', 'Ubuntu-22.04-Bio', '--cd', probe.wsl(probe.RESEARCH), '--exec',
                   '/opt/miniconda3/envs/pytorch/bin/python', '-B', '-m', 'vrms_cloud.raw_inference',
                   probe.wsl(worker_dir / 'request.json')]
        completed = subprocess.run(command, capture_output=True, timeout=240,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        (worker_dir / 'stderr.log').write_bytes(completed.stderr)
        worker = json.loads(completed.stdout.decode('utf-8'))
        probe.save(worker_dir / 'result.json', worker)
        assert worker['available'] and len(worker['segments']) == 1
        prediction = worker['segments'][0]
        assert prediction['fresh_encoded_windows'] == prior['accepted_windows']
        assert abs(prediction['mil_raw_high_probability'] - prior['mil_raw_high_probability']) < 1e-8
        assert numerical_match(prior['numeric_fusion'], prediction['numeric_fusion'])
        runtime = probe.ProbeRuntime(prediction, teaching_context=teaching)
        runtime.numerical_execution = ('fresh_preprocessing_and_encoding_before_cloud; callable_immutable_evidence'
            if cached is None else 'fresh_encoding_of_same_saved_raw_derived_windows; no_fresh_preprocessing')
        initial = runtime.initial_evidence()
        corrective = runtime.execute('corrective_evidence')
        context = next(x['training_case_context'] for x in corrective['actual_individual_evidence']
                       if x['tool'] == 'case_retrieval')
        assert len(teaching['input_definitions']) == 9
        assert context['available'] and len(context['cases']) == 12 and len(context['detailed_cases']) == 3
        protocol = {'schema': 'single_errorpath_high_rag_probe_v1', 'created_at': probe.stamp(),
            'path_key': row['path_key'], 'subject': fold, 'ordinal': ordinal, 'paths': 1,
            'profile_id': 'cpa_gpt', 'requested_model': 'gpt-6.1-sol', 'reasoning_effort': 'high',
            'maximum_api_requests': 2, 'maximum_output_tokens_per_request': 8192,
            'api_timeout_seconds': 120, 'automatic_retries': 0,
            'fused_class_withheld': True, 'fused_score_withheld': True,
            'query_truth_sent': False, 'previous_cloud_answer_sent': False,
            'fresh_preprocessing': cached is None, 'fresh_local_encoding': True,
            'saved_window_sha256': probe.sha(window_path),
            'original_saved_window_path': str(cached) if cached else None,
            'selection': 'one of ten fixed known errors', 'protected_file_hashes': protection}
        probe.save(case_dir / 'protocol.json', protocol)
        probe.save(case_dir / 'prediction.json', prediction)
        probe.save(case_dir / 'offline_preview.json', {'initial_evidence': initial, 'corrective_evidence': corrective})
        probe.save(case_dir / 'offline_checks.json', {'fresh_encoded_windows': prediction['fresh_encoded_windows'],
            'same_mil_and_all_original_numeric_evidence_within_1e_8': True,
            'nine_input_teaching_complete': True, 'retrieved_internal_subjects': 12, 'detailed_cases': 3,
            'cloud_payload_hidden_keys': [], 'api_requests': 0})
        probe.save(OUT / 'progress.json', {'stage': 'offline_preparing', 'prepared_paths': index,
                   'total': 10, 'api_requests': 0, 'updated_at': probe.stamp()})
        print(json.dumps({'stage': 'offline_prepared', 'index': index, 'path': row['path_key'],
              'windows': prediction['fresh_encoded_windows'], 'fresh_preprocessing': cached is None,
              'original_evidence_matches': True}, ensure_ascii=False), flush=True)
    probe.save(OUT / 'offline_ready.json', {'prepared_paths': 10, 'api_requests': 0})
    print('All ten prepared and verified; zero API requests.', flush=True)


def one(path_key):
    selected = probe.read(OUT / 'selection.json')
    row = next(r for r in selected if r['path_key'] == path_key)
    probe.OUT, probe.PATH_KEY = directory(row), path_key
    probe.run()


def aggregate(stage):
    selected = probe.read(OUT / 'selection.json')
    rows, usage = [], []
    for row in selected:
        case_dir = directory(row)
        if not (case_dir / 'judgment.json').exists():
            continue
        j = probe.read(case_dir / 'judgment.json')
        s = probe.read(case_dir / 'summary.json') if (case_dir / 'summary.json').exists() else {}
        u = probe.read(case_dir / 'api_usage.json')
        usage.append(u)
        rows.append({'path_key': row['path_key'], 'subject': int(row['subject']),
            'true_class': row['true_class'], 'previous_gpt_class': row['gpt_class'],
            'new_gpt_class': j['final_class'] if j['available'] else None,
            'corrected': bool(j['available'] and j['final_class'] == row['true_class']),
            'confidence': j.get('confidence'), 'available': j['available'],
            'evidence_use_verified': s.get('new_evidence_use_verified', False),
            'numeric_class': row['numeric_class'], 'api_requests': u['calls'],
            'api_failures': u['failures'], 'explanation': j.get('explanation', '')})
    corrected = sum(r['corrected'] for r in rows)
    result = {'stage': stage, 'total': 10, 'completed_paths': len(rows),
        'valid_gpt_paths': sum(r['available'] for r in rows), 'corrected_errors': corrected,
        'correction_fraction_fixed_denominator': corrected / 10,
        'evidence_use_verified_paths': sum(r['evidence_use_verified'] for r in rows),
        'api_requests': sum(u['calls'] for u in usage), 'api_failures': sum(u['failures'] for u in usage),
        'input_tokens': sum(u['input_tokens'] for u in usage), 'output_tokens': sum(u['output_tokens'] for u in usage),
        'automatic_retries': 0, 'updated_at': probe.stamp(), 'path_results': rows,
        'by_true_class': {c: {'selected': 5, 'corrected': sum(r['corrected'] for r in rows if r['true_class'] == c)}
                          for c in ('High', 'Low')},
        'limitation': 'Known error subset only; does not estimate full 146-path ACC or isolate RAG vs high effort.'}
    probe.save(OUT / 'progress.json', result)
    if stage == 'completed':
        probe.save(OUT / 'summary.json', result)
        write_csv(OUT / 'path_results.csv', rows)
    return result


def run():
    probe.read(OUT / 'offline_ready.json')
    if (OUT / 'api_started.json').exists():
        raise RuntimeError('Batch already attempted; no duplicate or retry requests')
    protocol = probe.read(OUT / 'protocol.json')
    assert all(probe.sha(f) == h for f, h in protocol['protected_file_hashes'].items())
    selected = probe.read(OUT / 'selection.json')
    assert len(selected) == 10 and all(not (directory(r) / 'api_started.json').exists() for r in selected)
    probe.save(OUT / 'api_started.json', {'started_at': probe.stamp(), 'maximum_requests': 20, 'retries': 0})

    def execute(row):
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                   '--stage', 'one', '--path-key', row['path_key']]
        completed = subprocess.run(command, capture_output=True, timeout=280,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        (directory(row) / 'run_process.log').write_bytes(completed.stdout + completed.stderr)
        return row['path_key'], completed.returncode

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute, r) for r in selected]
        for future in as_completed(futures):
            key, code = future.result()
            result = aggregate('running')
            print(json.dumps({'path': key, 'process_exit': code, 'completed_paths': result['completed_paths'],
                'corrected_errors': result['corrected_errors'], 'api_requests': result['api_requests']},
                ensure_ascii=False), flush=True)
    result = aggregate('completed')
    assert result['api_requests'] <= 20
    assert all(probe.sha(f) == h for f, h in protocol['protected_file_hashes'].items())
    print(json.dumps({k: v for k, v in result.items() if k != 'path_results'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run', 'one'), required=True)
    parser.add_argument('--path-key')
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'one': lambda: one(args.path_key)}[args.stage]()
