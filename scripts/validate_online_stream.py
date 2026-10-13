"""Check every raw CDT against current D2 offline preprocessing without inference.

The offline oracle may read the whole Trigger column. The streaming processor
receives only waveform blocks and never receives oracle events or labels.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from eeg_agent.config import file_hash, load_config
from eeg_agent.data import metadata
from eeg_agent.online.stream import StreamingProcessor
from eeg_agent.recordings import preprocess, state_segments_from_events, trigger_events


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_record(path, meta, block_samples, seconds=None):
    begin = time.perf_counter()
    count = meta['actual_samples']
    if seconds is not None:
        count = min(count, int(seconds * 1024))
    raw = np.memmap(path, dtype='<f4', mode='r', shape=(meta['actual_samples'], meta['n_channels']))[:count]
    channels = meta['channels']
    print(f'  offline oracle: {count / 1024:.3f} source seconds', flush=True)
    processed, names, compatible = preprocess(raw, channels, 1024)
    if not compatible:
        raise ValueError('Source does not match the current thirty-channel offline model input')
    marks = trigger_events(raw, channels.index('Trigger'))
    tasks, rests, oracle_warnings = state_segments_from_events(marks, count, 1024)
    expected_windows, expected_ends, expected_starts = {}, [], []
    for scope, segments in (('task', tasks), ('rest', rests)):
        for segment in segments:
            start, end, ordinal = segment['start_sample'], segment['end_sample'], segment['ordinal']
            expected_starts.append((scope, ordinal, start))
            expected_ends.append((scope, ordinal, start, end, segment['complete']))
            for index in range((end - start) // 5120):
                stamp = start + index * 5120
                expected_windows[scope, ordinal, stamp] = (stamp + 5120, index + 1)
    expected_starts.sort(key=lambda row: row[2])
    expected_ends.sort(key=lambda row: row[3])
    oracle_seconds = time.perf_counter() - begin
    print(f'  stream: {len(expected_windows)} expected task/rest windows; block={block_samples}', flush=True)
    processor = StreamingProcessor(channels)
    observed, actual_ends, actual_starts, mismatches = set(), [], [], []
    valid_counts = {'task': 0, 'rest': 0}
    window_counts = {'task': 0, 'rest': 0}
    valid_expected = {'task': 0, 'rest': 0}
    mismatch_count = 0
    max_error = 0.0
    warning_count = 0
    stream_begin = time.perf_counter()
    next_progress = stream_begin + 30

    def mismatch(kind, **details):
        nonlocal mismatch_count
        mismatch_count += 1
        if len(mismatches) < 20:
            mismatches.append({'kind': kind, **details})

    def consume(events):
        nonlocal max_error, warning_count
        for event in events:
            kind, scope = event['type'], event['scope']
            if kind == 'warning':
                warning_count += 1
            elif kind in ('path_start', 'rest_start'):
                actual_starts.append((scope, event['ordinal'], event['start_sample']))
            elif kind in ('path_end', 'rest_end'):
                actual_ends.append((scope, event['ordinal'], event['start_sample'],
                                   event['end_sample'], event['complete']))
            elif kind == 'window':
                window_counts[scope] += 1
                valid_counts[scope] += int(event['valid'])
                key = scope, event['ordinal'], event['start_sample']
                if key in observed:
                    mismatch('duplicate_window', key=key)
                observed.add(key)
                if key not in expected_windows:
                    mismatch('unexpected_window', key=key)
                    continue
                end, window_ordinal = expected_windows[key]
                if event['end_sample'] != end or event['window_ordinal'] != window_ordinal:
                    mismatch('window_boundary_or_ordinal', key=key)
                if event['observed_end_sample'] < end:
                    mismatch('early_window', key=key)
                left = (event['start_sample'] + 3) // 4
                expected = processed[:, left:left + 1280]
                actual = event['window']
                if expected.shape != (30, 1280) or actual.shape != expected.shape:
                    mismatch('window_shape', key=key)
                    continue
                if not np.array_equal(actual, expected, equal_nan=True):
                    mismatch('waveform', key=key)
                both_finite = np.isfinite(actual) & np.isfinite(expected)
                if both_finite.any():
                    max_error = max(max_error, float(np.max(np.abs(actual[both_finite] - expected[both_finite]))))
                expected_valid = bool(np.isfinite(expected).all() and
                    np.ptp(expected, axis=1).max() <= 2000 and
                    np.count_nonzero(expected.std(axis=1) >= .05) >= 28)
                valid_expected[scope] += int(expected_valid)
                if event['valid'] != expected_valid:
                    mismatch('quality_gate', key=key)

    for start in range(0, count, block_samples):
        stop = min(count, start + block_samples)
        consume(processor.feed(raw[start:stop], start))
        now = time.perf_counter()
        if now >= next_progress:
            print(f'  progress: {stop / 1024:.1f}/{count / 1024:.1f} source seconds; '
                  f'{sum(window_counts.values())} windows; {now - stream_begin:.1f}s processing', flush=True)
            next_progress = now + 30
    consume(processor.finish())
    absent = set(expected_windows) - observed
    if absent:
        mismatch('missing_windows', count=len(absent), examples=[list(key) for key in sorted(absent)[:10]])
    if actual_starts != expected_starts:
        mismatch('segment_starts', expected=expected_starts, actual=actual_starts)
    if actual_ends != expected_ends:
        mismatch('segment_ends_or_complete', expected=expected_ends, actual=actual_ends)
    result = {'status': 'passed' if not mismatch_count else 'failed',
              'source_samples': count, 'source_seconds': count / 1024,
              'expected_windows': len(expected_windows), 'observed_windows': sum(window_counts.values()),
              'window_counts': window_counts, 'valid_windows': valid_counts,
              'oracle_valid_windows': valid_expected,
              'task_segments': len(tasks), 'rest_segments': len(rests),
              'complete_tasks': sum(segment['complete'] for segment in tasks),
              'complete_rests': sum(segment['complete'] for segment in rests),
              'missing_windows': len(absent), 'mismatch_count': mismatch_count,
              'max_abs_finite_error_uv': max_error, 'exact_equal_nan': mismatch_count == 0,
              'oracle_seconds': oracle_seconds, 'stream_seconds': time.perf_counter() - stream_begin,
              'runtime_seconds': time.perf_counter() - begin,
              'stream_warning_count': warning_count, 'oracle_warnings': oracle_warnings,
              'source_warnings': meta['warnings'], 'mismatches': mismatches}
    del raw, processed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(load_config()['data_root']))
    parser.add_argument('--block-samples', type=int, default=256)
    parser.add_argument('--all-data', action='store_true', help='Validate every sample of all 26 recordings')
    parser.add_argument('--seconds', type=float, default=200, help='Smoke length per recording without --all-data')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    if args.block_samples < 1 or not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('block-samples and seconds must be positive')
    seconds = None if args.all_data else args.seconds
    destination = args.out or (PROJECT / 'outputs/online_implementation/20261011' /
        ('stream_validation.json' if args.all_data else 'stream_validation_smoke.json'))
    source_hashes = {str(path.relative_to(PROJECT)): file_hash(path) for path in
        (Path(__file__), PROJECT / 'eeg_agent/online/stream.py', PROJECT / 'eeg_agent/recordings.py')}
    previous = {}
    if destination.is_file() and not args.no_resume:
        saved = json.loads(destination.read_text(encoding='utf-8'))
        if (saved.get('source_hashes') == source_hashes and saved.get('block_samples') == args.block_samples
                and saved.get('seconds_limit') == seconds):
            previous = {row['acquisition']: row for row in saved.get('records', []) if row.get('status') == 'passed'}
    report = {'schema': 'current_raw_online_stream_validation_v1', 'status': 'running',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'validation_kind': 'accelerated numeric verification, not Pi realtime performance',
              'data_root': str(args.data_root.resolve()), 'block_samples': args.block_samples,
              'seconds_limit': seconds, 'expected_recordings': 26,
              'source_hashes': source_hashes, 'records': [],
              'api_calls': 0, 'inference_calls': 0, 'label_reads': 0, 'ceo_reads': 0}
    started = time.perf_counter()
    resumed = 0
    for acquisition in range(1, 27):
        print(f'[{acquisition:02d}/26] Acquisition {acquisition:02d}', flush=True)
        path = args.data_root / f'data/raw/Acquisition {acquisition:02d}.cdt'
        row = {'acquisition': acquisition, 'file': path.name}
        try:
            meta = metadata(path)
            identity = {'raw_sha256': file_hash(path), 'dpo_sha256': file_hash(str(path) + '.dpo')}
            if acquisition in previous and all(previous[acquisition].get(key) == value for key, value in identity.items()):
                row = {**previous[acquisition], 'resumed': True}
                resumed += 1
                print('  reused passed checkpoint with identical source and implementation hashes', flush=True)
            else:
                row.update(identity)
                before = (path.stat().st_size, path.stat().st_mtime_ns, Path(str(path) + '.dpo').stat().st_mtime_ns)
                row.update(validate_record(path, meta, args.block_samples, seconds))
                after = (path.stat().st_size, path.stat().st_mtime_ns, Path(str(path) + '.dpo').stat().st_mtime_ns)
                if before != after:
                    row.update(status='failed', error='Source changed during validation')
                row['resumed'] = False
                print(f"  {row['status']}: {row['observed_windows']} windows; "
                      f"{row['mismatch_count']} mismatches; {row['runtime_seconds']:.2f}s", flush=True)
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            print('  failed: ' + row['error'], flush=True)
        report['records'].append(row)
        report.update(completed_recordings=len(report['records']),
                      passed_recordings=sum(item['status'] == 'passed' for item in report['records']),
                      invocation_runtime_seconds=time.perf_counter() - started,
                      resumed_recordings=resumed)
        save_report(destination, report)
    report.update(status='passed' if report['passed_recordings'] == 26 else 'failed',
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  invocation_runtime_seconds=time.perf_counter() - started,
                  total_recorded_runtime_seconds=sum(row.get('runtime_seconds', 0) for row in report['records']),
                  total_source_seconds=sum(row.get('source_seconds', 0) for row in report['records']),
                  total_observed_windows=sum(row.get('observed_windows', 0) for row in report['records']))
    save_report(destination, report)
    print(f"{report['status']}: {report['passed_recordings']}/26 recordings; "
          f"{report['total_observed_windows']} windows; {report['invocation_runtime_seconds']:.2f}s this run", flush=True)
    print(str(destination.resolve()), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
