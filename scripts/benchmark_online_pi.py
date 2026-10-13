"""Measure the online API on a local Raspberry Pi 5; never substitutes PC timing.

Default: one complete Acquisition 01 replay at 1x, with cloud disabled.
--hours2 repeats fresh 1x sessions for two hours, then stops the final session.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import psutil

EXPECTED_BACKEND = 'mamba_minimal_cpu_fp32'
RESOURCE_GAP_LIMIT_SECONDS = 2.5


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def quantiles(values):
    if not values:
        return {'count': 0, 'p50': None, 'p95': None, 'max': None}
    ordered = sorted(values)
    def percentile(q):
        position = (len(ordered) - 1) * q
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)
    return {'count': len(values), 'p50': percentile(.5), 'p95': percentile(.95), 'max': ordered[-1]}


def api(base, path, document=None):
    body = None if document is None else json.dumps(document).encode('utf-8')
    request = Request(base + path, data=body, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def local_hardware():
    source = Path('/proc/device-tree/model')
    model = source.read_bytes().rstrip(b'\0').decode('utf-8', errors='replace') if source.is_file() else None
    eligible = platform.system() == 'Linux' and platform.machine() == 'aarch64' and bool(model and 'Raspberry Pi 5' in model)
    return {'model': model, 'platform': platform.platform(), 'machine': platform.machine(),
            'python': platform.python_version(), 'pi5_arm64': eligible,
            'total_memory_mb': psutil.virtual_memory().total / 1024**2}


def process_for_service(pid=None):
    if pid:
        result = psutil.Process(pid)
    else:
        found = subprocess.run(['systemctl', 'show', '-p', 'MainPID', '--value', 'eeg-agent-online.service'],
                               capture_output=True, text=True, timeout=5, check=False)
        if not found.stdout.strip().isdigit() or int(found.stdout.strip()) <= 0:
            raise RuntimeError('Cannot identify service process; supply --pid for a manually started server')
        result = psutil.Process(int(found.stdout.strip()))
    result.cpu_percent(None)
    psutil.cpu_percent(None)
    return result


def sample(process, elapsed, queue_depth):
    thermal = Path('/sys/class/thermal/thermal_zone0/temp')
    try:
        temperature = float(thermal.read_text().strip()) / 1000
    except (OSError, ValueError):
        temperature = None
    try:
        value = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True,
                               text=True, timeout=2, check=False)
        throttled = int(value.stdout.strip().split('=', 1)[1], 16) if value.returncode == 0 else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        throttled = None
    return {'elapsed_seconds': elapsed, 'service_rss_mb': process.memory_info().rss / 1024**2,
            'service_cpu_percent': process.cpu_percent(None), 'system_cpu_percent': psutil.cpu_percent(None),
            'system_used_mb': psutil.virtual_memory().used / 1024**2, 'temperature_c': temperature,
            'throttled_bits': throttled, 'analysis_queue_depth': queue_depth}


def session_checks(current, recording, received_windows, deadline_stop):
    metrics = current.get('metrics', {})
    samples = recording.get('sample_count')
    eof = bool(type(samples) is int and samples > 0 and metrics.get('source_samples') == samples
               and metrics.get('read_samples') == samples and metrics.get('reader_complete') is True
               and metrics.get('source_eof_reached') is True
               and math.isclose(current.get('position_seconds', -1), samples / 1024, rel_tol=0, abs_tol=1e-9))
    produced = metrics.get('produced_windows')
    return {'termination_valid': current.get('state') == 'completed' or (deadline_stop and current.get('state') == 'stopped'),
            'source_eof_reached': eof,
            'full_replay_valid': (current.get('state') == 'completed' and eof) or
                                 (deadline_stop and current.get('state') == 'stopped'),
            'window_counts_match': type(produced) is int and produced == metrics.get('total_windows') == received_windows,
            'error_free': not current.get('error') and not metrics.get('persistence_error'),
            'one_times_speed': current.get('speed') == 1,
            'realtime_mode': metrics.get('replay_mode') == 'realtime',
            'arm64_service': metrics.get('measurement_host') == 'aarch64',
            'cloud_disabled': current.get('cloud', {}).get('requested') is False
                              and not any(current['cloud'].get(key) for key in ('completed', 'pending', 'in_flight'))}


def run(args):
    hardware = local_hardware()
    result = {'schema': 'pi5_online_benchmark_v2', 'status': 'hardware_not_tested',
              'hardware_tested': False, 'realtime_passed': False, 'hardware': hardware,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'api_url': args.url, 'recording_id': args.recording_id,
              'speed': 1, 'cloud': False, 'requested_hours': 2 if args.hours2 else None,
              'sessions': [], 'api_calls_to_model_provider': 0}
    if not hardware['pi5_arm64']:
        result['reason'] = 'Raspberry Pi 5 aarch64 hardware is not available on this host; no replay or timing was executed'
        save(args.out, result)
        print(result['reason'])
        print(str(args.out.resolve()))
        return 2
    if urlsplit(args.url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Run this benchmark on the Pi against its loopback API')
    process = process_for_service(args.pid)
    result['service_pid'] = process.pid
    result['software'] = {}
    for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'joblib', 'fastapi', 'onnxruntime'):
        try:
            result['software'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['software'][name] = None
    catalog = api(args.url, '/api/online/catalog')
    recording = next((row for row in catalog['recordings'] if row['id'] == args.recording_id), None)
    if recording is None or not catalog['capabilities'].get('model_available'):
        raise RuntimeError('Raw recording or frozen CPU model is not available')
    if catalog['capabilities'].get('backend') != EXPECTED_BACKEND:
        raise RuntimeError('Unexpected numerical backend; expected ' + EXPECTED_BACKEND)
    result.update(hardware_tested=True, status='running', recording=recording,
                  pipeline_id=catalog['pipeline_id'], model_backend=catalog['capabilities'].get('backend'))
    save(args.out, result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples_path = args.out.with_suffix('.samples.jsonl')
    processing, completion_lags, rss, cpu, system_memory, temperatures, queues, throttling = [], [], [], [], [], [], [], []
    telemetry_missing = 0
    resource_missing = {key: 0 for key in ('service_rss_mb', 'service_cpu_percent', 'system_used_mb',
                                          'analysis_queue_depth', 'temperature_c', 'throttled_bits')}
    resource_samples = 0
    resource_intervals, resource_gaps = [], []
    windows = 0
    model_readouts = 0
    began = time.monotonic()
    last_resource_wall = began
    deadline = began + 7200 if args.hours2 else None
    last_sample = began - 1
    last_progress = began
    active = None
    try:
        with samples_path.open('w', encoding='utf-8') as telemetry:
            while deadline is None or time.monotonic() < deadline:
                current = api(args.url, '/api/online/sessions',
                              {'recording_id': args.recording_id, 'speed': 1, 'cloud': False})
                active = current['id']
                sequence, queue_depth, state = -1, 0, current['state']
                received_windows = 0
                deadline_stop = False
                print('1x replay started: ' + active, flush=True)
                with urlopen(args.url + '/api/online/sessions/' + active + '/events?after=-1', timeout=15) as response:
                    while True:
                        line = response.readline()
                        if not line:
                            break
                        now = time.monotonic()
                        if deadline is not None and now >= deadline and not deadline_stop:
                            current = api(args.url, '/api/online/sessions/' + active)
                            if current['state'] in ('created', 'running', 'paused', 'finalizing'):
                                current = api(args.url, '/api/online/sessions/' + active + '/control', {'action': 'stop'})
                                state = current['state']
                                deadline_stop = current['state'] == 'stopped'
                            # Drain remaining SSE through its terminal event so
                            # a window received at the cutoff is still counted.
                        if line.startswith(b'data: '):
                            event = json.loads(line[6:])
                            if event['sequence'] <= sequence:
                                continue
                            sequence = event['sequence']
                            payload, kind = event['payload'], event['type']
                            if kind == 'state':
                                state = payload['state']
                            elif kind == 'progress':
                                queue_depth = payload.get('queue_depth', queue_depth)
                            elif kind == 'window':
                                windows += 1
                                received_windows += 1
                                model_readouts += int(payload.get('vrms', {}).get('available', False))
                                timings = [payload.get('processing_seconds'), payload.get('completion_lag_seconds')]
                                if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in timings):
                                    telemetry_missing += 1
                                else:
                                    processing.append(float(payload['processing_seconds']))
                                    completion_lags.append(float(payload['completion_lag_seconds']))
                                    telemetry.write(json.dumps({'kind': 'window', 'session': active,
                                        'sequence': sequence, 'processing_seconds': processing[-1],
                                        'completion_lag_seconds': completion_lags[-1],
                                        'scope': payload['scope'], 'start_seconds': payload['start_seconds'],
                                        'end_seconds': payload['end_seconds']}) + '\n')
                            elif kind == 'snapshot':
                                # A lost SSE history cannot provide individual latency quantiles.
                                telemetry_missing += 1
                        if now - last_sample >= 1:
                            observed = sample(process, now - began, queue_depth)
                            sample_finished = time.monotonic()
                            interval = sample_finished - last_resource_wall
                            resource_intervals.append(interval)
                            if interval > RESOURCE_GAP_LIMIT_SECONDS:
                                gap = {'from_elapsed_seconds': last_resource_wall - began,
                                       'to_elapsed_seconds': sample_finished - began, 'seconds': interval}
                                resource_gaps.append(gap)
                                telemetry.write(json.dumps({'kind': 'resource_gap', **gap}) + '\n')
                            last_resource_wall = sample_finished
                            resource_samples += 1
                            for key in resource_missing:
                                value = observed.get(key)
                                if type(value) not in (int, float) or not math.isfinite(value) or (key != 'temperature_c' and value < 0):
                                    resource_missing[key] += 1
                            telemetry.write(json.dumps({'kind': 'resource', **observed}) + '\n')
                            telemetry.flush()
                            for key, values in (('service_rss_mb', rss), ('service_cpu_percent', cpu),
                                                ('system_used_mb', system_memory), ('analysis_queue_depth', queues),
                                                ('temperature_c', temperatures), ('throttled_bits', throttling)):
                                value = observed.get(key)
                                if type(value) in (int, float) and math.isfinite(value):
                                    values.append(value)
                            last_sample = sample_finished
                        if now - last_progress >= 30:
                            print(f'{now - began:.1f}s measured; {windows} windows; '
                                  f'lag P95={quantiles(completion_lags)["p95"]}; RSS={rss[-1] if rss else None} MB', flush=True)
                            last_progress = now
                current = api(args.url, '/api/online/sessions/' + active)
                if current['metrics'].get('total_windows') != received_windows:
                    telemetry_missing += abs(current['metrics'].get('total_windows', 0) - received_windows) or 1
                checks = session_checks(current, recording, received_windows, deadline_stop)
                result['sessions'].append({'id': active, 'state': current['state'],
                    'speed': current.get('speed'),
                    'position_seconds': current['position_seconds'], 'metrics': current['metrics'],
                    'cloud': current['cloud'], 'error': current.get('error'), 'sse_windows_received': received_windows,
                    'stopped_by_benchmark_at_deadline': deadline_stop, 'acceptance_checks': checks})
                required = ('termination_valid', 'full_replay_valid', 'window_counts_match', 'error_free',
                            'one_times_speed', 'realtime_mode', 'arm64_service', 'cloud_disabled')
                failed = [key for key in required if not checks[key]]
                if failed:
                    raise RuntimeError('Replay acceptance failed: ' + ', '.join(failed))
                active = None
                if deadline is None:
                    break
        result['status'] = 'measured'
    except KeyboardInterrupt:
        result.update(status='interrupted', reason='Benchmark interrupted by operator')
    except Exception as exc:
        result.update(status='failed', reason=type(exc).__name__ + ': ' + str(exc))
    finally:
        if active is not None:
            try:
                api(args.url, '/api/online/sessions/' + active + '/control', {'action': 'stop'})
            except Exception:
                pass
        finished_wall = time.monotonic()
        tail_interval = finished_wall - last_resource_wall
        resource_intervals.append(tail_interval)
        if tail_interval > RESOURCE_GAP_LIMIT_SECONDS:
            resource_gaps.append({'from_elapsed_seconds': last_resource_wall - began,
                                  'to_elapsed_seconds': finished_wall - began, 'seconds': tail_interval})
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - began,
                      windows=windows, model_available_readouts=model_readouts, missing_latency_events=telemetry_missing,
                      resource_samples=resource_samples, missing_resource_samples=resource_missing,
                      resource_sampling_gap_limit_seconds=RESOURCE_GAP_LIMIT_SECONDS,
                      resource_sampling_gaps=resource_gaps,
                      resource_sampling_intervals_seconds=quantiles(resource_intervals),
                      missing_resource_fractions={key: count / max(1, resource_samples) for key, count in resource_missing.items()},
                      window_processing_seconds=quantiles(processing), completion_lag_seconds=quantiles(completion_lags),
                      service_rss_mb=quantiles(rss), service_cpu_percent=quantiles(cpu),
                      system_used_mb=quantiles(system_memory),
                      temperature_c=quantiles(temperatures), inference_backlog_events=quantiles(queues),
                      throttled_bits_observed=sorted(set(throttling)), telemetry_file=str(samples_path),
                      latency_scope='source-clock due time to window completion; requires matching runtime telemetry',
                      cpu_note='process percent can exceed 100 on multiple cores',
                      power_watts=None, power_note='Electrical power requires a separate meter and is not inferred from CPU usage')
        p95 = result['completion_lag_seconds']['p95']
        processing_p95 = result['window_processing_seconds']['p95']
        result['acceptance_targets'] = {'processing_p95_seconds_lt': 5, 'completion_lag_p95_seconds_lt': 5,
                                        'service_rss_mb_lte': 4096, 'system_used_mb_lte': 6144,
                                        'stability_seconds_gte': 7200 if args.hours2 else None}
        result['realtime_passed'] = bool(result['status'] == 'measured' and windows and model_readouts
            and not telemetry_missing and resource_samples and not any(resource_missing.values()) and not resource_gaps
            and p95 is not None and p95 < 5
            and processing_p95 is not None and processing_p95 < 5
            and rss and max(rss) <= 4096 and system_memory and max(system_memory) <= 6144
            and queues and max(queues) < 8 and temperatures and throttling and not any(throttling)
            and (not args.hours2 or result['wall_seconds'] >= 7200)
            and all(row['cloud'].get('completed', 0) == 0 for row in result['sessions']))
        result['stability_passed'] = bool(args.hours2 and result['realtime_passed'])
        result['fault_injection_passed'] = None
        result['fault_injection_note'] = 'Cloud timeout/offline, browser reconnect, stop and service restart require the separate acceptance checklist.'
        save(args.out, result)
    print(json.dumps({'status': result['status'], 'hardware_tested': True,
                      'realtime_passed': result['realtime_passed'], 'windows': windows}, indent=2))
    return 0 if result['realtime_passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8880')
    parser.add_argument('--recording-id', default='subject-01')
    parser.add_argument('--hours2', action='store_true')
    parser.add_argument('--pid', type=int, help='Service PID when running outside systemd')
    parser.add_argument('--out', type=Path, default=Path('/var/lib/eeg-agent-online/benchmarks') /
                        ('pi5_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'))
    arguments = parser.parse_args()
    try:
        exit_code = run(arguments)
    except (OSError, ValueError, RuntimeError) as exc:
        save(arguments.out, {'schema': 'pi5_online_benchmark_v2', 'status': 'failed',
             'hardware_tested': False, 'realtime_passed': False, 'hardware': local_hardware(),
             'reason': type(exc).__name__ + ': ' + str(exc), 'cloud': False, 'speed': 1})
        print(type(exc).__name__ + ': ' + str(exc))
        exit_code = 1
    raise SystemExit(exit_code)
