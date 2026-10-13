"""Pi acceptance gates with an isolated API, fake hardware and fake clock."""
import copy
import io
import json
from types import SimpleNamespace

import pytest

from scripts import benchmark_online_pi as benchmark


@pytest.fixture
def benchmark_double(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'local_hardware', lambda: {'pi5_arm64': True, 'model': 'offline hardware double'})
    monkeypatch.setattr(benchmark, 'process_for_service', lambda pid: SimpleNamespace(pid=123))
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(benchmark.time, 'monotonic', lambda: clock.now)

    def execute(state='completed', seconds=30, hours2=False, damage=None, missing=None, missing_sse=False,
                external_deadline_stop=False, backend='mamba_minimal_cpu_fp32', telemetry_gap=False):
        recording = {'id': 'subject-01', 'duration_seconds': seconds, 'sample_count': seconds * 1024}
        current = {'id': 'fake-session', 'state': state, 'speed': 1, 'position_seconds': float(seconds), 'error': None,
                   'metrics': {'total_windows': 3, 'produced_windows': 3, 'source_samples': seconds * 1024,
                               'read_samples': seconds * 1024, 'reader_complete': True, 'source_eof_reached': True,
                               'measurement_host': 'aarch64', 'replay_mode': 'realtime'},
                   'cloud': {'requested': False, 'completed': 0, 'pending': 0, 'in_flight': False}}
        if damage:
            damage(current)
        calls, samples = [], []
        stopped_by_script = False
        clock.now = 0.

        def fake_api(base, path, document=None):
            nonlocal stopped_by_script
            calls.append((path, document))
            if path == '/api/online/catalog':
                return {'recordings': [recording], 'pipeline_id': 'offline-double',
                        'capabilities': {'model_available': True, 'backend': backend}}
            if path == '/api/online/sessions':
                assert document == {'recording_id': 'subject-01', 'speed': 1, 'cloud': False}
                return {**current, 'state': 'running'}
            if path.endswith('/control'):
                current['state'] = 'stopped'
                stopped_by_script = True
            if hours2 and state == 'stopped' and seconds >= 7200 and not stopped_by_script and not external_deadline_stop:
                return {**copy.deepcopy(current), 'state': 'running'}
            return copy.deepcopy(current)

        def fake_sample(process, elapsed, queue_depth):
            observed = {'elapsed_seconds': elapsed, 'service_rss_mb': 800., 'service_cpu_percent': 80.,
                        'system_used_mb': 2000., 'analysis_queue_depth': 0, 'temperature_c': 55., 'throttled_bits': 0}
            samples.append(observed)
            if missing and len(samples) > 1:
                observed[missing] = None
            return observed

        finish = 7200 if hours2 and state == 'stopped' and seconds >= 7200 else seconds
        entries = []
        sequence = 0
        for stamp in range(1, finish + 1):
            if telemetry_gap and 22 <= stamp <= 26:
                continue
            if stamp in (11, 16, 21) and not (missing_sse and stamp == 21):
                event = {'sequence': sequence, 'type': 'window', 'payload': {
                    'scope': 'task', 'start_seconds': stamp - 5, 'end_seconds': stamp,
                    'processing_seconds': .2, 'completion_lag_seconds': .3, 'vrms': {'available': True}}}
                entries.append((stamp, b'data: ' + json.dumps(event).encode() + b'\n'))
                sequence += 1
            else:
                entries.append((stamp, b': heartbeat\n'))
        entries.append((finish + .1, b'data: ' + json.dumps({
            'sequence': sequence, 'type': 'state', 'payload': {'state': state}}).encode() + b'\n'))

        class Response(io.BytesIO):
            def __init__(self):
                self.entries = iter(entries)

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                pass

            def readline(self):
                entry = next(self.entries, None)
                if entry is None:
                    return b''
                clock.now, line = entry
                return line

        monkeypatch.setattr(benchmark, 'api', fake_api)
        monkeypatch.setattr(benchmark, 'sample', fake_sample)
        monkeypatch.setattr(benchmark, 'urlopen', lambda *args, **kwargs: Response())
        out = tmp_path / 'benchmark.json'
        args = SimpleNamespace(url='http://127.0.0.1:8880', recording_id='subject-01',
                               hours2=hours2, pid=None, out=out)
        code = benchmark.run(args)
        return code, json.loads(out.read_text(encoding='utf-8')), calls

    return execute


def test_full_replay_passes_only_with_eof_and_matching_producer_consumer_sse_counts(benchmark_double):
    code, result, _ = benchmark_double()
    assert code == 0 and result['realtime_passed'] is True
    assert all(result['sessions'][0]['acceptance_checks'].values())
    assert result['missing_resource_samples']['temperature_c'] == 0


@pytest.mark.parametrize('hours2', [False, True])
def test_external_premature_stop_can_never_pass(benchmark_double, hours2):
    def early(current):
        current['position_seconds'] = 21.
        current['metrics'].update(read_samples=21 * 1024, reader_complete=False, source_eof_reached=False)
    code, result, _ = benchmark_double(state='stopped', seconds=60, hours2=hours2, damage=early)
    assert code == 1 and result['status'] == 'failed' and result['realtime_passed'] is False
    assert result['sessions'][0]['stopped_by_benchmark_at_deadline'] is False


@pytest.mark.parametrize('damage', [
    lambda row: row.update(position_seconds=21.),
    lambda row: row['metrics'].update(source_eof_reached=False),
    lambda row: row['metrics'].update(reader_complete=False),
    lambda row: row['metrics'].update(read_samples=29 * 1024),
    lambda row: row['metrics'].update(produced_windows=4),
    lambda row: row['metrics'].pop('produced_windows'),
    lambda row: row.update(error='analysis_error:test'),
])
def test_completed_state_cannot_hide_eof_or_dropped_window_failure(benchmark_double, damage):
    code, result, _ = benchmark_double(damage=damage)
    assert code == 1 and result['status'] == 'failed' and result['realtime_passed'] is False


def test_sse_window_loss_fails_even_when_producer_and_consumer_agree(benchmark_double):
    code, result, _ = benchmark_double(missing_sse=True)
    assert code == 1 and result['missing_latency_events'] == 1
    assert result['sessions'][0]['acceptance_checks']['window_counts_match'] is False


@pytest.mark.parametrize('missing', ['temperature_c', 'throttled_bits', 'service_rss_mb'])
def test_later_missing_resource_samples_do_not_disappear_from_acceptance(benchmark_double, missing):
    code, result, _ = benchmark_double(missing=missing)
    assert code == 1 and result['status'] == 'measured' and result['realtime_passed'] is False
    assert result['resource_samples'] == 30
    assert result['missing_resource_samples'][missing] == 29
    assert result['missing_resource_fractions'][missing] == pytest.approx(29 / 30)


def test_only_benchmark_deadline_can_stop_the_last_two_hour_replay(benchmark_double):
    def partial(current):
        current.update(position_seconds=7200.)
        current['metrics'].update(read_samples=7200 * 1024, reader_complete=False, source_eof_reached=False)
    code, result, calls = benchmark_double(state='stopped', seconds=9000, hours2=True, damage=partial)
    assert code == 0 and result['stability_passed'] is True
    assert result['sessions'][0]['stopped_by_benchmark_at_deadline'] is True
    assert result['sessions'][0]['acceptance_checks']['source_eof_reached'] is False
    assert sum(path.endswith('/control') for path, _ in calls) == 1


def test_non_pi_host_never_requests_a_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'local_hardware', lambda: {'pi5_arm64': False})
    monkeypatch.setattr(benchmark, 'api', lambda *args: pytest.fail('Non-Pi benchmark requested an API'))
    out = tmp_path / 'not-tested.json'
    args = SimpleNamespace(url='http://127.0.0.1:8880', recording_id='subject-01', hours2=False, out=out)
    assert benchmark.run(args) == 2
    assert json.loads(out.read_text(encoding='utf-8'))['hardware_tested'] is False


def test_external_stop_already_terminal_at_deadline_is_not_attributed_to_benchmark(benchmark_double):
    code, result, _ = benchmark_double(state='stopped', seconds=7200, hours2=True, external_deadline_stop=True)
    assert code == 1 and result['status'] == 'failed'
    assert result['sessions'][0]['stopped_by_benchmark_at_deadline'] is False


@pytest.mark.parametrize('damage', [
    lambda row: row.update(speed=10),
    lambda row: row['metrics'].update(replay_mode='accelerated_validation'),
    lambda row: row['metrics'].update(measurement_host='AMD64'),
    lambda row: row['metrics'].pop('measurement_host'),
    lambda row: row['cloud'].update(requested=True),
    lambda row: row['cloud'].pop('requested'),
    lambda row: row['cloud'].update(in_flight=True),
])
def test_requested_parameters_do_not_substitute_for_actual_runtime_evidence(benchmark_double, damage):
    code, result, _ = benchmark_double(damage=damage)
    assert code == 1 and result['status'] == 'failed' and result['realtime_passed'] is False


def test_wrong_catalog_backend_is_rejected_before_replay(benchmark_double):
    with pytest.raises(RuntimeError, match='backend'):
        benchmark_double(backend='cuda_bf16')


def test_telemetry_time_gaps_fail_even_when_every_collected_resource_sample_is_present(benchmark_double):
    code, result, _ = benchmark_double(telemetry_gap=True)
    assert code == 1 and result['realtime_passed'] is False
    assert not any(result['missing_resource_samples'].values())
    assert result['resource_sampling_gaps'][0]['seconds'] == pytest.approx(6)
