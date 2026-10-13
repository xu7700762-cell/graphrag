"""Replay lifecycle and truthful path-end cloud contracts, with no model/API load."""
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from eeg_agent.config import EEG_CHANNELS, load_config
from eeg_agent.online.runtime import ACTIVE, OnlineManager, OnlineSession, TERMINAL

CHANNELS = EEG_CHANNELS + ['M1', 'M2', 'VEOG', 'HEOG', 'ECG', 'EGG', 'Trigger']


def write_recording(cfg, seconds=30, marks=((6, 20), (21, 22), (27, 20))):
    folder = Path(cfg['data_root']) / 'data/raw'
    folder.mkdir(parents=True, exist_ok=True)
    count = int(seconds * 1024)
    raw = np.random.default_rng(14).normal(0, 10, (count, len(CHANNELS))).astype('<f4')
    raw[:, -1] = 0
    for stamp, mark in marks:
        position = int(stamp * 1024)
        raw[position:position + 8, -1] = mark
    path = folder / 'Acquisition 10.cdt'
    raw.tofile(path)
    header = ('NumChannels = 37\nNumSamples = ' + str(count) + '\nSampleFreqHz = 1024\n'
              'DataByteOrder = INTEL\nDataSampOrder = SAMP\nDataFormat = 6\nDataUnit = uV\n'
              'CommonScale = 1\nCommonOffset = 0\nLABELS START_LIST\n' + '\n'.join(CHANNELS) + '\nLABELS END_LIST\n')
    Path(str(path) + '.dpo').write_text(header, encoding='utf-8')
    return path


class FakePath:
    def __init__(self):
        self.count = 0

    def add(self, window):
        assert window.shape == (30, 1280) and np.isfinite(window).all()
        self.count += 1
        return self.result()

    def result(self):
        return {'available': self.count >= 3, 'mil_raw_high_probability': .7, 'mil_class': 'High',
                'numeric_fusion': {'raw_combined_score': .8, 'predicted_class': 'High'}}


class FakeModel:
    def __init__(self):
        self.folds = []

    def start_path(self, fold):
        self.folds.append(fold)
        return FakePath()


class FakeCloud:
    def __init__(self, calls, fail=False):
        self.calls, self.fail = calls, fail

    def report(self, row, prediction, cancelled=None):
        assert row['scope'] == 'task' and row['complete'] is True
        assert row['data_cutoff_seconds'] >= row['end_seconds']
        self.calls.append((row, prediction))
        if self.fail:
            raise TimeoutError('fake offline timeout')
        return {'status': 'verified', 'final_class': 'Low', 'explanation': '离线测试回复'}


@pytest.fixture
def online_cfg(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.update(data_root=str(tmp_path / 'data'), output_root=str(tmp_path / 'offline'),
               online_output_root=str(tmp_path / 'online'), online_model_manifest=str(tmp_path / 'missing-model.json'),
               cloud_config_path=str(tmp_path / '.env'))
    write_recording(cfg)
    monkeypatch.setattr(OnlineManager, 'cloud_configured', lambda self: False)
    return cfg


def wait_finished(session, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if session.state in TERMINAL and session._reader_done.is_set() and session._processor_done.is_set() and session._cloud_done.is_set():
            for worker in (session.reader, session.worker, session.cloud_worker):
                worker.join(.5)
            return session.snapshot()
        time.sleep(.01)
    pytest.fail('Replay workers did not finish: ' + str(session.snapshot(False)))


def manual_session(cfg, monkeypatch, cloud=False, configured=False, fail=False):
    monkeypatch.setattr(OnlineManager, 'cloud_configured', lambda self: configured)
    calls, model = [], FakeModel()
    manager = OnlineManager(cfg, model_factory=lambda: model, cloud_factory=lambda emit: FakeCloud(calls, fail))
    session = OnlineSession(manager, {'recording_id': 'subject-10', 'speed': 1, 'cloud': cloud, 'max_seconds': None})
    manager.sessions[session.id] = session
    return manager, session, calls, model


def feed_path(session, ordinal=1, complete=True, windows=3, scope='task'):
    start = 6.
    session._process({'type': 'path_start' if scope == 'task' else 'rest_start', 'scope': scope,
                      'ordinal': ordinal, 'start_seconds': start, 'end_seconds': start})
    for index in range(windows):
        session._process({'type': 'window', 'scope': scope, 'ordinal': ordinal,
                          'start_seconds': start + index * 5, 'end_seconds': start + (index + 1) * 5,
                          'observed_end_sample': int((start + (index + 1) * 5) * 1024),
                          'valid': True, 'reason': '', 'window': np.random.default_rng(index).normal(0, 10, (30, 1280)).astype('float32'),
                          '_ready_wall': time.monotonic()})
    session._process({'type': 'path_end' if scope == 'task' else 'rest_end', 'scope': scope,
                      'ordinal': ordinal, 'start_seconds': start, 'end_seconds': start + windows * 5 + .5,
                      'observed_end_sample': int((start + windows * 5 + .5) * 1024) + 1,
                      'complete': complete, 'reason': '' if complete else 'missing_end_marker'})


def test_create_pause_resume_stop_and_single_active_session(online_cfg, monkeypatch):
    monkeypatch.setattr(OnlineSession, 'start', lambda self: self._set_state('running'))
    manager = OnlineManager(online_cfg, model_factory=FakeModel)
    request = {'recording_id': 'subject-10', 'speed': 1, 'cloud': False, 'max_seconds': None}
    created = manager.create(request)
    session = manager.get(created['id'])
    assert created['state'] == 'running' and created['recording']['acquisition'] == 10
    assert session.control('pause')['state'] == 'paused'
    with pytest.raises(RuntimeError, match='在线回放'):
        manager.create(request)
    assert session.control('resume')['state'] == 'running'
    assert session.control('stop')['state'] == 'stopped'
    assert session.stop_event.is_set()
    with pytest.raises(ValueError):
        session.control('resume')
    assert manager.create(request)['id'] != session.id
    manager.shutdown()


def test_real_reader_eof_keeps_incomplete_task_unclassified(online_cfg):
    model = FakeModel()
    manager = OnlineManager(online_cfg, model_factory=lambda: model)
    value = manager.create({'recording_id': 'subject-10', 'speed': 1000, 'cloud': False, 'max_seconds': None})
    result = wait_finished(manager.get(value['id']))
    assert result['state'] == 'completed', result['error']
    tasks = [row for row in result['paths'] if row['scope'] == 'task']
    assert len(tasks) == 2
    assert tasks[0]['complete'] and tasks[0]['accepted_windows'] == 3
    assert tasks[0]['vrms']['mil_class'] == 'High'
    assert tasks[1]['complete'] is False and tasks[1]['vrms']['mil_class'] is None
    assert result['position_seconds'] == 30
    assert result['metrics']['reader_complete'] is True and result['metrics']['source_eof_reached'] is True
    assert result['metrics']['read_samples'] == result['recording']['sample_count'] == 30 * 1024
    assert result['metrics']['produced_windows'] == result['metrics']['total_windows'] == 4
    assert all(fold == 10 for fold in model.folds)
    assert not any(event['type'].startswith('cloud_') for event in result['events'])


def test_service_restart_marks_active_sessions_interrupted(online_cfg, monkeypatch):
    manager, session, _, _ = manual_session(online_cfg, monkeypatch)
    session._set_state('paused')
    restarted = OnlineManager(online_cfg)
    saved = restarted.get(session.id)
    assert saved['state'] == 'interrupted'
    assert '重启' in saved['error']
    persisted = json.loads((session.directory / 'session.json').read_text(encoding='utf-8'))
    assert persisted['state'] == 'interrupted'
    assert persisted['position_seconds'] == 0
    assert restarted.cloud_in_use() is False


def test_queue_overrun_stops_explicitly_without_dropping_a_window(online_cfg, monkeypatch):
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    session._set_state('running')
    for _ in range(session._queue.maxsize):
        assert session._enqueue({'type': 'warning'}) is True
    assert session._enqueue({'type': 'window'}) is False
    assert session._queue.qsize() == 8
    assert session.state == 'failed' and 'overrun' in session.error
    assert session.stop_event.is_set()


@pytest.mark.parametrize('fail', [False, True])
def test_cloud_is_path_end_only_and_failure_cannot_replace_its_class(online_cfg, monkeypatch, fail):
    _, session, calls, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True, fail=fail)
    session._set_state('running')
    feed_path(session, complete=True)
    assert calls == []  # Cloud work is queued, never performed in the numerical worker.
    assert session.paths[0]['cloud']['status'] == 'pending'
    session._processor_done.set()
    session._cloud_consume()
    assert len(calls) == 1
    row = session.paths[0]
    assert row['vrms']['mil_class'] == 'High' and row['vrms']['numeric_class'] == 'High'
    assert row['cloud']['status'] == ('fallback' if fail else 'verified')
    assert row['cloud']['final_class'] == (None if fail else 'Low')
    assert session.cloud['pending'] == 0
    assert session.cloud['failed' if fail else 'completed'] == 1
    types = [event['type'] for event in session.events]
    assert types.index('cloud_pending') > types.index('path_completed')
    assert types[-1] == ('cloud_fallback' if fail else 'cloud_verified')


def test_rest_and_incomplete_tasks_do_not_queue_cloud(online_cfg, monkeypatch):
    _, session, calls, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    feed_path(session, complete=False)
    feed_path(session, scope='rest', complete=True)
    assert session._cloud_queue.empty() and calls == []
    assert all(row['cloud']['status'] == 'not_requested' for row in session.paths)
    assert session.paths[0]['vrms']['available'] is False


def test_no_cloud_configuration_retains_local_output_and_returns_no_cloud_class(online_cfg, monkeypatch):
    _, session, calls, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=False)
    feed_path(session)
    assert session.paths[0]['vrms']['mil_class'] == 'High'
    assert session.paths[0]['cloud']['status'] == 'fallback'
    assert session.paths[0]['cloud']['final_class'] is None
    assert session._cloud_queue.empty() and calls == []


def test_source_validation_never_accepts_uploaded_or_arbitrary_paths(online_cfg):
    manager = OnlineManager(online_cfg)
    for identity in ('../Acquisition 10.cdt', 'edf-private', 'subject-1', 'subject-10/../../other'):
        with pytest.raises(ValueError):
            manager.source_path(identity)
    assert manager.catalog()['defaults']['cloud_trigger'] == 'path_end'


def test_pause_freezes_reader_position_resume_and_stop_join_workers(online_cfg):
    manager = OnlineManager(online_cfg, model_factory=FakeModel)
    value = manager.create({'recording_id': 'subject-10', 'speed': 1, 'cloud': False, 'max_seconds': 2})
    session = manager.get(value['id'])
    session.control('pause')
    time.sleep(.1)
    position = session.position_seconds
    time.sleep(.15)
    assert session.state == 'paused' and session.position_seconds == position
    session.control('resume')
    time.sleep(.35)
    assert session.position_seconds > position
    session.control('stop')
    result = wait_finished(session)
    assert result['state'] == 'stopped'
    assert result['position_seconds'] < 2


def test_slow_cloud_does_not_block_arrived_window_analysis(online_cfg, monkeypatch):
    manager, session, calls, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    started, release = threading.Event(), threading.Event()

    class DelayedCloud(FakeCloud):
        def report(self, row, prediction, cancelled=None):
            started.set()
            assert release.wait(2)
            return super().report(row, prediction, cancelled=cancelled)

    manager._cloud_factory = lambda emit: DelayedCloud(calls)
    session._set_state('running')
    feed_path(session, ordinal=1)
    worker = threading.Thread(target=session._cloud_consume, daemon=True)
    worker.start()
    try:
        assert started.wait(1)
        feed_path(session, ordinal=2, windows=1, complete=False)
        assert session.metrics['total_windows'] == 4
        assert session.latest['ordinal'] == 2
        assert session.cloud['pending'] == 1 and calls == []
    finally:
        session._processor_done.set()
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert session.paths[0]['cloud']['final_class'] == 'Low'
    assert session.paths[1]['cloud']['final_class'] is None


def test_cloud_queue_limit_preserves_local_results_and_explicit_failure(online_cfg, monkeypatch):
    _, session, _, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    for ordinal in range(1, 6):
        feed_path(session, ordinal=ordinal)
    assert session._cloud_queue.qsize() == 4
    assert session.cloud['pending'] == 4
    assert session.paths[-1]['cloud']['status'] == 'fallback'
    assert session.paths[-1]['cloud']['final_class'] is None
    assert all(row['vrms']['mil_class'] == 'High' for row in session.paths)
    assert session.cloud['failed'] == 1


def test_missing_model_still_publishes_descriptive_indicators(online_cfg, monkeypatch):
    manager = OnlineManager(online_cfg)
    session = OnlineSession(manager, {'recording_id': 'subject-10', 'speed': 1, 'cloud': False, 'max_seconds': None})
    feed_path(session)
    row = session.paths[0]
    assert row['vrms']['available'] is False and row['vrms']['mil_class'] is None
    assert row['fatigue']['available'] and row['emotion']['available']
    assert np.isfinite(row['fatigue']['theta_alpha'])


def test_immediate_cloud_cannot_publish_before_pending(online_cfg, monkeypatch):
    _, session, _, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    started, applied = threading.Event(), threading.Event()
    workers = []
    original_put, original_emit = session._cloud_queue.put_nowait, session._emit

    def enqueue(job):
        original_put(job)
        def immediate():
            started.set()
            session._apply_cloud(job['row'], {'status': 'verified', 'final_class': 'High'})
            applied.set()
        worker = threading.Thread(target=immediate, daemon=True)
        workers.append(worker)
        worker.start()
        assert started.wait(1)

    def emit(kind, payload):
        if kind == 'cloud_pending':
            # The immediate reporter is waiting for this publication lock.
            assert not applied.wait(.03)
        return original_emit(kind, payload)

    monkeypatch.setattr(session._cloud_queue, 'put_nowait', enqueue)
    monkeypatch.setattr(session, '_emit', emit)
    feed_path(session)
    for worker in workers:
        worker.join(1)
    assert applied.is_set()
    kinds = [event['type'] for event in session.events]
    assert kinds.index('path_completed') < kinds.index('cloud_pending') < kinds.index('cloud_verified')
    assert session.cloud['pending'] == 0


def test_processor_cloud_finish_transition_is_atomic(online_cfg, monkeypatch):
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    session._set_state('running')
    session.cloud['pending'] = 1
    session._reader_done.set()
    started, settled = threading.Event(), threading.Event()
    workers = []
    original_set = session._processor_done.set

    def done():
        original_set()
        def settle():
            started.set()
            with session.lock:
                session.cloud['pending'] = 0
                if session.state == 'finalizing':
                    session._set_state('completed')
            settled.set()
        worker = threading.Thread(target=settle, daemon=True)
        workers.append(worker)
        worker.start()
        assert started.wait(1)
        assert not settled.wait(.03)

    monkeypatch.setattr(session._processor_done, 'set', done)
    session._consume()
    for worker in workers:
        worker.join(1)
    assert settled.is_set() and session.state == 'completed'


def test_stopped_state_cannot_be_overwritten_by_late_completion(online_cfg, monkeypatch):
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    session._set_state('running')
    session.control('stop')
    session._set_state('completed')
    session._consume()
    session._cloud_consume()
    assert session.state == 'stopped'


def test_asset_failure_is_visible_without_publishing_a_class(online_cfg, monkeypatch):
    def broken_factory():
        raise ValueError('Frozen model asset hash differs')
    manager = OnlineManager(online_cfg, model_factory=broken_factory)
    assert manager.model() is None
    capabilities = manager.catalog()['capabilities']
    assert capabilities['model_available'] is False
    assert '哈希' in capabilities['model_error']
    class BrokenFold(FakeModel):
        def start_path(self, fold):
            raise ValueError('No frozen held-out model for this subject; no replacement is allowed')
    manager._model, manager._model_error = BrokenFold(), None
    session = OnlineSession(manager, {'recording_id': 'subject-10', 'speed': 1, 'cloud': False})
    feed_path(session)
    assert '对应折 10' in session.paths[0]['vrms']['reason']
    assert '禁止使用其他折' in session.latest['vrms']['reason']
    assert session.paths[0]['vrms']['mil_class'] is None


def test_old_finished_sessions_are_read_from_disk_and_list_is_bounded(online_cfg, monkeypatch):
    online_cfg['online_max_resident_sessions'] = 3
    monkeypatch.setattr(OnlineSession, 'start', lambda self: self._set_state('running'))
    manager = OnlineManager(online_cfg, model_factory=FakeModel)
    request = {'recording_id': 'subject-10', 'speed': 1, 'cloud': False}
    identities = []
    for _ in range(7):
        value = manager.create(request)
        identities.append(value['id'])
        manager.get(value['id']).control('stop')
    assert sum(isinstance(item, OnlineSession) for item in manager.sessions.values()) <= 3
    assert isinstance(manager.sessions[identities[0]], Path)
    assert manager.get(identities[0])['state'] == 'stopped'
    page = manager.list(limit=2, offset=1)
    assert [row['id'] for row in page] == identities[-2:-4:-1]
    assert all(not {'latest', 'paths', 'events'} & row.keys() for row in page)
    restored = OnlineManager(online_cfg)
    assert all(isinstance(item, Path) for item in restored.sessions.values())
    assert restored.get(identities[0])['state'] == 'stopped'


def test_evidence_log_rotates_with_a_fixed_segment_budget(online_cfg, monkeypatch):
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    session._log_limit = 1
    for index in range(8):
        session._emit('window', {'ordinal': index, 'waveform': {'values': [1]}})
    files = sorted(session.directory.glob('evidence*.jsonl'))
    assert len(files) == 3
    for name, ordinal in (('evidence.jsonl', 7), ('evidence.2.jsonl', 5)):
        payload = json.loads((session.directory / name).read_text(encoding='utf-8'))['payload']
        assert payload['ordinal'] == ordinal and 'waveform' not in payload


@pytest.mark.parametrize('action', ['stop', 'fail'])
def test_terminal_cleanup_closes_every_pending_path_once(online_cfg, monkeypatch, action):
    _, session, calls, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    session._set_state('running')
    feed_path(session, ordinal=1)
    feed_path(session, ordinal=2)
    local = [dict(row['vrms']) for row in session.paths]
    if action == 'stop':
        session.control('stop')
    else:
        session._fail('analysis_error:test')
    session._cloud_consume()
    session._apply_cloud(session.paths[0], {'status': 'verified', 'final_class': 'Low'})
    session.control('stop')
    saved = json.loads((session.directory / 'session.json').read_text(encoding='utf-8'))
    assert calls == [] and session._cloud_done.is_set()
    assert session._cloud_queue.unfinished_tasks == 0
    assert saved['cloud']['pending'] == 0 and saved['cloud']['failed'] == 2
    assert all(row['cloud']['generation_status'] == 'cancelled' and row['cloud']['final_class'] is None
               for row in saved['paths'])
    assert [row['vrms'] for row in saved['paths']] == local
    terminal = next(event for event in session.events if event['type'] == 'state' and event['payload']['state'] in TERMINAL)
    closed = [event for event in session.events if event['type'] == 'cloud_fallback']
    assert len(closed) == 2 and max(event['sequence'] for event in closed) < terminal['sequence']
    assert terminal['payload']['cloud']['pending'] == 0
    assert terminal['payload']['error'] == saved['error']


@pytest.mark.parametrize('saved_state', ['finalizing', 'failed', 'stopped', 'interrupted'])
def test_restore_migrates_stale_pending_reports_idempotently(online_cfg, monkeypatch, saved_state):
    _, session, _, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    feed_path(session)
    session.state = saved_state
    session.persist()
    restored = OnlineManager(online_cfg).get(session.id)
    assert restored['state'] == ('interrupted' if saved_state == 'finalizing' else saved_state)
    assert restored['cloud']['pending'] == 0 and restored['cloud']['failed'] == 1
    assert restored['paths'][0]['cloud']['generation_status'] == 'cancelled'
    assert restored['paths'][0]['vrms']['mil_class'] == 'High'
    assert OnlineManager(online_cfg).get(session.id) == restored


def test_sse_window_and_terminal_events_carry_live_metrics_and_error(online_cfg, monkeypatch):
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    feed_path(session)
    last_window = [event for event in session.events if event['type'] == 'window'][-1]
    assert last_window['payload']['metrics']['window_processing_p95_seconds'] is not None
    assert last_window['payload']['metrics']['total_windows'] == 3
    session._fail('analysis_error:test')
    terminal = session.events[-1]
    assert terminal['payload']['error'] == session.snapshot(False)['error']
    assert terminal['payload']['metrics'] == session.metrics


def test_sse_cached_frames_and_empty_poll_never_snapshot(online_cfg, monkeypatch):
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    feed_path(session)
    session._set_state('completed')
    monkeypatch.setattr(session, 'snapshot', lambda *a, **k: pytest.fail('Ordinary SSE poll copied a snapshot'))
    packets, terminal = session.sse_batch(-1)
    assert terminal and len(packets) == len(session.events)
    assert [sequence for sequence, _ in packets] == [event['sequence'] for event in session.events]
    for (sequence, frame), event in zip(packets, session.events):
        assert frame.startswith('id: ' + str(sequence) + '\n')
        assert json.loads(frame.split('data: ', 1)[1]) == event
    assert session.sse_batch(packets[-1][0]) == ([], True)


def test_duplicate_cloud_completion_cannot_republish_or_decrement(online_cfg, monkeypatch):
    _, session, _, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    feed_path(session)
    result = {'status': 'verified', 'final_class': 'Low'}
    session._apply_cloud(session.paths[0], result)
    before = session.snapshot()
    session._apply_cloud(session.paths[0], result)
    assert session.snapshot() == before


def test_replay_limit_cannot_be_reported_as_source_eof(online_cfg):
    manager = OnlineManager(online_cfg, model_factory=FakeModel)
    value = manager.create({'recording_id': 'subject-10', 'speed': 1000, 'cloud': False, 'max_seconds': 20})
    result = wait_finished(manager.get(value['id']))
    assert result['state'] == 'completed' and result['position_seconds'] == 20
    assert result['metrics']['reader_complete'] is True and result['metrics']['source_eof_reached'] is False
    assert result['metrics']['read_samples'] == 20 * 1024 < result['metrics']['source_samples']
    assert result['metrics']['produced_windows'] == result['metrics']['total_windows'] == 2


def test_incremental_band_means_match_every_original_prefix_and_ignore_rejected_windows(online_cfg, monkeypatch):
    from eeg_agent.online import runtime
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    history, rng = [], np.random.default_rng(73)
    bands = ('delta', 'theta', 'alpha', 'beta', 'gamma')
    session._process({'type': 'rest_start', 'scope': 'rest', 'ordinal': 1, 'start_seconds': 6.})
    for index in range(40):
        valid = index % 5 != 0
        powers = {key: rng.uniform(.01, 100, 30) for key in bands}
        monkeypatch.setattr(runtime, '_spectral', lambda window: (None, None, powers))
        if valid:
            history.append(powers)
        session._window({'scope': 'rest', 'ordinal': 1, 'start_seconds': 6 + index * 5,
                         'end_seconds': 11 + index * 5, 'observed_end_sample': (11 + index * 5) * 1024,
                         'valid': valid, 'reason': '' if valid else 'quality_rejected',
                         'window': np.zeros((30, 1280), np.float32), '_ready_wall': time.monotonic()})
        for actual, expected in zip((session.latest['fatigue'], session.latest['emotion']), runtime._indicators(history)):
            assert actual.keys() == expected.keys()
            for key in actual:
                assert actual[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-12)
    assert len(session._band_sums) == 5 and session._accepted == 32


def test_inflight_cancelled_result_keeps_usage_without_reopening_counts_or_queued_jobs(online_cfg, monkeypatch):
    manager, session, calls, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    session._set_state('running')
    feed_path(session, ordinal=1)
    feed_path(session, ordinal=2)

    class CancelCloud(FakeCloud):
        def report(self, row, prediction, cancelled=None):
            calls.append(row['ordinal'])
            session.control('stop')
            assert cancelled()
            return {'status': 'fallback', 'generation_status': 'cancelled', 'final_class': None,
                    'reports': {'vrms': {'available': True, 'final_class': 'Low'}}, 'usage': {'calls': 1}}

    manager._cloud_factory = lambda emit: CancelCloud(calls)
    session._cloud_consume()
    assert calls == [1] and session._cloud_queue.unfinished_tasks == 0
    assert session.state == 'stopped' and session.cloud['pending'] == 0 and session.cloud['failed'] == 2
    assert session.paths[0]['cloud']['usage'] == {'calls': 1}
    assert session.paths[0]['cloud']['final_class'] is None
    assert not any(event['type'] == 'cloud_verified' for event in session.events)
    assert session.events[-1]['type'] == 'state'


def test_model_loading_keeps_controls_available_and_cannot_publish_after_stop(online_cfg, monkeypatch):
    manager, session, _, model = manual_session(online_cfg, monkeypatch)
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    loads = []

    def factory():
        loads.append(1)
        entered.set()
        assert release.wait(3)
        return model

    manager._model_factory = factory
    session._set_state('running')
    event = {'type': 'path_start', 'scope': 'task', 'ordinal': 1, 'start_seconds': 6.}
    loader = threading.Thread(target=session._process, args=(event,), daemon=True)
    loader.start()
    assert entered.wait(1)

    def stop():
        manager.get(session.id).control('stop')
        stopped.set()

    controller = threading.Thread(target=stop, daemon=True)
    controller.start()
    try:
        assert stopped.wait(.5), 'Model loading blocked the session control lock'
        terminal_sequence = session.sequence - 1
    finally:
        release.set()
        loader.join(2)
        controller.join(2)
    assert not loader.is_alive() and not controller.is_alive()
    assert session.sequence - 1 == terminal_sequence
    assert manager.model() is model and len(loads) == 1
    assert model.folds == []


def test_fold_loading_cannot_publish_a_warning_or_start_after_stop(online_cfg, monkeypatch):
    manager, session, _, _ = manual_session(online_cfg, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    class SlowFold(FakeModel):
        def start_path(self, fold):
            entered.set()
            assert release.wait(2)
            raise ValueError('No frozen held-out model for this subject')

    manager._model_factory = SlowFold
    session._set_state('running')
    worker = threading.Thread(target=session._process, args=({'type': 'path_start', 'scope': 'task',
                               'ordinal': 1, 'start_seconds': 6.},), daemon=True)
    worker.start()
    try:
        assert entered.wait(1)
        session.control('stop')
        sequence = session.sequence
    finally:
        release.set()
        worker.join(2)
    assert session.sequence == sequence
    assert not worker.is_alive()


def test_pause_during_enqueue_is_counted_even_when_reader_never_observes_paused_flag(online_cfg, monkeypatch):
    from eeg_agent.online import runtime
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    session.request.update(speed=10, max_seconds=1)
    session._set_state('running')
    clock, fed = [0.], []
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(session.stop_event, 'wait', lambda duration: clock.__setitem__(0, clock[0] + duration) or False)

    class Processor:
        def __init__(self, channels):
            pass
        def feed(self, block, start):
            fed.append(clock[0])
            return [{'type': 'warning', 'observed_end_sample': start + len(block)}]
        def finish(self):
            return []

    monkeypatch.setattr(runtime, 'StreamingProcessor', Processor)
    original = session._enqueue
    def enqueue(event):
        if len(fed) == 1:
            session.control('pause')
            clock[0] += 2
            session.control('resume')
        return original(event)
    monkeypatch.setattr(session, '_enqueue', enqueue)
    session._read()
    assert fed == pytest.approx([.025, 2.05, 2.075, 2.1])
    assert session._queue.get_nowait()['_source_due_wall'] == pytest.approx(.025)


def test_pause_acknowledgement_waits_for_inprogress_input_block_commit(online_cfg, monkeypatch):
    from eeg_agent.online import runtime
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    session.request.update(speed=1000, max_seconds=1)
    session._set_state('running')
    entered, release, paused = threading.Event(), threading.Event(), threading.Event()

    class Processor:
        def __init__(self, channels):
            pass
        def feed(self, block, start):
            entered.set()
            assert release.wait(2)
            return []
        def finish(self):
            return []

    monkeypatch.setattr(runtime, 'StreamingProcessor', Processor)
    reader = threading.Thread(target=session._read, daemon=True)
    reader.start()
    assert entered.wait(1)
    result = []
    def pause():
        result.append(session.control('pause'))
        paused.set()
    controller = threading.Thread(target=pause, daemon=True)
    controller.start()
    try:
        assert not paused.wait(.05), 'Pause acknowledged before the arrived block committed'
        release.set()
        assert paused.wait(1)
        position = result[0]['position_seconds']
        time.sleep(.03)
        assert position >= .25 and session.position_seconds == position
    finally:
        release.set()
        session.control('stop')
        reader.join(2)
        controller.join(2)
    assert not reader.is_alive()


def test_stopped_session_reserves_cloud_settings_until_inflight_request_returns(online_cfg, monkeypatch):
    manager, session, calls, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    entered, release = threading.Event(), threading.Event()
    class SlowCloud(FakeCloud):
        def report(self, row, prediction, cancelled=None):
            entered.set()
            assert release.wait(2)
            return super().report(row, prediction, cancelled)
    manager._cloud_factory = lambda emit: SlowCloud(calls)
    session._set_state('running')
    feed_path(session)
    worker = threading.Thread(target=session._cloud_consume, daemon=True)
    worker.start()
    try:
        assert entered.wait(1)
        session.control('stop')
        assert manager.cloud_in_use(), 'Synchronous request is still in flight'
        assert session.snapshot(False)['cloud']['in_flight'] is True
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and not manager.cloud_in_use()
    saved = json.loads((session.directory / 'session.json').read_text(encoding='utf-8'))
    assert saved['cloud']['in_flight'] is False and saved['cloud']['pending'] == 0
    assert saved['paths'][0]['cloud']['final_class'] is None


def test_cloud_publication_failure_settles_finalizing_session_without_losing_local_results(online_cfg, monkeypatch):
    _, session, _, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    feed_path(session, ordinal=1)
    feed_path(session, ordinal=2)
    local = [dict(row['vrms']) for row in session.paths]
    session._processor_done.set()
    session._set_state('finalizing')
    def broken_publication(*unused):
        raise OSError('controlled persistence failure')
    monkeypatch.setattr(session, '_apply_cloud', broken_publication)
    session._cloud_consume()
    assert session.state == 'failed' and session.error == 'cloud_worker_error:OSError'
    assert session.cloud['pending'] == 0 and session._cloud_queue.unfinished_tasks == 0
    assert session._cloud_done.is_set()
    assert [row['vrms'] for row in session.paths] == local


def test_failure_still_sets_terminal_state_when_cancel_logging_fails(online_cfg, monkeypatch):
    _, session, _, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    feed_path(session)
    emit = session._emit
    def broken_log(kind, payload):
        if kind == 'cloud_fallback':
            raise OSError('controlled event log failure')
        return emit(kind, payload)
    monkeypatch.setattr(session, '_emit', broken_log)
    session._fail('analysis_error:test')
    assert session.state == 'failed' and session.stop_event.is_set()
    assert session.cloud['pending'] == 0 and session.cloud['failed'] == 1
    assert session.snapshot(False)['metrics']['persistence_error'] == 'OSError'


def test_graceful_shutdown_marks_active_session_interrupted_and_preserves_manual_stop(online_cfg, monkeypatch):
    manager, session, _, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    feed_path(session)
    session._set_state('paused')
    manager.shutdown()
    assert session.state == 'interrupted' and session.stop_event.is_set()
    saved = json.loads((session.directory / 'session.json').read_text(encoding='utf-8'))
    assert saved['state'] == 'interrupted' and '预热' in saved['error']
    assert saved['cloud']['pending'] == 0 and saved['paths'][0]['vrms']['mil_class'] == 'High'
    sequence = session.sequence
    manager.shutdown()
    assert session.sequence == sequence
    restored = OnlineManager(online_cfg)
    assert restored.get(session.id)['state'] == 'interrupted'
    _, manually_stopped, _, _ = manual_session(online_cfg, monkeypatch)
    manually_stopped.control('stop')
    manually_stopped.manager.shutdown()
    assert manually_stopped.state == 'stopped'


def test_huge_finite_replay_limit_is_clamped_before_sample_conversion(online_cfg):
    manager = OnlineManager(online_cfg, model_factory=FakeModel)
    created = manager.create({'recording_id': 'subject-10', 'speed': 1000, 'cloud': False, 'max_seconds': 1e308})
    result = wait_finished(manager.get(created['id']))
    assert result['state'] == 'completed' and result['metrics']['source_eof_reached'] is True


def test_pause_before_reader_initialization_does_not_double_count_previous_time(online_cfg, monkeypatch):
    from eeg_agent.online import runtime
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    session.request.update(speed=10, max_seconds=.25)
    session._set_state('running')
    clock, fed = [0.], []
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: clock[0])
    session.control('pause')
    clock[0] = 2.
    def wait(duration):
        clock[0] += duration
        if session.pause_event.is_set() and clock[0] >= 2.1 - 1e-9:
            session.control('resume')
        return False
    monkeypatch.setattr(session.stop_event, 'wait', wait)
    class Processor:
        def __init__(self, channels):
            pass
        def feed(self, block, start):
            fed.append(clock[0])
            return []
        def finish(self):
            return []
    monkeypatch.setattr(runtime, 'StreamingProcessor', Processor)
    session._read()
    assert fed == pytest.approx([2.125])


def test_accelerated_enqueue_uses_success_time_and_never_enqueues_after_stop(online_cfg, monkeypatch):
    from eeg_agent.online import runtime
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    session.request['speed'] = 10
    for _ in range(8):
        session._queue.put_nowait({'type': 'warning'})
    clock = [0.]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: clock[0])
    def wait(duration):
        clock[0] += 2
        session._queue.get_nowait()
        session._queue.task_done()
        return False
    monkeypatch.setattr(session.stop_event, 'wait', wait)
    event = {'type': 'warning'}
    assert session._enqueue(event) and event['_ready_wall'] == 2
    session.control('stop')
    before = session._queue.qsize()
    assert session._enqueue({'type': 'warning'}) is False and session._queue.qsize() == before


@pytest.mark.parametrize('state', ['running', 'stopped'])
def test_restart_resets_inflight_metadata_in_active_and_terminal_snapshots(online_cfg, monkeypatch, state):
    _, session, _, _ = manual_session(online_cfg, monkeypatch, cloud=True, configured=True)
    session.state = state
    session.cloud['in_flight'] = True
    session.persist()
    restored = OnlineManager(online_cfg)
    saved = restored.get(session.id)
    assert saved['cloud']['in_flight'] is False and not restored.cloud_in_use()
    assert saved['state'] == ('interrupted' if state == 'running' else state)


@pytest.mark.parametrize('worker', ['_consume', '_cloud_consume'])
def test_completion_snapshot_write_failure_cannot_leave_a_successful_or_live_session(online_cfg, monkeypatch, worker):
    from eeg_agent.online import runtime
    _, session, _, _ = manual_session(online_cfg, monkeypatch)
    feed_path(session)
    session._reader_done.set()
    if worker == '_cloud_consume':
        session._processor_done.set()
    session._set_state('finalizing' if worker == '_cloud_consume' else 'running')
    def disk_failure(*unused):
        raise OSError('controlled permanently unavailable snapshot storage')
    monkeypatch.setattr(runtime, 'atomic_json', disk_failure)
    getattr(session, worker)()
    assert session.state == 'failed' and session.stop_event.is_set()
    assert session.metrics['persistence_error'] == 'OSError'
    assert session.error == ('cloud_worker_error:OSError' if worker == '_cloud_consume' else 'analysis_error:OSError')
    assert session.paths[0]['vrms']['mil_class'] == 'High'
    packets, terminal = session.sse_batch(-1)
    assert terminal and '"state": "completed"' not in ''.join(frame for _, frame in packets)
