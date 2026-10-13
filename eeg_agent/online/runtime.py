"""Bounded replay sessions, independent of the existing offline repository."""
from collections import deque
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import queue
import re
import threading
import time
import uuid

import numpy as np
import psutil

from ..cloud_config import public_settings
from ..config import EEG_CHANNELS, PROJECT
from ..data import metadata
from ..tools import _spectral
from .cloud import timeout_seconds
from .stream import StreamingProcessor

PIPELINE = 'raw_eeg_stream_minimal_cpu_seed2026_v1'
TERMINAL = {'completed', 'failed', 'stopped', 'interrupted'}
ACTIVE = {'created', 'running', 'paused', 'finalizing'}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(str(temporary), str(path))


def _cancel_pending(paths, cloud, reason):
    """Settle pending paths once, including snapshots from an older runtime."""
    closed = []
    for row in paths:
        if row.get('cloud', {}).get('status') != 'pending':
            continue
        row['cloud'].update(status='fallback', generation_status='cancelled',
                            explanation_status='cancelled', final_class=None, reason=reason)
        closed.append(row)
    cloud['pending'] = 0
    cloud['failed'] = cloud.get('failed', 0) + len(closed)
    cloud['cancelled'] = cloud.get('cancelled', 0) + len(closed)
    return closed


def _indicators(powers):
    if not powers:
        return {'available': False}, {'available': False}
    bands = {key: np.mean([row[key] for row in powers], axis=0) for key in powers[0]}
    return _band_indicators(bands)


def _band_indicators(bands):
    if not bands:
        return {'available': False}, {'available': False}
    def ratio(a, b):
        return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 1e-12 else None
    frontal = [EEG_CHANNELS.index(name) for name in ('F3', 'Fz', 'F4')]
    parietal = [EEG_CHANNELS.index(name) for name in ('P3', 'Pz', 'P4')]
    alpha3, alpha4 = (bands['alpha'][EEG_CHANNELS.index(name)] for name in ('F3', 'F4'))
    fatigue = {'available': True, 'theta_alpha': ratio(bands['theta'].mean(), bands['alpha'].mean()),
               'theta_beta': ratio(bands['theta'].mean(), bands['beta'].mean()),
               'beta_alpha': ratio(bands['beta'].mean(), bands['alpha'].mean())}
    emotion = {'available': True, 'workload_index': ratio(bands['theta'][frontal].mean(), bands['alpha'][parietal].mean()),
               'faa': float(np.log(alpha4) - np.log(alpha3)) if alpha3 > 1e-12 and alpha4 > 1e-12 else None}
    return fatigue, emotion


def _public_vrms(result, status='provisional'):
    numeric = result.get('numeric_fusion') or {}
    return {'available': bool(result.get('available')), 'status': status,
            'probability_high': result.get('mil_raw_high_probability'),
            'mil_class': result.get('mil_class'), 'numeric_class': numeric.get('predicted_class'),
            'numeric_score': numeric.get('raw_combined_score'), 'calibrated': False,
            'reason': result.get('reason'), 'backend': 'mamba_minimal_cpu_fp32',
            'assessment_scope': 'path_prefix' if status == 'provisional' else 'path_end',
            'provenance': {key: value for key, value in result.items() if key.endswith('_sha256')}}


def _asset_error(exc, fold=None):
    # Do not expose local paths or deserialization details in public responses.
    message = str(exc).lower()
    reason = ('缺少该被试冻结折，禁止使用其他折' if 'no frozen held-out' in message or 'unavailable for this held-out' in message
              else '冻结资产哈希校验失败' if 'hash differs' in message
              else '冻结资产文件缺失' if isinstance(exc, FileNotFoundError)
              else '冻结资产格式或隔离校验失败')
    return ('对应折 ' + str(fold) + '：' if fold is not None else 'CPU模型未加载：') + reason + ' (' + type(exc).__name__ + ')'


class OnlineSession:
    def __init__(self, manager, request):
        self.manager, self.request = manager, dict(request)
        from ..default_rag import resolve
        self.rag_policy = copy.deepcopy(resolve(request.get('rag_mode', 'default')))
        self.request['rag_policy'] = copy.deepcopy(self.rag_policy)
        self.id = uuid.uuid4().hex
        self.directory = manager.root / self.id
        self.directory.mkdir(parents=True)
        self.recording = manager.recording(request['recording_id'])
        self.state, self.error = 'created', None
        self.lock, self._persist_lock = threading.RLock(), threading.Lock()
        self.stop_event, self.pause_event = threading.Event(), threading.Event()
        self._pause_started, self._paused_seconds = None, 0.
        self._cloud_inflight = threading.Event()
        self.events, self.sequence = deque(maxlen=256), 0
        self._sse_packets = deque(maxlen=256)
        self.paths, self.latest, self.position_seconds = [], None, 0.
        self._queue = queue.Queue(maxsize=8)
        self._cloud_queue = queue.Queue(maxsize=4)
        self._reader_done, self._processor_done = threading.Event(), threading.Event()
        self._cloud_done = threading.Event()
        self._segment, self._path_model, self._band_sums = None, None, {}
        self._path_model_error = None
        self._total, self._accepted, self._latencies = 0, 0, deque(maxlen=10000)
        self._wall_started = time.monotonic()
        self._process_monitor = psutil.Process()
        self._process_monitor.cpu_percent()
        configured = manager.cloud_configured()
        self.cloud = {'requested': bool(request['cloud']), 'configured': configured,
                      'pending': 0, 'completed': 0, 'failed': 0, 'cancelled': 0,
                      'in_flight': False,
                      'status': 'ready' if configured else 'not_configured'}
        self.metrics = {'total_windows': 0, 'accepted_windows': 0, 'queue_depth': 0,
                        'lag_seconds': 0., 'window_processing_p95_seconds': None,
                        'produced_windows': 0, 'read_samples': 0, 'source_samples': self.recording['sample_count'],
                        'reader_complete': False, 'source_eof_reached': False}
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._log_limit = int(manager.cfg.get('online_max_event_log_mb', 64)) * 1024**2
        self._emit('state', {'state': self.state})
        self.persist()

    def _emit(self, kind, payload):
        with self.lock:
            if kind in ('state', 'progress', 'window', 'cloud_pending', 'cloud_verified', 'cloud_fallback'):
                payload = dict(payload, metrics=dict(self.metrics), cloud=dict(self.cloud),
                               error=self.error, position_seconds=self.position_seconds)
            event = {'id': self.id, 'sequence': self.sequence, 'type': kind, 'payload': plain(payload)}
            self.sequence += 1
            self.events.append(event)
            encoded = json.dumps(event, ensure_ascii=False, allow_nan=False)
            self._sse_packets.append((event['sequence'], 'id: ' + str(event['sequence']) + '\ndata: ' + encoded + '\n\n'))
            log = self.directory / 'events.jsonl'
            if log.exists() and log.stat().st_size >= self._log_limit:
                os.replace(str(log), str(self.directory / 'events.previous.jsonl'))
            with log.open('a', encoding='utf-8') as handle:
                handle.write(encoded + '\n')
            if kind in ('window', 'path_completed', 'rest_completed', 'cloud_verified', 'cloud_fallback'):
                evidence = copy.deepcopy(event)
                evidence['payload'].pop('waveform', None)
                evidence_log = self.directory / 'evidence.jsonl'
                if evidence_log.exists() and evidence_log.stat().st_size >= self._log_limit:
                    previous = self.directory / 'evidence.1.jsonl'
                    if previous.exists():
                        os.replace(str(previous), str(self.directory / 'evidence.2.jsonl'))
                    os.replace(str(evidence_log), str(previous))
                with evidence_log.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(evidence, ensure_ascii=False, allow_nan=False) + '\n')
        return event

    def _set_state(self, state):
        with self.lock:
            if self.state in TERMINAL and self.state != state:
                return
            previous, state_sequence = self.state, self.sequence
            self.state = state
            self.metrics.update(elapsed_seconds=time.monotonic() - self._wall_started,
                                queue_depth=self._queue.qsize())
            try:
                self._emit('state', {'state': state, 'position_seconds': self.position_seconds})
                self.persist()
            except OSError as exc:
                self.metrics['persistence_error'] = type(exc).__name__
                if state == 'completed':
                    # Do not send a successful terminal SSE frame when the
                    # completion snapshot could not be saved. Keep sequence
                    # numbers monotone even if the event log already wrote it.
                    self.state = previous
                    if self.events and self.events[-1]['sequence'] == state_sequence:
                        self.events.pop()
                        self._sse_packets.pop()
                raise

    def snapshot(self, include_events=True):
        with self.lock:
            result = {'id': self.id, 'state': self.state, 'recording': self.recording,
                      'speed': self.request['speed'], 'position_seconds': self.position_seconds,
                      'latest': copy.deepcopy(self.latest), 'paths': copy.deepcopy(self.paths),
                      'metrics': dict(self.metrics), 'cloud': dict(self.cloud), 'error': self.error,
                      'created_at': self.created_at, 'pipeline_id': PIPELINE,
                      'rag_mode': self.request.get('rag_mode', 'default'),
                      'rag_policy': copy.deepcopy(self.rag_policy),
                      'last_sequence': self.sequence - 1, 'request': dict(self.request)}
            if include_events:
                result['events'] = list(self.events)
            return plain(result)

    def sse_batch(self, after):
        """Reuse serialized events; only an expired cursor needs a snapshot."""
        with self.lock:
            if self._sse_packets and after < self._sse_packets[0][0] - 1:
                sequence = self.sequence - 1
                event = {'id': self.id, 'sequence': sequence, 'type': 'snapshot', 'payload': self.snapshot(False)}
                frame = 'id: ' + str(sequence) + '\ndata: ' + json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n\n'
                return [(sequence, frame)], self.state in TERMINAL
            return [(sequence, frame) for sequence, frame in self._sse_packets if sequence > after], self.state in TERMINAL

    def persist(self):
        # Keep lock order consistent when an atomic transition also persists.
        with self.lock:
            with self._persist_lock:
                try:
                    atomic_json(self.directory / 'session.json', self.snapshot(False))
                except OSError as exc:
                    self.metrics['persistence_error'] = type(exc).__name__
                    raise

    def start(self):
        self._set_state('running')
        self.worker = threading.Thread(target=self._consume, name='online-model-' + self.id[:6], daemon=True)
        self.cloud_worker = threading.Thread(target=self._cloud_consume, name='online-cloud-' + self.id[:6], daemon=True)
        self.reader = threading.Thread(target=self._read, name='online-replay-' + self.id[:6], daemon=True)
        self.worker.start()
        self.cloud_worker.start()
        self.reader.start()

    def control(self, action):
        with self.lock:
            if action == 'stop':
                self._terminate('stopped', '任务已停止，未完成的云端报告已取消')
            elif action == 'pause' and self.state == 'running':
                self._pause_started = time.monotonic()
                self.pause_event.set()
                self._set_state('paused')
            elif action == 'resume' and self.state == 'paused':
                self._paused_seconds += time.monotonic() - self._pause_started
                self._pause_started = None
                self.pause_event.clear()
                self._set_state('running')
            else:
                raise ValueError('当前任务状态不支持该操作')
        return self.snapshot()

    def _cancel_cloud(self, reason):
        with self.lock:
            closed = _cancel_pending(self.paths, self.cloud, reason)
            while True:
                try:
                    self._cloud_queue.get_nowait()
                except queue.Empty:
                    break
                self._cloud_queue.task_done()
            for row in closed:
                self._emit('cloud_fallback', {'ordinal': row['ordinal'], 'scope': 'path_end',
                           'data_cutoff_seconds': row['data_cutoff_seconds'], 'result': row['cloud']})

    def _fail(self, code):
        self._terminate('failed', '在线分析失败，未完成的云端报告已取消', code)

    def interrupt(self):
        self._terminate('interrupted', '服务正在关闭，未完成的云端报告已取消',
                        '服务已关闭；新建回放并重新预热')

    def _terminate(self, state, reason, error=None):
        with self.lock:
            self.stop_event.set()
            self.pause_event.clear()
            self._pause_started = None
            if self.state in TERMINAL:
                return
            if error is not None:
                self.error = error
            try:
                self._cancel_cloud(reason)
            except OSError as exc:
                self.metrics['persistence_error'] = type(exc).__name__
            finally:
                # A failed event log must not leave a live/finalizing task.
                try:
                    self._set_state(state)
                except OSError as exc:
                    self.metrics['persistence_error'] = type(exc).__name__
                    try:
                        self.persist()
                    except OSError:
                        pass  # The in-memory snapshot explicitly reports the disk failure.

    def _enqueue(self, event):
        while True:
            with self.lock:
                if self.stop_event.is_set():
                    return False
                try:
                    event['_ready_wall'] = time.monotonic()
                    self._queue.put_nowait(event)
                    return True
                except queue.Full:
                    if self.request['speed'] == 1:
                        self._fail('overrun: 在线计算队列已满，回放已停止；未静默丢弃窗口')
                        return False
            # Accelerated validation waits outside the control lock. Timestamp
            # queue arrival only when capacity is actually available.
            self.stop_event.wait(.05)

    def _read(self):
        try:
            path = self.manager.source_path(self.request['recording_id'])
            meta = metadata(path)
            raw = np.memmap(path, mode='r', dtype='<f4', shape=(meta['actual_samples'], meta['n_channels']))
            processor = StreamingProcessor(meta['channels'])
            limit = len(raw)
            if self.request.get('max_seconds') is not None:
                limit = int(min(self.request['max_seconds'], limit / 1024) * 1024)
            with self.lock:
                clock_origin = time.monotonic()
                pause_baseline = self._paused_seconds
                if self._pause_started is not None:
                    pause_baseline += clock_origin - self._pause_started
            block_origin = clock_origin
            for start in range(0, limit, 256):
                end = min(start + 256, limit)
                events = None
                while not self.stop_event.is_set():
                    with self.lock:
                        if self.stop_event.is_set():
                            break
                        if self.pause_event.is_set():
                            remaining = .05
                        else:
                            block_origin = clock_origin + self._paused_seconds - pause_baseline
                            remaining = block_origin + end / (1024 * self.request['speed']) - time.monotonic()
                            if remaining <= 0:
                                # Pause acknowledgement waits for this small
                                # arrived block and its position to commit.
                                events = processor.feed(np.asarray(raw[start:end]), start)
                                if self.stop_event.is_set():
                                    events = None
                                    break
                                self.position_seconds = end / 1024
                                self.metrics.update(queue_depth=self._queue.qsize(), read_samples=end,
                                    produced_windows=self.metrics['produced_windows'] + sum(event['type'] == 'window' for event in events))
                                break
                    self.stop_event.wait(min(.05, remaining))
                if events is None:
                    break
                for event in events:
                    event['_source_due_wall'] = block_origin + event.get('observed_end_sample', end) / (1024 * self.request['speed'])
                    if not self._enqueue(event):
                        break
                with self.lock:
                    if not self.stop_event.is_set():
                        self.metrics['queue_depth'] = self._queue.qsize()
                        self._emit('progress', {'position_seconds': self.position_seconds,
                                               'queue_depth': self._queue.qsize()})
            if not self.stop_event.is_set():
                with self.lock:
                    if self.stop_event.is_set():
                        return
                    events = processor.finish()
                    self.metrics['produced_windows'] += sum(event['type'] == 'window' for event in events)
                for event in events:
                    event['_source_due_wall'] = block_origin + event.get('observed_end_sample', limit) / (1024 * self.request['speed'])
                    if not self._enqueue(event):
                        break
                with self.lock:
                    if not self.stop_event.is_set():
                        self.metrics.update(reader_complete=True, source_eof_reached=limit == len(raw))
            del raw
        except Exception as exc:
            self._fail('replay_error:' + type(exc).__name__)
        finally:
            self._reader_done.set()

    def _consume(self):
        try:
            while not self.stop_event.is_set():
                try:
                    event = self._queue.get(timeout=.1)
                except queue.Empty:
                    if self._reader_done.is_set():
                        break
                    continue
                self._process(event)
                self._queue.task_done()
        except Exception as exc:
            self._fail('analysis_error:' + type(exc).__name__)
        finally:
            self._path_model, self._band_sums, self._segment = None, {}, None
            with self.lock:
                self._processor_done.set()
                if self.state not in TERMINAL:
                    try:
                        self._set_state('finalizing' if self.cloud['pending'] else 'completed')
                    except Exception as exc:
                        self._fail('analysis_error:' + type(exc).__name__)
            try:
                self.persist()
            except OSError:
                pass  # The terminal snapshot still exposes persistence_error.

    def _process(self, event):
        if self.stop_event.is_set():
            return
        kind = event['type']
        if kind in ('path_start', 'rest_start'):
            self._segment, self._band_sums = dict(event), {}
            self._total, self._accepted, self._path_model = 0, 0, None
            self._path_model_error = None
            if event['scope'] == 'task':
                model = self.manager.model()
                if self.stop_event.is_set():
                    return
                if model is not None:
                    try:
                        self._path_model = model.start_path(self.recording['acquisition'])
                    except (KeyError, FileNotFoundError, ValueError) as exc:
                        self._path_model = None
                        self._path_model_error = _asset_error(exc, self.recording['acquisition'])
            with self.lock:
                if self.stop_event.is_set():
                    self._path_model = None
                    return
                if self._path_model_error:
                    self._emit('warning', {'code': 'fold_asset_unavailable',
                                          'fold': self.recording['acquisition'],
                                          'reason': self._path_model_error})
                self._emit(kind, {k: v for k, v in event.items() if not k.startswith('_')})
        elif kind == 'window':
            self._window(event)
        elif kind in ('path_end', 'rest_end'):
            self._end(event)
        elif kind == 'warning':
            with self.lock:
                if not self.stop_event.is_set():
                    self._emit('warning', {k: v for k, v in event.items() if not k.startswith('_')})

    def _window(self, event):
        begin = time.monotonic()
        self._total += 1
        prediction = {'available': False, 'reason': self._path_model_error or self.manager._model_error or '未配置对应冻结模型或窗口不足'}
        if event['valid']:
            self._accepted += 1
            if self._accepted > self.manager.max_path_windows:
                raise ValueError('Online segment exceeds bounded window budget')
            frequencies, psd, powers = _spectral(event['window'])
            for key, values in powers.items():
                if key not in self._band_sums:
                    self._band_sums[key] = np.array(values, dtype=np.float64, copy=True)
                else:
                    self._band_sums[key] += values
            if event['scope'] == 'task' and self._path_model is not None:
                prediction = (self._path_model.add_shared(event['window'], frequencies, psd)
                              if hasattr(self._path_model, 'add_shared') else self._path_model.add(event['window']))
        fatigue, emotion = self._indicators()
        sampled = event['window'][:, ::4]
        payload = {key: event[key] for key in ('scope', 'ordinal', 'start_seconds', 'end_seconds', 'valid')}
        payload.update(coverage=self._accepted / self._total, accepted_windows=self._accepted,
                       total_windows=self._total, qc={'valid': event['valid'], 'reason': event['reason']},
                       vrms=_public_vrms(prediction), fatigue=fatigue, emotion=emotion,
                       waveform={'channels': EEG_CHANNELS, 'sampling_hz': 64, 'values': plain(sampled),
                                 'start_seconds': event['start_seconds'], 'end_seconds': event['end_seconds']},
                       data_cutoff_seconds=event['observed_end_sample'] / 1024,
                       completed_at=datetime.now(timezone.utc).isoformat())
        if self.stop_event.is_set():
            return
        elapsed = time.monotonic() - begin
        payload['processing_seconds'] = elapsed
        payload['queue_processing_lag_seconds'] = time.monotonic() - event['_ready_wall']
        payload['completion_lag_seconds'] = max(0., time.monotonic() - event.get('_source_due_wall', event['_ready_wall']))
        self._latencies.append(elapsed)
        process = self._process_monitor
        with self.lock:
            if self.stop_event.is_set():
                return
            self.latest = plain(payload)
            self.metrics.update(total_windows=self.metrics['total_windows'] + 1,
                                accepted_windows=self.metrics['accepted_windows'] + int(event['valid']),
                                window_processing_p95_seconds=float(np.percentile(self._latencies, 95)),
                                window_processing_p50_seconds=float(np.median(self._latencies)),
                                window_processing_max_seconds=float(max(self._latencies)),
                                lag_seconds=payload['completion_lag_seconds'],
                                last_processing_seconds=elapsed, last_completion_lag_seconds=payload['completion_lag_seconds'],
                                replay_mode='realtime' if self.request['speed'] == 1 else 'accelerated_validation',
                                queue_depth=self._queue.qsize(),
                                rss_mb=process.memory_info().rss / 1024**2,
                                cpu_percent=process.cpu_percent(), system_used_mb=psutil.virtual_memory().used / 1024**2,
                                elapsed_seconds=time.monotonic() - self._wall_started,
                                measurement_host=platform.machine())
            self._emit('window', payload)

    def _indicators(self):
        return _band_indicators({key: values / self._accepted for key, values in self._band_sums.items()})

    def _end(self, event):
        prediction = self._path_model.result() if self._path_model is not None else {
            'available': False, 'reason': self._path_model_error or self.manager._model_error}
        coverage = self._accepted / max(1, self._total)
        permitted = bool(event['complete'] and self._accepted >= 3 and coverage >= .8 and prediction.get('available'))
        fatigue, emotion = self._indicators()
        row = {key: event[key] for key in ('scope', 'ordinal', 'start_seconds', 'end_seconds', 'complete')}
        row.update(assessment_scope='path_end' if event['scope'] == 'task' else 'rest_end',
                   accepted_windows=self._accepted, total_windows=self._total, coverage=coverage,
                   reason=event.get('reason', ''), vrms=_public_vrms(prediction, 'path_end'),
                   fatigue=fatigue, emotion=emotion, data_cutoff_seconds=event['observed_end_sample'] / 1024,
                   cloud={'status': 'not_requested', 'final_class': None})
        if not permitted:
            row['vrms'].update(available=False, probability_high=None, mil_class=None, numeric_class=None,
                               reason=event.get('reason') or prediction.get('reason') or '完整性、窗口数量、覆盖率或模型未满足发布要求')
        else:
            row['vrms']['reason'] = None
        if event['scope'] == 'task' and event['complete'] and self.cloud['requested']:
            if self.cloud['configured']:
                row['cloud']['status'] = 'pending'
            else:
                row['cloud'].update(status='fallback', reason='未配置云端接口，保留本地结果')
        with self.lock:
            if self.stop_event.is_set():
                return
            self.paths.append(plain(row))
            self._emit('path_completed' if event['scope'] == 'task' else 'rest_completed', row)
            if row['cloud']['status'] == 'pending':
                job = {'row': copy.deepcopy(row), 'prediction': copy.deepcopy(prediction) if permitted else None}
                try:
                    self.cloud['pending'] += 1
                    try:
                        self._cloud_queue.put_nowait(job)
                    except queue.Full:
                        self.cloud['pending'] -= 1
                        raise
                    self._emit('cloud_pending', {'ordinal': row['ordinal'], 'scope': 'path_end',
                                                 'data_cutoff_seconds': row['data_cutoff_seconds']})
                except queue.Full:
                    self._apply_cloud(row, {'status': 'fallback', 'reason': '云端队列已满，保留本地结果', 'final_class': None}, decrement=False)
        self._path_model, self._band_sums, self._segment = None, {}, None
        self.persist()

    def _apply_cloud(self, row, result, decrement=True):
        with self.lock:
            target = next(item for item in self.paths if item['scope'] == row['scope'] and item['ordinal'] == row['ordinal'])
            if self.stop_event.is_set():
                if result.get('generation_status') == 'cancelled' and target['cloud'].get('generation_status') == 'cancelled':
                    # A request already in flight may finish later; retain its
                    # audit/usage without reopening the settled path or counts.
                    for key in ('reports', 'usage', 'session_cumulative_usage', 'provider_model'):
                        if key in result:
                            target['cloud'][key] = plain(result[key])
                    self.persist()
                return
            if target['cloud'].get('status') != 'pending':
                return
            target['cloud'] = plain(result)
            if decrement:
                self.cloud['pending'] -= 1
            self.cloud['completed' if result.get('status') == 'verified' else 'failed'] += 1
            self._emit('cloud_verified' if result.get('status') == 'verified' else 'cloud_fallback',
                       {'ordinal': row['ordinal'], 'scope': 'path_end', 'data_cutoff_seconds': row['data_cutoff_seconds'], 'result': result})
        self.persist()

    def _cloud_consume(self):
        reporter = None
        try:
            while not self.stop_event.is_set():
                try:
                    job = self._cloud_queue.get(timeout=.1)
                except queue.Empty:
                    if self._processor_done.is_set():
                        break
                    continue
                try:
                    with self.lock:
                        if self.stop_event.is_set():
                            continue
                        self._cloud_inflight.set()
                        self.cloud['in_flight'] = True
                    try:
                        if reporter is None:
                            reporter = self.manager.cloud_reporter(self._emit, rag_policy=self.rag_policy)
                        if self.stop_event.is_set():
                            continue
                        result = reporter.report(job['row'], job['prediction'], cancelled=self.stop_event.is_set)
                    except Exception as exc:
                        result = {'status': 'fallback', 'final_class': None, 'reason': '云端报告未完成', 'error_type': type(exc).__name__}
                    self._apply_cloud(job['row'], result)
                finally:
                    self._cloud_queue.task_done()
                    with self.lock:
                        self.cloud['in_flight'] = False
                        self._cloud_inflight.clear()
                        self.persist()
        except Exception as exc:
            self._fail('cloud_worker_error:' + type(exc).__name__)
        finally:
            with self.lock:
                self._cloud_done.set()
                if self.state == 'finalizing' and self._processor_done.is_set() and not self.cloud['pending']:
                    try:
                        self._set_state('completed')
                    except Exception as exc:
                        self._fail('cloud_worker_error:' + type(exc).__name__)

    def report(self):
        result = self.snapshot(False)
        result['limitations'] = ['过程模型输出是未校准的整路径前缀估计；High/Low目标为路径结束问卷类别',
                                '疲劳与情绪领域输出EEG指标及解释', '运行日志不读取问卷、目标标签或历史查询预测']
        return result


class OnlineManager:
    def __init__(self, cfg, model_factory=None, cloud_factory=None):
        self.cfg, self.lock = dict(cfg), threading.RLock()
        self._model_lock = threading.Lock()
        self.root = Path(cfg.get('online_output_root', Path(cfg['output_root']).parent / 'online_sessions'))
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = Path(cfg.get('online_model_manifest', PROJECT / 'outputs/online_assets/seed2026/manifest.json'))
        self.max_path_windows = int(cfg.get('online_max_path_windows', 1000))
        self.max_resident_sessions = max(2, min(32, int(cfg.get('online_max_resident_sessions', 8))))
        self._model_factory, self._cloud_factory, self._model = model_factory, cloud_factory, None
        self._model_error, self.sessions = None, {}
        for path in sorted(self.root.glob('*/session.json'), key=lambda item: item.stat().st_mtime):
            try:
                saved = json.loads(path.read_text(encoding='utf-8'))
                changed = False
                if saved.get('state') in ACTIVE:
                    saved['state'] = 'interrupted'
                    saved['error'] = '服务已重启；新建回放并重新预热'
                    changed = True
                if saved.get('state') in TERMINAL:
                    cloud = saved.setdefault('cloud', {})
                    if cloud.get('in_flight'):
                        cloud['in_flight'] = False
                        changed = True
                    if cloud.get('pending') or any(row.get('cloud', {}).get('status') == 'pending' for row in saved.get('paths', [])):
                        _cancel_pending(saved.get('paths', []), cloud, '会话已终止，未完成的云端报告已取消；本地结果保留')
                        changed = True
                if changed:
                    atomic_json(path, saved)
                self.sessions[saved['id']] = path
            except (OSError, ValueError, KeyError):
                continue

    def source_path(self, recording_id):
        if not re.fullmatch(r'subject-\d{2}', recording_id):
            raise ValueError('请选择现有CDT记录')
        number = int(recording_id[-2:])
        return Path(self.cfg['data_root']) / 'data/raw' / ('Acquisition %02d.cdt' % number)

    def recording(self, recording_id):
        path = self.source_path(recording_id)
        meta = metadata(path)
        return {'id': recording_id, 'title': 'Acquisition ' + recording_id[-2:],
                'acquisition': int(recording_id[-2:]), 'sampling_hz': 1024,
                'sample_count': meta['actual_samples'],
                'duration_seconds': meta['actual_samples'] / 1024, 'warnings': meta['warnings']}

    def cloud_configured(self):
        return public_settings(self.cfg.get('cloud_config_path'))['configured']

    def catalog(self):
        from ..default_rag import catalog_fields
        recordings = []
        for path in sorted((Path(self.cfg['data_root']) / 'data/raw').glob('Acquisition ??.cdt')):
            try:
                recordings.append(self.recording('subject-' + path.stem[-2:]))
            except (ValueError, OSError):
                continue
        return {'recordings': recordings, 'pipeline_id': PIPELINE, **catalog_fields(),
                'capabilities': {'model_available': self._model_error is None and bool(self._model_factory or (self.manifest.is_file() and importlib.util.find_spec('torch'))),
                                 'model_error': self._model_error, 'cloud_configured': self.cloud_configured(),
                                 'online_cloud_timeout_seconds': timeout_seconds(self.cfg),
                                 'backend': 'mamba_minimal_cpu_fp32'},
                'defaults': {'speed': 1, 'window_seconds': 5, 'block_samples': 256, 'cloud_trigger': 'path_end', 'cloud': True,
                             'rag_mode': 'default'}}

    def model(self):
        with self._model_lock:
            if self._model is None and self._model_error is None:
                try:
                    if self._model_factory:
                        self._model = self._model_factory()
                    elif self.manifest.is_file():
                        from .model import OnlineModel
                        self._model = OnlineModel(self.manifest, threads=int(self.cfg.get('online_cpu_threads', 2)))
                    else:
                        self._model_error = '冻结在线资产未配置；描述性工具继续运行'
                except Exception as exc:
                    self._model_error = _asset_error(exc)
            return self._model

    def cloud_reporter(self, emit, rag_policy=None):
        if self._cloud_factory:
            return self._cloud_factory(emit)
        from .cloud import OnlineCloudReporter
        return OnlineCloudReporter(self.cfg, emit, rag_policy=rag_policy)

    def create(self, request):
        with self.lock:
            if any(isinstance(item, OnlineSession) and item.state in ACTIVE for item in self.sessions.values()):
                raise RuntimeError('已有在线回放；停止后再新建')
            self._compact_finished()
            if sum(isinstance(item, OnlineSession) for item in self.sessions.values()) >= self.max_resident_sessions:
                raise RuntimeError('此前回放线程仍在退出，请稍后新建')
            session = OnlineSession(self, request)
            self.sessions[session.id] = session
            session.start()
            return session.snapshot()

    def get(self, identity):
        with self.lock:
            value = self.sessions[identity]
            if isinstance(value, Path):
                try:
                    return json.loads(value.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    raise KeyError(identity)
            return value

    def _compact_finished(self):
        residents = [key for key, value in self.sessions.items() if isinstance(value, OnlineSession)]
        for key in residents[:]:
            if len(residents) < self.max_resident_sessions:
                break
            value = self.sessions[key]
            workers = [getattr(value, name, None) for name in ('reader', 'worker', 'cloud_worker')]
            if value.state in TERMINAL and all(worker is None or not worker.is_alive() for worker in workers):
                value.persist()
                self.sessions[key] = value.directory / 'session.json'
                residents.remove(key)

    def list(self, limit=20, offset=0):
        with self.lock:
            limit, offset = max(1, min(100, int(limit))), max(0, int(offset))
            identities = list(reversed(self.sessions))[offset:offset + limit]
            summaries = []
            for identity in identities:
                item = self.get(identity)
                value = item.snapshot(False) if isinstance(item, OnlineSession) else item
                summaries.append({key: copy.deepcopy(value.get(key)) for key in (
                    'id', 'state', 'recording', 'created_at', 'position_seconds', 'metrics', 'cloud', 'error', 'last_sequence',
                    'rag_mode', 'rag_policy')})
            return summaries

    def cloud_in_use(self):
        with self.lock:
            return any(isinstance(item, OnlineSession) and (item._cloud_inflight.is_set() or
                       (item.state in ACTIVE and item.cloud['requested'] and item.cloud['configured']))
                       for item in self.sessions.values())

    def shutdown(self):
        with self.lock:
            for item in self.sessions.values():
                if isinstance(item, OnlineSession) and item.state in ACTIVE:
                    item.interrupt()
