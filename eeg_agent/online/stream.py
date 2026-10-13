"""Arrived-only version of the current raw recording preprocessing contract.

No labels, resting reference, EOG regression, or whole-record quality gate are
used. Filter state is continuous across task/rest boundaries; window grids are
anchored to their original Trigger markers even after missing samples.
"""
import numpy as np
from scipy import signal

from ..config import EEG_CHANNELS


class StreamingProcessor:
    FS = 1024
    WINDOW_SAMPLES = 5120
    OUTPUT_SAMPLES = 1280

    def __init__(self, channels, fs=1024):
        self.channels = list(channels)
        if fs != self.FS:
            raise ValueError('Online raw VRMS input requires 1024 Hz')
        if len(set(self.channels)) != len(self.channels):
            raise ValueError('Channel names must be unique')
        required = EEG_CHANNELS + ['M1', 'M2', 'Trigger']
        if any(name not in self.channels for name in required):
            raise ValueError('Missing required EEG, M1/M2, or Trigger channel')
        self._picks = [self.channels.index(name) for name in EEG_CHANNELS]
        self._refs = [self.channels.index('M1'), self.channels.index('M2')]
        self._trigger = self.channels.index('Trigger')
        self._sos = np.vstack([
            signal.tf2sos(*signal.iirnotch(50, 30, fs=fs)),
            signal.butter(4, [.5, 45], btype='bandpass', fs=fs, output='sos'),
            signal.butter(8, 70, btype='lowpass', fs=fs, output='sos'),
        ])
        self._state = np.zeros((len(self._sos), 2, 30))
        self._warm_until = self.WINDOW_SAMPLES
        self._last_finite = True
        self._previous_trigger = 0
        self._cursor = None
        self._active = None
        self._ordinals = {'task': 0, 'rest': 0}
        self._window = np.full((30, self.OUTPUT_SAMPLES), np.nan, np.float32)
        self._window_missing = False
        self._finished = False

    @property
    def buffered_samples(self):
        """Upper bound on retained waveform samples, independent of run length."""
        return self.OUTPUT_SAMPLES if self._active is not None else 0

    def _event(self, kind, scope, ordinal, start, end, complete, **extra):
        return {'type': kind, 'scope': scope, 'ordinal': ordinal,
                'start_sample': int(start), 'end_sample': int(end),
                'start_seconds': start / self.FS, 'end_seconds': end / self.FS,
                'complete': bool(complete), **extra}

    def _warning(self, events, reason, start, end):
        active = self._active or {}
        events.append(self._event('warning', active.get('scope'),
            active.get('ordinal'), start, end, False, reason=reason,
            observed_end_sample=int(end)))

    def _close(self, stamp, complete, events, reason=''):
        active = self._active
        if active is None:
            return
        if stamp > active['start_sample']:
            kind = 'path_end' if active['scope'] == 'task' else 'rest_end'
            events.append(self._event(kind, active['scope'], active['ordinal'],
                active['start_sample'], stamp, complete, reason=reason,
                total_windows=active['window_ordinal'],
                observed_end_sample=int(stamp + (1 if complete else 0))))
        self._active = None
        self._window.fill(np.nan)
        self._window_missing = False

    def _mark(self, stamp, mark, events):
        scope = 'task' if mark == 20 else 'rest'
        if self._active is None and mark == 22:
            self._warning(events, '结束事件前未找到开始事件', stamp, stamp + 1)
        elif self._active is not None:
            complete = self._active['scope'] != scope
            reason = '' if complete else (
                '重复开始事件：前片段缺少结束标记' if mark == 20 else
                '重复结束事件：前休息段缺少下一开始标记')
            if reason:
                self._warning(events, reason, stamp, stamp + 1)
            self._close(stamp, complete, events, reason)
        self._ordinals[scope] += 1
        self._active = {'scope': scope, 'ordinal': self._ordinals[scope],
                        'start_sample': stamp, 'next_window': stamp,
                        'window_ordinal': 0}
        self._window.fill(np.nan)
        self._window_missing = False
        kind = 'path_start' if scope == 'task' else 'rest_start'
        events.append(self._event(kind, scope, self._ordinals[scope], stamp,
            stamp, False, observed_end_sample=int(stamp + 1)))

    def _flush(self, observed_end, events):
        if self._active is None:
            return
        active = self._active
        while active['next_window'] + self.WINDOW_SAMPLES <= observed_end:
            start = active['next_window']
            end = start + self.WINDOW_SAMPLES
            window = self._window.copy()
            reason = ''
            if not np.isfinite(window).all():
                reason = 'missing_samples' if self._window_missing else 'warmup_or_nonfinite'
            elif np.ptp(window, axis=1).max() > 2000:
                reason = 'amplitude'
            elif np.count_nonzero(window.std(axis=1) >= .05) < 28:
                reason = 'flat_channels'
            active['window_ordinal'] += 1
            events.append(self._event('window', active['scope'], active['ordinal'],
                start, end, True, window=window, valid=not reason, reason=reason,
                window_ordinal=active['window_ordinal'],
                segment_start_sample=active['start_sample'],
                observed_end_sample=int(end),
                output_start_sample=(start + 3) // 4,
                output_end_sample=(start + 3) // 4 + self.OUTPUT_SAMPLES))
            active['next_window'] = end
            self._window.fill(np.nan)
            self._window_missing = False

    def _filter(self, raw, start, events):
        x = np.asarray(raw[:, self._picks], dtype=np.float64)
        x = x - np.asarray(raw[:, self._refs], dtype=np.float64).mean(axis=1, keepdims=True)
        finite = np.isfinite(x).all(axis=1)
        result = np.full(x.shape, np.nan, np.float32)
        edges = np.r_[0, np.flatnonzero(finite[1:] != finite[:-1]) + 1, len(x)]
        for left, right in zip(edges[:-1], edges[1:]):
            if not finite[left]:
                if self._last_finite:
                    self._warning(events, 'nonfinite_signal', start + left, start + left + 1)
                self._state.fill(0)
                self._last_finite = False
                continue
            if not self._last_finite:
                self._warm_until = start + left + self.WINDOW_SAMPLES
            filtered, self._state = signal.sosfilt(
                self._sos, x[left:right], axis=0, zi=self._state)
            result[left:right] = filtered
            result[left:min(right, max(left, self._warm_until - start))] = np.nan
            self._last_finite = True
        samples = np.arange(start, start + len(raw))
        take = samples % 4 == 0
        if self._active is not None:
            first = 4 * ((self._active['next_window'] + 3) // 4)
            positions = (samples[take] - first) // 4
            keep = (positions >= 0) & (positions < self.OUTPUT_SAMPLES)
            self._window[:, positions[keep]] = result[take][keep].T

    def _interval(self, raw, start, events):
        offset = 0
        while offset < len(raw):
            left = start + offset
            right = min(start + len(raw), left + 16384)
            if self._active is not None:
                right = min(right, self._active['next_window'] + self.WINDOW_SAMPLES)
            self._filter(raw[offset:offset + right - left], left, events)
            offset += right - left
            self._flush(right, events)

    def feed(self, block, start_sample):
        """Consume samples [start_sample, start_sample + len(block))."""
        if self._finished:
            raise RuntimeError('The processor has finished; create a new processor to replay')
        if isinstance(start_sample, (bool, np.bool_)) or not isinstance(start_sample, (int, np.integer)) or start_sample < 0:
            raise ValueError('start_sample must be a nonnegative integer')
        raw = np.asarray(block)
        if raw.ndim != 2 or raw.shape[1] != len(self.channels):
            raise ValueError('Expected samples-by-channel input matching channel names')
        start_sample = int(start_sample)
        if self._cursor is not None and start_sample < self._cursor:
            raise ValueError('Overlapping or out-of-order sample block')
        if not len(raw):
            return []
        events = []
        if self._cursor is None:
            self._warm_until = start_sample + self.WINDOW_SAMPLES
        elif start_sample != self._cursor:
            self._warning(events, 'missing_samples', self._cursor, start_sample)
            # Missing samples occupy the original grid. Never join both sides
            # into a fresh, apparently valid five-second task window.
            while (self._active is not None and
                   self._active['next_window'] + self.WINDOW_SAMPLES <= start_sample):
                self._window_missing = True
                self._flush(self._active['next_window'] + self.WINDOW_SAMPLES, events)
            if self._active is not None and start_sample > self._active['next_window']:
                self._window_missing = True
            self._state.fill(0)
            self._warm_until = start_sample + self.WINDOW_SAMPLES
            self._last_finite = True
            self._previous_trigger = 0
        trigger = raw[:, self._trigger]
        finite = np.isfinite(trigger)
        rounded = np.rint(np.where(finite, trigger, 0)).astype(np.int64)
        previous = np.r_[self._previous_trigger, rounded[:-1]]
        hits = np.flatnonzero(finite & (rounded != previous) & np.isin(rounded, (20, 22)))
        pointer = 0
        for offset in hits:
            offset = int(offset)
            self._interval(raw[pointer:offset], start_sample + pointer, events)
            self._mark(start_sample + offset, int(rounded[offset]), events)
            pointer = offset
        self._interval(raw[pointer:], start_sample + pointer, events)
        self._previous_trigger = int(rounded[-1])
        self._cursor = start_sample + len(raw)
        return events

    def finish(self):
        """Close an unmarked final segment as incomplete; idempotent."""
        if self._finished:
            return []
        self._finished = True
        events = []
        if self._active is not None:
            reason = ('记录结束前缺少结束标记' if self._active['scope'] == 'task'
                      else '记录结束前缺少下一开始标记')
            self._flush(self._cursor, events)
            self._close(self._cursor, False, events, reason)
        return events
