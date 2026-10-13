"""Local EEG measurements and deterministic analysis routing.

All numerical inputs are unlabeled, contract-matched P0/P1 windows in microvolts. Times are source
seconds and intervals are half open. A measurement is not a symptom label.
"""
from collections import deque
import math
import numpy as np
from scipy import signal
from scipy.integrate import trapezoid

from .config import EEG_CHANNELS

TOOL_VERSION = 'vrms-tools-2.0'
FS = 256
SAMPLES = 1280
BANDS = {'delta': (.5, 4.), 'theta': (4., 8.), 'alpha': (8., 13.),
         'beta': (13., 30.), 'gamma_low': (30., 45.)}
REGIONS = {'all': list(range(30)),
           'frontal': [EEG_CHANNELS.index(c) for c in ('F3', 'Fz', 'F4')],
           'occipital': [EEG_CHANNELS.index(c) for c in ('O1', 'Oz', 'O2')]}
TOOL_CATALOG = [
    {'id': 'EEGPreprocessor', 'version': TOOL_VERSION, 'input': 'arrived raw CDT samples/events',
     'output': 'P0 30x1280 windows and quality; optional causal-EOG-RLS P1; source intervals; baseline state',
     'requires': [], 'unit': 'uV', 'profile': 'P0'},
    {'id': 'SignalQualityChecker', 'version': TOOL_VERSION, 'input': 'raw and processed arrived samples',
     'output': 'quality measurements and usable gate', 'requires': ['EEGPreprocessor'], 'unit': 'uV'},
    {'id': 'EOGArtifactCleaner', 'version': 'causal-eog-rls-v1', 'input': 'arrived P0 EEG/EOG; prior-baseline initializer; causal per-sample RLS state',
     'output': 'P1 EEG; predict-before-update; pre/post measurements and usable gate; unavailable initialization blocks P1 inference',
     'requires': ['EEGPreprocessor','SignalQualityChecker'], 'unit': 'uV', 'profile': 'P1'},
    {'id': 'DeepVRMSDetector', 'version': TOOL_VERSION, 'input': 'quality-valid model-contract-matched P0/P1 30x1280 EEG',
     'output': 'selected frozen model probability and availability',
     'requires': ['SignalQualityChecker'], 'unit': 'probability', 'profiles': ['P0','P1']},
    {'id': 'BiomarkerCalculator', 'version': TOOL_VERSION, 'input': 'quality-valid P0/P1 window; optional same-profile frozen baseline',
     'output': 'Welch PSD, band powers, ratios and prior-baseline changes; no symptom labels',
     'requires': ['SignalQualityChecker'], 'unit': 'uV^2 and uV^2/Hz'},
    {'id': 'CovarianceAnalyzer', 'version': TOOL_VERSION, 'input': 'quality-valid P0/P1 window; optional same-profile frozen baseline',
     'output': '10% trace-shrunk covariance summary and unsigned AIRM state distance',
     'requires': ['SignalQualityChecker'], 'unit': 'uV^2; dimensionless distance'},
    {'id': 'TemporalAnalyzer', 'version': TOOL_VERSION, 'input': 'arrived selected-model probabilities and true intervals',
     'output': '30s mean/slope/coverage, volatility, jumps and persistent model-high candidate events',
     'requires': ['DeepVRMSDetector'], 'unit': 'probability; probability/minute; seconds'},
    {'id': 'VRMSReportGenerator', 'version': TOOL_VERSION, 'input': 'validated evidence and cited local knowledge',
     'output': 'Chinese report and traceable JSON; categories rendered from local evidence',
     'requires': [], 'unit': None},
]


def _window(x):
    a = np.asarray(x, dtype=np.float64)
    if a.shape == (SAMPLES, 30):
        a = a.T
    if a.shape != (30, SAMPLES):
        raise ValueError('requires 30 EEG channels and 1280 samples (P0, 256 Hz, 5 seconds)')
    if not np.isfinite(a).all():
        raise ValueError('nonfinite EEG window')
    return a


def _failure(tool, reason):
    return {'tool': tool, 'version': TOOL_VERSION, 'available': False, 'reason': str(reason)}


def _json_values(a):
    a = np.asarray(a)
    return [float(v) if np.isfinite(v) else None for v in a.ravel()]


def _ratio(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or b <= 1e-12:
        return None
    v = float(a / b)
    return v if math.isfinite(v) else None


def _spectral(x):
    frequencies, psd = signal.welch(x, fs=FS, window='hann', nperseg=512,
                                    noverlap=256, detrend='constant', axis=-1,
                                    scaling='density')
    if not np.isfinite(psd).all():
        raise ValueError('nonfinite spectral computation')
    power = {}
    for band, (low, high) in BANDS.items():
        # Include exact edges. Adjacent bands share boundary points, not area.
        take = (frequencies >= low) & (frequencies <= high)
        power[band] = trapezoid(psd[:, take], frequencies[take], axis=-1)
    return frequencies, psd, power


def _covariance(x):
    centered = x - x.mean(axis=1, keepdims=True)
    raw = centered @ centered.T / (x.shape[1] - 1)
    scale = float(np.trace(raw) / 30.)
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError('insufficient covariance variance')
    c = .9 * raw + .1 * scale * np.eye(30)
    return (c + c.T) / 2.


class BaselineCollector:
    """Freeze only quality-valid, complete pre-task windows; never retain EEG."""
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.min_windows = 2
        self.max_windows = 12
        self.frozen = False
        self.cutoff = None
        self.available = False
        self.power_by_band = None
        self.covariance_matrix = None
        self._windows = []
        self._last_end = None
        self.rejected_windows = 0

    def observe(self, x, quality, start, end):
        if self.frozen:
            return False
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or abs(end-start-5.) > 1e-6:
            self.rejected_windows += 1
            return False
        if self._last_end is not None and start < self._last_end - 1e-6:
            self.rejected_windows += 1
            return False
        self._last_end = end
        good = bool(quality.get('valid', quality.get('usable', False))) if isinstance(quality, dict) else bool(quality)
        if not good or len(self._windows) >= self.max_windows:
            if not good:
                self.rejected_windows += 1
            return False
        try:
            a = _window(x)
            _, _, powers = _spectral(a)
            cov = _covariance(a)
        except (ValueError, TypeError):
            self.rejected_windows += 1
            return False
        self._windows.append({'start': start, 'end': end, 'power': powers, 'covariance': cov})
        return True

    def freeze(self, cutoff):
        if self.frozen:
            return self.summary()
        cutoff = float(cutoff)
        if not math.isfinite(cutoff):
            raise ValueError('baseline cutoff must be finite source seconds')
        self.frozen = True
        self.cutoff = cutoff
        # Defensive arrival guard: a task event can occur inside a tentative window.
        self._windows = [w for w in self._windows if w['end'] <= cutoff + 1e-9]
        self.available = len(self._windows) >= self.min_windows
        if self.available:
            self.power_by_band = {b: np.mean([w['power'][b] for w in self._windows], axis=0) for b in BANDS}
            self.covariance_matrix = np.mean([w['covariance'] for w in self._windows], axis=0)
        return self.summary()

    def summary(self):
        return {'version': TOOL_VERSION, 'frozen': self.frozen, 'available': self.available,
                'reason': 'frozen prior unlabeled baseline' if self.available else
                          ('insufficient eligible pre-task windows' if self.frozen else 'collecting pre-task windows'),
                'windows': len(self._windows), 'seconds': len(self._windows) * 5,
                'minimum_windows': self.min_windows, 'maximum_windows': self.max_windows,
                'cutoff_seconds': self.cutoff, 'rejected_windows': self.rejected_windows,
                'z_scores_available': False}


def biomarkers(x, baseline=None):
    try:
        a = _window(x)
        frequencies, psd, powers = _spectral(a)
    except (ValueError, TypeError) as exc:
        return _failure('BiomarkerCalculator', exc)
    total = np.sum(list(powers.values()), axis=0)
    relative = {b: [_ratio(v, den) for v, den in zip(power, total)] for b, power in powers.items()}
    regions = {}
    for region, indices in REGIONS.items():
        p = {b: float(power[indices].mean()) for b, power in powers.items()}
        den = sum(p.values())
        regions[region] = {'absolute_power_uv2': p,
                           'relative_power': {b: _ratio(v, den) for b, v in p.items()},
                           'ratios': {'theta_alpha': _ratio(p['theta'], p['alpha']),
                                      'theta_beta': _ratio(p['theta'], p['beta']),
                                      'slow_fast': _ratio(p['delta'] + p['theta'], p['alpha'] + p['beta'])}}
    prior = baseline.power_by_band if isinstance(baseline, BaselineCollector) and baseline.available else None
    changes = None
    if prior is not None:
        changes = {'per_channel': {b: [_ratio(v-r, r) for v, r in zip(powers[b], prior[b])] for b in BANDS},
                   'regions': {region: {b: _ratio(float(powers[b][ix].mean()-prior[b][ix].mean()),
                                                  float(prior[b][ix].mean())) for b in BANDS}
                               for region, ix in REGIONS.items()}}
    selected = frequencies <= 45.
    return {'tool': 'BiomarkerCalculator', 'version': TOOL_VERSION, 'available': True,
            'channels': list(EEG_CHANNELS), 'sample_rate_hz': FS, 'window_samples': SAMPLES,
            'unit': 'uV^2', 'psd_unit': 'uV^2/Hz', 'frequencies_hz': frequencies[selected].tolist(),
            'psd': psd[:, selected].tolist(), 'absolute_power_uv2': {b: _json_values(v) for b, v in powers.items()},
            'relative_power': relative, 'regions': regions,
            'baseline_available': prior is not None, 'baseline_change': changes,
            'reason': None if prior is not None else 'baseline changes unavailable; absolute measurements remain available',
            'interpretation': 'EEG measurement; not a validated directional VR motion-sickness biomarker'}


def covariance(x, baseline=None):
    try:
        c = _covariance(_window(x))
    except (ValueError, TypeError) as exc:
        return _failure('CovarianceAnalyzer', exc)
    eig = np.linalg.eigvalsh(c)
    ref = baseline.covariance_matrix if isinstance(baseline, BaselineCollector) and baseline.available else None
    distance = None
    if ref is not None:
        values, vectors = np.linalg.eigh(ref)
        inverse_root = (vectors * (1. / np.sqrt(values))[None, :]) @ vectors.T
        relative = inverse_root @ c @ inverse_root
        relative = (relative + relative.T) / 2.
        relative_values = np.linalg.eigvalsh(relative)
        if np.isfinite(relative_values).all() and (relative_values > 0).all():
            distance = float(np.linalg.norm(np.log(relative_values)))
    return {'tool': 'CovarianceAnalyzer', 'version': TOOL_VERSION, 'available': True,
            'unit': 'uV^2', 'distance_unit': 'dimensionless', 'shrinkage': .1,
            'trace_uv2': float(np.trace(c)), 'minimum_eigenvalue': float(eig.min()),
            'maximum_eigenvalue': float(eig.max()), 'baseline_available': ref is not None,
            'baseline_windows': len(baseline._windows) if isinstance(baseline, BaselineCollector) else 0,
            'airm_distance': distance, 'z_score': None,
            'reason': None if distance is not None else 'prior baseline covariance unavailable',
            'interpretation': 'unsigned EEG spatial/amplitude state change; does not determine high/low class'}


class TemporalAnalyzer:
    def __init__(self, threshold):
        self.threshold = float(threshold)
        if not 0. <= self.threshold <= 1. or not math.isfinite(self.threshold):
            raise ValueError('invalid frozen model threshold')
        self.history = deque(maxlen=6)
        self.path_id = None
        self.last_end = None
        self.high_run = 0
        self.high_start = None
        self.last_reset_reason = 'initial'

    def reset(self, reason='explicit_reset'):
        self.history.clear()
        self.last_end = None
        self.high_run = 0
        self.high_start = None
        self.path_id = None
        self.last_reset_reason = reason

    def observe(self, start, end, probability, path_id):
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError('invalid source interval')
        if self.last_end is not None and start < self.last_end-1e-6:
            raise ValueError('out-of-order temporal interval')
        if self.path_id is not None and path_id != self.path_id:
            self.reset('path_switch')
        elif self.last_end is not None and abs(start-self.last_end) > 1e-6:
            self.reset('stream_gap')
        self.path_id = path_id
        self.last_end = end
        p = None
        if probability is not None:
            value = float(probability)
            if math.isfinite(value) and 0. <= value <= 1. and abs(end-start-5.) <= 1e-6:
                p = value
        self.history.append({'start': start, 'end': end, 'probability': p})
        while self.history and self.history[0]['end'] <= end-30. + 1e-9:
            self.history.popleft()
        events = []
        if p is not None and p >= self.threshold:
            if self.high_run == 0:
                self.high_start = start
            self.high_run += 1
            if self.high_run == 3:
                events.append({'type': 'persistent_model_high_candidate',
                               'first_high_start_seconds': self.high_start,
                               'confirmation_seconds': end, 'confirmed_windows': 3,
                               'threshold': self.threshold,
                               'interpretation': 'model-high candidate; physiological symptom onset unvalidated'})
        else:
            self.high_run = 0
            self.high_start = None
        valid = [w for w in self.history if w['probability'] is not None]
        values = np.asarray([w['probability'] for w in valid], dtype=np.float64)
        n = len(valid)
        duration = end-self.history[0]['start'] if self.history else 0.
        valid_seconds = sum(w['end']-w['start'] for w in valid)
        slope, trend = None, None
        if n >= 3:
            times = np.asarray([(w['start']+w['end'])/2 for w in valid], dtype=np.float64)
            t = times-times.mean()
            slope = float(np.dot(t, values-values.mean()) / np.dot(t, t) * 60.)
            trend = 'increasing' if slope > 1e-12 else ('decreasing' if slope < -1e-12 else 'stable')
        adjacent = []
        rows = [dict(w) for w in self.history]
        for previous, current in zip(rows, rows[1:]):
            if previous['probability'] is not None and current['probability'] is not None and abs(current['start']-previous['end']) <= 1e-6:
                adjacent.append(abs(current['probability']-previous['probability']))
        std = float(values.std(ddof=0)) if n >= 2 else None
        return {'tool': 'TemporalAnalyzer', 'version': TOOL_VERSION, 'available': n >= 3,
                'reason': None if n >= 3 else 'requires at least 3 valid arrived windows for trend',
                'current_probability': p, 'mean_30s': float(values.mean()) if n else None,
                'slope_probability_per_minute': slope, 'slope': slope,
                'trend': trend, 'std_probability': std, 'volatility': std,
                'max_jump': float(max(adjacent)) if adjacent else None,
                'valid_windows': n, 'slots': len(rows), 'analysis_duration_seconds': duration,
                'valid_duration_seconds': valid_seconds, 'coverage': valid_seconds/duration if duration else 0.,
                'consecutive_high_windows': self.high_run, 'persistent_duration_seconds': self.high_run*5.,
                'threshold': self.threshold, 'events': events, 'history': rows,
                'last_reset_reason': self.last_reset_reason}


def analysis_policy(strategy, deep, temporal, full_requested=False):
    if strategy not in ('adaptive', 'deep_only', 'full_evidence'):
        raise ValueError('unknown analysis strategy')
    if not deep or not deep.get('available', True):
        return {'tools': [], 'reasons': ['deep_prediction_unavailable'], 'early_exit': True}
    p = deep.get('probability_high', deep.get('probability', deep.get('vrms_probability')))
    threshold = deep.get('threshold', deep.get('decision_threshold', temporal.get('threshold', .5)))
    if p is None or not math.isfinite(float(p)) or not 0. <= float(p) <= 1.:
        return {'tools': [], 'reasons': ['deep_prediction_unavailable'], 'early_exit': True}
    p, threshold = float(p), float(threshold)
    if strategy == 'deep_only' and not full_requested:
        return {'tools': [], 'reasons': ['deep_only_strategy'], 'early_exit': True}
    if strategy == 'full_evidence' or full_requested:
        return {'tools': ['BiomarkerCalculator', 'CovarianceAnalyzer'],
                'reasons': ['full_assessment_requested' if full_requested else 'full_evidence_strategy'], 'early_exit': False}
    mean = temporal.get('mean_30s')
    opposite = mean is not None and ((p >= threshold) != (float(mean) >= threshold))
    std = temporal.get('std_probability', temporal.get('volatility'))
    jump = temporal.get('max_jump')
    reasons = []
    if int(temporal.get('valid_windows', 0)) < 3:
        reasons.append('insufficient_temporal_history')
    # Preserve the strict mathematical boundaries despite decimal roundoff.
    if abs(p-threshold) < .10-1e-12:
        reasons.append('margin_below_0.10')
    if opposite:
        reasons.append('current_mean_threshold_conflict')
    if std is not None and float(std) > .15+1e-12:
        reasons.append('temporal_std_above_0.15')
    if jump is not None and float(jump) > .20+1e-12:
        reasons.append('adjacent_jump_above_0.20')
    selected = ['BiomarkerCalculator'] if reasons else []
    if opposite or (jump is not None and float(jump) > .20+1e-12):
        selected.append('CovarianceAnalyzer')
    return {'tools': selected, 'reasons': reasons or ['deep_temporal_stable'], 'early_exit': not selected,
            'policy_version': 'auxiliary-adaptive-1.0', 'changes_final_class': False}
