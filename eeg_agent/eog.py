"""Frozen, prior-baseline EOG regression and its classification guardrails.

This removes only the component predicted by measured VEOG/HEOG using a
fixed two-channel linear mixing estimate. It does not claim that unrelated
EEG/EOG correlation proves ocular contamination or that all ocular effects
have been removed. Calibration never uses labels or a task window.
"""
from copy import deepcopy

import numpy as np

from . import config as configuration
from .signal import Calibration, _eog_correlation, _measurements


PROCESSING_VERSION = 'frozen-eog-ridge-v1'
ADAPTIVE_PROCESSING_VERSION = 'causal-eog-rls-v1'
WINDOW_SAMPLES = 1280
def fit_eog_calibration(values, cutoff, cfg):
    """Fit from the already-arrived QC-valid complete pre-task windows.

    The caller supplies these windows, excluding warm-up, in arrival order.
    Only the first configured maximum duration is used. Range is checked
    separately per window: gaps between accepted windows must not create a
    fictitious within-window amplitude excursion.
    """
    data = np.asarray(values, dtype=np.float64)
    c = Calibration(cutoff_sample=int(cutoff))
    c.processing_version = PROCESSING_VERSION
    if data.ndim != 2 or data.shape[1] != 32:
        c.reason = 'invalid baseline shape'; return c
    maximum = min(60.0, float(cfg['calibration_max_seconds']))
    available = min(len(data) // WINDOW_SAMPLES,
                    int(maximum * 256) // WINDOW_SAMPLES) * WINDOW_SAMPLES
    data = data[:available]
    c.samples = len(data)
    if len(data) < max(10.0, float(cfg['calibration_min_seconds'])) * 256:
        c.reason = 'insufficient past baseline'; return c
    if not np.isfinite(data).all():
        c.reason = 'nonfinite prior baseline'; return c
    x, e = data[:, :30], data[:, 30:]
    ranges = np.ptp(x.reshape(-1, WINDOW_SAMPLES, 30), axis=1)
    if np.any(ranges > cfg['qc']['peak_to_peak_uv']):
        c.reason = 'invalid baseline amplitude'; return c
    p = configuration.eog_policy(cfg)
    if np.any(e.std(axis=0) < p['minimum_eog_std_uv']):
        c.reason = 'insufficient EOG variation'; return c
    e0 = e - e.mean(axis=0)
    covariance = e0.T @ e0 / len(e0)
    condition = float(np.linalg.cond(covariance))
    c.condition = condition if np.isfinite(condition) else 0.0
    if not np.isfinite(condition) or condition > p['maximum_condition']:
        c.reason = 'ill-conditioned EOG baseline'; return c
    ridge = p['ridge_fraction'] * float(np.trace(covariance)) / 2.0
    try:
        mixing = np.linalg.solve(covariance + ridge * np.eye(2),
                                 e0.T @ (x - x.mean(axis=0)) / len(x)).T
    except np.linalg.LinAlgError:
        c.reason = 'unstable regression'; return c
    if not np.isfinite(mixing).all() or np.max(np.abs(mixing)) > p['maximum_coefficient']:
        c.reason = 'unstable regression'; return c
    c.available = True
    c.reason = 'frozen prior unlabeled baseline'
    c.B = mixing
    c.mean = e.mean(axis=0)
    c.eog_covariance = covariance
    c.eeg_mean = x.mean(axis=0)
    return c


class AdaptiveEOGRegression:
    """Causal reference RLS, initialized from the accepted prior baseline.

    Each output uses coefficients and centering means from *past* samples.
    Only after producing that output does the current unlabeled sample update
    the state. No batch fit, window look-ahead, label or model prediction is
    used. Correlated neural activity remains a limitation of reference-based
    regression, even when controlled injection is recovered correctly.
    """
    def __init__(self, calibration, cfg):
        self.cfg = cfg
        self.calibration = calibration
        self.policy = configuration.eog_policy(cfg)
        self.memory_seconds = float(self.policy.get('adaptive_memory_seconds', 10.0))
        self.forgetting = float(np.exp(-1.0 / (256.0 * self.memory_seconds)))
        self.mean_step = 1.0 - self.forgetting
        self.samples = 0
        self.skipped_updates = 0
        self.amplitude_limit_uv = float(cfg['qc']['peak_to_peak_uv'])
        self.available = bool(calibration.available)
        self.reason = 'causal reference RLS active' if self.available else calibration.reason
        self.B = None
        self.inverse = None
        self.eog_mean = None
        self.eeg_mean = None
        self.condition = 0.0
        if not self.available: return
        covariance = getattr(calibration, 'eog_covariance', None)
        eeg_mean = getattr(calibration, 'eeg_mean', None)
        if (np.shape(covariance) != (2, 2) or np.shape(eeg_mean) != (30,) or
                np.shape(calibration.B) != (30, 2) or np.shape(calibration.mean) != (2,) or
                not all(np.isfinite(v).all() for v in (covariance, eeg_mean, calibration.B, calibration.mean))):
            self.available = False; self.reason = 'missing adaptive initializer'; return
        if (np.linalg.cond(covariance) > self.policy['maximum_condition'] or
                np.abs(calibration.B).max() > self.policy['maximum_coefficient']):
            self.available = False; self.reason = 'unstable adaptive initializer'; return
        ridge = self.policy['ridge_fraction'] * float(np.trace(covariance)) / 2.0
        # RLS uses inverse *information*, not inverse covariance. Initializing
        # with cov^-1 alone would give a single task sample excessive weight.
        try:
            self.inverse = np.linalg.inv(covariance + ridge * np.eye(2)) / max(1, calibration.samples)
        except np.linalg.LinAlgError:
            self.available = False; self.reason = 'unstable adaptive initializer'; return
        self.B = calibration.B.copy()
        self.eog_mean = calibration.mean.copy()
        self.eeg_mean = np.asarray(eeg_mean).copy()
        self.condition = float(np.linalg.cond(self.inverse))

    def reset(self):
        """Discard adaptive history at a gap; reuse only its prior initializer."""
        self.__init__(self.calibration, self.cfg)

    def summary(self):
        return {'version': ADAPTIVE_PROCESSING_VERSION, 'available': self.available,
                'reason': self.reason, 'processed_samples': int(self.samples + self.skipped_updates),
                'samples': int(self.samples + self.skipped_updates), 'updates': int(self.samples),
                'skipped_updates': int(self.skipped_updates),
                'coefficient_version': int(self.samples),
                'memory_seconds': self.memory_seconds,
                'forgetting_seconds': self.memory_seconds,
                'method': 'causal_reference_rls',
                'ridge_fraction': self.policy['ridge_fraction'],
                'forgetting_factor': self.forgetting,
                'prediction_before_update': True,
                'labels_used': False, 'parameters_frozen': True,
                'coefficients_frozen': False,
                'condition': _finite_number(self.condition),
                'maximum_absolute_coefficient': float(np.abs(self.B).max()) if self.B is not None else None,
                'coefficient_max_abs': float(np.abs(self.B).max()) if self.B is not None else None,
                'initialization_cutoff_sample': int(self.calibration.cutoff_sample)}

    def clean_samples(self, x, e, valid=None):
        x = np.asarray(x, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 30 or e.shape != (len(x), 2):
            raise ValueError('Adaptive EOG input requires paired N x 30 EEG and N x 2 EOG')
        valid = np.ones(len(x), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        if valid.shape != (len(x),): raise ValueError('Adaptive validity mask must have length N')
        output = x.copy()
        if not self.available: return output
        for i in range(len(x)):
            if not valid[i]:
                self.skipped_updates += 1
                continue
            if not np.isfinite(x[i]).all() or not np.isfinite(e[i]).all():
                self.available = False; self.reason = 'nonfinite adaptive input'; break
            eye = e[i] - self.eog_mean
            neural = x[i] - self.eeg_mean
            prediction = self.B @ eye
            # Publish using the prior state. The following update cannot alter
            # this sample or any earlier output.
            output[i] = x[i] - prediction
            # A finite electrode excursion must not alter the future mixing
            # estimate or either centering mean. Later complete-window QC
            # still rejects its amplitude; this only protects causal state.
            if np.any(np.abs(neural) > self.amplitude_limit_uv):
                self.skipped_updates += 1
                continue
            residual = neural - prediction
            gain_numerator = self.inverse @ eye
            gain = gain_numerator / (self.forgetting + eye @ gain_numerator)
            coefficients = self.B + residual[:, None] * gain[None, :]
            inverse = (self.inverse - np.outer(gain, eye @ self.inverse)) / self.forgetting
            inverse = (inverse + inverse.T) * .5
            tr = float(np.trace(inverse)); det = float(np.linalg.det(inverse))
            root = np.sqrt(max(0.0, tr * tr - 4.0 * det))
            condition = (tr + root) / max(tr - root, 1e-30)
            if (not np.isfinite(coefficients).all() or not np.isfinite(inverse).all() or
                    det <= 0 or condition > self.policy['maximum_condition'] or
                    np.abs(coefficients).max() > self.policy['maximum_coefficient']):
                self.available = False; self.reason = 'unstable adaptive regression'; break
            self.B = coefficients
            self.inverse = inverse
            self.condition = float(condition)
            self.eog_mean += self.mean_step * eye
            self.eeg_mean += self.mean_step * neural
            self.samples += 1
        return output


def _finite_number(value):
    return float(value) if np.isfinite(value) else None


def _summary(c):
    result = c.summary()
    result['processing_version'] = PROCESSING_VERSION
    return result


def process_window(w, cfg, profile='P1', method=None):
    """Produce a cleaned complete window without mutating the source window.

    A failed P1 request never becomes valid P0 input for a P1-only model.
    Raw saturation, discontinuity, warm-up, flat-channel and jump guardrails
    survive cleaning; processed amplitude is measured again after cleaning.
    The residual correlation and excessive-removal limits are engineering
    acceptance rules, not validated symptom or artifact probabilities.
    """
    if profile not in ('P0', 'P1'):
        raise ValueError('Unsupported EOG processing profile')
    result = dict(w)
    q = deepcopy(w.get('quality', {'valid': False, 'reasons': ['missing_quality']}))
    q['reasons'] = list(q.get('reasons', []))
    c = w.get('calibration') or Calibration()
    x = np.asarray(w.get('x', []), dtype=np.float64)
    e = np.asarray(w.get('e', []), dtype=np.float64)
    shape_ok = x.shape == (WINDOW_SAMPLES, 30) and e.shape == (WINDOW_SAMPLES, 2)
    finite = shape_ok and np.isfinite(x).all() and np.isfinite(e).all()
    pre_corr = _eog_correlation(x, e) if finite else None
    pre_p2p = float(np.ptp(x, axis=0).max()) if finite else None
    requested_method=method or configuration.eog_policy(cfg).get('method','frozen')
    adaptive_requested=requested_method in ('causal_rls','causal_reference_rls') and not w.get('calibration_reference',False)
    detail = {
        'processing_version': ADAPTIVE_PROCESSING_VERSION if adaptive_requested else PROCESSING_VERSION,
        'method': 'causal_reference_rls' if adaptive_requested else 'frozen_reference_ridge',
        'parameters_frozen': True,
        'coefficients_frozen': not adaptive_requested,
        'requested_profile': profile,
        'applied_profile': None,
        'applied': False,
        'status': 'unavailable',
        'reason': '',
        'reasons': [],
        'input_available': False,
        'pre_frontal_eog_correlation': pre_corr,
        'post_frontal_eog_correlation': pre_corr,
        'pre_peak_to_peak_uv': pre_p2p,
        'post_peak_to_peak_uv': pre_p2p,
        'removed_rms_ratio': 0.0,
        'removed_rms_uv': 0.0,
        'calibration': _summary(c),
    }
    result.update(x=x.copy(), quality=q, eog_cleaning=detail)

    def fail(reason):
        detail['reason'] = reason
        detail['reasons'] = [reason]
        q['valid'] = False
        if reason not in q['reasons']: q['reasons'].append(reason)
        result['profile'] = None
        return result

    if not shape_ok: return fail('eog_invalid_window_shape')
    if not finite: return fail('eog_nonfinite_window')
    if profile == 'P0':
        detail.update(applied_profile='P0', status='bypassed', reason='P0 requested',
                      input_available=bool(q.get('valid', False)))
        result['profile'] = 'P0'
        return result
    if not c.available: return fail('eog_calibration_unavailable: ' + c.reason)
    if np.shape(c.B) != (30, 2) or np.shape(c.mean) != (2,) or not np.isfinite(c.B).all() or not np.isfinite(c.mean).all():
        return fail('eog_invalid_calibration')
    reference = bool(w.get('calibration_reference', False))
    start = int(w.get('start_sample', -1))
    path_start = w.get('path_start_sample')
    if reference:
        if int(w.get('observed_end_sample', -1)) < c.cutoff_sample:
            return fail('eog_calibration_not_yet_arrived')
    elif c.cutoff_sample > start or (path_start is not None and c.cutoff_sample > int(path_start)):
        return fail('eog_calibration_from_future')
    hard_reasons = [r for r in q['reasons'] if r != 'amplitude']
    if hard_reasons or (not q.get('valid', False) and not q['reasons']):
        return fail('eog_source_quality_rejected')
    adaptive = w.get('eog_adaptive_state') if requested_method in ('causal_rls','causal_reference_rls') else None
    selected_method = method or configuration.eog_policy(cfg).get('method', 'frozen')
    if selected_method not in ('frozen', 'frozen_reference_ridge', 'causal_rls', 'causal_reference_rls'):
        return fail('eog_unknown_processing_method')
    if selected_method in ('causal_rls', 'causal_reference_rls') and adaptive is None and not reference:
        return fail('eog_adaptive_stream_unavailable')
    if adaptive is not None:
        detail.update(processing_version=ADAPTIVE_PROCESSING_VERSION,
                      method='causal_reference_rls', adaptive=deepcopy(adaptive),
                      coefficients_frozen=False)
        if not adaptive.get('available', False):
            return fail('eog_adaptive_unavailable: ' + adaptive.get('reason', 'unknown'))
        cleaned = np.asarray(w.get('eog_adaptive_x', []), dtype=np.float64)
        if cleaned.shape != x.shape: return fail('eog_invalid_adaptive_shape')
    else:
        cleaned = c.clean(x, e)
    if not np.isfinite(cleaned).all(): return fail('eog_nonfinite_output')
    removed = x - cleaned
    # Demeaned RMS avoids large electrode offsets hiding an excessive removal.
    removed_rms = np.sqrt(np.mean((removed - removed.mean(axis=0)) ** 2, axis=0))
    pre_rms = x.std(axis=0)
    ratio = float(np.max(removed_rms / np.maximum(pre_rms, 1e-12)))
    metrics = _measurements(cleaned, 256, cfg['qc'])
    post_corr = _eog_correlation(cleaned, e)
    metrics['frontal_eog_correlation'] = post_corr
    q['reasons'] = []
    p = configuration.eog_policy(cfg)
    if metrics['peak_to_peak_uv'] > cfg['qc']['peak_to_peak_uv']: q['reasons'].append('amplitude')
    if metrics['flat_channel_fraction'] > 0: q['reasons'].append('processed_flat_required_channel')
    if metrics['jump_fraction'] > 0.001: q['reasons'].append('processed_abrupt_jump')
    if post_corr > p['residual_correlation_max']: q['reasons'].append('residual_eog_correlation')
    if ratio > p['removed_rms_ratio_max']: q['reasons'].append('excessive_eog_removal')
    q.update(valid=not q['reasons'], processed_metrics=metrics,
             processed_peak_to_peak_uv=metrics['peak_to_peak_uv'],
             frontal_eog_correlation=post_corr,
             processing_profile='P1')
    detail.update(applied=True, applied_profile='P1',
                  status='cleaned' if q['valid'] else 'cleaned_rejected',
                  reason=('accepted causal reference RLS' if adaptive is not None else 'accepted frozen EOG regression') if q['valid'] else '; '.join(q['reasons']),
                  reasons=list(q['reasons']),
                  input_available=bool(q['valid'] and not reference),
                  post_frontal_eog_correlation=post_corr,
                  post_peak_to_peak_uv=metrics['peak_to_peak_uv'],
                  removed_rms_ratio=_finite_number(ratio),
                  removed_rms_uv=_finite_number(float(removed_rms.max())))
    result.update(x=cleaned, profile='P1')
    return result
