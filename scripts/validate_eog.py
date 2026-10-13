"""Controlled EOG-injection validation with a known clean EEG reference.

Run from the project: python scripts/validate_eog.py --output <new-report.json>
Synthetic results test numerical recovery, not natural-data artifact truth.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
from scipy import signal, integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_agent.config import load_config, write_json, eog_policy, file_hash
from eeg_agent.eog import (PROCESSING_VERSION, ADAPTIVE_PROCESSING_VERSION,
                           AdaptiveEOGRegression, fit_eog_calibration, process_window)
from eeg_agent.signal import quality


def controlled_signals(seconds, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 256)) / 256.0
    phase = rng.uniform(0, 2 * np.pi, (2, 30))
    clean = (8 * np.sin(2 * np.pi * 10 * t[:, None] + phase[0]) +
             3 * np.sin(2 * np.pi * 20 * t[:, None] + phase[1]) +
             rng.normal(0, 0.35, (len(t), 30)))
    eye = np.column_stack([8 * np.sin(2 * np.pi * 1.1 * t + 0.3),
                           7 * np.sin(2 * np.pi * 1.7 * t + 1.2)])
    for center in np.arange(0.6, seconds, 2.7):
        eye[:, 0] += 30 * np.exp(-0.5 * ((t - center) / 0.13) ** 2)
    for center in np.arange(1.1, seconds, 3.3):
        eye[:, 1] += 17 * np.exp(-0.5 * ((t - center) / 0.20) ** 2)
    # Synthetic zero-offset sources represent the AC signal after high-pass
    # filtering. The regression itself still uses only the prior EOG mean.
    return clean, eye - eye.mean(axis=0)


def _band_power(x, low, high):
    f, psd = signal.welch(x, fs=256, window='hann', nperseg=512,
                          noverlap=256, detrend='constant', axis=0)
    mask = (f >= low) & (f < high)
    return integrate.trapezoid(psd[mask], f[mask], axis=0)


def _rmse(x, reference):
    return float(np.sqrt(np.mean((x - reference) ** 2)))


def _snr(x, reference):
    return float(10 * np.log10(np.mean(reference ** 2) /
                              max(np.mean((x - reference) ** 2), 1e-30)))


def make_window(x, e, c, cfg):
    start = c.cutoff_sample
    raw = np.repeat(x, 4, axis=0)
    raw_e = np.repeat(e, 4, axis=0)
    return {'x': x, 'e': e, 'calibration': c, 'start_sample': start,
            'end_sample': start + 5120, 'path_start_sample': start,
            'quality': quality(x, e, raw, cfg, start + 5120, raw_eog=raw_e),
            'profile': 'P0'}


def validate(cfg=None):
    cfg = cfg or load_config()
    calibration_clean, calibration_eye = controlled_signals(60, 42)
    held_clean, held_eye = controlled_signals(5, 43)
    # Frontal channels receive stronger independent ocular contamination.
    mixing = np.column_stack([np.linspace(1.0, 0.2, 30),
                              np.linspace(-0.45, 0.35, 30)])
    prior = calibration_clean + calibration_eye @ mixing.T
    c = fit_eog_calibration(np.column_stack([prior, calibration_eye]), 60 * 1024, cfg)
    noisy = held_clean + held_eye @ mixing.T
    processed = process_window(make_window(noisy, held_eye, c, cfg), cfg, method='frozen')
    cleaned = processed['x']
    fidelity = {}
    for band, low, high in [('alpha', 8, 13), ('beta', 13, 30)]:
        before = _band_power(held_clean, low, high)
        after = _band_power(cleaned, low, high)
        error = np.abs(after / before - 1)
        fidelity[band] = {'mean_absolute_relative_error': float(error.mean()),
                          'maximum_absolute_relative_error': float(error.max())}
    # Negative control: eye channels remain active, but EEG has no ocular mix
    # in either its separate prior calibration or the held-out clean window.
    negative_c = fit_eog_calibration(np.column_stack([calibration_clean, calibration_eye]),
                                     60 * 1024, cfg)
    negative = process_window(make_window(held_clean, held_eye, negative_c, cfg), cfg, method='frozen')
    no_mix_error = _rmse(negative['x'], held_clean) / float(np.sqrt(np.mean(held_clean ** 2)))
    no_eye = process_window(make_window(held_clean, np.tile(c.mean, (1280, 1)), c, cfg), cfg, method='frozen')
    failures = {}
    too_short = fit_eog_calibration(np.zeros((1280, 32)), 60 * 1024, cfg)
    flat_eye = fit_eog_calibration(np.column_stack([calibration_clean, np.zeros_like(calibration_eye)]), 60 * 1024, cfg)
    collinear = fit_eog_calibration(np.column_stack([calibration_clean, calibration_eye[:, 0], calibration_eye[:, 0]]), 60 * 1024, cfg)
    for name, failed in [('insufficient', too_short), ('flat_EOG', flat_eye), ('collinear_EOG', collinear)]:
        failures[name] = {'available': failed.available, 'reason': failed.reason}
    rmse_before, rmse_after = _rmse(noisy, held_clean), _rmse(cleaned, held_clean)
    snr_before, snr_after = _snr(noisy, held_clean), _snr(cleaned, held_clean)
    checks = {
        'calibration_available': c.available,
        'held_window_accepted': processed['quality']['valid'],
        'artifact_rmse_reduction_at_least_90_percent': rmse_after <= 0.1 * rmse_before,
        'snr_improvement_at_least_20_db': snr_after - snr_before >= 20,
        'alpha_and_beta_power_error_at_most_2_percent': all(v['maximum_absolute_relative_error'] <= 0.02 for v in fidelity.values()),
        'negative_control_rms_change_at_most_1_percent': no_mix_error <= 0.01,
        'zero_centered_eog_exactly_preserves_eeg': bool(np.array_equal(no_eye['x'], held_clean)),
        'degenerate_calibrations_rejected': all(not v['available'] for v in failures.values()),
    }
    return {'processing_version': PROCESSING_VERSION, 'policy': eog_policy(cfg),
            'seed_calibration': 42, 'seed_held_out': 43, 'calibration_seconds': 60,
            'held_out_seconds': 5, 'clean_reference_known': True,
            'scope': 'controlled linear EOG injection; no natural artifact or symptom ground truth',
            'artifact_rmse_uv_before': rmse_before, 'artifact_rmse_uv_after': rmse_after,
            'artifact_rmse_reduction_fraction': 1 - rmse_after / rmse_before,
            'snr_db_before': snr_before, 'snr_db_after': snr_after,
            'snr_improvement_db': snr_after - snr_before,
            'band_power_fidelity': fidelity,
            'negative_control_relative_rms_change': no_mix_error,
            'cleaning': processed['eog_cleaning'],
            'calibration_failures': failures, 'checks': checks,
            'passed': all(checks.values())}


def validate_adaptive(cfg=None):
    """Held-out clean-reference checks plus an explicit neural-loss control."""
    cfg = cfg or load_config()
    prior_clean, prior_eye = controlled_signals(60, 42)
    held_clean, held_eye = controlled_signals(60, 43)
    mixing = np.column_stack([np.linspace(1.0, 0.2, 30),
                              np.linspace(-0.45, 0.35, 30)])
    c = fit_eog_calibration(np.column_stack([prior_clean + prior_eye @ mixing.T, prior_eye]),
                            60 * 1024, cfg)
    # A predetermined 80% step in electrode mixing tests adaptation. It is
    # not estimated from natural subject 1 or from a classification label.
    noisy = held_clean + held_eye @ (mixing * 1.8).T
    cleaner = AdaptiveEOGRegression(c, cfg)
    output = cleaner.clean_samples(noisy, held_eye)
    tail = slice(20 * 256, None)
    fidelity = {}
    for name, low, high in [('alpha', 8, 13), ('beta', 13, 30)]:
        error = np.abs(_band_power(output, low, high) / _band_power(held_clean, low, high) - 1)
        fidelity[name] = {'mean_absolute_relative_error': float(error.mean()),
                          'maximum_absolute_relative_error': float(error.max())}
    negative_c = fit_eog_calibration(np.column_stack([prior_clean, prior_eye]), 60 * 1024, cfg)
    negative = AdaptiveEOGRegression(negative_c, cfg).clean_samples(held_clean, held_eye)
    negative_rms = _rmse(negative, held_clean) / float(np.sqrt(np.mean(held_clean ** 2)))
    # A neural component correlated with the ocular reference is not
    # identifiable from those two sources alone. Expose the loss rather
    # than calling correlation decrease clean-EEG fidelity.
    prior_correlated_neural = prior_clean + 0.12 * prior_eye[:, :1]
    held_correlated_neural = held_clean + 0.12 * held_eye[:, :1]
    correlated_c = fit_eog_calibration(np.column_stack([
        prior_correlated_neural + prior_eye @ mixing.T, prior_eye]), 60 * 1024, cfg)
    correlated = AdaptiveEOGRegression(correlated_c, cfg).clean_samples(
        held_correlated_neural + held_eye @ mixing.T, held_eye)
    low_ratio = float(np.mean(_band_power(correlated, .5, 4) /
                              _band_power(held_correlated_neural, .5, 4)))
    all_before, all_after = _rmse(noisy, held_clean), _rmse(output, held_clean)
    tail_before, tail_after = _rmse(noisy[tail], held_clean[tail]), _rmse(output[tail], held_clean[tail])
    tail_snr = _snr(output[tail], held_clean[tail]) - _snr(noisy[tail], held_clean[tail])
    checks = {
        'calibration_and_adaptive_state_available': bool(c.available and cleaner.available),
        'drift_entire_recording_rmse_reduction_at_least_75_percent': all_after <= .25 * all_before,
        'drift_after_first_20_seconds_rmse_reduction_at_least_90_percent': tail_after <= .10 * tail_before,
        'drift_after_first_20_seconds_snr_improvement_at_least_20db': tail_snr >= 20,
        'alpha_beta_power_error_at_most_2_percent': all(v['maximum_absolute_relative_error'] <= .02 for v in fidelity.values()),
        'no_mix_negative_control_rms_change_at_most_1_percent': negative_rms <= .01,
        'correlated_neural_loss_explicitly_demonstrated': low_ratio < .5,
    }
    return {'processing_version': ADAPTIVE_PROCESSING_VERSION, 'policy': eog_policy(cfg),
            'seed_calibration': 42, 'seed_held_out': 43, 'calibration_seconds': 60,
            'held_out_seconds': 60, 'mixing_step_factor': 1.8,
            'scope': 'causal unlabeled subject adaptation; controlled injection, not natural artifact truth',
            'prediction_before_update': True, 'labels_used': False,
            'entire_recording': {'artifact_rmse_uv_before': all_before, 'artifact_rmse_uv_after': all_after,
                                 'artifact_rmse_reduction_fraction': 1 - all_after / all_before,
                                 'snr_improvement_db': _snr(output, held_clean) - _snr(noisy, held_clean)},
            'after_first_20_seconds': {'analysis_seconds': 40,
                                       'artifact_rmse_uv_before': tail_before, 'artifact_rmse_uv_after': tail_after,
                                       'artifact_rmse_reduction_fraction': 1 - tail_after / tail_before,
                                       'snr_improvement_db': tail_snr},
            'band_power_fidelity': fidelity,
            'no_mix_negative_control_relative_rms_change': negative_rms,
            'correlated_neural_limit': {
                'description': 'If genuine low-frequency neural activity shares the EOG reference, regression also removes that component.',
                'low_band_power_retained_fraction': low_ratio,
                'clean_reference_rmse_uv_after': _rmse(correlated, held_correlated_neural),
                'consequence': 'Do not interpret correlation decrease as proof all EEG is preserved; no perfect ocular separation claim.'},
            'adaptive': cleaner.summary(), 'checks': checks, 'passed': all(checks.values())}


def measure_natural(cfg, subject=1, seconds=130):
    """Actual arrived-only P0/frozen/adaptive comparison; no clean reference."""
    from eeg_agent.data import CDTSource
    from eeg_agent.signal import StreamPipeline
    source = CDTSource(cfg['data_root'], subject)
    stream_cfg = {**cfg, 'classification_profile': 'P1'}
    rows = []

    def arrived(w):
        original = {key: value for key, value in w.items()
                    if key not in ('eog_adaptive_x', 'eog_adaptive_state')}
        frozen = process_window(original, stream_cfg, method='frozen')
        adaptive = process_window(w, stream_cfg, method='causal_rls')
        a, f = adaptive['eog_cleaning'], frozen['eog_cleaning']
        rows.append({'start_sample': w['start_sample'], 'end_sample': w['end_sample'],
                     'start_seconds': w['start_sample'] / 1024,
                     'P0_quality_valid': bool(w['quality']['valid']),
                     'P0_frontal_eog_correlation': a['pre_frontal_eog_correlation'],
                     'frozen_frontal_eog_correlation': f['post_frontal_eog_correlation'],
                     'adaptive_frontal_eog_correlation': a['post_frontal_eog_correlation'],
                     'frozen_quality_valid': bool(frozen['quality']['valid']),
                     'adaptive_quality_valid': bool(adaptive['quality']['valid']),
                     'adaptive_rejection_reasons': adaptive['quality']['reasons'],
                     'adaptive_removed_rms_ratio': a['removed_rms_ratio'],
                     'adaptive': a.get('adaptive'),
                     'calibration': a['calibration']})

    stream = StreamPipeline(source.meta['channels'], stream_cfg, arrived, lambda _: None)
    for start, raw, events in source.chunks(256, seconds):
        stream.feed(start, raw, events)
    stream.eof()
    summaries = {}
    for name, field in [('P0', 'P0_frontal_eog_correlation'),
                        ('frozen', 'frozen_frontal_eog_correlation'),
                        ('adaptive', 'adaptive_frontal_eog_correlation')]:
        values = np.asarray([r[field] for r in rows if r[field] is not None])
        summaries[name] = {'valid_windows': sum(r[name + '_quality_valid'] for r in rows),
                           'correlation_mean': float(values.mean()) if len(values) else None,
                           'correlation_median': float(np.median(values)) if len(values) else None,
                           'correlation_minimum': float(values.min()) if len(values) else None,
                           'correlation_maximum': float(values.max()) if len(values) else None}
    return {'processing_version': ADAPTIVE_PROCESSING_VERSION, 'policy': eog_policy(cfg),
            'local_subject_reference': int(subject), 'source_seconds': float(seconds),
            'raw_sha256': file_hash(source.path),
            'processing_code_sha256': file_hash(Path(__file__).resolve().parents[1] / 'eeg_agent/eog.py'),
            'scope': 'actual CDT numerical replay; no labels, model, cloud API or clean artifact reference',
            'interpretation': 'Correlation decrease is a measurement only and cannot prove ocular truth or preservation of all neural activity.',
            'total_complete_windows': len(rows), 'summary': summaries, 'windows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--method', choices=['frozen', 'adaptive'], default='adaptive')
    parser.add_argument('--natural-subject', type=int)
    parser.add_argument('--natural-seconds', type=float, default=130)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new output path to preserve earlier validation evidence')
    report = (validate_adaptive if args.method == 'adaptive' else validate)(load_config(args.config))
    if args.natural_subject is not None:
        report['natural_measurements'] = measure_natural(load_config(args.config), args.natural_subject, args.natural_seconds)
    report['processing_code_sha256'] = file_hash(Path(__file__).resolve().parents[1] / 'eeg_agent/eog.py')
    write_json(output, report)
    print('Controlled EOG validation:', 'PASS' if report['passed'] else 'FAIL')
    metrics = report.get('entire_recording', report)
    print('RMSE reduction: {:.3%}; SNR improvement: {:.2f} dB'.format(
        metrics['artifact_rmse_reduction_fraction'], metrics['snr_improvement_db']))
    print('Report:', output.resolve())
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
