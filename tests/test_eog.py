import json

import numpy as np

from eeg_agent.config import EEG_CHANNELS, load_config
from eeg_agent.eog import AdaptiveEOGRegression, fit_eog_calibration, process_window
from eeg_agent.signal import Calibration, StreamPipeline, quality


def _prior(seconds=60, seed=4):
    rng = np.random.default_rng(seed)
    n = int(seconds * 256)
    eye = rng.normal(size=(n, 2)) * 8
    clean = rng.normal(size=(n, 30)) * 3
    mixing = np.column_stack([np.linspace(0.8, 0.2, 30),
                              np.linspace(-0.2, 0.4, 30)])
    return clean + (eye - eye.mean(axis=0)) @ mixing.T, eye, mixing


def _window(x, e, c, cfg, start=None):
    start = c.cutoff_sample if start is None else start
    return {'x': x, 'e': e, 'profile': 'P0', 'calibration': c,
            'start_sample': start, 'end_sample': start + 5120,
            'path_start_sample': start,
            'quality': quality(x, e, np.repeat(x, 4, axis=0), cfg, start + 5120)}


def test_known_clean_heldout_recovery_and_negative_control():
    # A separate held-out recording, not fitting and measuring on the same EEG.
    from scripts.validate_eog import validate
    report = validate()
    assert report['passed'], report['checks']
    assert report['artifact_rmse_reduction_fraction'] > .90
    assert report['snr_improvement_db'] > 20
    assert report['negative_control_relative_rms_change'] < .01


def test_calibration_checks_each_baseline_window_not_disjoint_ranges():
    cfg = load_config()
    x, e, _ = _prior(10)
    x[:1280] += 1000
    x[1280:] -= 1000
    # QC passed these two separate windows; their different DC levels must not
    # be interpreted as a 2,000 uV within-window excursion.
    c = fit_eog_calibration(np.column_stack([x, e]), 20480, cfg)
    assert c.available, c.reason
    x[100, 0] += 1000
    bad = fit_eog_calibration(np.column_stack([x, e]), 20480, cfg)
    assert not bad.available and bad.reason == 'invalid baseline amplitude'


def test_only_first_maximum_complete_prior_windows_enter_fit():
    cfg = load_config()
    x, e, _ = _prior(80)
    baseline = np.column_stack([x, e])
    c1 = fit_eog_calibration(baseline, 100000, cfg)
    baseline[60 * 256:] = np.nan
    c2 = fit_eog_calibration(baseline, 100000, cfg)
    assert c1.available and c2.available and c1.samples == 60 * 256
    np.testing.assert_array_equal(c1.B, c2.B)
    np.testing.assert_array_equal(c1.mean, c2.mean)


def test_degenerate_and_unstable_calibration_are_not_available():
    cfg = load_config()
    x, e, _ = _prior(10)
    assert not fit_eog_calibration(np.zeros((2560, 32)), 20480, cfg).available
    assert not fit_eog_calibration(np.column_stack([x, e[:, 0], e[:, 0]]), 20480, cfg).available
    assert not fit_eog_calibration(np.column_stack([x, e])[:1280], 20480, cfg).available
    nonfinite = np.column_stack([x, e]); nonfinite[100, 0] = np.inf
    assert not fit_eog_calibration(nonfinite, 20480, cfg).available
    assert not fit_eog_calibration(np.zeros((2560, 31)), 20480, cfg).available
    e *= .025
    x = e[:, :1] * np.ones((1, 30)) * 100
    bad = fit_eog_calibration(np.column_stack([x, e]), 20480, cfg)
    assert not bad.available and bad.reason == 'unstable regression'


def test_failed_p1_never_silently_becomes_classifier_p0():
    cfg = load_config()
    x, e, _ = _prior(5)
    w = _window(x, e, Calibration(), cfg, start=20480)
    processed = process_window(w, cfg, method='frozen')
    assert not processed['quality']['valid']
    assert processed['profile'] is None
    assert not processed['eog_cleaning']['applied']
    assert not processed['eog_cleaning']['input_available']
    assert w['quality']['valid'] and w['profile'] == 'P0'
    np.testing.assert_array_equal(processed['x'], w['x'])


def test_calibration_from_future_rejected_and_reference_reprocessing_explicit():
    cfg = load_config()
    x, e, _ = _prior(10)
    c = fit_eog_calibration(np.column_stack([x, e]), 20480, cfg)
    w = _window(x[:1280], e[:1280], c, cfg, start=10240)
    assert 'eog_calibration_from_future' in process_window(w, cfg, method='frozen')['quality']['reasons']
    w.update(calibration_reference=True, observed_end_sample=20479)
    assert 'eog_calibration_not_yet_arrived' in process_window(w, cfg, method='frozen')['quality']['reasons']
    w['observed_end_sample'] = 20480
    reference = process_window(w, cfg, method='frozen')
    assert reference['quality']['valid']
    assert reference['eog_cleaning']['applied']
    assert not reference['eog_cleaning']['input_available']


def test_raw_guards_survive_cleaning_and_json_summary_contains_no_arrays():
    cfg = load_config()
    x, e, _ = _prior(10)
    c = fit_eog_calibration(np.column_stack([x, e]), 20480, cfg)
    w = _window(x[:1280], e[:1280], c, cfg)
    w['quality'].update(valid=False, reasons=['raw_range_exceeded'])
    out = process_window(w, cfg, method='frozen')
    assert not out['quality']['valid']
    assert 'raw_range_exceeded' in out['quality']['reasons']
    assert not out['eog_cleaning']['applied']
    summary = json.dumps(out['eog_cleaning'], allow_nan=False)
    assert 'raw_range' not in summary  # failure reason refers to the source QC
    assert '"B"' not in summary and '"mean"' not in summary


def test_bad_residual_flat_and_excessive_removal_reject_without_refitting():
    cfg = load_config()
    x, e, mixing = _prior(10)
    c = fit_eog_calibration(np.column_stack([x, e]), 20480, cfg)
    rng = np.random.default_rng(17)
    eye = rng.normal(size=(1280, 2)) * 8
    clean = rng.normal(size=(1280, 30)) * 3
    before = c.B.copy()
    residual = process_window(_window(clean + 4 * eye @ mixing.T, eye, c, cfg), cfg, method='frozen')
    assert 'residual_eog_correlation' in residual['quality']['reasons']
    excessive = process_window(_window(clean, eye, c, cfg), cfg, method='frozen')
    assert 'excessive_eog_removal' in excessive['quality']['reasons']
    flat = process_window(_window((eye - c.mean) @ c.B.T, eye, c, cfg), cfg, method='frozen')
    assert 'processed_flat_required_channel' in flat['quality']['reasons']
    np.testing.assert_array_equal(c.B, before)


def test_post_clean_amplitude_can_recover_without_mutating_raw_quality():
    cfg = load_config()
    x, e, mixing = _prior(10)
    c = fit_eog_calibration(np.column_stack([x, e]), 20480, cfg)
    rng = np.random.default_rng(42)
    eye = rng.normal(size=(1280, 2)) * 130
    clean = rng.normal(size=(1280, 30)) * 3
    # Raw input contract guardrail remains within 5,000uV; processed-only
    # 500uV amplitude is deliberately exceeded by a known ocular component.
    noisy = clean + (eye - c.mean) @ c.B.T
    w = _window(noisy, eye, c, cfg)
    # A smooth real signal has no raw jumps; focus this numerical guard test
    # on the processed amplitude that can be recovered by exact regression.
    w['quality'].update(valid=False, reasons=['amplitude'])
    out = process_window(w, cfg, method='frozen')
    assert out['quality']['valid'] and out['eog_cleaning']['input_available']
    np.testing.assert_allclose(out['x'], clean, atol=1e-12)
    assert w['quality']['reasons'] == ['amplitude']


def test_complete_stream_partition_equivalence_and_future_isolation():
    cfg = load_config(); cfg['classification_profile'] = 'P1'
    n = 45 * 1024; t = np.arange(n) / 1024
    phase = np.linspace(0, 2, 30)
    eye = np.column_stack([15 * np.sin(2 * np.pi * 1.1 * t),
                           12 * np.sin(2 * np.pi * 1.7 * t)])
    eeg = (8 * np.sin(2 * np.pi * 10 * t[:, None] + phase) +
           4 * np.sin(2 * np.pi * 20 * t[:, None] - phase) +
           eye @ np.column_stack([np.linspace(.8, .2, 30), np.linspace(-.2, .4, 30)]).T)
    raw = np.column_stack([eeg, np.zeros((n, 2)), eye, np.zeros((n, 3))])
    channels = EEG_CHANNELS + ['M1', 'M2', 'VEOG', 'HEOG', 'ECG', 'EGG', 'Trigger']
    events = [(20 * 1024, 20), (45 * 1024, 22)]

    def run(values, sizes):
        windows = []
        pipe = StreamPipeline(channels, cfg, lambda w: windows.append(process_window(w, cfg, method='frozen')), lambda _: None)
        start = 0; k = 0
        while start < len(values):
            end = min(len(values), start + sizes[k % len(sizes)])
            arrived = [(s, mark) for s, mark in events if start <= s < end or s == end == len(values)]
            pipe.feed(start, values[start:end], arrived)
            start = end; k += 1
        return windows

    a = run(raw, [256]); b = run(raw, [17, 513, 127])
    assert len(a) == len(b) == 5
    for wa, wb in zip(a, b):
        assert wa['quality']['valid'] and wb['quality']['valid']
        np.testing.assert_array_equal(wa['x'], wb['x'])
        assert wa['eog_cleaning'] == wb['eog_cleaning']
    future = raw.copy(); future[35 * 1024:] *= 4
    changed = run(future, [256])
    for before, after in zip(a, changed):
        if before['end_sample'] <= 35 * 1024:
            np.testing.assert_array_equal(before['x'], after['x'])
            assert before['eog_cleaning'] == after['eog_cleaning']


def test_adaptive_drift_fidelity_and_correlated_neural_limit():
    from scripts.validate_eog import validate_adaptive
    result = validate_adaptive()
    assert result['passed'], result['checks']
    assert result['no_mix_negative_control_relative_rms_change'] < .01
    assert result['correlated_neural_limit']['low_band_power_retained_fraction'] < .5


def test_adaptive_predicts_before_update_and_is_partition_invariant():
    cfg = load_config()
    x, e, mixing = _prior(10)
    c = fit_eog_calibration(np.column_stack([x, e]), 20480, cfg)
    noisy, eyes, _ = _prior(10, seed=19)
    a = AdaptiveEOGRegression(c, cfg)
    expected = a.clean_samples(noisy, eyes)
    b = AdaptiveEOGRegression(c, cfg)
    parts = []
    for start in range(0, len(noisy), 17):
        parts.append(b.clean_samples(noisy[start:start + 17], eyes[start:start + 17]))
    np.testing.assert_array_equal(expected, np.concatenate(parts))
    assert a.summary() == b.summary()
    changed = noisy.copy(); changed[500:] *= 2
    future = AdaptiveEOGRegression(c, cfg).clean_samples(changed, eyes)
    np.testing.assert_array_equal(expected[:500], future[:500])
    # The sample at index 500 contributes to the update only after its own
    # ocular prediction. Its input change therefore cannot change removal.
    np.testing.assert_allclose(noisy[500] - expected[500], changed[500] - future[500], atol=1e-14)


def test_adaptive_invalid_and_extreme_samples_do_not_poison_next_state():
    cfg = load_config()
    x, e, _ = _prior(10)
    c = fit_eog_calibration(np.column_stack([x, e]), 20480, cfg)
    clean, eyes, _ = _prior(5, seed=19)
    validity = np.ones(len(clean), dtype=bool); validity[500] = False
    a = AdaptiveEOGRegression(c, cfg)
    b = AdaptiveEOGRegression(c, cfg)
    altered = clean.copy(); altered[500] = 1e6
    expected = a.clean_samples(clean, eyes, valid=validity)
    actual = b.clean_samples(altered, eyes)
    np.testing.assert_array_equal(expected[501:], actual[501:])
    np.testing.assert_array_equal(a.B, b.B)
    assert a.samples == b.samples == 1279 and b.skipped_updates == 1
    b.reset()
    fresh = AdaptiveEOGRegression(c, cfg)
    np.testing.assert_array_equal(b.B, fresh.B)
    np.testing.assert_array_equal(b.inverse, fresh.inverse)
    np.testing.assert_array_equal(b.eeg_mean, fresh.eeg_mean)
    assert b.summary() == fresh.summary()


def test_adaptive_contract_requires_continuous_stream_output():
    cfg = load_config()
    x, e, _ = _prior(10)
    c = fit_eog_calibration(np.column_stack([x, e]), 20480, cfg)
    w = _window(x[:1280], e[:1280], c, cfg)
    out = process_window(w, cfg, method='causal_rls')
    assert not out['quality']['valid']
    assert 'eog_adaptive_stream_unavailable' in out['quality']['reasons']
    assert not out['eog_cleaning']['input_available']
