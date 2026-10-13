"""Raw conversation inference must never turn recording onset into a baseline."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from eeg_agent import raw_model


def test_raw_model_never_reads_or_sends_recording_onset_reference(tmp_path, monkeypatch):
    worker_root = tmp_path / 'worker'
    head = worker_root / 'outputs/vrms_model_ablation/v3_20261008/model_results/checkpoints/outer_subject_10_femba_frozen_mil.pt'
    head.parent.mkdir(parents=True)
    head.touch()
    monkeypatch.setattr(raw_model, 'ROOT', worker_root)
    windows = np.random.default_rng(8).normal(size=(3, 30, 1280)).astype(np.float32)
    segment = {'ordinal': 1, 'start_seconds': 20, 'end_seconds': 35,
               'start_clock': '00:20.000', 'end_clock': '00:35.000', 'complete': True,
               'stats': {'window_coverage': 1}}
    record = {'signal_sha256': 'raw-waveform-hash', 'task_count': 1,
              'model_compatible': True, 'segments': [segment]}
    repository = SimpleNamespace(root=tmp_path / 'analysis',
        cfg={'minimum_path_windows': 3, 'minimum_path_coverage': .8},
        model_inputs=lambda _: [(windows, [20, 25, 30])],
        model_baseline=lambda _: pytest.fail('Recording onset cannot be used as model baseline'))
    observed = []

    def run_worker(command, **kwargs):
        folders = list((repository.root / 'inference').iterdir())
        assert len(folders) == 1
        folder = folders[0]
        request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
        observed.append((request, folder))
        prediction = {'ordinal': 1, 'predicted_class': 'High',
                      'numeric_fusion': {'raw_combined_score': .6, 'reference_available': False}}
        result = {'available': True, 'segments': [prediction],
                  'signal_sha256': record['signal_sha256'], 'fresh_raw_encoding': True,
                  'model_sha256': 'head', 'encoder_sha256': 'encoder', 'normalization_sha256': 'training-normalization',
                  'numeric_first_sha256': 'first', 'numeric_second_sha256': 'second', 'numeric_selector_sha256': 'selector'}
        return SimpleNamespace(stdout=json.dumps(result).encode('utf-8'))

    monkeypatch.setattr(raw_model.subprocess, 'run', run_worker)
    result = raw_model.assess(repository, 'subject-10', {'recording': record})
    assert len(observed) == 1
    request, folder = observed[0]
    assert request['baseline_file'] is False
    assert not (folder / 'baseline.npy').exists()
    npt.assert_array_equal(np.load(folder / 'segment-001.npy'), windows)
    assert result['available'] and result['classifier_available']
    assert result['measurements']['segments'][0]['numeric_fusion']['reference_available'] is False
    assert result['normalization_sha256'] == 'training-normalization'


def test_frozen_numeric_missing_reference_keeps_missing_slots_and_within_path_changes(monkeypatch):
    # Import the worker's feature construction without loading any checkpoints,
    # training records, questionnaire files, CUDA models, or API clients.
    monkeypatch.syspath_prepend(str(raw_model.ROOT))
    from vrms_cloud import raw_numeric
    numeric = object.__new__(raw_numeric.FrozenNumeric)
    captured = {}
    first_names = ('femba_features', 'spectral', 'spatial_covariance')
    second_names = ('femba_mean', 'femba_temporal', 'spectral_compact', 'filterbank_csp')

    class Identity:
        def transform(self, values):
            return np.asarray(values)

    class Selector:
        def decision_function(self, values):
            assert np.isfinite(values).all()
            return np.array([0.])

    numeric.first = {'tools': {name: {'model': name} for name in first_names}}
    numeric.first['tools']['case_retrieval'] = {
        'scaler': Identity(), 'reference_features': np.array([[1., 2., 3., 4.], [4., 3., 2., 1.]]),
        'reference_labels': np.array([0, 1]), 'reference_subjects': np.array([1, 2]), 'config': (2, False)}
    numeric.second = {'tools': {name: {'model': name} for name in second_names}}
    numeric.selector = {'feature_names': ['baseline'] + list(first_names) + list(second_names) + ['case_retrieval'],
                        'model': Selector()}

    def score(name, values):
        captured[name] = np.array(values[0])
        return np.array([.2])

    monkeypatch.setattr(raw_numeric, 'predict_score', score)
    calls = []
    spectral = raw_numeric.spectral_features

    def tracked_spectrum(waveform):
        calls.append(np.array(waveform))
        return spectral(waveform)

    monkeypatch.setattr(raw_numeric, 'spectral_features', tracked_spectrum)
    windows = np.random.default_rng(12).normal(size=(3, 30, 1280))
    windows *= np.array([1., 2., 4.])[:, None, None]
    features = np.arange(12, dtype=float).reshape(3, 4)
    result = numeric.predict(windows, features, .65, baseline=None)

    # Exactly the three path windows are measured. No first window or task
    # prefix is silently measured a second time as an EEG reference.
    assert len(calls) == len(windows)
    for actual, expected in zip(calls, windows):
        npt.assert_array_equal(actual, expected)
    npt.assert_array_equal(captured['spectral'][-121:], np.zeros(121))
    npt.assert_array_equal(captured['spectral_compact'][-25:], np.zeros(25))
    absolute = np.array([spectral(window)[0] for window in windows])
    npt.assert_allclose(captured['spectral'][480:600], absolute[-1, :120] - absolute[0, :120])
    npt.assert_allclose(captured['femba_temporal'][-4:], features[-1] - features[0])
    assert result['reference_available'] is False
    assert result['predicted_class'] == 'High'
    assert result['raw_combined_score'] == .5


@pytest.mark.parametrize('baseline_flag', [True, 1, 'false'])
def test_worker_rejects_reference_request_before_loading_models(tmp_path, monkeypatch, baseline_flag):
    monkeypatch.syspath_prepend(str(raw_model.ROOT))
    from vrms_cloud import raw_inference
    request_path = (tmp_path / 'request.json').resolve()
    request = {'fold': 10, 'signal_sha256': 'raw', 'segments': [], 'baseline_file': baseline_flag}

    def read_request_only(path):
        assert Path(path).resolve() == request_path, 'Untrusted reference request reached model loading'
        return request

    monkeypatch.setattr(raw_inference, 'read', read_request_only)
    with pytest.raises(ValueError):
        raw_inference.infer(request_path)
