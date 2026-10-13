"""Measure actual spectral changes without symptom labels or rest assumptions."""
import copy
import hashlib
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eeg_agent import semantic_observations as observations
from eeg_agent.online.features import spectral_features


def waveform(alpha_amplitude=1.):
    time = np.arange(1280) / 256.
    value = (np.sin(2 * np.pi * 6 * time) +
             alpha_amplitude * np.sin(2 * np.pi * 10 * time) +
             .75 * np.sin(2 * np.pi * 20 * time))
    return np.tile(value, (30, 1)).astype(np.float32)


def test_real_alpha_change_does_not_become_a_rest_or_class_prediction():
    windows = np.stack([waveform(2.)] * 2 + [waveform()] * 2 + [waveform(.5)] * 2)
    measured = observations.observations_from_windows(windows, quality_valid=True)
    assert measured['evidence_status'] == 'available'
    assert measured['scope'] == 'whole_path_first_last_third_no_rest'
    assert measured['resting_reference_available'] is False
    items = {item['biomarker']: item for item in measured['items']}
    assert items['alpha']['value'] == pytest.approx(np.log(1 / 16), abs=1e-6)
    assert items['alpha']['direction'] == 'decrease'
    assert items['theta']['direction'] == items['beta']['direction'] == 'stable'
    assert set(items) == {'alpha', 'theta', 'beta'}
    for item in items.values():
        assert item['baseline'] == 'path_first_third'
        assert item['comparison'] == 'path_last_third'
        assert item['baseline_window_range'] == [0, 2]
        assert item['comparison_window_range'] == [4, 6]
        assert item['evidence_level'] == 'low'
        unhashed = {key: value for key, value in item.items() if key != 'content_sha256'}
        assert item['content_sha256'] == hashlib.sha256(json.dumps(
            unhashed, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert not any(key in json.dumps(measured) for key in ('predicted_class', 'subject_id', 'questionnaire'))


@pytest.mark.parametrize('count,complete,quality,reason', [
    (2, True, True, 'fewer_than_three_accepted_windows'),
    (3, False, True, 'whole_path_not_complete'),
    (3, True, False, 'window_quality_unavailable'),
])
def test_observations_require_real_quality_and_complete_sufficient_path(count, complete, quality, reason):
    result = observations.observations_from_windows(
        np.stack([waveform()] * count), quality_valid=quality, complete=complete)
    assert result['items'] == [] and result['evidence_status'] == 'unavailable'
    assert result['fallback_reason'] == reason


def test_zero_nonfinite_and_missing_cached_power_are_not_invented_evidence():
    result = observations.observations_from_windows(np.zeros((3, 30, 1280)), quality_valid=True)
    assert result['fallback_reason'] == 'insufficient_measured_band_power'
    assert result['items'] == []
    windows = np.stack([waveform()] * 3)
    windows[1, 0, 10] = np.nan
    assert observations.observations_from_windows(windows, True)['items'] == []
    measured = observations.SpectralObservations()
    measured.add(waveform(), None, quality_valid=True)
    assert measured.result()['fallback_reason'] == 'measured_band_powers_unavailable'


def test_cached_observations_reuse_powers_and_hash_actual_samples(monkeypatch):
    windows = [waveform(value) for value in (1., 2., 3.)]
    cached = [spectral_features(window)[1] for window in windows]
    expected = observations.observations_from_windows(np.stack(windows), quality_valid=True)
    from eeg_agent.online import features
    monkeypatch.setattr(features.signal, 'welch', lambda *args, **kw: pytest.fail('Recomputed cached Welch'))
    measured = observations.SpectralObservations()
    for window, powers in zip(windows, cached):
        measured.add(window, powers, quality_valid=True)
    assert measured.result() == expected
    assert expected['sample_sha256'] == hashlib.sha256(
        np.stack(windows).astype('<f4').tobytes(order='C')).hexdigest()
    assert expected['source_sha256'] == hashlib.sha256(
        np.stack(cached).astype('<f8').tobytes(order='C')).hexdigest()
    changed = observations.SpectralObservations()
    for window, powers in zip(windows, cached):
        changed.add(window * -1, powers, quality_valid=True)
    assert changed.result()['source_sha256'] == expected['source_sha256']
    assert changed.result()['sample_sha256'] != expected['sample_sha256']


def test_online_prefix_keeps_measured_state_unavailable_until_path_completion(monkeypatch):
    from eeg_agent.online import model
    calls = []
    def cached(window):
        calls.append(True)
        return {'powers': spectral_features(window)[1]}
    monkeypatch.setattr(model, 'cache_window', cached)
    encoder = SimpleNamespace(encode_windows=lambda value: torch.ones((1, 525)))
    numeric = SimpleNamespace(hashes={}, predict=lambda *args: {
        'predicted_class': 'Low', 'raw_combined_score': .4})
    owner = SimpleNamespace(lock=threading.RLock(), encoder=encoder, encoder_sha256='encoder')
    bundle = {'fold': 1, 'mean': torch.zeros(1, 30, 1), 'scale': torch.ones(1, 30, 1),
              'head': lambda value: torch.zeros(()), 'numeric': numeric,
              'model_sha256': 'model', 'normalization_sha256': 'normalization'}
    path = model.OnlinePathModel(owner, bundle)
    for amplitude in (1., 1., 2.):
        result = path.add(waveform(amplitude))
        assert result['eeg_observations']['items'] == []
    final = path.observations(complete=True)
    assert final['evidence_status'] == 'available'
    assert next(item for item in final['items'] if item['biomarker'] == 'alpha')['direction'] == 'increase'
    assert result['numeric_fusion']['predicted_class'] == 'Low'
    saved = copy.deepcopy(final)
    final['items'][0]['value'] = 999
    assert path.observations(complete=True) == saved
    assert len(calls) == 3


def test_session_log_stays_inside_session_and_preserves_safe_projection(tmp_path):
    directory = tmp_path / 'session'; directory.mkdir()
    evidence = {'schema': 'session_model_evidence_v1', 'prediction': {'final_class': 'Low'},
                'evidence_strength': 'low', 'sources': []}
    observations.append_session_evidence(directory, evidence)
    observations.append_session_evidence(directory, evidence)
    rows = [json.loads(row) for row in (directory / 'model_evidence.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows == [evidence, evidence]
    assert list(tmp_path.glob('**/model_evidence.jsonl')) == [directory / 'model_evidence.jsonl']
    with pytest.raises(ValueError, match='schema'):
        observations.append_session_evidence(directory, {'schema': 'static_graph'})


@pytest.mark.parametrize('damage', ['content', 'field', 'direction', 'band', 'hash', 'incomplete', 'window_range'])
def test_validation_rejects_modified_or_injected_observations(damage):
    measured = observations.observations_from_windows(np.stack([waveform()] * 3), True)
    item = measured['items'][0]
    if damage == 'content':
        item['value'] = 2.
    elif damage == 'field':
        item['subject_id'] = 'must-never-be-forwarded'
    elif damage == 'direction':
        item['direction'] = 'increase'
    elif damage == 'band':
        item['band_hz'] = [2, 50]
    elif damage == 'hash':
        item['source_sha256'] = 'not-actual-data'
    elif damage == 'incomplete':
        measured['whole_path_complete'] = False
    elif damage == 'window_range':
        item['baseline_window_range'] = [0, 2]
    with pytest.raises(ValueError):
        observations.validate_observations(measured)


def test_validated_observations_are_copied_and_missing_or_unavailable_are_empty():
    measured = observations.observations_from_windows(np.stack([waveform()] * 3), True)
    validated = observations.validate_observations(measured)
    assert validated == measured['items']
    validated[0]['value'] = 999
    assert observations.validate_observations(measured)[0]['value'] != 999
    assert observations.validate_observations(None) == []
    assert observations.validate_observations(observations.SpectralObservations().result()) == []


def semantic_prediction():
    """Exercise the real graph projection instead of copying arbitrary context."""
    from eeg_agent.state_graph_rag import explain, load_assets
    from tests.test_raw_independent_judge import prediction
    numeric = prediction()
    numeric['eeg_observations'] = observations.observations_from_windows(
        np.stack([waveform(2.), waveform(), waveform(.5)]), True)
    numeric['subject_id'] = 'private-identity'
    numeric['true_label'] = 'High'
    numeric['questionnaire'] = {'ssq': 99}
    numeric['current_raw_contrastive_context'] = {'state_evidence': explain(numeric, load_assets())}
    return numeric


def test_online_cloud_session_record_survives_until_session_jsonl(tmp_path, monkeypatch):
    from eeg_agent.online import cloud, runtime
    from tests.test_online_cloud import completed_path, ProviderDouble, KnowledgeDouble
    from eeg_agent.config import load_config
    monkeypatch.setattr(cloud, 'HierarchicalKnowledge', lambda cfg: KnowledgeDouble())
    monkeypatch.setattr(cloud, 'build_teaching_context', lambda cfg: {'available': False})
    monkeypatch.setattr(cloud, 'attach_default_rag', lambda fold, row, **kwargs: copy.deepcopy(row))
    monkeypatch.setattr(cloud, 'classify', lambda *args, **kwargs: {
        'available': True, 'final_class': 'Low', 'confidence': 'low', 'explanation': 'Measured evidence'})
    cfg = {**load_config(), 'output_root': str(tmp_path / 'offline')}
    reporter = cloud.OnlineCloudReporter(cfg, lambda *args: None, provider=ProviderDouble(),
                                        rag_policy={'rag_mode': 'semantic_graph'})
    result = reporter.report(completed_path(), semantic_prediction())
    assert result['session_model_evidence']['schema'] == 'session_model_evidence_v1'
    assert result['session_model_evidence']['prediction']['final_class'] == 'Low'
    assert result['session_model_evidence']['observed_eeg_changes']
    session = object.__new__(runtime.OnlineSession)
    session.lock = threading.RLock()
    session.directory = tmp_path / 'session'; session.directory.mkdir()
    session.paths = [dict(completed_path(), cloud={'status': 'pending'})]
    session.rag_policy = {'rag_mode': 'semantic_graph'}
    session.stop_event = threading.Event()
    session.cloud = {'pending': 1, 'completed': 0, 'failed': 0}
    session._emit = lambda *args: None
    session.persist = lambda: None
    session._apply_cloud(completed_path(), result)
    saved = json.loads((session.directory / 'model_evidence.jsonl').read_text(encoding='utf-8'))
    assert saved == result['session_model_evidence']
    serialized = json.dumps(saved)
    assert 'private-identity' not in serialized
    assert not any(word in serialized for word in ('true_label', 'questionnaire', 'numeric_fusion'))
    # A repeated settled cloud completion does not duplicate the audit row.
    session._apply_cloud(completed_path(), result)
    assert len((session.directory / 'model_evidence.jsonl').read_text(encoding='utf-8').splitlines()) == 1


def test_brain_logs_real_semantic_context_before_public_result_removes_it(tmp_path, monkeypatch):
    from eeg_agent import brain
    numeric = semantic_prediction()
    numeric.update(cloud_judgment={'available': True, 'final_class': 'Low', 'confidence': 'low'},
                   predicted_class='Low', ordinal=1)
    monkeypatch.setattr(brain, 'assess_raw_model', lambda *args, **kwargs: {
        'available': True, 'cloud_requested': True, 'measurements': {'segments': [numeric]}})
    monkeypatch.setattr(brain, 'execute_tool', lambda *args: {'available': False, 'reason': 'not requested'})
    session = object.__new__(brain.BrainSession)
    session.lock = threading.RLock()
    session.directory = tmp_path / 'brain'; session.directory.mkdir()
    session.backend, session.result, session.provider = 'mock', None, None
    session.recording_id = 'subject-10'
    session.rag_policy = {'rag_mode': 'semantic_graph'}
    session.manager = SimpleNamespace(recordings=object(), knowledge=SimpleNamespace(
        search=lambda *args, **kwargs: {'results': []}))
    session.emit = lambda *args: None
    session._specialist('vrms', '分析完整路径', {'recording': {}}, 1)
    saved = json.loads((session.directory / 'model_evidence.jsonl').read_text(encoding='utf-8'))
    assert saved['prediction']['final_class'] == 'Low'
    assert saved['observed_eeg_changes']
    serialized = json.dumps(saved)
    assert 'private-identity' not in serialized
    assert not any(word in serialized for word in ('true_label', 'questionnaire', 'numeric_fusion'))


@pytest.mark.parametrize('complete,scope,expected', [(True, 'task', True),
                                                   (False, 'task', False), (True, 'rest', False)])
def test_online_semantic_path_without_llm_only_audits_complete_tasks(tmp_path, monkeypatch, complete, scope, expected):
    from eeg_agent import default_rag
    from eeg_agent.online import runtime
    numeric = semantic_prediction()
    monkeypatch.setattr(default_rag, 'attach', lambda fold, row, **kwargs: copy.deepcopy(row))
    session = object.__new__(runtime.OnlineSession)
    session.lock = threading.RLock()
    session.directory = tmp_path / 'session'; session.directory.mkdir()
    session.rag_policy = {'rag_mode': 'semantic_graph'}
    session.stop_event = threading.Event()
    session._path_model = SimpleNamespace(result=lambda: dict(numeric, available=True))
    session.manager = SimpleNamespace(_model_error=None)
    session._accepted, session._total = 3, 3
    session._indicators = lambda: ({}, {})
    session.cloud = {'requested': False, 'configured': False}
    session.paths = []
    session._emit = lambda *args: None
    session.persist = lambda: None
    session._end({'scope': scope, 'ordinal': 1, 'start_seconds': 0., 'end_seconds': 15.,
                  'complete': complete, 'observed_end_sample': 15 * 1024})
    log = session.directory / 'model_evidence.jsonl'
    assert log.exists() is expected
    if expected:
        saved = json.loads(log.read_text(encoding='utf-8'))
        assert saved['prediction'] is None
        assert saved['observed_eeg_changes']
        assert 'numeric_fusion' not in json.dumps(saved)


def test_raw_semantic_mode_attaches_actual_windows_without_cloud_or_rest_input(tmp_path, monkeypatch):
    from eeg_agent import raw_model
    worker_root = tmp_path / 'worker'
    head = worker_root / 'outputs/vrms_model_ablation/v3_20261008/model_results/checkpoints/outer_subject_10_femba_frozen_mil.pt'
    head.parent.mkdir(parents=True); head.touch()
    monkeypatch.setattr(raw_model, 'ROOT', worker_root)
    windows = np.stack([waveform(2.), waveform(), waveform(.5)])
    segment = {'ordinal': 1, 'start_seconds': 0., 'end_seconds': 15.,
               'start_clock': '00:00.000', 'end_clock': '00:15.000', 'complete': True,
               'stats': {'window_coverage': 1}}
    record = {'signal_sha256': 'anonymous-waveform-source', 'task_count': 1,
              'model_compatible': True, 'segments': [segment]}
    repository = SimpleNamespace(root=tmp_path / 'analysis',
        cfg={'minimum_path_windows': 3, 'minimum_path_coverage': .8},
        model_inputs=lambda _: [(windows, [])],
        model_baseline=lambda _: pytest.fail('Opening windows cannot become rest'))
    def worker(*args, **kwargs):
        return SimpleNamespace(stdout=json.dumps({
            'available': True, 'segments': [{'ordinal': 1, 'predicted_class': 'High',
                'numeric_fusion': {'raw_combined_score': .6}}],
            'signal_sha256': record['signal_sha256'], 'fresh_raw_encoding': True,
            'model_sha256': 'head', 'encoder_sha256': 'encoder', 'normalization_sha256': 'norm',
            'numeric_first_sha256': 'first', 'numeric_second_sha256': 'second',
            'numeric_selector_sha256': 'selector'}).encode())
    attached = []
    def attach(fold, row, rag_policy):
        attached.append(copy.deepcopy(row))
        assert rag_policy['rag_mode'] == 'semantic_graph'
        return row
    monkeypatch.setattr(raw_model.subprocess, 'run', worker)
    monkeypatch.setattr(raw_model, 'attach_default_rag', attach)
    result = raw_model.assess(repository, 'subject-10', {'recording': record},
                              cloud=False, rag_policy={'rag_mode': 'semantic_graph'})
    assert result['available'] and len(attached) == 1
    measured = result['measurements']['segments'][0]['eeg_observations']
    assert observations.validate_observations(measured)
    assert measured['sample_sha256'] == hashlib.sha256(windows.tobytes()).hexdigest()
    assert not any(word in json.dumps(measured) for word in ('subject-10', 'true_label', 'questionnaire'))
