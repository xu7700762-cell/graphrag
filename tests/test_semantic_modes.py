"""Semantic mode pins versions and is selectable without promoting the default."""
import copy
import sys
from types import SimpleNamespace

import pytest

import eeg_agent
from eeg_agent import default_rag
from eeg_agent.brain import BrainManager
from tests.test_brain_workspace import workspace
from tests.test_online_service import online_api
from tests.test_online_runtime import online_cfg


@pytest.fixture
def semantic_module(monkeypatch):
    hashes = {'semantic_manifest_sha256': 'a' * 64, 'semantic_code_sha256': 'b' * 64,
              'observation_code_sha256': 'e' * 64, 'semantic_judge_code_sha256': 'f' * 64,
              'semantic_schema': 'raw_eeg_state_graph_v1', 'semantic_variant': 'state_reasoning_v1',
              'graph_manifest_sha256': 'c' * 64, 'graph_code_sha256': 'd' * 64,
              'graph_schema': 'raw_eeg_graph_rag_v3', 'graph_variant': 'graph_v3'}
    calls = []
    def retrieve(fold, prediction, semantic_manifest_sha256=None):
        calls.append((fold, copy.deepcopy(prediction), semantic_manifest_sha256))
        return {'available': True, 'schema': 'raw_eeg_state_graph_v1', 'variant': 'state_reasoning_v1',
                'state_evidence': {'supporting_paths': [], 'contradictory_paths': [],
                                   'evidence_strength': 'low', 'confidence_basis': [],
                                   'evidence_status': 'unavailable', 'fallback_reason': 'no_reviewed_match'}}
    module = SimpleNamespace(SCHEMA='raw_eeg_state_graph_v1', VARIANT='state_reasoning_v1',
                             asset_policy=lambda: copy.deepcopy(hashes), retrieve=retrieve,
                             hashes=hashes, calls=calls)
    monkeypatch.setitem(sys.modules, 'eeg_agent.state_graph_rag', module)
    monkeypatch.setattr(eeg_agent, 'state_graph_rag', module, raising=False)
    return module


def test_semantic_policy_pins_assets_and_contract_without_changing_default(semantic_module):
    regular = default_rag.resolve('default')
    policy = default_rag.resolve('semantic_graph')
    assert regular['rag_mode'] == 'regular' and regular['variant'] == 'percentile_group'
    assert policy['rag_mode'] == 'semantic_graph'
    assert policy['judge_profile'] == 'semantic_state_high_v1'
    assert policy['teaching_sha256'] is None
    assert default_rag.validate_policy(policy) == policy
    assert default_rag.judge_contract(policy) == default_rag.judge_contract(default_rag.resolve('regular'))


@pytest.mark.parametrize('field', ['semantic_manifest_sha256', 'semantic_code_sha256',
                                  'semantic_schema', 'semantic_variant', 'graph_manifest_sha256',
                                  'graph_code_sha256', 'graph_schema', 'graph_variant',
                                  'observation_code_sha256', 'semantic_judge_code_sha256',
                                  'judge_profile', 'teaching_sha256'])
def test_semantic_saved_policy_rejects_changed_pins(field, semantic_module):
    policy = default_rag.resolve('semantic_graph')
    policy[field] = 'matched_decisive_high_v1' if field == 'judge_profile' else 'changed'
    with pytest.raises(ValueError):
        default_rag.validate_policy(policy)


def test_semantic_attach_routes_pinned_manifest_and_preserves_numeric_model(semantic_module):
    prediction = {'numeric_fusion': {'predicted_class': 'High', 'score': .8}}
    original = copy.deepcopy(prediction)
    policy = default_rag.resolve('semantic_graph')
    attached = default_rag.attach(1, prediction, policy)
    assert attached['numeric_fusion'] == original['numeric_fusion'] and prediction == original
    assert attached['current_raw_contrastive_context']['variant'] == 'state_reasoning_v1'
    assert semantic_module.calls == [(1, original, policy['semantic_manifest_sha256'])]
    semantic_module.hashes['semantic_manifest_sha256'] = 'future-version'
    unavailable = default_rag.retrieve(1, prediction, policy)
    assert unavailable['available'] is False
    assert unavailable['state_evidence']['evidence_status'] == 'unavailable'
    assert not unavailable['state_evidence']['supporting_paths']


def test_semantic_pin_failure_preserves_numeric_and_records_unavailable_state(monkeypatch):
    from eeg_agent import state_graph_rag
    prediction = {'numeric_fusion': {'predicted_class': 'High', 'score': .8}}
    policy = default_rag.resolve('semantic_graph')
    policy['semantic_manifest_sha256'] = 'wrong-version'
    attached = default_rag.attach(1, prediction, policy)
    record = state_graph_rag.session_evidence(attached)
    assert attached['numeric_fusion'] == prediction['numeric_fusion']
    assert attached['current_raw_contrastive_context']['available'] is False
    assert record['evidence_status'] == 'unavailable' and record['evidence_strength'] == 'low'
    assert record['observed_eeg_changes'] == []
    assert record['prediction'] is None


def test_both_catalogs_and_status_show_semantic_version_without_promotion(workspace, online_api, semantic_module):
    brain_client, _, _ = workspace
    online_client, _, _ = online_api
    for value in (brain_client.get('/api/brain/catalog').json(),
                  brain_client.get('/api/rag/version').json(),
                  online_client.get('/api/online/catalog').json()):
        assert value['default_rag_mode'] == 'regular'
        assert 'semantic_graph' in [mode['id'] for mode in value['rag_modes']]
        assert value['semantic_graph_rag']['promotion_status'] == 'not_evaluated'
        assert value['semantic_graph_rag']['semantic_manifest_sha256'] == 'a' * 64


def test_homepage_semantic_session_restores_pinned_policy(workspace, semantic_module):
    client, manager, tasks = workspace
    response = client.post('/api/brain/sessions', json={'backend': 'mock', 'rag_mode': 'semantic_graph'})
    assert response.status_code == 200
    saved = response.json()
    expected = copy.deepcopy(saved['rag_policy'])
    semantic_module.hashes['semantic_manifest_sha256'] = 'future-version'
    restored = BrainManager(manager.cfg, tasks).get(saved['id']).snapshot()
    assert restored['rag_mode'] == 'semantic_graph' and restored['rag_policy'] == expected


def test_online_semantic_session_pins_and_reports_version(online_api, semantic_module):
    client, _, _ = online_api
    response = client.post('/api/online/sessions', json={
        'recording_id': 'subject-10', 'cloud': False, 'rag_mode': 'semantic_graph'})
    assert response.status_code == 200
    saved = response.json()
    policy = copy.deepcopy(saved['rag_policy'])
    semantic_module.hashes['semantic_manifest_sha256'] = 'future-version'
    snapshot = client.get('/api/online/sessions/' + saved['id']).json()
    assert snapshot['rag_policy'] == policy == snapshot['request']['rag_policy']
    report = client.get('/api/online/sessions/' + saved['id'] + '/report.md').text
    assert 'semantic_graph' in report and 'semantic_state_high_v1' in report


def test_semantic_profile_cannot_be_assigned_to_old_graph(semantic_module):
    policy = default_rag.resolve('graph')
    policy['judge_profile'] = 'semantic_state_high_v1'
    with pytest.raises(ValueError):
        default_rag.validate_policy(policy)


def test_pages_include_semantic_mode(workspace, online_api, semantic_module):
    client, _, _ = workspace
    online_client, _, _ = online_api
    for page in (client.get('/').text, online_client.get('/online').text):
        assert 'value="semantic_graph"' in page and '状态推理实验' in page
