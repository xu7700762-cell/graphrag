"""Mode selection pins each session and keeps comparison display read-only."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from pydantic import ValidationError

from eeg_agent import default_rag
from eeg_agent.brain import BrainManager
from eeg_agent.online.router import OnlineRequest
from eeg_agent.online.runtime import OnlineManager
from eeg_agent.service import BrainRequest
from tests.test_brain_workspace import Tasks, workspace
from tests.test_online_runtime import online_cfg
from tests.test_online_service import online_api
from tests.test_online_cloud import reporter_factory, ProviderDouble, completed_path
from tests.test_raw_gpt_tool_agent import prediction


def policy(mode='default', version='one'):
    return {'rag_mode': 'regular' if mode == 'default' else mode,
            'requested_mode': mode, 'judge_profile': 'matched_decisive_high_v1',
            'reference_manifest_sha256': version, 'graph_manifest_sha256': version if mode == 'graph' else None}


@pytest.mark.parametrize('mode', ['default', 'regular', 'graph', 'semantic_graph'])
def test_both_creation_schemas_accept_modes_and_omit_defaults_compatibly(mode):
    assert BrainRequest(rag_mode=mode).rag_mode == mode
    assert OnlineRequest(recording_id='subject-10', rag_mode=mode).rag_mode == mode
    assert BrainRequest().rag_mode == 'default'
    assert OnlineRequest(recording_id='subject-10').rag_mode == 'default'


@pytest.mark.parametrize('model, values', [(BrainRequest, {}), (OnlineRequest, {'recording_id': 'subject-10'})])
def test_unknown_mode_and_extra_creation_fields_are_rejected(model, values):
    with pytest.raises(ValidationError):
        model(**values, rag_mode='changed-by-filename')
    with pytest.raises(ValidationError):
        model(**values, graph_database='arbitrary')


def test_homepage_creation_and_restore_keep_the_original_resolved_version(workspace, monkeypatch):
    client, manager, tasks = workspace
    original = policy('graph')
    monkeypatch.setattr(default_rag, 'resolve', lambda mode='default': original)
    response = client.post('/api/brain/sessions', json={'backend': 'mock', 'rag_mode': 'graph'})
    assert response.status_code == 200
    identity = response.json()['id']
    expected = copy.deepcopy(original)
    original['reference_manifest_sha256'] = 'changed outside session'
    assert client.get('/api/brain/sessions/' + identity).json()['rag_policy'] == expected
    monkeypatch.setattr(default_rag, 'resolve', lambda mode='default': policy(mode, 'new-default'))
    restored = BrainManager(manager.cfg, tasks).get(identity)
    assert restored.snapshot()['rag_policy'] == expected
    assert restored.snapshot()['rag_mode'] == 'graph'
    assert manager.list()[0]['rag_policy'] == expected
    assert client.post('/api/brain/sessions', json={'backend': 'mock', 'rag_mode': 'bad'}).status_code == 422


def test_old_homepage_history_does_not_invent_a_current_version(workspace, monkeypatch):
    client, manager, tasks = workspace
    identity = client.post('/api/brain/sessions', json={'backend': 'mock'}).json()['id']
    path = manager.root / identity / 'conversation.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    saved.pop('rag_mode', None)
    saved.pop('rag_policy', None)
    path.write_text(json.dumps(saved), encoding='utf-8')
    monkeypatch.setattr(default_rag, 'resolve', lambda *a, **k: pytest.fail('Historical version resolved to the present'))
    snapshot = BrainManager(manager.cfg, tasks).get(identity).snapshot()
    assert snapshot['rag_policy'] is None and snapshot['rag_mode'] is None


def test_unrecorded_old_session_new_classification_is_guarded_without_adopting_default(workspace, monkeypatch):
    from eeg_agent import raw_independent_judge
    from tests.test_default_rag import recorded_prediction
    from tests.test_raw_independent_judge import Provider, final
    client, manager, tasks = workspace
    identity = client.post('/api/brain/sessions', json={'backend': 'mock'}).json()['id']
    path = manager.root / identity / 'conversation.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    saved.pop('rag_mode', None)
    saved.pop('rag_policy', None)
    path.write_text(json.dumps(saved), encoding='utf-8')
    monkeypatch.setattr(default_rag, 'resolve', lambda *a, **k: pytest.fail('Old session adopted current default'))
    restored = BrainManager(manager.cfg, tasks).get(identity)
    row = recorded_prediction()
    guarded = default_rag.attach(1, row, restored._classification_policy())
    provider = Provider(final())
    result = raw_independent_judge.classify(provider, guarded)
    assert result['validation_code'] == 'historical_rag_version_unrecorded'
    assert result['error_stage'] == 'local_rag_validation' and not provider.calls
    assert guarded['numeric_fusion'] == row['numeric_fusion']
    assert restored.snapshot()['rag_policy'] is None
    assert restored.snapshot()['rag_mode'] is None


def test_online_creation_pins_mode_before_any_path_end_and_returns_policy(online_api, monkeypatch):
    client, manager, _ = online_api
    initial = policy('graph')
    monkeypatch.setattr(default_rag, 'resolve', lambda mode='default': initial)
    response = client.post('/api/online/sessions', json={'recording_id': 'subject-10', 'cloud': False, 'rag_mode': 'graph'})
    assert response.status_code == 200
    snapshot = response.json()
    initial['reference_manifest_sha256'] = 'future-default'
    pinned = client.get('/api/online/sessions/' + snapshot['id']).json()
    assert pinned['rag_mode'] == 'graph'
    assert pinned['rag_policy']['reference_manifest_sha256'] == 'one'
    assert pinned['request']['rag_policy'] == pinned['rag_policy']
    text = client.get('/api/online/sessions/' + snapshot['id'] + '/report.md').text
    assert 'graph' in text and 'matched_decisive_high_v1' in text
    assert client.post('/api/online/sessions', json={'recording_id': 'subject-10', 'rag_mode': 'bad'}).status_code == 422


def test_online_legacy_reporter_factory_is_still_injectable(online_cfg):
    reporter = object()
    manager = OnlineManager(online_cfg, cloud_factory=lambda emit: reporter)
    assert manager.cloud_reporter(lambda *a: None, rag_policy=policy('graph')) is reporter


def test_online_reporter_passes_pinned_policy_to_attachment(reporter_factory, monkeypatch):
    from eeg_agent.online import cloud
    reporter = reporter_factory(ProviderDouble())
    selected = policy('graph')
    reporter.rag_policy = copy.deepcopy(selected)
    calls = []
    def attach(fold, row, rag_policy=None):
        calls.append(copy.deepcopy(rag_policy))
        return copy.deepcopy(row)
    monkeypatch.setattr(cloud, 'attach_default_rag', attach)
    result = reporter.report(completed_path(), prediction())
    assert calls == [selected]
    assert result['rag_policy'] == selected and result['rag_mode'] == 'graph'


def test_comparison_page_reads_saved_results_without_starting_a_session(workspace, monkeypatch):
    client, manager, _ = workspace
    saved = {'available': True, 'status': 'completed',
             'arms': [{'id': 'regular', 'correct': 107, 'n': 146}], 'notes': ['stored test output']}
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        if path.name == 'comparison.json':
            return json.dumps(saved)
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    assert client.get('/api/rag/comparison').json() == saved
    page = client.get('/rag-comparison')
    assert page.status_code == 200 and 'RAG 前后对比' in page.text
    assert manager.list() == []


def test_online_browser_request_carries_explicit_mode_and_rejects_invalid_mode():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is unavailable')
    root = Path(__file__).resolve().parents[1]
    script = """
const assert=require('node:assert/strict');
const ui=require('./eeg_agent/web/online.js');
for (const mode of ['default','regular','graph','semantic_graph']) {
  assert.equal(ui.requestBody('subject-10','1',true,true,mode).rag_mode,mode);
}
assert.throws(()=>ui.requestBody('subject-10','1',true,true,'bad'));
assert.equal('rag_mode' in ui.requestBody('subject-10','1',true,true),false);
"""
    result = subprocess.run([node, '-e', script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
