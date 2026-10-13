"""The removed research entry cannot execute through creation, restoration or HTTP."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from eeg_agent.brain import BrainManager
from eeg_agent.config import load_config
from eeg_agent.service import create_app


class Tasks:
    def list(self):
        return []

    def get(self, task_id):
        raise KeyError(task_id)


def workspace(tmp_path):
    cfg = {**load_config(), 'output_root': str(tmp_path / 'output'),
           'cloud_config_path': str(tmp_path / '.env')}
    tasks = Tasks()
    manager = BrainManager(cfg, tasks)
    return cfg, tasks, manager, TestClient(create_app(cfg, task_manager=tasks,
        knowledge_index=object(), brain_manager=manager))


@pytest.mark.parametrize('legacy', [
    {'vrms_path_id': 'path-000'},
    {'pipeline_id': 'femba_mil_tools_gpt_seed2026'},
    {'input_policy': 'research_path'},
])
def test_legacy_creation_is_rejected_before_inference(tmp_path, monkeypatch, legacy):
    _, _, manager, client = workspace(tmp_path)
    monkeypatch.setattr('eeg_agent.vrms_runtime._worker', lambda *a, **k: pytest.fail('Retired worker executed'))
    response = client.post('/api/brain/sessions', json={'backend': 'mock', **legacy})
    assert response.status_code == 422
    assert not manager.list()
    with pytest.raises(ValueError, match='旧研究入口已移除'):
        manager.create({'backend': 'mock', 'domains': ['vrms'], **legacy})


@pytest.mark.parametrize('explicit_policy', [True, False])
def test_old_history_is_read_only_and_preserves_answers_after_restart(tmp_path, monkeypatch, explicit_policy):
    cfg, tasks, manager, _ = workspace(tmp_path)
    session_id = 'abcdef123456'
    directory = manager.root / session_id
    directory.mkdir()
    result = {'text': '历史模型结果', 'markdown': '# 历史报告\n\nMIL=0.33',
              'provider': {'calls': 8}, 'agent_reports': []}
    saved = {'id': session_id, 'backend': 'api', 'domains': ['vrms'], 'state': 'idle',
             'pipeline_id': 'femba_mil_tools_gpt_seed2028', 'vrms_path_id': 'path-000',
             'turn': 1, 'created': 1., 'messages': [{'role': 'assistant', **result}],
             'events': [], 'result': copy.deepcopy(result)}
    if explicit_policy:
        saved['input_policy'] = 'research_path'
    (directory / 'conversation.json').write_text(json.dumps(saved, ensure_ascii=False), encoding='utf-8')
    monkeypatch.setattr('eeg_agent.vrms_runtime._worker', lambda *a, **k: pytest.fail('Retired worker executed'))
    restored = BrainManager(cfg, tasks, provider_factory=lambda *a, **k: pytest.fail('Retired cloud request'))
    client = TestClient(create_app(cfg, task_manager=tasks, knowledge_index=object(), brain_manager=restored))
    snapshot = client.get('/api/brain/sessions/' + session_id).json()
    assert snapshot['read_only'] and snapshot['messages'] == saved['messages'] and snapshot['result'] == result
    assert snapshot['pipeline_id'] == saved['pipeline_id'] and snapshot['vrms_path_id'] == saved['vrms_path_id']
    assert client.get('/api/brain/sessions').json()[0]['read_only']
    assert client.get('/api/brain/sessions/' + session_id + '/report').json() == result
    response = client.post('/api/brain/sessions/' + session_id + '/messages', json={'text': '继续推理'})
    assert response.status_code == 409 and '旧研究入口已移除' in response.text
    assert restored.get(session_id).turn == 1
    with pytest.raises(RuntimeError, match='旧研究入口已移除'):
        restored.get(session_id).message('直接调用也不能续跑')


def test_current_catalog_and_method_chat_only_offer_raw_vrms_tools(tmp_path):
    _, _, manager, client = workspace(tmp_path)
    catalog = client.get('/api/brain/catalog').json()
    tools = {agent['id']: [tool['id'] for tool in agent['tools']] for agent in catalog['agents']}
    assert tools == {'vrms': ['vrms.raw_recording', 'vrms.raw_model'],
                     'fatigue': ['fatigue.raw_spectrum'], 'emotion': ['emotion.raw_workload']}
    response = client.post('/api/brain/sessions', json={'backend': 'mock', 'input_policy': 'raw_eeg_only'})
    assert response.status_code == 200 and response.json()['input_policy'] == 'raw_eeg_only'
    assert not response.json()['read_only'] and response.json()['vrms_path_id'] is None
    assert not hasattr(manager, 'vrms_assess') and not hasattr(manager, 'path_evidence')
