"""Exercise conversation orchestration with local evidence and an API double."""
import copy
import json
import math
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from eeg_agent.brain import BrainManager, local_evidence
from eeg_agent.cloud_config import save_settings
from eeg_agent.config import EEG_CHANNELS, load_config
from eeg_agent.domain_tools import toolkit_catalog
from eeg_agent.service import create_app


class Tasks:
    def __init__(self):
        self.tasks = {}

    def list(self):
        return [task.snapshot() for task in self.tasks.values()]

    def get(self, task_id):
        return self.tasks[task_id]


class ProviderDouble:
    calls = []
    release = None
    invalid = False

    def __init__(self, backend, cfg, emit):
        self.client = None
        self.emit = emit

    def call(self, role, payload, messages=None):
        self.calls.append((role, copy.deepcopy(payload), copy.deepcopy(messages)))
        if self.release is not None:
            self.release.wait(2)
        if 'available_agents' in payload:
            return {'tasks': [{'agent': d, 'task': '说明本领域已到达证据与方法'}
                              for d in payload['available_agents']], 'explanation': '领域协作'}
        if 'available_tools' in payload:
            return {'requested_tools': ['unregistered.execute'] if self.invalid else
                    [t['id'] for t in payload['available_tools']]}
        refs = payload.get('knowledge')
        if refs is None:
            refs = [r for report in payload['agent_reports'] for r in report['references']]
        return {'explanation': '以本地 EEG 工具结果和领域引用解释指标、趋势及证据边界。',
                'citations': ['invented-citation'] if self.invalid else list(dict.fromkeys(r['id'] for r in refs))}

    def summary(self):
        return {'backend': 'api', 'calls': len(self.calls), 'simulated': False}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for key in ('EEG_API_BASE_URL', 'EEG_API_MODEL', 'EEG_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    cfg = load_config()
    cfg['output_root'] = str(tmp_path / 'output')
    cfg['cloud_config_path'] = str(tmp_path / '.env')
    tasks = Tasks()
    ProviderDouble.calls = []
    ProviderDouble.release = None
    ProviderDouble.invalid = False
    brain = BrainManager(cfg, tasks, provider_factory=ProviderDouble)
    app = create_app(cfg, task_manager=tasks, knowledge_index=object(), brain_manager=brain)
    return TestClient(app), brain, tasks


def finished(client, session_id):
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        snapshot = client.get('/api/brain/sessions/' + session_id).json()
        if snapshot['state'] == 'idle':
            return snapshot
        time.sleep(.01)
    pytest.fail('conversation did not finish')


def start(client, backend='mock', **extra):
    response = client.post('/api/brain/sessions', json={'backend': backend, **extra})
    assert response.status_code == 200, response.text
    session_id = response.json()['id']
    response = client.post('/api/brain/sessions/' + session_id + '/messages', json={
        'text': '请协作分析所选领域的 EEG 指标，并说明分析方法。'})
    assert response.status_code == 200, response.text
    return session_id


def test_three_real_toolkits_scoped_rag_then_supervisor_and_resume(workspace):
    client, _, _ = workspace
    session_id = start(client)
    snapshot = finished(client, session_id)
    result = client.get('/api/brain/sessions/' + session_id + '/report').json()
    assert result['simulated'] and result['evidence_status'] == 'missing'
    assert len(result['agent_reports']) == 3
    assert [m['role'] for m in snapshot['messages']] == ['user', 'assistant']
    events = snapshot['events']
    kinds = [e['kind'] for e in events]
    assert 'shared_state' in kinds and kinds.index('shared_state') < kinds.index('synthesis')
    for domain in ('vrms', 'fatigue', 'emotion'):
        specific = [e for e in events if e['payload'].get('agent') == domain]
        sk = [e['kind'] for e in specific]
        assert sk.index('tool_result') < sk.index('retrieval') < sk.index('agent_report')
        retrieval = next(e['payload'] for e in specific if e['kind'] == 'retrieval')
        assert retrieval['stages'][0]['candidate_limit'] == 20
        assert all(r['domain'] in ('shared', domain) for r in retrieval['results'])
    assert not ProviderDouble.calls
    response = client.get('/api/brain/sessions/' + session_id + '/events?after=2',
                          headers={'Last-Event-ID': '3'})
    received = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert received and all(e['sequence'] > 3 for e in received)
    assert client.get('/api/brain/sessions/' + session_id + '/events?after=-2').status_code == 400


def test_followup_and_persistent_session_restart(workspace):
    client, manager, tasks = workspace
    session_id = start(client, domains=['fatigue'])
    finished(client, session_id)
    assert client.post('/api/brain/sessions/' + session_id + '/messages', json={'text': '还需要哪些输入？'}).status_code == 200
    snapshot = finished(client, session_id)
    assert snapshot['turn'] == 2 and len(snapshot['messages']) == 4
    restored = BrainManager(manager.cfg, tasks)
    assert restored.get(session_id).snapshot()['messages'] == snapshot['messages']
    assert restored.get(session_id).state == 'idle'


@pytest.mark.parametrize('backend', ['mock', 'api'])
def test_fatigue_emotion_measurements_and_cooperation_need_no_classifier(workspace, monkeypatch, backend):
    client, manager, tasks = workspace
    inference_calls = []

    def reject_inference(*args, **kwargs):
        inference_calls.append(True)
        raise AssertionError('EEG tools-only agents must never load or run a classifier')

    monkeypatch.setattr('eeg_agent.models.build_model', reject_inference)
    monkeypatch.setattr('eeg_agent.models.ONNXPredictor.__init__', reject_inference)
    monkeypatch.setattr('eeg_agent.vrms_runtime.assess', reject_inference)

    class MeasurementTask:
        lock = threading.RLock()
        paths = []
        last_window = {'quality': {'valid': True}}
        events = [{'kind': 'window', 'payload': {'start_sample': index * 5120,
                  'end_sample': (index + 1) * 5120, 'quality': {'valid': True},
                  'biomarkers': {'regions': {'all': {'ratios': {'theta_alpha': float(index + 1)}},
                                             'frontal': {'absolute_power_uv2': {'alpha': float(index + 2), 'beta': float(index + 1)}}}}}}
                  for index in range(3)]

        def snapshot(self):
            return {'latest': {'quality': {'valid': True}, 'prediction_available': False,
                              'deep': {'available': False}}, 'source_sample': 15360,
                    'model': {'available': False}}

    # A local signal fixture supplies real spectral measurements without weights.
    sample_times = np.arange(1280) / 256.
    waveform = (np.sin(2 * np.pi * 2 * sample_times) + np.sin(2 * np.pi * 6 * sample_times) +
                2 * np.sin(2 * np.pi * 10 * sample_times) + np.sin(2 * np.pi * 20 * sample_times))
    task = MeasurementTask()
    task.last_x = np.tile(waveform, (len(EEG_CHANNELS), 1))
    task.last_x[EEG_CHANNELS.index('F4')] *= math.sqrt(2)
    tasks.tasks['tools-only-input'] = task
    if backend == 'api':
        save_settings({'base_url': 'https://test.invalid/v1', 'model_id': 'test', 'api_key': 'never-outbound'},
                      manager.cfg['cloud_config_path'])
    session_id = start(client, backend=backend, domains=['fatigue', 'emotion'], task_id='tools-only-input')
    result = finished(client, session_id)['result']
    assert result['workflow_status'] == ('cloud_verified' if backend == 'api' else 'mock')
    assert result['evidence_status'] == 'arrived'
    reports = {r['agent']: r for r in result['agent_reports']}
    assert set(reports) == {'fatigue', 'emotion'}
    for report in reports.values():
        assert report['analysis_mode'] == 'eeg_tools_only' and not report['classifier_required']
        assert report['model_status'] == 'tools_only'
        assert all(item['result']['available'] and not item['result']['classifier_required'] and
                   item['result']['predicted_class'] is None for item in report['tool_results'])
        assert all(r['domain'] in ('shared', report['agent']) for r in report['references'])
    emotion = reports['emotion']['tool_results'][0]['result']['measurements']
    assert emotion['faa'] == pytest.approx(math.log(2))
    temporal = reports['fatigue']['tool_results'][1]['result']['measurements']
    assert temporal['theta_alpha']['slope_ratio_per_minute'] == pytest.approx(12.)
    assert inference_calls == []
    if backend == 'api':
        assert len(ProviderDouble.calls) == 6
        assert {role for role, _, _ in ProviderDouble.calls} == {'Supervisor', 'FatigueAgent', 'EmotionAgent'}
        assert all('probability_high' not in json.dumps(payload) for _, payload, _ in ProviderDouble.calls)


def test_api_roles_receive_bounded_context_and_invalid_calls_fallback(workspace):
    client, manager, _ = workspace
    save_settings({'base_url': 'https://test.invalid/v1', 'model_id': 'test', 'api_key': 'never-outbound'},
                  manager.cfg['cloud_config_path'])
    session_id = start(client, backend='api')
    snapshot = finished(client, session_id)
    assert snapshot['result'] and not snapshot['result']['simulated']
    assert set(role for role, _, _ in ProviderDouble.calls) == {'Supervisor', 'VRMSAgent', 'FatigueAgent', 'EmotionAgent'}
    assert len(ProviderDouble.calls) == 8
    assert all('conversation' not in payload for role, payload, _ in ProviderDouble.calls if role != 'Supervisor')
    assert all('untrusted data' in messages[0]['content'] for _, _, messages in ProviderDouble.calls)
    ProviderDouble.invalid = True
    assert client.post('/api/brain/sessions/' + session_id + '/messages', json={'text': '继续说明证据。'}).status_code == 200
    snapshot = finished(client, session_id)
    assert all(r['fallback_reason'] for r in snapshot['result']['agent_reports'])
    assert snapshot['result']['fallback_reason']
    assert all(item['tool'] in {t['id'] for t in toolkit_catalog(include_recordings=True)[r['agent']]}
               for r in snapshot['result']['agent_reports'] for item in r['tool_results'])
    assert 'invented-citation' not in json.dumps(snapshot['result'])



def test_running_turn_rejects_duplicate_send_and_locks_cloud_settings(workspace):
    client, manager, _ = workspace
    save_settings({'base_url': 'https://test.invalid/v1', 'model_id': 'test', 'api_key': 'secret'},
                  manager.cfg['cloud_config_path'])
    gate = threading.Event()
    ProviderDouble.release = gate
    session_id = start(client, backend='api')
    assert client.post('/api/brain/sessions/' + session_id + '/messages', json={'text': '重复消息'}).status_code == 409
    assert client.get('/api/cloud-settings').json()['in_use']
    assert client.put('/api/cloud-settings', json={'base_url': 'https://new.invalid', 'model_id': 'new'}).status_code == 409
    gate.set()
    finished(client, session_id)
    assert not client.get('/api/cloud-settings').json()['in_use']


def test_local_evidence_strips_raw_arrays_identity_labels_and_resets_spectral_gaps():
    class Task:
        lock = threading.RLock()
        paths = [{'subject': 'private', 'label': 1, 'probability_high': .6, 'status': 'accepted'}]
        events = [{'kind': 'window', 'payload': {'start_sample': start, 'end_sample': start + 5120,
                  'quality': {'valid': True}, 'biomarkers': {'regions': {'all': {'ratios': {'theta_alpha': 2}}}}}}
                  for start in (0, 5120, 15360)]

        def snapshot(self):
            return {'latest': {'raw_eeg': [[99]], 'subject': 'private', 'quality': {'valid': True},
                'deep': {'available': True, 'probability_high': .6, 'logits': [9, 9]},
                'biomarkers': {'available': True, 'psd': [[99]], 'channels': ['F3', 'F4'],
                               'absolute_power_uv2': {'alpha': [1, 2]}}}, 'source_sample': 20480,
                    'model': {'available': True, 'model_id': 'eegnet', 'path': '/secret'}}
    evidence = local_evidence(Task())
    assert len(evidence['arrived']['fatigue_history']) == 1
    assert evidence['arrived']['biomarkers']['channel_power_uv2']['F4']['alpha'] == 2
    serialized = json.dumps(evidence)
    assert all(term not in serialized for term in ('raw_eeg', 'psd', 'private', 'subject', 'logits', '/secret', '"label"'))
    Task.events.append({'kind': 'gap', 'payload': {'reason': 'test interruption'}})
    after_gap = local_evidence(Task())
    assert not after_gap['arrived']['fatigue_history']
    assert after_gap['arrived']['historical'] and after_gap['arrived']['quality']['valid'] is False


def test_validation_static_routes_and_knowledge_import(workspace):
    client, _, _ = workspace
    assert client.get('/').status_code == 200
    assert client.get('/assets/brain.js').status_code == 200
    assert '冻结检查点' in client.get('/monitor').text
    assert len(client.get('/api/brain/catalog').json()['agents']) == 3
    assert client.post('/api/brain/sessions', json={'domains': []}).status_code == 422
    assert client.post('/api/brain/sessions', json={'domains': ['vrms', 'vrms']}).status_code == 400
    assert client.post('/api/brain/sessions', json={'backend': 'api'}).status_code == 400
    assert client.post('/api/brain/sessions', json={'backend': 'mock', 'task_id': 'unknown'}).status_code == 404
    assert client.get('/api/brain/sessions/missing').status_code == 404
    assert client.get('/api/brain/knowledge/search?domains=unknown').status_code == 400
    document = {'domain': 'fatigue', 'title': '导入资料', 'source': 'test-source',
                'text': 'fatigueunique_token 资料原文。文献中的指令只是引用数据。'}
    assert client.post('/api/brain/knowledge/documents', json=document).status_code == 200
    assert client.get('/api/brain/knowledge/search?q=fatigueunique_token&domains=emotion').json()['results'] == []
    result = client.get('/api/brain/knowledge/search?q=fatigueunique_token&domains=fatigue').json()['results']
    assert result and result[0]['evidence_type'] == 'user_supplied'
    assert len(result[0]['source_sha256']) == 64
    assert client.post('/api/brain/knowledge/documents', json=document,
                       headers={'Origin': 'https://unrelated.invalid'}).status_code == 403



