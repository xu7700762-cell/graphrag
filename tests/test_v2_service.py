"""HTTP contract checks use a monitor stub, without training or cloud traffic."""
import json
import threading

import pytest
from fastapi.testclient import TestClient

from eeg_agent.config import load_config
from eeg_agent.service import create_app


class TaskStub:
    def __init__(self, request):
        self.id = 'task-stub'
        self.request = request.copy()
        self.state = 'completed'
        self.lock = threading.RLock()
        self.stop_requested = False
        self.events = [
            {'sequence': 0, 'kind': 'plan', 'payload': {'steps': ['quality', 'detector']}},
            {'sequence': 1, 'kind': 'window', 'payload': {'prediction_available': False, 'probability_high': None}},
            {'sequence': 2, 'kind': 'report', 'payload': {'markdown': '预测不可用'}},
            {'sequence': 3, 'kind': 'status', 'payload': {'state': 'completed'}},
        ]

    def snapshot(self):
        return {'id': self.id, 'state': self.state, 'request': self.request.copy(),
                'event_count': len(self.events), 'latest': self.events[1]['payload']}

    def request_stop(self):
        self.stop_requested = True

    def report(self):
        return {'markdown': '预测不可用', 'paths': [], 'cloud': {'simulated': True}}

    def message(self, text):
        self.events.append({'sequence': len(self.events), 'kind': 'message', 'payload': {'role': 'user', 'text': text}})
        self.events.append({'sequence': len(self.events), 'kind': 'language_report', 'payload': {'phase': 'followup', 'explanation': '当前模型不可用'}})
        return {'status': 'accepted'}


class ManagerStub:
    def __init__(self):
        self.tasks = {}

    def create(self, request):
        if any(t.state == 'running' for t in self.tasks.values()):
            raise RuntimeError('已有任务正在运行')
        if request.get('checkpoint_id') == '../unsafe':
            raise ValueError('Invalid checkpoint_id')
        task = TaskStub(request)
        self.tasks[task.id] = task
        return task.snapshot()

    def list(self):
        return [t.snapshot() for t in self.tasks.values()]

    def get(self, task_id):
        return self.tasks[task_id]


class KnowledgeStub:
    def search(self, query, limit=5):
        matches = [] if query == 'unknown-token' else [{'id': 'protocol-p0', 'text': 'P0 is the classification input', 'source': 'protocol', 'query': query}][:limit]
        return {'results': matches, 'status': 'matched' if matches else 'insufficient', 'reason': None if matches else 'no evidence'}


@pytest.fixture
def api(tmp_path, monkeypatch):
    cfg = load_config()
    cfg['output_root'] = str(tmp_path / 'outputs')
    cfg['data_root'] = str(tmp_path / 'data')
    raw = tmp_path / 'data/data/raw'
    raw.mkdir(parents=True)
    (raw / 'Acquisition 01.cdt').write_bytes(b'')
    manager = ManagerStub()
    for name in ('EEG_API_BASE_URL', 'EEG_API_MODEL', 'EEG_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    client = TestClient(create_app(cfg, task_manager=manager, knowledge_index=KnowledgeStub(),
                                  cloud_config_path=tmp_path / '.env'))
    return client, manager


def test_task_model_selection_and_quality_only_http_contract(api):
    client, manager = api
    response = client.post('/api/tasks', json={'subject': 1, 'model_id': 'eegconformer', 'strategy': 'adaptive'})
    assert response.status_code == 200
    value = response.json()
    assert value['request']['model_id'] == 'eegconformer'
    assert value['request']['checkpoint_id'] is None
    assert value['latest']['prediction_available'] is False
    assert value['latest']['probability_high'] is None
    assert client.get('/api/tasks').json()[0]['id'] == value['id']
    assert client.get('/api/tasks/' + value['id']).json()['request'] == value['request']
    assert client.get('/api/tasks/' + value['id'] + '/report').json()['markdown'] == '预测不可用'
    assert client.post('/api/tasks/' + value['id'] + '/stop').status_code == 200
    assert manager.get(value['id']).stop_requested


def test_sse_event_ids_and_resume_do_not_repeat(api):
    client, _ = api
    task_id = client.post('/api/tasks', json={}).json()['id']
    response = client.get('/api/tasks/' + task_id + '/events')
    assert response.headers['content-type'].startswith('text/event-stream')
    assert [line for line in response.text.splitlines() if line.startswith('id: ')] == ['id: 0', 'id: 1', 'id: 2', 'id: 3']
    resumed = client.get('/api/tasks/' + task_id + '/events?after=0', headers={'Last-Event-ID': '2'})
    events = [json.loads(line[6:]) for line in resumed.text.splitlines() if line.startswith('data: ')]
    assert [e['sequence'] for e in events] == [3]
    assert client.get('/api/tasks/' + task_id + '/events?after=3').text == ''
    assert client.get('/api/tasks/' + task_id + '/events', headers={'Last-Event-ID': 'invalid'}).status_code == 400
    assert client.get('/api/tasks/' + task_id + '/events?after=-2').status_code == 400
    assert client.get('/api/tasks/' + task_id + '/events', headers={'Last-Event-ID': '-2'}).status_code == 400


def test_validation_missing_task_cloud_configuration_and_conflicts(api):
    client, manager = api
    assert client.post('/api/tasks', json={'model_id': 'unregistered'}).status_code == 422
    assert client.post('/api/tasks', json={'strategy': 'override_classifier'}).status_code == 422
    assert client.post('/api/tasks', json={'start_sample': -1}).status_code == 422
    assert client.post('/api/tasks', json={'backend': 'api'}).status_code == 400
    assert client.post('/api/tasks', json={'checkpoint_id': '../unsafe'}).status_code == 400
    assert client.post('/api/tasks', json={'subject': 2}).status_code == 400
    assert client.get('/api/tasks/missing').status_code == 404
    task_id = client.post('/api/tasks', json={}).json()['id']
    manager.get(task_id).state = 'running'
    assert client.post('/api/tasks', json={}).status_code == 409
    assert client.post('/api/runs', json={}).status_code == 409


def test_model_tools_knowledge_messages_and_static_ui(api):
    client, _ = api
    models = client.get('/api/models').json()
    assert [m['model_id'] for m in models] == ['eegnet', 'eegconformer']
    assert all(not m['available'] for m in models)
    assert client.get('/api/config').json()['model']['available'] is False
    assert len(client.get('/api/tools').json()) >= 7
    assert client.get('/api/knowledge/search?q=P0&limit=100').json()['matches'][0]['id'] == 'protocol-p0'
    assert not client.get('/api/knowledge/search').json()['evidence_available']
    unmatched = client.get('/api/knowledge/search?q=unknown-token').json()
    assert not unmatched['evidence_available'] and unmatched['matches'] == []
    assert unmatched['reason'] == 'no evidence'
    task_id = client.post('/api/tasks', json={}).json()['id']
    assert client.post('/api/tasks/' + task_id + '/messages', json={'text': ' 为什么未预测？ '}).json()['status'] == 'accepted'
    assert client.get('/api/tasks/' + task_id + '/messages').json()[0]['payload']['text'] == '为什么未预测？'
    assert client.get('/api/tasks/' + task_id + '/messages').json()[1]['payload']['phase'] == 'followup'
    assert client.post('/api/tasks/' + task_id + '/messages', json={'text': ' '}).status_code == 400
    page = client.get('/monitor').text
    assert 'EEGConformer' in page and '冻结检查点' in page
    assert client.get('/assets/app.js').status_code == 200
    assert client.get('/api/health').json()['version'] == '0.2.0'
