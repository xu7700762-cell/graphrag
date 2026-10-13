"""Independent online HTTP/SSE contracts and preserved offline routes."""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from eeg_agent.online.router import mount_online
from eeg_agent.online.runtime import OnlineManager, OnlineSession
from tests.test_online_runtime import FakeModel, online_cfg, feed_path  # Shared synthetic waveform fixture.


@pytest.fixture
def online_api(online_cfg, monkeypatch):
    monkeypatch.setattr(OnlineSession, 'start', lambda self: self._set_state('running'))
    manager = OnlineManager(online_cfg, model_factory=FakeModel)
    app = FastAPI()
    app.state.cloud_testing = False
    mount_online(app, online_cfg, manager)
    with TestClient(app) as client:
        yield client, manager, app


def create(client, **updates):
    return client.post('/api/online/sessions', json={
        'recording_id': 'subject-10', 'speed': 1, 'cloud': False, 'max_seconds': None, **updates})


def decode_sse(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]


def test_catalog_create_pause_resume_stop_reports_and_single_session(online_api):
    client, manager, _ = online_api
    catalog = client.get('/api/online/catalog').json()
    assert catalog['recordings'][0]['id'] == 'subject-10'
    assert catalog['defaults']['speed'] == 1 and catalog['defaults']['window_seconds'] == 5
    assert catalog['capabilities']['model_available'] is True
    response = create(client)
    assert response.status_code == 200
    identity = response.json()['id']
    base = '/api/online/sessions/' + identity
    assert response.json()['request']['cloud'] is False
    assert client.get(base).json()['pipeline_id'].startswith('raw_eeg_stream_')
    assert client.get('/api/online/sessions').json()[0]['id'] == identity
    assert create(client).status_code == 409
    assert client.post(base + '/control', json={'action': 'pause'}).json()['state'] == 'paused'
    assert client.post(base + '/control', json={'action': 'resume'}).json()['state'] == 'running'
    assert client.post(base + '/control', json={'action': 'stop'}).json()['state'] == 'stopped'
    assert client.post(base + '/control', json={'action': 'resume'}).status_code == 409
    assert client.get(base + '/report').json()['state'] == 'stopped'
    report = client.get(base + '/report.md')
    assert report.status_code == 200 and report.headers['content-type'].startswith('text/markdown')
    assert '未校准' in report.text
    assert manager.get(identity).stop_event.is_set()
    assert create(client).status_code == 200


def test_max_speed_uses_the_same_explicit_verification_mode(online_api):
    client, _, _ = online_api
    response = create(client, speed='max')
    assert response.status_code == 200
    assert response.json()['speed'] == 1000


def test_markdown_report_keeps_three_agent_explanations_and_cutoff(online_api):
    client, manager, _ = online_api
    identity = create(client).json()['id']
    session = manager.get(identity)
    feed_path(session)
    session.paths[0]['cloud'] = {'status': 'verified', 'final_class': 'Low', 'reports': {
        'vrms': {'explanation': 'VRMS 已核验解释'}, 'fatigue': {'explanation': '疲劳频谱解释'},
        'emotion': {'explanation': '额区与负荷解释'}, 'supervisor': {'explanation': '路径汇总解释'}}}
    text = client.get('/api/online/sessions/' + identity + '/report.md').text
    for value in ('VRMS 已核验解释', '疲劳频谱解释', '额区与负荷解释', '路径汇总解释', '数据截止：'):
        assert value in text


@pytest.mark.parametrize('updates', [
    {'recording_id': '../secret.cdt'}, {'recording_id': 'edf-a'}, {'speed': 0}, {'speed': 1001},
    {'max_seconds': -1}, {'target_label': 'High'}, {'path_score': 99},
])
def test_input_contract_rejects_paths_labels_and_invalid_replay_parameters(online_api, updates):
    client, manager, _ = online_api
    response = create(client, **updates)
    assert response.status_code in (400, 422)
    assert manager.list() == []


def test_missing_records_unknown_sessions_and_invalid_controls_are_explicit(online_api):
    client, _, _ = online_api
    assert create(client, recording_id='subject-99').status_code == 400
    assert client.get('/api/online/sessions/unknown').status_code == 404
    assert client.get('/api/online/sessions/unknown/events').status_code == 404
    identity = create(client).json()['id']
    assert client.post('/api/online/sessions/' + identity + '/control', json={'action': 'rewind'}).status_code == 422


def test_cross_origin_cannot_create_or_control_online_sessions(online_api):
    client, manager, _ = online_api
    body = {'recording_id': 'subject-10', 'speed': 1, 'cloud': False}
    for origin in ('http://hostile.example', 'https://testserver', 'null'):
        response = client.post('/api/online/sessions', json=body, headers={'Origin': origin})
        assert response.status_code == 403
    assert manager.list() == []
    response = client.post('/api/online/sessions', json=body, headers={'Origin': 'http://testserver'})
    assert response.status_code == 200
    identity = response.json()['id']
    denied = client.post('/api/online/sessions/' + identity + '/control', json={'action': 'stop'}, headers={'Origin': 'http://hostile.example'})
    assert denied.status_code == 403
    assert manager.get(identity).state == 'running'


def test_cloud_connection_testing_only_blocks_cloud_requested_sessions(online_api):
    client, _, app = online_api
    app.state.cloud_testing = True
    assert create(client, cloud=True).status_code == 409
    assert create(client, cloud=False).status_code == 200


def test_markdown_renders_retained_transport_failures_and_unclosed_reason(online_api):
    client, manager, _ = online_api
    identity = create(client).json()['id']
    session = manager.get(identity)
    feed_path(session)
    row = session.paths[0]
    row['complete'] = False
    row['reason'] = '记录结束前缺少结束标记'
    row['cloud'] = {'status': 'fallback', 'final_class': None, 'reports': {
        'vrms': {'available': False, 'error_type': 'APITimeoutError', 'reason': '独立 LLM 判断未完成'},
        'supervisor': {'available': False, 'error_type': 'APITimeoutError', 'reason': '云端解释未通过来源核验；保留本地指标'}}}
    original = json.dumps(row, ensure_ascii=False)
    text = client.get('/api/online/sessions/' + identity + '/report.md').text
    assert 'API请求超时' in text
    assert '未通过来源核验' not in text
    assert '路径未闭合' in text and row['reason'] in text
    assert json.dumps(row, ensure_ascii=False) == original


def test_sse_resume_is_monotone_deduplicated_and_rejects_invalid_cursors(online_api):
    client, manager, _ = online_api
    identity = create(client).json()['id']
    session = manager.get(identity)
    session._emit('window', {'end_seconds': 5, 'scope': 'task'})
    session._emit('window', {'end_seconds': 10, 'scope': 'task'})
    session._set_state('completed')
    base = '/api/online/sessions/' + identity + '/events'
    response = client.get(base)
    assert response.headers['content-type'].startswith('text/event-stream')
    events = decode_sse(response)
    sequences = [event['sequence'] for event in events]
    assert sequences == sorted(set(sequences))
    resumed = decode_sse(client.get(base + '?after=0', headers={'Last-Event-ID': str(sequences[-2])}))
    assert [event['sequence'] for event in resumed] == sequences[-1:]
    assert client.get(base + '?after=' + str(sequences[-1])).text == ''
    assert client.get(base, headers={'Last-Event-ID': 'invalid'}).status_code == 400
    assert client.get(base + '?after=-2').status_code == 400


def test_evicted_cursor_sends_current_snapshot_instead_of_partial_history(online_api):
    client, manager, _ = online_api
    identity = create(client).json()['id']
    session = manager.get(identity)
    for index in range(300):
        session._emit('progress', {'position_seconds': index / 4})
    session._set_state('completed')
    response = client.get('/api/online/sessions/' + identity + '/events?after=0')
    events = decode_sse(response)
    assert len(events) == 1 and events[0]['type'] == 'snapshot'
    assert events[0]['sequence'] == session.sequence - 1
    assert events[0]['payload']['state'] == 'completed'
    assert 'events' not in events[0]['payload']
    assert client.get('/api/online/sessions/' + identity + '/events?after=' + str(events[0]['sequence'])).text == ''
    assert len(session._sse_packets) == len(session.events) == 256


def test_sse_route_uses_cached_frames_without_snapshot_or_reencoding(online_api, monkeypatch):
    client, manager, _ = online_api
    identity = create(client).json()['id']
    session = manager.get(identity)
    session._emit('window', {'end_seconds': 5, 'scope': 'task'})
    session._set_state('completed')
    monkeypatch.setattr(session, 'snapshot', lambda *a, **k: pytest.fail('SSE route requested a full snapshot'))
    events = decode_sse(client.get('/api/online/sessions/' + identity + '/events'))
    assert events[-1]['type'] == 'state' and events[-1]['payload']['state'] == 'completed'
    assert client.get('/api/online/sessions/' + identity + '/events?after=' + str(events[-1]['sequence'])).text == ''


def test_interrupted_historical_session_is_read_only(online_api):
    client, manager, _ = online_api
    identity = create(client).json()['id']
    session = manager.get(identity)
    session._set_state('paused')
    restored = OnlineManager(manager.cfg)
    saved = restored.get(identity)
    assert saved['state'] == 'interrupted'
    manager.sessions[identity] = saved
    response = client.post('/api/online/sessions/' + identity + '/control', json={'action': 'resume'})
    assert response.status_code == 409
    assert client.get('/api/online/sessions/' + identity).json()['state'] == 'interrupted'


def test_additive_online_routes_preserve_offline_request_contract(online_cfg, monkeypatch):
    from eeg_agent.brain import BrainManager
    from eeg_agent.service import create_app
    import eeg_agent.online.router as online_router

    class Tasks:
        def list(self):
            return []
        def get(self, identity):
            raise KeyError(identity)

    for key in ('EEG_API_BASE_URL', 'EEG_API_MODEL', 'EEG_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    tasks = Tasks()
    brain = BrainManager(online_cfg, tasks)
    online = OnlineManager(online_cfg, model_factory=FakeModel)
    monkeypatch.setattr(online_router, 'OnlineManager', lambda cfg: online)
    app = create_app(online_cfg, task_manager=tasks, knowledge_index=object(), brain_manager=brain)
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/monitor').status_code == 200
        page = client.get('/online')
        assert page.status_code == 200 and '/assets/online.js' in page.text
        assert client.get('/assets/online.js').status_code == 200
        assert client.get('/api/brain/recordings').json()['input_policy'] == 'raw_eeg_only'
        catalog = client.get('/api/brain/catalog').json()
        assert {agent['id'] for agent in catalog['agents']} == {'vrms', 'fatigue', 'emotion'}
        raw = client.post('/api/brain/sessions', json={'backend': 'mock', 'input_policy': 'raw_eeg_only'})
        assert raw.status_code == 200 and raw.json()['input_policy'] == 'raw_eeg_only'
        assert client.post('/api/brain/sessions', json={'backend': 'mock', 'input_policy': 'research_path'}).status_code == 422
        assert client.post('/api/brain/sessions', json={'backend': 'mock', 'vrms_path_id': 'path-000'}).status_code == 422
        assert online.list() == []  # Offline session creation does not start a replay.


@pytest.mark.parametrize('value', ['Infinity', 'NaN', .0001])
def test_nonfinite_or_subsample_replay_limit_is_rejected_before_start(online_api, value):
    client, manager, _ = online_api
    response = create(client, max_seconds=value)
    assert response.status_code == 422
    assert manager.list() == []


@pytest.mark.parametrize('header', [False, True])
def test_sse_rejects_future_cursor_in_live_and_historical_sessions(online_api, header):
    client, manager, _ = online_api
    identity = create(client).json()['id']
    session = manager.get(identity)
    session.control('stop')
    cursor = session.sequence + 100
    url = '/api/online/sessions/' + identity + '/events'
    headers = {'Last-Event-ID': str(cursor)} if header else {}
    if not header:
        url += '?after=' + str(cursor)
    assert client.get(url, headers=headers).status_code == 400
    manager.sessions[identity] = session.directory / 'session.json'
    assert client.get(url, headers=headers).status_code == 400
