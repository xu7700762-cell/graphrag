"""Cloud settings tests use temporary files and an in-process API double."""
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APITimeoutError

from eeg_agent import cloud_config
from eeg_agent.config import load_config
from eeg_agent.service import create_app


ENV_NAMES = ('EEG_API_BASE_URL', 'EEG_API_MODEL', 'EEG_API_KEY')
SECRET = 'sk-test-private-never-return'
VALUES = {'base_url': 'https://cloud.example/v1', 'model_id': 'vr-test', 'api_key': SECRET}


class ManagerStub:
    def __init__(self):
        self.items = []

    def list(self):
        return self.items


@pytest.fixture
def api(tmp_path, monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    cfg = load_config()
    cfg['output_root'] = str(tmp_path / 'outputs')
    cfg['data_root'] = str(tmp_path / 'data')
    manager = ManagerStub()
    path = tmp_path / '.env'
    app = create_app(cfg, task_manager=manager, knowledge_index=object(), cloud_config_path=path)
    return TestClient(app), manager, path, app


def fake_provider(monkeypatch, error=None, response=None):
    calls = {'client': [], 'requests': []}

    def create(**kwargs):
        calls['requests'].append(kwargs)
        if error is not None:
            raise error
        return response if response is not None else SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='OK'), finish_reason='stop')])

    def client(**kwargs):
        calls['client'].append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                               responses=SimpleNamespace(create=create))

    monkeypatch.setattr('openai.OpenAI', client)
    return calls


def test_save_read_public_and_preserve_key(api):
    client, _, path, _ = api
    path.write_text('# retained comment\nOTHER_OPTION=unchanged\n', encoding='utf-8')
    assert not client.get('/api/cloud-settings').json()['configured']
    saved = client.put('/api/cloud-settings', json=VALUES)
    assert saved.status_code == 200
    value = saved.json()
    assert value['base_url'] == VALUES['base_url']
    assert value['model_id'] == VALUES['model_id']
    assert value['key_configured'] and value['configured'] and not value['in_use']
    assert value['sources'] == {'base_url': 'local_file', 'model_id': 'local_file', 'api_key': 'local_file'}
    assert SECRET not in saved.text
    assert SECRET not in client.get('/api/cloud-settings').text
    text = path.read_text(encoding='utf-8')
    assert '# retained comment' in text and 'OTHER_OPTION=unchanged' in text
    assert cloud_config.read_settings(path)['EEG_API_KEY'] == SECRET
    for payload in ({'base_url': VALUES['base_url'], 'model_id': 'next-model'},
                    {'base_url': VALUES['base_url'], 'model_id': 'next-model', 'api_key': ''}):
        assert client.put('/api/cloud-settings', json=payload).status_code == 200
        assert cloud_config.read_settings(path)['EEG_API_KEY'] == SECRET


def test_settings_can_return_profile_keys_for_visible_local_editor(api):
    client, _, _, _ = api
    client.put('/api/cloud-settings', json=VALUES)
    response = client.get('/api/cloud-settings?include_keys=true')
    assert response.status_code == 200
    assert response.json()['profiles'][response.json()['active_profile']]['api_key'] == SECRET
    assert response.headers['cache-control'] == 'no-store'
    assert SECRET not in client.get('/api/cloud-settings').text
    assert client.get('/api/cloud-settings?include_keys=true', headers={'Origin': 'https://other.example'}).status_code == 403


def test_unsaved_input_test_uses_only_input_and_does_not_save_or_switch(api, monkeypatch):
    client, _, path, app = api
    client.put('/api/cloud-settings', json=VALUES)
    before = path.read_bytes(), path.with_name('cloud_profiles.json').read_bytes()
    calls = fake_provider(monkeypatch)
    payload = {'base_url': 'https://draft.example/v1', 'model_id': 'gpt-draft',
               'api_key': 'draft-private-key', 'protocol': 'chat_completions'}
    response = client.post('/api/cloud-settings/test-input', json=payload)
    assert response.status_code == 200 and response.json()['ok']
    assert response.json()['model_id'] == 'gpt-draft'
    assert calls['client'][0]['base_url'] == payload['base_url']
    assert calls['client'][0]['api_key'] == payload['api_key']
    assert calls['requests'][0]['messages'][0]['content'].endswith('Reply OK.')
    assert len(calls['requests']) == 1
    assert before == (path.read_bytes(), path.with_name('cloud_profiles.json').read_bytes())
    assert not app.state.cloud_testing
    assert payload['api_key'] not in response.text


@pytest.mark.parametrize('payload', [[], {}, {**VALUES, 'api_key': ''},
    {**VALUES, 'base_url': 'file:///secret'}, {**VALUES, 'protocol': 'invalid'}])
def test_invalid_input_test_does_not_call_provider_or_echo_key(api, monkeypatch, payload):
    client, _, path, app = api
    calls = fake_provider(monkeypatch)
    response = client.post('/api/cloud-settings/test-input', json=payload)
    assert response.status_code == 400 and SECRET not in response.text
    assert not calls['requests'] and not path.exists() and not app.state.cloud_testing


def test_input_test_blocks_busy_and_rejects_external_origin(api, monkeypatch):
    client, manager, _, _ = api
    calls = fake_provider(monkeypatch)
    manager.items = [{'id': 'active', 'state': 'running', 'request': {'backend': 'api'}}]
    assert client.post('/api/cloud-settings/test-input', json=VALUES).status_code == 409
    manager.items = []
    assert client.post('/api/cloud-settings/test-input', json=VALUES,
                       headers={'Origin': 'https://other.example'}).status_code == 403
    assert not calls['requests']


def test_file_values_override_environment_and_missing_values_fall_back(api, monkeypatch):
    client, _, path, _ = api
    for name, value in zip(ENV_NAMES, ('https://environment.example/v1', 'env-model', 'env-secret')):
        monkeypatch.setenv(name, value)
    value = client.get('/api/cloud-settings').json()
    assert value['configured'] and value['model_id'] == 'env-model'
    assert set(value['sources'].values()) == {'environment'}
    assert 'env-secret' not in client.get('/api/cloud-settings').text
    path.write_text('EEG_API_MODEL=local-model\n', encoding='utf-8')
    value = client.get('/api/cloud-settings').json()
    assert value['model_id'] == 'local-model'
    assert value['sources']['model_id'] == 'local_file'
    assert value['sources']['api_key'] == 'environment'
    assert client.put('/api/cloud-settings', json=VALUES).status_code == 200
    assert cloud_config.read_settings(path)['EEG_API_KEY'] == SECRET
    assert cloud_config.read_settings(path)['EEG_API_MODEL'] == VALUES['model_id']


@pytest.mark.parametrize('override', [
    {'base_url': 'file:///tmp/key'},
    {'base_url': 'https://user:password@cloud.example/v1'},
    {'base_url': 'https://cloud.example/v1\nINJECTED=yes'},
    {'model_id': 'model\nINJECTED=yes'},
    {'api_key': SECRET + '\r\nINJECTED=yes'},
    {'api_key': 123},
])
def test_invalid_settings_do_not_echo_secret_or_replace_file(api, override):
    client, _, path, _ = api
    assert client.put('/api/cloud-settings', json=VALUES).status_code == 200
    before = path.read_bytes()
    response = client.put('/api/cloud-settings', json={**VALUES, **override})
    assert response.status_code == 400
    assert SECRET not in response.text and 'INJECTED' not in response.text
    assert path.read_bytes() == before


def test_invalid_json_and_non_object_request_are_sanitized(api):
    client, _, path, _ = api
    malformed = client.put('/api/cloud-settings', content='{"api_key":"' + SECRET,
                           headers={'Content-Type': 'application/json'})
    assert malformed.status_code == 400 and SECRET not in malformed.text
    value = client.put('/api/cloud-settings', json=[SECRET])
    assert value.status_code == 400 and SECRET not in value.text
    assert not path.exists()


@pytest.mark.parametrize('state', ['created', 'running', 'finalizing', 'stopping'])
def test_active_api_task_blocks_save_and_test(api, state):
    client, manager, path, _ = api
    assert client.put('/api/cloud-settings', json=VALUES).status_code == 200
    before = path.read_bytes()
    manager.items = [{'id': 'active', 'state': state, 'request': {'backend': 'api'}}]
    assert client.get('/api/cloud-settings').json()['in_use']
    assert client.put('/api/cloud-settings', json=VALUES).status_code == 409
    assert client.post('/api/cloud-settings/test').status_code == 409
    assert path.read_bytes() == before


def test_legacy_api_run_blocks_save_and_test(api):
    client, _, _, app = api
    app.state.legacy_runs['legacy'] = SimpleNamespace(state='running', backend='api')
    assert client.get('/api/cloud-settings').json()['in_use']
    assert client.put('/api/cloud-settings', json=VALUES).status_code == 409
    assert client.post('/api/cloud-settings/test').status_code == 409


def test_active_mock_task_can_save_and_connection_uses_one_fixed_request(api, monkeypatch):
    client, manager, _, _ = api
    calls = fake_provider(monkeypatch)
    manager.items = [{'id': 'mock', 'state': 'running', 'request': {'backend': 'mock'}}]
    assert client.put('/api/cloud-settings', json=VALUES).status_code == 200
    assert not client.get('/api/cloud-settings').json()['in_use']
    response = client.post('/api/cloud-settings/test')
    assert response.status_code == 200
    value = response.json()
    assert value['ok'] and value['model_id'] == VALUES['model_id']
    assert value['latency_seconds'] >= 0
    assert SECRET not in response.text
    assert len(calls['client']) == 1 and len(calls['requests']) == 1
    options = calls['client'][0]
    assert options['base_url'] == VALUES['base_url'] and options['api_key'] == SECRET
    assert options['max_retries'] == 0 and options['timeout'] == 10
    assert calls['requests'][0]['model'] == VALUES['model_id']
    assert calls['requests'][0]['max_tokens'] == 64
    assert SECRET not in str(calls['requests'])


@pytest.mark.parametrize('content, reasoning, finish, expected_ok', [
    ('OK ' + SECRET, None, 'stop', True),
    ('', 'private reasoning ' + SECRET, 'length', True),
    ('', None, 'stop', False),
])
def test_connection_distinguishes_text_reasoning_limit_and_empty_response(
        api, monkeypatch, content, reasoning, finish, expected_ok):
    client, _, _, app = api
    assert client.put('/api/cloud-settings', json=VALUES).status_code == 200
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, reasoning_content=reasoning), finish_reason=finish)])
    calls = fake_provider(monkeypatch, response=response)
    result = client.post('/api/cloud-settings/test')
    assert result.status_code == 200
    value = result.json()
    assert value['ok'] is expected_ok
    assert value['finish_reason'] == finish
    assert SECRET not in result.text and 'private reasoning' not in result.text
    if expected_ok:
        assert value['reason'] is None
    else:
        assert value['reason'] == '接口未返回有效文本响应'
    if reasoning and finish == 'length':
        assert value['note'] and '上限' in value['note'] and '验证' in value['note']
    else:
        assert not value.get('note')
    assert len(calls['requests']) == 1 and calls['requests'][0]['max_tokens'] == 64
    assert calls['client'][0]['max_retries'] == 0
    assert not app.state.cloud_testing
    assert not client.get('/api/cloud-settings').json()['in_use']


@pytest.mark.parametrize('error', [
    RuntimeError('provider response containing ' + SECRET),
    APITimeoutError(request=httpx.Request('POST', 'https://cloud.example/v1')),
])
def test_connection_failure_is_sanitized_and_not_retried(api, monkeypatch, error):
    client, _, _, _ = api
    assert client.put('/api/cloud-settings', json=VALUES).status_code == 200
    calls = fake_provider(monkeypatch, error)
    response = client.post('/api/cloud-settings/test')
    assert response.status_code == 200
    assert not response.json()['ok'] and response.json()['reason']
    assert SECRET not in response.text and 'provider response' not in response.text
    assert len(calls['requests']) == 1


def test_unconfigured_test_does_not_make_a_cloud_request(api, monkeypatch):
    client, _, _, _ = api
    calls = fake_provider(monkeypatch)
    response = client.post('/api/cloud-settings/test')
    assert response.status_code in (200, 400)
    assert not response.json().get('ok', False)
    assert not calls['client'] and not calls['requests']


def test_gpt_connection_test_uses_responses_without_echoing_content(api, monkeypatch):
    client, _, _, _ = api
    assert client.put('/api/cloud-settings', json={**VALUES, 'model_id': 'gpt-6.1-sol'}).status_code == 200
    calls = fake_provider(monkeypatch, response=SimpleNamespace(status='completed', output_text='OK ' + SECRET))
    response = client.post('/api/cloud-settings/test')
    assert response.json()['ok'] and response.json()['finish_reason'] == 'completed'
    assert SECRET not in response.text
    request = calls['requests'][0]
    assert request['model'] == 'gpt-6.1-sol' and request['store'] is False
    assert request['max_output_tokens'] == 64 and 'messages' not in request
