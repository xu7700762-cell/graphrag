"""Independent project provider keys, switching and active-protocol tests."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from eeg_agent import cloud_config
from eeg_agent.config import load_config
from eeg_agent.service import create_app

DEEPSEEK = {'profile_id': 'deepseek', 'base_url': 'https://api.deepseek.com',
            'model_id': 'deepseek-flash', 'api_key': 'DEEPSEEK_PROJECT_KEY'}
CPA = {'profile_id': 'cpa_gpt', 'base_url': 'http://127.0.0.1:8317/v1',
       'model_id': 'gpt-6.1-sol', 'api_key': 'CPA_PROJECT_KEY'}
RELAY = {'profile_id': 'relay', 'base_url': 'https://relay.example/v1',
         'model_id': 'gpt-relay', 'api_key': 'RELAY_PROJECT_KEY', 'protocol': 'chat_completions'}


class Tasks:
    items = []
    def list(self):
        return self.items


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    for name in cloud_config.NAMES.values():
        monkeypatch.delenv(name, raising=False)
    cfg = load_config()
    cfg['output_root'] = str(tmp_path / 'outputs')
    target = tmp_path / '.env'
    tasks = Tasks()
    tasks.items = []
    app = create_app(cfg, task_manager=tasks, knowledge_index=object(), cloud_config_path=target)
    return TestClient(app), target, tasks


def test_read_does_not_create_files_and_defaults_have_no_keys(profiles):
    client, target, _ = profiles
    value = client.get('/api/cloud-settings').json()
    assert value['active_profile'] == 'deepseek'
    assert set(value['profiles']) == {'deepseek', 'cpa_gpt', 'relay'}
    assert value['profiles']['cpa_gpt']['base_url'] == CPA['base_url']
    assert value['profiles']['cpa_gpt']['model_id'] == CPA['model_id']
    assert all(not p['configured'] and not p['key_configured'] for p in value['profiles'].values())
    assert not target.exists() and not target.with_name('cloud_profiles.json').exists()


def test_legacy_deepseek_import_is_read_only_and_inactive_cpa_save_preserves_it(profiles):
    client, target, _ = profiles
    original = '# original\nOTHER=keep\n' + '\n'.join(
        name + '=' + json.dumps(DEEPSEEK[field]) for field, name in cloud_config.NAMES.items()) + '\n'
    target.write_text(original, encoding='utf-8')
    value = client.get('/api/cloud-settings').json()
    assert value['active_profile'] == 'deepseek' and value['profiles']['deepseek']['configured']
    assert not value['profiles']['cpa_gpt']['configured']
    assert target.read_text(encoding='utf-8') == original
    assert not target.with_name('cloud_profiles.json').exists()
    saved = client.put('/api/cloud-settings', json={**CPA, 'activate': False})
    assert saved.status_code == 200, saved.text
    assert saved.json()['active_profile'] == 'deepseek'
    assert target.read_text(encoding='utf-8') == original
    for key in (DEEPSEEK['api_key'], CPA['api_key']):
        assert key not in saved.text and key not in client.get('/api/cloud-settings').text
    stored = json.loads(target.with_name('cloud_profiles.json').read_text(encoding='utf-8'))
    assert stored['profiles']['deepseek']['api_key'] == DEEPSEEK['api_key']
    assert stored['profiles']['cpa_gpt']['api_key'] == CPA['api_key']


def test_switch_roundtrip_retains_independent_keys_and_current_snapshot(profiles):
    client, target, _ = profiles
    assert client.put('/api/cloud-settings', json=DEEPSEEK).status_code == 200
    assert client.put('/api/cloud-settings', json={**CPA, 'activate': False}).status_code == 200
    before = cloud_config.read_settings(target)
    switched = client.put('/api/cloud-settings', json={'active_profile': 'cpa_gpt'})
    assert switched.status_code == 200
    assert switched.json()['active_profile'] == 'cpa_gpt'
    assert cloud_config.read_settings(target)['EEG_API_KEY'] == CPA['api_key']
    assert before['EEG_API_KEY'] == DEEPSEEK['api_key']
    assert client.put('/api/cloud-settings', json={**CPA, 'api_key': '', 'model_id': 'gpt-6.1-sol'}).status_code == 200
    assert cloud_config.read_settings(target)['EEG_API_KEY'] == CPA['api_key']
    assert client.put('/api/cloud-settings', json={'active_profile': 'deepseek'}).status_code == 200
    assert cloud_config.read_settings(target)['EEG_API_KEY'] == DEEPSEEK['api_key']
    assert cloud_config.read_settings(target)['EEG_API_MODEL'] == DEEPSEEK['model_id']


def test_empty_cpa_key_is_never_borrowed_from_deepseek(profiles):
    client, target, _ = profiles
    client.put('/api/cloud-settings', json=DEEPSEEK)
    before = target.read_bytes()
    response = client.put('/api/cloud-settings', json={**CPA, 'api_key': ''})
    assert response.status_code == 400 and '独立API密钥' in response.json()['detail']
    assert target.read_bytes() == before
    stored = client.put('/api/cloud-settings', json={**CPA, 'api_key': '', 'activate': False})
    assert stored.status_code == 200
    assert not stored.json()['profiles']['cpa_gpt']['key_configured']
    response = client.put('/api/cloud-settings', json={'active_profile': 'cpa_gpt'})
    assert response.status_code == 400 and target.read_bytes() == before
    assert cloud_config.read_settings(target)['EEG_API_KEY'] == DEEPSEEK['api_key']


@pytest.mark.parametrize('payload', [
    {'active_profile': []}, {'active_profile': 'unknown'},
    {'active_profile': 'deepseek', 'api_key': 'MUST_NOT_ECHO'},
    {**CPA, 'profile_id': []}, {**CPA, 'activate': 'yes'},
    {**CPA, 'api_key': 'MUST_NOT_ECHO\ninvalid'},
])
def test_invalid_profile_requests_are_sanitized_and_do_not_mutate(profiles, payload):
    client, target, _ = profiles
    client.put('/api/cloud-settings', json=DEEPSEEK)
    before = target.read_bytes()
    profiles_before = target.with_name('cloud_profiles.json').read_bytes()
    response = client.put('/api/cloud-settings', json=payload)
    assert response.status_code == 400
    assert 'MUST_NOT_ECHO' not in response.text and CPA['api_key'] not in response.text
    assert target.read_bytes() == before
    assert target.with_name('cloud_profiles.json').read_bytes() == profiles_before


def test_active_cloud_use_blocks_switch_save_and_test(profiles):
    client, target, tasks = profiles
    client.put('/api/cloud-settings', json=DEEPSEEK)
    client.put('/api/cloud-settings', json={**CPA, 'activate': False})
    before = target.read_bytes()
    profiles_before = target.with_name('cloud_profiles.json').read_bytes()
    tasks.items = [{'id': 'active', 'state': 'running', 'request': {'backend': 'api'}}]
    assert client.put('/api/cloud-settings', json={'active_profile': 'cpa_gpt'}).status_code == 409
    assert client.put('/api/cloud-settings', json=CPA).status_code == 409
    assert client.post('/api/cloud-settings/test').status_code == 409
    assert target.read_bytes() == before
    assert target.with_name('cloud_profiles.json').read_bytes() == profiles_before


def test_connection_test_uses_only_active_profile_and_matching_api_protocol(profiles, monkeypatch):
    client, _, _ = profiles
    requests, clients = [], []
    def chat(**kwargs):
        requests.append(('chat', kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content='OK'))])
    def responses(**kwargs):
        requests.append(('responses', kwargs))
        return SimpleNamespace(status='completed', output_text='OK')
    def provider(**kwargs):
        clients.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=chat)),
                               responses=SimpleNamespace(create=responses))
    monkeypatch.setattr('openai.OpenAI', provider)
    client.put('/api/cloud-settings', json=DEEPSEEK)
    client.put('/api/cloud-settings', json={**CPA, 'activate': False})
    assert client.post('/api/cloud-settings/test').json()['ok']
    assert clients[-1]['api_key'] == DEEPSEEK['api_key'] and requests[-1][0] == 'chat'
    client.put('/api/cloud-settings', json={'active_profile': 'cpa_gpt'})
    tested = client.post('/api/cloud-settings/test')
    assert tested.json()['ok']
    assert clients[-1]['api_key'] == CPA['api_key'] and requests[-1][0] == 'responses'
    assert all(key not in tested.text for key in (CPA['api_key'], DEEPSEEK['api_key']))


def test_failed_env_write_rolls_back_profile_metadata(profiles, monkeypatch):
    client, target, _ = profiles
    client.put('/api/cloud-settings', json=DEEPSEEK)
    client.put('/api/cloud-settings', json={**CPA, 'activate': False})
    before = target.read_bytes()
    profiles_before = target.with_name('cloud_profiles.json').read_bytes()
    def cannot_write(*args):
        raise OSError('private filesystem detail')
    monkeypatch.setattr(cloud_config, '_write_active', cannot_write)
    response = client.put('/api/cloud-settings', json={'active_profile': 'cpa_gpt'})
    assert response.status_code == 500 and 'private filesystem detail' not in response.text
    assert target.read_bytes() == before
    assert target.with_name('cloud_profiles.json').read_bytes() == profiles_before


def test_relay_migration_and_switching_preserve_three_independent_keys(profiles):
    client, target, _ = profiles
    client.put('/api/cloud-settings', json=DEEPSEEK)
    client.put('/api/cloud-settings', json=CPA)
    stored = json.loads(target.with_name('cloud_profiles.json').read_text(encoding='utf-8'))
    del stored['profiles']['relay']  # Existing two-channel installations.
    target.with_name('cloud_profiles.json').write_text(json.dumps(stored), encoding='utf-8')
    before = target.read_bytes()
    assert not client.get('/api/cloud-settings').json()['profiles']['relay']['key_configured']
    rejected = client.put('/api/cloud-settings', json={**RELAY, 'api_key': ''})
    assert rejected.status_code == 400 and target.read_bytes() == before
    saved = client.put('/api/cloud-settings', json=RELAY)
    assert saved.status_code == 200 and saved.json()['active_profile'] == 'relay'
    assert cloud_config.read_settings(target)['EEG_API_PROTOCOL'] == 'chat_completions'
    for profile in (DEEPSEEK, CPA, RELAY):
        assert client.put('/api/cloud-settings', json={'active_profile': profile['profile_id']}).status_code == 200
        assert cloud_config.read_settings(target)['EEG_API_KEY'] == profile['api_key']
        assert profile['api_key'] not in client.get('/api/cloud-settings').text
    assert client.put('/api/cloud-settings', json={**RELAY, 'api_key': '', 'protocol': 'responses'}).status_code == 200
    assert cloud_config.read_settings(target)['EEG_API_KEY'] == RELAY['api_key']
    assert cloud_config.read_settings(target)['EEG_API_PROTOCOL'] == 'responses'


@pytest.mark.parametrize('model', ['gpt-relay', 'custom-model'])
@pytest.mark.parametrize('protocol', ['responses', 'chat_completions'])
def test_relay_test_and_agent_use_selected_endpoint_key_and_protocol(profiles, monkeypatch, model, protocol):
    import httpx
    import openai
    from eeg_agent.agents import CloudProvider

    client, target, _ = profiles
    sdk = openai.OpenAI
    outbound = []
    def transport(request):
        body = json.loads(request.content)
        outbound.append((request, body))
        if request.url.path.endswith('/responses'):
            reply = {'id': 'resp_test', 'object': 'response', 'created_at': 0, 'status': 'completed',
                     'model': model, 'output': [{'id': 'msg_test', 'type': 'message', 'role': 'assistant',
                         'status': 'completed', 'content': [{'type': 'output_text', 'text': '{"ok":true}', 'annotations': []}]}]}
        else:
            reply = {'id': 'chat_test', 'object': 'chat.completion', 'created': 0, 'model': model,
                     'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': '{"ok":true}'}}]}
        return httpx.Response(200, json=reply)
    monkeypatch.setattr(openai, 'OpenAI', lambda **options: sdk(
        **options, http_client=httpx.Client(transport=httpx.MockTransport(transport))))
    assert client.put('/api/cloud-settings', json={**RELAY, 'model_id': model, 'protocol': protocol}).status_code == 200
    assert client.post('/api/cloud-settings/test').json()['ok']
    cfg = load_config()
    cfg['cloud_config_path'] = str(target)
    provider = CloudProvider('api', cfg, lambda *args: None)
    try:
        assert provider.call('Supervisor', {'generation_phase': 'planning'}) == {'ok': True}
        assert provider.protocol == protocol
        assert provider.calls[0]['api_protocol'] == protocol
    finally:
        provider.client.close()
    assert len(outbound) == 2
    for request, body in outbound:
        assert str(request.url) == RELAY['base_url'] + ('/responses' if protocol == 'responses' else '/chat/completions')
        assert request.headers['authorization'] == 'Bearer ' + RELAY['api_key']
        assert body['model'] == model
        assert ('input' in body) == (protocol == 'responses')
        assert 'thinking' not in body


@pytest.mark.parametrize('protocol', ['invalid', [], None])
def test_invalid_relay_protocol_does_not_mutate_saved_configuration(profiles, protocol):
    client, target, _ = profiles
    client.put('/api/cloud-settings', json=CPA)
    before = target.read_bytes(), target.with_name('cloud_profiles.json').read_bytes()
    assert client.put('/api/cloud-settings', json={**RELAY, 'protocol': protocol}).status_code == 400
    assert before == (target.read_bytes(), target.with_name('cloud_profiles.json').read_bytes())
