"""Adapter contracts use process doubles and never contact an API."""
import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from eeg_agent import vrms_runtime as runtime


@pytest.fixture
def worker(monkeypatch, tmp_path):
    executable = tmp_path / 'python.exe'
    executable.touch()
    monkeypatch.setattr(runtime, 'WORKER_PYTHON', executable)
    monkeypatch.setattr(runtime, 'WORKER_ROOT', tmp_path)
    monkeypatch.setattr(runtime, '_cache', None)
    monkeypatch.setattr(runtime, '_cache_time', 0)
    calls = []
    local = {'available': True, 'status': 'completed', 'measurements': {
        'mil': {'probability_high': .6, 'predicted_class': 'High'},
        'numeric_fusion': {'uncalibrated_combined_score': .7, 'raw_combined_class': 'High'},
        'cloud_judgment': {'final_class': None, 'generation_status': 'not_requested'}},
        'limitations': [], 'provenance': {'run_id': 'fresh-run'}}

    def fake(command, **kwargs):
        calls.append((command, kwargs))
        value = {'pipeline_id': runtime.PIPELINE_ID, 'paths': [
            {'id': 'path-000', 'label': '研究路径 001', 'true_high': True, 'score': 50},
            {'id': 'path-145', 'label': '研究路径 146'}],
            'capabilities': {'fresh_mil_head': True}} if '--catalog' in command else copy.deepcopy(local)
        return SimpleNamespace(returncode=0, stdout=json.dumps(value, ensure_ascii=False), stderr='')

    monkeypatch.setattr(runtime.subprocess, 'run', fake)
    return calls, local


def test_catalog_anonymous_cached_copy(worker):
    calls, _ = worker
    value = runtime.catalog()
    assert value['status'] == 'ready'
    assert value['paths'][0] == {'id': 'path-000', 'title': '研究路径 001', 'source': runtime.PIPELINE_ID}
    assert 'true_high' not in json.dumps(value)
    value['paths'].clear()
    assert len(runtime.catalog()['paths']) == 2
    assert len(calls) == 1
    assert calls[0][0][-1] == '--catalog'
    assert calls[0][1]['timeout'] == 30


@pytest.mark.parametrize('path_id', ['../../secret', 'path-146', 'path-001', 'path-0', None, 0])
def test_reject_paths_outside_actual_allowlist(worker, path_id):
    with pytest.raises(ValueError, match='有效匿名研究路径'):
        runtime.validate_path(path_id)
    assert runtime.assess(path_id)['available'] is False
    assert not any('--catalog' not in command for command, _ in worker[0])


def test_cloud_credential_only_in_stdin(worker):
    calls, local = worker
    local['measurements']['cloud_judgment'] = {'final_class': 'Low', 'generation_status': 'cloud_verified'}
    result = runtime.assess('path-000', {'base_url': 'https://gateway.example/v1',
                                        'model': 'gpt-6.1-sol', 'api_key': 'private-secret'})
    command, options = calls[-1]
    payload = json.loads(options['input'])
    assert payload['cloud'] is True
    assert payload['provider']['api_key'] == 'private-secret'
    assert 'private-secret' not in ' '.join(command)
    assert 'env' not in options
    assert options['timeout'] == 180 and options['capture_output'] is True
    assert result['measurements']['cloud_judgment']['generation_status'] == 'cloud_verified'
    assert result['pipeline_id'] == runtime.PIPELINE_ID


@pytest.mark.parametrize('provider', [None, {'model': 'unregistered-model', 'api_key': 'secret'},
                                      {'model': 'gpt-6.1-sol', 'base_url': 'https://gateway.example/v1'},
                                      {'model': 'gpt-6.1-sol', 'base_url': 'https://user:pass@example/v1', 'api_key': 'secret'}])
def test_missing_or_wrong_provider_keeps_local_without_cloud(worker, provider):
    result = runtime.assess('path-000', provider)
    payload = json.loads(worker[0][-1][1]['input'])
    assert payload['cloud'] is False and 'provider' not in payload
    assert result['available'] is True
    assert result['measurements']['mil']['predicted_class'] == 'High'
    assert result['measurements']['cloud_judgment']['generation_status'] == 'fallback'
    assert result['measurements']['cloud_judgment']['final_class'] is None
    assert result['limitations']


def test_explicit_local_is_not_cloud_failure(worker):
    result = runtime.assess('path-145', cloud=False)
    assert result['measurements']['cloud_judgment']['generation_status'] == 'not_requested'
    assert result['limitations'] == []


@pytest.mark.parametrize('model', ['deepseek-flash', 'deepseek-pro', 'deepseek-chat', 'deepseek-reasoner'])
def test_selected_deepseek_kept_without_model_substitution(worker, model):
    runtime.assess('path-000', {'base_url': 'https://api.deepseek.com', 'model': model, 'api_key': 'deepseek-secret'})
    payload = json.loads(worker[0][-1][1]['input'])
    assert payload['cloud'] is True
    assert payload['provider']['model'] == model
    assert payload['provider']['api_key'] == 'deepseek-secret'


@pytest.mark.parametrize('failure', ['timeout', 'process_error', 'invalid_json', 'reported_error'])
def test_failures_sanitized(worker, monkeypatch, failure):
    runtime.catalog()

    def fail(*args, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(args[0], 180, output='private-secret')
        if failure == 'process_error':
            return SimpleNamespace(returncode=1, stdout='private-secret', stderr='private-secret')
        if failure == 'reported_error':
            return SimpleNamespace(returncode=0, stdout=json.dumps({'available': False, 'reason': 'private-secret'}), stderr='')
        return SimpleNamespace(returncode=0, stdout='private-secret', stderr='')

    monkeypatch.setattr(runtime.subprocess, 'run', fail)
    result = runtime.assess('path-000', {'base_url': 'https://gateway.example/v1',
                                        'model': 'gpt-6.1-sol', 'api_key': 'private-secret'})
    assert result['available'] is False
    assert 'private-secret' not in json.dumps(result)


def test_worker_output_redacts_credential(worker):
    worker[1]['measurements']['cloud_judgment']['explanation'] = 'private-secret'
    worker[1]['api_key'] = 'private-secret'
    result = runtime.assess('path-000', {'base_url': 'https://gateway.example/v1',
                                        'model': 'gpt-6.1-sol', 'api_key': 'private-secret'})
    assert 'private-secret' not in json.dumps(result)
    assert 'api_key' not in result


@pytest.mark.parametrize('change', ['bad_measurements', 'wrong_path', 'wrong_pipeline', 'bad_provenance'])
def test_invalid_worker_identity_or_shape_not_published(worker, change):
    local = worker[1]
    if change == 'bad_measurements':
        local['measurements'] = None
    elif change == 'wrong_path':
        local['provenance']['path_id'] = 'path-145'
    elif change == 'wrong_pipeline':
        local['measurements']['pipeline'] = {'id': 'old-eegnet'}
    else:
        local['provenance'] = None
    assert runtime.assess('path-000', cloud=False)['available'] is False


def test_seed_catalogs_have_separate_cache_and_forward_selected_pipeline(worker, monkeypatch):
    calls = []

    def fake(command, **options):
        calls.append((command, options))
        if '--catalog' in command:
            seed = int(command[command.index('--seed') + 1]) if '--seed' in command else 2028
            result = {'pipeline_id': runtime.PIPELINE_IDS[seed], 'paths': [{'id': 'path-000'}]}
        else:
            payload = json.loads(options['input'])
            result = copy.deepcopy(worker[1])
            result['measurements']['pipeline'] = {'id': payload['pipeline_id'], 'seed': 2026}
        return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr='')

    monkeypatch.setattr(runtime.subprocess, 'run', fake)
    assert runtime.catalog()['seed'] == 2026
    assert runtime.catalog(2028)['pipeline_id'] == runtime.PIPELINE_IDS[2028]
    assert runtime.catalog()['seed'] == 2026
    assert len(calls) == 2
    result = runtime.assess('path-000', cloud=False, pipeline_id=runtime.PIPELINE_IDS[2026])
    assert result['seed'] == 2026
    assert json.loads(calls[-1][1]['input'])['pipeline_id'] == runtime.PIPELINE_IDS[2026]
    assert runtime.assess('path-000', cloud=False)['available'] is True
    assert runtime.assess('path-000', cloud=False, pipeline_id=runtime.PIPELINE_IDS[2028])['available'] is False
