"""Provider options are verified without cloud traffic or real credentials."""
from types import SimpleNamespace

import pytest

from eeg_agent.agents import CloudProvider
from eeg_agent.cloud_config import save_settings
from eeg_agent.config import load_config
from eeg_agent import raw_tool_agent


@pytest.mark.parametrize('base,model,disabled', [
    ('https://api.deepseek.com', 'deepseek-flash', True),
    ('https://api.deepseek.com/v1', 'deepseek-pro', True),
    ('https://gateway.example/v1', 'deepseek-flash', False),
    ('https://api.deepseek.com', 'other-model', False),
])
def test_native_deepseek_structured_workflow_uses_non_thinking(base, model, disabled, tmp_path, monkeypatch):
    path = tmp_path / '.env'
    save_settings({'base_url': base, 'model_id': model, 'api_key': 'TEST_ONLY'}, path)
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
            choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='{"explanation":"ok"}'))])
    monkeypatch.setattr('openai.OpenAI', lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    cfg = load_config()
    cfg['cloud_config_path'] = str(path)
    provider = CloudProvider('api', cfg, lambda *args: None)
    assert provider.call('Supervisor', {}) == {'explanation': 'ok'}
    assert len(sent) == 1 and sent[0]['max_tokens'] == 700
    if disabled:
        assert sent[0]['extra_body'] == {'thinking': {'type': 'disabled'}}
    else:
        assert 'extra_body' not in sent[0]
    summary = provider.summary()
    assert summary['thinking_disabled'] is disabled
    assert summary['model_id'] == model
    assert summary['records'][0]['finish_reason'] == 'stop'
    assert not summary['records'][0]['output_truncated']
    assert 'TEST_ONLY' not in str(sent) and 'TEST_ONLY' not in str(summary)


def test_deepseek_classifier_sends_native_tools_and_keeps_tool_call_ids(tmp_path, monkeypatch):
    path = tmp_path / '.env'
    save_settings({'base_url': 'https://api.deepseek.com', 'model_id': 'deepseek-flash', 'api_key': 'TEST_ONLY'}, path)
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        value = {'model': 'deepseek-flash', 'choices': [{'finish_reason': 'tool_calls',
            'message': {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'ds-call1', 'type': 'function', 'function': {'name': 'corrective_evidence',
                 'arguments': '{"query":"qsingle"}'}}]}}]}
        return SimpleNamespace(model='deepseek-flash', usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
            choices=[SimpleNamespace(finish_reason='tool_calls', message=SimpleNamespace(content=None))],
            model_dump=lambda **_: value)
    monkeypatch.setattr('openai.OpenAI', lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    cfg = load_config()
    cfg['cloud_config_path'] = str(path)
    provider = CloudProvider('api', cfg, lambda *args: None)
    result = provider.call('VRMSAgent', {'generation_phase': 'vrms_tool_agent', 'step': 0},
                           tools=raw_tool_agent.TOOLS, messages=[{'role': 'user', 'content': 'anonymous'}])
    assert sent[0]['tools'][0]['function']['parameters']['required'] == ['query']
    assert sent[0]['tool_choice'] == {'type': 'function', 'function': {'name': 'corrective_evidence'}}
    assert sent[0]['max_tokens'] == 4096 and 'response_format' not in sent[0]
    assert sent[0]['extra_body'] == {'thinking': {'type': 'disabled'}}
    assert result['choices'][0]['message']['tool_calls'][0]['id'] == 'ds-call1'
    assert provider.summary()['records'][0]['api_protocol'] == 'chat_completions'
