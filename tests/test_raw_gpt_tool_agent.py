"""Real tool-protocol routing and class provenance, using a deterministic API double."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from eeg_agent.agents import CloudProvider
from eeg_agent import raw_model, raw_tool_agent


def prediction(ordinal=1):
    return {'ordinal': ordinal, 'mil_raw_high_probability': .4, 'mil_class': 'Low',
            'accepted_windows': 3, 'predicted_class': 'High', 'classification_layer': 'numeric_fusion',
            'numeric_fusion': {'raw_combined_score': .6, 'predicted_class': 'High',
                'actual_tools': [{'tool': name, 'predicted_class': 'High', 'raw_decision_score': .2}
                                 for name in raw_tool_agent.TOOL_NAMES]}}


def final(cls='Low', tools=('corrective_evidence',)):
    return {'id': 'qsingle', 'final_class': cls, 'confidence': 'low', 'tools_used': list(tools),
            'supporting_evidence': ['实际频谱证据'], 'conflicting_evidence': ['工具方向存在冲突'],
            'missing_evidence': [], 'explanation': '最终判为Low，保留工具冲突并降低定性置信度。'}


class Provider:
    model = 'gpt-6.1-sol'

    def __init__(self, final_value=None, fail=False, direct=False, name='corrective_evidence'):
        self.calls = []
        self.final_value = final_value or final()
        self.fail, self.direct, self.name = fail, direct, name

    def call(self, role, payload, tools=None, messages=None):
        self.calls.append(copy.deepcopy({'role': role, 'payload': payload, 'tools': tools, 'messages': messages}))
        if self.fail:
            raise RuntimeError('secret-from-a-provider-exception')
        if len(self.calls) == 1 and not self.direct:
            output = [{'type': 'function_call', 'name': self.name, 'call_id': 'call1',
                       'arguments': json.dumps({'query': 'qsingle'})}]
        else:
            output = [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(self.final_value)}]}]
        return {'status': 'completed', 'output': output}


def test_gpt_discrete_class_can_differ_without_mutating_numeric_evidence_or_sending_identity():
    row = prediction()
    row.update(subject_id='secret-subject', questionnaire=99, target_label='High', waveform=np.ones((30, 1280)))
    numeric = copy.deepcopy(row['numeric_fusion'])
    provider = Provider()
    result = raw_tool_agent.classify(provider, row)
    assert result['available'] and result['final_class'] == 'Low'
    assert result['generation_status'] == 'cloud_verified'
    assert result['actual_tool_calls'] == [*raw_tool_agent.TOOL_NAMES, 'corrective_evidence']
    assert row['numeric_fusion'] == numeric and row['predicted_class'] == 'High'
    outbound = json.dumps(provider.calls)
    assert not any(value in outbound for value in ('secret-subject', 'questionnaire": 99', 'target_label', 'waveform'))
    tool_output = provider.calls[1]['messages'][-1]
    assert tool_output['type'] == 'function_call_output'
    assert json.loads(tool_output['output'])['raw_combined_class'] == 'High'


@pytest.mark.parametrize('provider', [Provider(fail=True), Provider(direct=True),
    Provider(name='unregistered'), Provider(final_value={**final(), 'llm_probability': .7}),
    Provider(final_value=final(tools=('unexecuted',)))])
def test_failed_invalid_or_unexecuted_tool_reply_never_becomes_a_final_class(provider):
    result = raw_tool_agent.classify(provider, prediction())
    assert not result['available'] and result['final_class'] is None
    assert result['generation_status'] == 'fallback'
    assert 'secret-from-a-provider-exception' not in json.dumps(result)


def test_unsupported_configuration_never_becomes_a_cloud_class():
    provider = Provider()
    provider.model = 'unsupported-model'
    result = raw_tool_agent.classify(provider, prediction())
    assert not result['available'] and not provider.calls and 'GPT 或 DeepSeek' in result['reason']


class DeepSeekProvider(Provider):
    model = 'deepseek-flash'

    def call(self, role, payload, tools=None, messages=None):
        self.calls.append(copy.deepcopy({'role': role, 'payload': payload, 'tools': tools, 'messages': messages}))
        if self.fail:
            raise RuntimeError('secret-from-a-provider-exception')
        if len(self.calls) == 1 and not self.direct:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call1', 'type': 'function', 'function': {'name': self.name,
                 'arguments': json.dumps({'query': 'qsingle'})}}]}
        else:
            message = {'role': 'assistant', 'content': json.dumps(self.final_value)}
        return {'choices': [{'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop', 'message': message}]}


def test_deepseek_reads_actual_tools_and_keeps_its_class_separate_from_numeric_scores():
    row = prediction()
    before = copy.deepcopy(row)
    provider = DeepSeekProvider()
    result = raw_tool_agent.classify(provider, row)
    assert result['available'] and result['final_class'] == 'Low'
    assert result['provider_model'] == 'deepseek-flash' and result['protocol'] == 'chat_completions'
    assert result['actual_tool_calls'] == [*raw_tool_agent.TOOL_NAMES, 'corrective_evidence']
    assert row == before
    messages = provider.calls[1]['messages']
    assert messages[-2]['role'] == 'assistant' and messages[-2]['tool_calls'][0]['id'] == 'call1'
    assert messages[-1]['role'] == 'tool' and messages[-1]['tool_call_id'] == 'call1'
    assert json.loads(messages[-1]['content'])['raw_combined_class'] == 'High'
    initial = json.loads(messages[1]['content'])
    assert initial['final_response_schema'] == raw_tool_agent.SCHEMA
    assert 'never translated or combined' in messages[0]['content']
    assert 'responses' not in json.dumps(provider.calls)


@pytest.mark.parametrize('model', ['gpt-relay', 'custom-model'])
def test_relay_chat_protocol_runs_complete_tool_conversation(model):
    provider = DeepSeekProvider()
    provider.model, provider.protocol = model, 'chat_completions'
    result = raw_tool_agent.classify(provider, prediction())
    assert result['available'] and result['final_class'] == 'Low'
    assert result['provider_model'] == model and result['protocol'] == 'chat_completions'
    assert 'corrective_evidence' in result['actual_tool_calls']
    assert provider.calls[1]['messages'][-1]['role'] == 'tool'


def test_relay_responses_protocol_accepts_custom_model_name():
    provider = Provider()
    provider.model, provider.protocol = 'custom-model', 'responses'
    result = raw_tool_agent.classify(provider, prediction())
    assert result['available'] and result['protocol'] == 'responses'
    assert provider.calls[1]['messages'][-1]['type'] == 'function_call_output'


@pytest.mark.parametrize('provider', [DeepSeekProvider(fail=True), DeepSeekProvider(direct=True),
    DeepSeekProvider(name='unregistered'), DeepSeekProvider(final_value={**final(), 'llm_probability': .7}),
    DeepSeekProvider(final_value=final(tools=('unexecuted',)))])
def test_deepseek_failure_or_missing_tool_evidence_never_falls_back_to_a_numeric_class(provider):
    result = raw_tool_agent.classify(provider, prediction())
    assert not result['available'] and result['final_class'] is None
    assert result['generation_status'] == 'fallback'
    assert 'secret-from-a-provider-exception' not in json.dumps(result)


def test_public_report_omits_numerical_fusion_and_keeps_internal_records():
    row = prediction()
    row['cloud_judgment'] = {'available': True, 'final_class': 'Low', 'confidence': 'low',
                             'rounds': [{}], 'tool_outputs': {'corrective_evidence': {'raw_combined_class': 'High'}}}
    row['predicted_class'] = 'Low'
    row['classification_layer'] = 'gpt_tool_agent'
    result = {'tool': 'vrms.raw_model', 'gpt_requested': True, 'measurements': {'segments': [row]}}
    public = raw_model.public_result(result)
    assert public['measurements']['segments'][0]['predicted_class'] == 'Low'
    assert not any(key in json.dumps(public) for key in ('numeric_fusion', 'mil_class', 'raw_combined_class', 'rounds'))
    assert result['measurements']['segments'][0]['numeric_fusion']['predicted_class'] == 'High'


def test_sdk_uses_responses_function_calls_and_shared_budget():
    provider = object.__new__(CloudProvider)
    provider.backend, provider.model = 'api', 'gpt-6.1-sol'
    provider.cfg = {'api_max_calls': 1}
    provider.calls, provider.lock, provider.budget_rejections = [], threading.Lock(), 0
    provider.emit = lambda *args: None
    observed = []

    def create(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(status='completed', usage=None,
            model_dump=lambda **_: {'status': 'completed', 'output': []})

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = provider.call('VRMSAgent', {'generation_phase': 'vrms_tool_agent', 'step': 0},
                           tools=raw_tool_agent.TOOLS, messages=[{'role': 'user', 'content': 'anonymous'}])
    assert result['status'] == 'completed'
    assert observed[0]['tool_choice'] == 'required' and observed[0]['store'] is False
    assert observed[0]['text']['format']['type'] == 'json_schema'
    assert provider.calls[0]['api_protocol'] == 'responses'
    with pytest.raises(RuntimeError, match='budget'):
        provider.call('VRMSAgent', {'generation_phase': 'vrms_tool_agent', 'step': 1})
    assert len(observed) == 1 and provider.budget_rejections == 1


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('model', ['gpt-6.1-sol', 'deepseek-flash'])
def test_raw_recording_first_high_and_published_classes_come_from_selected_cloud(tmp_path, monkeypatch, failure, model):
    head = tmp_path / 'worker/outputs/vrms_model_ablation/v3_20261008/model_results/checkpoints/outer_subject_10_femba_frozen_mil.pt'
    head.parent.mkdir(parents=True)
    head.touch()
    monkeypatch.setattr(raw_model, 'ROOT', tmp_path / 'worker')
    rows = [{'ordinal': ordinal, 'start_seconds': ordinal * 20, 'end_seconds': ordinal * 20 + 15,
             'start_clock': f'00:{ordinal * 20}.000', 'end_clock': f'00:{ordinal * 20 + 15}.000',
             'complete': True, 'stats': {'window_coverage': 1}} for ordinal in (1, 2)]
    record = {'signal_sha256': 'raw', 'task_count': 2, 'model_compatible': True, 'segments': rows}
    repository = SimpleNamespace(root=tmp_path / 'analysis',
        cfg={'minimum_path_windows': 3, 'minimum_path_coverage': .8},
        model_inputs=lambda _: [(np.ones((3, 30, 1280)), []) for _ in rows])

    def worker(*args, **kwargs):
        return SimpleNamespace(stdout=json.dumps({'available': True, 'segments': [prediction(1), prediction(2)],
            'signal_sha256': 'raw', 'fresh_raw_encoding': True, 'model_sha256': 'head', 'encoder_sha256': 'encoder',
            'normalization_sha256': 'normalization', 'numeric_first_sha256': 'first',
            'numeric_second_sha256': 'second', 'numeric_selector_sha256': 'selector'}).encode())

    def classify(provider, row, teaching_context=None):
        assert teaching_context['available']
        assert len(teaching_context['input_definitions']) == 9
        return {'available': not failure, 'final_class': None if failure else 'Low' if row['ordinal'] == 1 else 'High',
                'generation_status': 'fallback' if failure else 'cloud_verified', 'reason': 'GPT unavailable' if failure else ''}

    monkeypatch.setattr(raw_model.subprocess, 'run', worker)
    monkeypatch.setattr(raw_tool_agent, 'classify', classify)
    result = raw_model.assess(repository, 'subject-10', {'recording': record}, provider=SimpleNamespace(model=model), cloud=True)
    assert result['measurements']['readout'] == 'whole_path_' + ('gpt' if model.startswith('gpt-') else 'deepseek') + '_tool_agent'
    assert all(row['numeric_fusion']['predicted_class'] == 'High' for row in result['measurements']['segments'])
    if failure:
        assert result['status'] == 'partial_fallback' and result['cloud_failed_paths'] == 2
        assert result['measurements']['first_high'] is None
        assert all(row['predicted_class'] is None for row in result['measurements']['segments'])
    else:
        assert result['cloud_successful_paths'] == 2
        assert result['measurements']['first_high']['ordinal'] == 2
        assert result['measurements']['segments'][0]['predicted_class'] == 'Low'


def test_method_corpus_contains_current_raw_gpt_contract(tmp_path):
    from eeg_agent.domain_knowledge import HierarchicalKnowledge
    knowledge = HierarchicalKnowledge({'output_root': str(tmp_path)})
    matches = knowledge.search('GPT 工具 Agent 最终分类', domains=['vrms'], method_only=True)['results']
    assert any('GPT' in item['text'] and 'raw_model' in item['source'] for item in matches)
