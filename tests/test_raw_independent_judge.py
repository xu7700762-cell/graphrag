"""Offline contract tests for the production independent VRMS judge."""
import copy
import json
import threading
from types import SimpleNamespace

from eeg_agent import raw_independent_judge as judge, raw_tool_agent
from eeg_agent.agents import CloudProvider


def prediction():
    tools = []
    for name in raw_tool_agent.TOOL_NAMES:
        tools.append({'tool': name, 'predicted_class': 'High',
                      'raw_decision_score': .2, 'calibration_usable': False,
                      'calibrated_high_probability': None})
    return {'ordinal': 1, 'mil_raw_high_probability': .4, 'accepted_windows': 3,
            'numeric_fusion': {'raw_combined_score': .6, 'predicted_class': 'High',
                'actual_tools': tools, 'tool_validation_profiles': {}}}


class Provider:
    model = 'deepseek-flash'

    def __init__(self, value):
        self.value = value
        self.calls = []

    def call(self, role, payload, tools=None, messages=None):
        self.calls.append({'role': role, 'payload': payload, 'tools': tools,
                           'messages': copy.deepcopy(messages)})
        return {'id': 'offline-fake', 'model': self.model,
                'choices': [{'finish_reason': 'stop', 'message': {
                    'content': json.dumps(self.value, ensure_ascii=False)}}]}


def final(citations=None):
    return {'final_class': 'Low', 'confidence': 'low',
            'explanation': '八路证据存在冲突，结合可用参考后由模型独立判为Low。',
            'supporting_tools': ['MIL'], 'counterevidence_tools': ['spectral'],
            'rag_citations': citations or []}


def test_independent_judge_hides_fusion_and_copies_six_fields():
    row = prediction()
    before = copy.deepcopy(row)
    provider = Provider(final())
    result = judge.classify(provider, row, teaching_context={'available': False})
    assert result['available'] and result['final_class'] == 'Low'
    assert result['raw_llm_final'] == final()
    assert row == before
    outbound = json.dumps(provider.calls, ensure_ascii=False)
    assert 'raw_combined_class' not in outbound and 'raw_combined_score' not in outbound
    assert 'numeric_fusion' not in outbound
    assert provider.calls[0]['tools'] is None
    assert 'thinking' not in outbound  # provider adds thinking at SDK boundary


def test_current_raw_context_is_citable_and_kept_separate_from_query_numbers():
    row = prediction()
    context = {'schema': 'raw_current_contrastive_rag_v2', 'available': True,
        'provenance': 'fresh_same_fold_frozen_inference_on_saved_current_raw_policy_windows',
        'validation_scope': 'head_and_calibrator_subject_heldout_selector_reused',
        'retrieval_scope': 'eight_raw_tool_outputs_only; MIL excluded from selection',
        'cases': [{'case_id': 'current-contrastive-case-01', 'reference_class': 'High'}],
        'source_hashes': {'bank': 'a' * 64}}
    row['current_raw_contrastive_context'] = context
    provider = Provider(final(['current-contrastive-case-01']))
    result = judge.classify(provider, row, teaching_context={'available': False})
    assert result['available'] and result['rag_context_available']
    assert result['rag_citations'] == ['current-contrastive-case-01']
    assert result['rag_schema'] == context['schema']
    assert result['rag_source_hashes'] == context['source_hashes']
    payload = json.loads(provider.calls[0]['messages'][1]['content'])
    assert payload['initial_evidence']['policy_heldout_case_context']['schema'] == context['schema']
    assert 'raw_combined_class' not in json.dumps(payload, ensure_ascii=False)


def test_invalid_or_incomplete_final_stays_unavailable():
    for value in (dict(final(), extra='bad'), dict(final(), supporting_tools=['not-a-tool']),
                  dict(final(), final_class='bad')):
        result = judge.classify(Provider(value), prediction(), teaching_context={'available': False})
        assert not result['available'] and result['final_class'] is None

    class Incomplete(Provider):
        def call(self, *args, **kwargs):
            return {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}

    result = judge.classify(Incomplete(final()), prediction(), teaching_context={'available': False})
    assert not result['available'] and result['final_class'] is None


def test_deepseek_sdk_boundary_is_disabled_temperature_zero_single_call():
    value = final()
    observed = []
    response_json = {'id': 'offline-sdk', 'model': 'deepseek-flash',
        'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}

    def create(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(model='deepseek-flash', usage=None,
            choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                content=json.dumps(value), tool_calls=None))],
            model_dump=lambda **unused: response_json)

    provider = object.__new__(CloudProvider)
    provider.backend, provider.model = 'api', 'deepseek-flash'
    provider.cfg = {'api_max_calls': 1}
    provider.calls, provider.lock, provider.budget_rejections = [], threading.Lock(), 0
    provider.emit = lambda *args: None
    provider.thinking_disabled = True
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = judge.classify(provider, prediction(), teaching_context={'available': False})
    assert result['available'] and result['final_class'] == 'Low'
    assert len(observed) == 1
    request = observed[0]
    assert request['temperature'] == 0 and request['max_tokens'] == 8192
    assert request['extra_body'] == {'thinking': {'type': 'disabled'}}
    assert 'tools' not in request and request['response_format'] == {'type': 'json_object'}
    assert provider.calls[0]['generation_phase'] == 'vrms_independent_judge'
