"""Capture the actual pinned judge request and protect the unchanged baseline."""
import copy
import json
import threading
from types import SimpleNamespace

from eeg_agent import default_rag, raw_independent_judge
from eeg_agent.agents import CloudProvider
from tests.test_default_rag import recorded_prediction
from tests.test_raw_independent_judge import Provider, final


def test_regular_uses_frozen_historical_prompt_without_current_teaching():
    policy = default_rag.resolve('regular')
    prediction = default_rag.attach(1, recorded_prediction(), policy)
    provider = Provider(final())
    result = raw_independent_judge.classify(provider, prediction, {'malicious_new_teaching': True})
    assert result['available'] and result['judge_profile'] == 'matched_decisive_high_v1'
    request = provider.calls[0]
    assert request['messages'][0]['content'] == default_rag.judge_contract(policy)['system']
    data = json.loads(request['messages'][1]['content'])
    assert 'evidence_teaching' not in data['initial_evidence']
    assert request['payload']['matched_judge_parameters']['reasoning'] == {'effort': 'high'}
    assert request['payload']['matched_judge_parameters']['text']['format']['type'] == 'json_schema'
    assert not raw_independent_judge._hidden(data)


def test_default_teaching_remains_the_frozen_legacy_snapshot():
    policy = default_rag.resolve('default')
    assert policy['judge_profile'] == 'legacy_production'
    provider = Provider(final())
    attached = default_rag.attach(1, recorded_prediction(), policy)
    result = raw_independent_judge.classify(provider, attached, {'changed': True})
    assert result['available']
    data = json.loads(provider.calls[0]['messages'][1]['content'])
    assert data['initial_evidence']['evidence_teaching'] == default_rag.frozen_teaching(policy)


def test_pinned_high_contract_reaches_sdk_without_affecting_other_calls():
    captured = []
    response = SimpleNamespace(status='completed', output_text=json.dumps(final()), usage=None,
                               id='fake-sdk', model='gpt-6.1-sol')
    def create(**kwargs):
        captured.append(copy.deepcopy(kwargs))
        return response
    provider = object.__new__(CloudProvider)
    provider.backend, provider.model, provider.protocol = 'api', 'gpt-6.1-sol', 'responses'
    provider.cfg = {'api_max_calls': 1}
    provider.calls, provider.lock, provider.budget_rejections = [], threading.Lock(), 0
    provider.emit = lambda *unused: None
    provider.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    policy = default_rag.resolve('regular')
    prediction = default_rag.attach(1, recorded_prediction(), policy)
    result = raw_independent_judge.classify(provider, prediction)
    assert result['available'] and len(captured) == 1
    call = captured[0]
    assert call['model'] == 'gpt-6.1-sol' and call['reasoning'] == {'effort': 'high'}
    assert call['max_output_tokens'] == 8192 and call['store'] is False
    assert call['text']['format'] == default_rag.judge_contract(policy)['request_parameters']['text']['format']


def test_pinned_policy_tampering_does_not_trigger_any_cloud_request():
    policy = default_rag.resolve('regular')
    attached = default_rag.attach(1, recorded_prediction(), policy)
    attached['rag_policy']['reference_manifest_sha256'] = '0' * 64
    provider = Provider(final())
    result = raw_independent_judge.classify(provider, attached)
    assert not result['available'] and provider.calls == []
