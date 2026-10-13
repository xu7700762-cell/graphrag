"""Verify real-language evidence validation and failure visibility without API fees."""
import copy
import json
import time
from types import SimpleNamespace

import pytest

from eeg_agent.agents import CloudProvider
from eeg_agent.brain import BrainManager, _generation_failure, _language, _tool_facts
from eeg_agent.config import load_config


REFERENCES = [{'id': 'emotion-ref-01', 'text': 'F3/F4 的 alpha 常用 8–13 Hz，FAA 是特征，不能直接确定情绪标签。',
               'domain': 'emotion', 'source': 'source', 'section': 'method'}]
RESULTS = [{'tool': 'emotion.faa', 'result': {'available': True, 'measurements': {'faa': .62}}}]
CLAIM = {'source': 'emotion.faa', 'path': 'measurements.faa', 'value': .62}


def report(text, claims=(), references=REFERENCES):
    return {'explanation': text, 'citations': [r['id'] for r in references],
            'measurement_claims': list(claims)}


def test_channel_names_cited_band_numbers_and_verified_percent_are_accepted():
    text = 'F3/F4 的 alpha 频段为 8–13 Hz。当前 FAA 测量值为 0.62，只能作为特征，不能确诊情绪。'
    assert _language(report(text, [CLAIM]), REFERENCES, RESULTS)[0] == text
    probability = [{'tool': 'vrms.femba_pipeline', 'result': {'measurements': {'mil': {'score': .62}}}}]
    claim = {'source': 'vrms.femba_pipeline', 'path': 'measurements.mil.score', 'value': .62}
    assert _language(report('当前模型预测概率为 62%。', [claim]), REFERENCES, probability)


def test_verified_exact_claims_allow_rounded_display_but_not_changed_measurements():
    actual = .6296296296296297
    tools = [{'tool': 'vrms.femba_pipeline', 'result': {'measurements': {'audit': {'acc': actual}}}}]
    claim = {'source': 'vrms.femba_pipeline', 'path': 'measurements.audit.acc', 'value': actual}
    assert _language(report('本次审核结果 ACC 为 0.63。', [claim]), REFERENCES, tools)
    assert _language(report('本次审核结果 ACC 为 62.96%。', [claim]), REFERENCES, tools)
    with pytest.raises(ValueError):
        _language(report('本次审核结果 ACC 为 0.64。', [claim]), REFERENCES, tools)
    with pytest.raises(ValueError, match='conflicts'):
        _language(report('本次审核结果 ACC 为 0.63。', [{**claim, 'value': .63}]), REFERENCES, tools)


@pytest.mark.parametrize('minus', ['−', '－', '﹣'])
def test_scientific_minus_signs_keep_verified_direction_and_reject_reversed_sign(minus):
    negative_tools = [{'tool': 'emotion.faa', 'result': {'measurements': {'faa': -.62}}}]
    negative_claim = {**CLAIM, 'value': -.62}
    assert _language(report('当前 FAA 测量值为 ' + minus + '0.62。', [negative_claim]), REFERENCES, negative_tools)
    with pytest.raises(ValueError, match='numerical statement'):
        _language(report('当前 FAA 测量值为 ' + minus + '0.62。', [CLAIM]), REFERENCES, RESULTS)
    with pytest.raises(ValueError):
        _language(report('当前 FAA 测量值为 +0.62。', [negative_claim]), REFERENCES, negative_tools)


def test_audit_counts_and_layer_classes_use_their_actual_tool_sources():
    tools = [{'tool': 'vrms.femba_pipeline', 'result': {'measurements': {
        'mil': {'predicted_class': 'Low'},
        'numeric_fusion': {'independent_nested_audit': {'correction': {'corrected': 0, 'damaged': 2}}},
        'cloud_judgment': {'available': True, 'final_class': 'High'}}}}]
    assert _language(report('MIL预测为Low，当前审核结果纠正0条、损伤2条。'), REFERENCES, tools)
    with pytest.raises(ValueError):
        _language(report('当前审核结果纠正1条。'), REFERENCES, tools)
    with pytest.raises(ValueError, match='conflicts'):
        _language(report('最终分类为Low。'), REFERENCES, tools)


@pytest.mark.parametrize('text,claims', [
    ('当前 FAA 测量值为 0.72。', [CLAIM]),
    ('当前 FAA 测量值为 0.62。', []),
    ('当前 FAA 测量值为 0.62。', [{**CLAIM, 'value': .63}]),
    ('当前 FAA 测量值为 0.62。', [{**CLAIM, 'path': 'measurements.probability_high'}]),
    ('当前受试者处于重度疲劳。', []),
    ('情绪类别为焦虑。', []),
])
def test_hallucinated_results_misbinding_and_diagnoses_rejected(text, claims):
    with pytest.raises(ValueError):
        _language(report(text, claims), REFERENCES, RESULTS)


def test_reference_measurement_cannot_be_represented_as_current_measurement():
    refs = [{**REFERENCES[0], 'text': '先前研究的 FAA 为 0.62。'}]
    with pytest.raises(ValueError, match='current measurement'):
        _language(report('当前 FAA 测量值为 0.62。', references=refs), refs)
    assert _language(report('先前研究的 FAA 为 0.62。', references=refs), refs)


def test_final_vrms_class_must_match_available_cloud_judgment():
    tools = [{'tool': 'vrms.femba_pipeline', 'result': {'measurements': {
        'mil': {'class': 'Low', 'score': .4},
        'cloud_judgment': {'available': True, 'final_class': 'High'}}}}]
    final = {'source': 'vrms.femba_pipeline', 'path': 'measurements.cloud_judgment.final_class', 'value': 'High'}
    assert _language(report('最终分类结果为 High。', [final]), REFERENCES, tools)
    with pytest.raises(ValueError, match='conflicts'):
        _language(report('最终分类结果为 Low。', [final]), REFERENCES, tools)
    with pytest.raises(ValueError, match='conflicts'):
        _language(report('最终分类结果为 Low。', [{'source': 'vrms.femba_pipeline',
                  'path': 'measurements.mil.class', 'value': 'Low'}]), REFERENCES, tools)
    tools[0]['result']['measurements']['cloud_judgment']['available'] = False
    with pytest.raises(ValueError, match='unknown measurement'):
        _language(report('最终分类结果为 High。', [final]), REFERENCES, tools)
    assert not any('cloud_judgment' in f['path'] for f in _tool_facts(tools))


@pytest.mark.parametrize('text', [
    # Exact failure wording: the previous regex extracted 低类 from 高低类.
    '当前未提供 VRMSAgent 的执行结果；无法列出 VRMS 模型判定的高低类路径编号和首条高类起止区间。',
    '当前无法给出 VRMS 模型分类为 High 的结果。',
    '最终分类不能判为 Low。',
    'VRMS 模型不是 High。',
    'VRMS 模型尚未输出 High。',
    'VRMS 模型输出的高类标签是工具类别名。',
    'VRMS 模型可输出 High/Low 两种类别。',
    'VRMS 模型给出高类或低类标签。',
    '本次没有 VRMS 分类结果，无法给出高类路径。',
])
def test_missing_classification_and_label_names_are_not_class_assertions(text):
    assert _language(report(text), REFERENCES, RESULTS)[0] == text


@pytest.mark.parametrize('text', [
    '最终分类为 High。',
    'VRMS 模型输出 Low。',
    '本次判定高类。',
    'VRMS 模型给出高类。',
    'MIL 预测为 Low。',
    '当前没有云端结果，但最终分类为 High。',
    'VRMS 模型无法确诊症状，但是最终分类为 High。',
])
def test_actual_class_assertions_still_require_the_correct_tool_source(text):
    with pytest.raises(ValueError, match='lacks cloud judgment tool source'):
        _language(report(text), REFERENCES, RESULTS)


def test_denial_does_not_hide_a_later_conflicting_class_assertion():
    tools = [{'tool': 'vrms.femba_pipeline', 'result': {'measurements': {
        'mil': {'predicted_class': 'Low'},
        'numeric_fusion': {'raw_combined_class': 'Low'},
        'cloud_judgment': {'available': True, 'final_class': 'High'}}}}]
    assert _language(report('VRMS 模型不能确诊症状，但 MIL 预测为 Low。'), REFERENCES, tools)
    with pytest.raises(ValueError, match='conflicts'):
        _language(report('MIL 预测为 Low，但是最终分类为 Low。'), REFERENCES, tools)
    with pytest.raises(ValueError, match='conflicts'):
        _language(report('当前无法给出症状诊断，但融合分类为 High。'), REFERENCES, tools)


def test_unknown_and_missing_citations_are_rejected():
    value = report('F3/F4 alpha 特征。')
    for citations in ([], ['made-up'], [{}]):
        with pytest.raises(ValueError):
            _language({**value, 'citations': citations}, REFERENCES)


class Knowledge:
    def search(self, *args, **kwargs):
        return {'results': copy.deepcopy(REFERENCES), 'stages': []}


class Tasks:
    def get(self, task_id):
        raise KeyError(task_id)


class LanguageProvider:
    invalid_supervisor = False
    fail_all = False

    def __init__(self, backend, cfg, emit):
        self.client = None
        self.calls = []
        self.instructions = []
        self.cfg = cfg

    def call(self, role, payload, messages=None):
        self.calls.append(copy.deepcopy(payload))
        self.instructions.append(messages[0]['content'])
        if self.fail_all:
            class AuthenticationError(Exception):
                pass
            raise AuthenticationError('SECRET_KEY and raw_EEG must never reach events')
        if 'available_agents' in payload:
            return {'tasks': [{'agent': d, 'task': payload['question']} for d in payload['available_agents']],
                    'explanation': '先由情感工具读取测量，再汇总。'}
        if 'available_tools' in payload:
            return {'requested_tools': [payload['available_tools'][0]['id']]}
        fact = next(f for f in payload['tool_facts'] if f['path'] == 'measurements.faa')
        value = report('当前 FAA 测量值为 0.62；F3/F4 的 alpha 频段为 8–13 Hz，不能诊断为某种情绪。', [fact])
        if role == 'Supervisor' and self.invalid_supervisor:
            value['explanation'] = '当前 FAA 测量值为 0.99。'
        return value

    def summary(self):
        return {'backend': 'api', 'calls': len(self.calls), 'simulated': False}


def conversation(tmp_path, monkeypatch, provider=LanguageProvider):
    cfg = load_config()
    cfg['output_root'] = str(tmp_path / 'output')
    monkeypatch.setattr('eeg_agent.brain.execute_tool', lambda tool, evidence:
                        {'available': True, 'measurements': {'faa': .62}})
    manager = BrainManager(cfg, Tasks(), knowledge=Knowledge(), provider_factory=provider)
    session = manager.get(manager.create({'backend': 'api', 'domains': ['emotion']})['id'])
    session.message('请解释 F3/F4 的 FAA。')
    deadline = time.monotonic() + 5
    while session.state == 'running' and time.monotonic() < deadline:
        time.sleep(.01)
    assert session.state == 'idle'
    return session.snapshot(), session


def test_success_means_verified_cloud_answer_and_bounded_report_phase(tmp_path, monkeypatch):
    snapshot, session = conversation(tmp_path, monkeypatch)
    result = snapshot['result']
    assert result['generation_status'] == 'cloud_verified'
    assert result['workflow_status'] == 'cloud_verified'
    assert '当前 FAA 测量值为 0.62' in result['text']
    assert session.provider.cfg['api_timeout_seconds'] == 60
    assert session.cfg['api_timeout_seconds'] != 60
    assert [c['generation_phase'] for c in session.provider.calls] == [
        'planning', 'planning', 'specialist_report', 'supervisor_report']
    assert '600 Chinese characters' in session.provider.instructions[2]
    assert '900 Chinese characters' in session.provider.instructions[3]
    assert all('at most three short paragraphs' in instruction for instruction in session.provider.instructions[2:])


def test_invalid_cloud_synthesis_is_explicit_fallback_not_success(tmp_path, monkeypatch):
    class InvalidSupervisor(LanguageProvider):
        invalid_supervisor = True
    snapshot, _ = conversation(tmp_path, monkeypatch, InvalidSupervisor)
    result = snapshot['result']
    assert result['generation_status'] == result['workflow_status'] == 'fallback'
    assert result['fallback_reason'] and '真实 Supervisor 汇总未完成' in result['text']
    assert '0.99' not in result['text']
    assert any(e['kind'] == 'generation_error' and e['payload']['phase'] == 'supervisor_report'
               for e in snapshot['events'])


def test_api_failure_is_visible_and_does_not_echo_error_body(tmp_path, monkeypatch):
    class FailedProvider(LanguageProvider):
        fail_all = True
    snapshot, _ = conversation(tmp_path, monkeypatch, FailedProvider)
    assert snapshot['result']['generation_status'] == 'fallback'
    assert snapshot['result']['agent_reports'][0]['generation_status'] == 'fallback'
    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert '认证失败' in serialized and 'AuthenticationError' in serialized
    assert 'SECRET_KEY' not in serialized and 'raw_EEG' not in serialized


def test_gpt_uses_responses_with_phase_budgets_and_no_storage(monkeypatch, tmp_path):
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(status='completed', output_text='{"explanation":"报告","citations":[]}',
                               usage=SimpleNamespace(input_tokens=9, output_tokens=12))
    import openai
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs:
                        SimpleNamespace(responses=SimpleNamespace(create=create)))
    monkeypatch.setenv('EEG_API_BASE_URL', 'https://test.invalid/v1')
    monkeypatch.setenv('EEG_API_KEY', 'TEST_ONLY')
    monkeypatch.setenv('EEG_API_MODEL', 'gpt-6.1-sol')
    cfg = load_config()
    cfg['cloud_config_path'] = str(tmp_path / '.env')
    provider = CloudProvider('api', cfg, lambda *args: None)
    for phase in ('planning', 'specialist_report', 'supervisor_report'):
        provider.call('Supervisor', {'generation_phase': phase})
    assert [r['max_output_tokens'] for r in requests] == [700, 2000, 3000]
    assert all(r['store'] is False and r['reasoning'] == {'effort': 'low'} for r in requests)
    assert all(r['text']['format']['type'] == 'json_object' for r in requests)
    assert provider.summary()['input_tokens'] == 27
    assert all(c['api_protocol'] == 'responses' for c in provider.summary()['records'])
    assert 'TEST_ONLY' not in json.dumps(requests)


@pytest.mark.parametrize('model', ['gpt-6.1-sol', 'deepseek-flash'])
def test_truncated_json_output_cannot_be_recorded_as_success(model, monkeypatch, tmp_path):
    import openai
    response = SimpleNamespace(status='incomplete', output_text='{"explanation":"cut"}', usage=None,
                               choices=[SimpleNamespace(finish_reason='length',
                                   message=SimpleNamespace(content='{"explanation":"cut"}'))])
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response),
                             chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: client)
    monkeypatch.setenv('EEG_API_BASE_URL', 'https://test.invalid/v1')
    monkeypatch.setenv('EEG_API_KEY', 'TEST_ONLY')
    monkeypatch.setenv('EEG_API_MODEL', model)
    cfg = load_config()
    cfg['cloud_config_path'] = str(tmp_path / '.env')
    provider = CloudProvider('api', cfg, lambda *args: None)
    with pytest.raises(ValueError, match='Incomplete API output'):
        provider.call('Supervisor', {})
    assert provider.summary()['failures'] == 1
    assert provider.summary()['records'][0]['output_truncated']


def test_local_validation_codes_are_visible_but_provider_bodies_are_sanitized():
    error = _generation_failure(ValueError('Incomplete API output'), 'supervisor_report')
    assert error['validation_code'] == 'output_truncated' and '上限' in error['reason']
    error = _generation_failure(ValueError('SECRET provider body'), 'supervisor_report')
    assert 'validation_code' not in error and 'SECRET' not in json.dumps(error)
