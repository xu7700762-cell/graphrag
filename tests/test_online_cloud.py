"""Exercise real six-field judgement and grounded domain reports without networking."""
import copy
import json
import socket
import threading
import urllib.request

import pytest

from eeg_agent.config import load_config
from eeg_agent.online import cloud
from tests.test_raw_independent_judge import final, prediction


class KnowledgeDouble:
    def search(self, query, domains=None, limit=3, **unused):
        domain = (domains or ['shared'])[0]
        return {'results': [{'id': 'method-' + domain, 'title': '已核验方法',
                             'source': 'reviewed-method', 'text': 'EEG指标不等于主观问卷类别或临床诊断。'}]}


class ProviderDouble:
    model = 'deepseek-flash'
    protocol = 'chat_completions'

    def __init__(self, judge_mode='valid', fail_role=None, alteration=None):
        self.calls = []
        self.judge_mode, self.fail_role, self.alteration = judge_mode, fail_role, alteration

    def call(self, role, payload, tools=None, messages=None):
        self.calls.append(copy.deepcopy({'role': role, 'payload': payload, 'tools': tools, 'messages': messages}))
        if role == 'VRMSIndependentJudge':
            if self.judge_mode == 'exception':
                raise TimeoutError('private-provider-diagnostic')
            value = final()
            if self.judge_mode == 'extra_field':
                value['llm_probability'] = .99
            elif self.judge_mode == 'unknown_citation':
                value['rag_citations'] = ['unsupplied-case']
            elif self.judge_mode == 'invalid_class':
                value['final_class'] = 'Maybe'
            return {'choices': [{'finish_reason': 'length' if self.judge_mode == 'unfinished' else 'stop',
                                  'message': {'content': json.dumps(value, ensure_ascii=False)}}]}

        facts = payload['tool_facts']
        wanted = ['measurements.theta_alpha'] if role == 'FatigueAgent' else (
            ['measurements.workload_index', 'measurements.faa'] if role == 'EmotionAgent' else
            ['measurements.theta_alpha', 'measurements.workload_index', 'measurements.segments.0.predicted_class'])
        selected = [fact['id'] for fact in facts if fact['path'] in wanted]
        if role == 'FatigueAgent':
            text = '当前θ/α测量值为1.50，是疲劳相关EEG指标。'
        elif role == 'EmotionAgent':
            text = '当前负荷指标测量值为2.25，FAA为-0.25，均为EEG指标。'
        else:
            text = ('当前完整路径云端判为Low；疲劳指标θ/α为1.50，负荷指标为2.25。'
                    if any(fact['path'] == 'measurements.segments.0.predicted_class' for fact in facts)
                    else '当前云端类别未取得有效回复；疲劳指标θ/α为1.50，负荷指标为2.25。')
        response = {'explanation': text, 'citations': [payload['knowledge'][0]['id']],
                    'measurement_claim_ids': selected}
        if role == self.fail_role:
            if self.alteration == 'exception':
                raise TimeoutError('private-provider-diagnostic')
            if self.alteration == 'invented_number':
                response['explanation'] = text.replace('1.50', '99.00').replace('2.25', '99.00')
            elif self.alteration == 'unknown_citation':
                response['citations'] = ['not-in-retrieval']
            elif self.alteration == 'unknown_fact':
                response['measurement_claim_ids'] = ['F9999']
            elif self.alteration == 'changed_claim':
                response['measurement_claims'] = [{'source': 'fatigue.raw_spectrum',
                    'path': 'measurements.theta_alpha', 'value': 99.0}]
        return response

    def summary(self):
        return {'calls': len(self.calls), 'source': 'offline_provider_double'}


def completed_path():
    return {'scope': 'task', 'ordinal': 1, 'complete': True,
            'start_seconds': 6., 'end_seconds': 21., 'data_cutoff_seconds': 21.0009765625,
            'accepted_windows': 3, 'total_windows': 3, 'coverage': 1.,
            'vrms': {'available': True, 'mil_class': 'High', 'numeric_class': 'High'},
            'fatigue': {'available': True, 'theta_alpha': 1.5, 'theta_beta': 2., 'beta_alpha': .75},
            'emotion': {'available': True, 'workload_index': 2.25, 'faa': -.25}}


@pytest.fixture
def reporter_factory(tmp_path, monkeypatch):
    cfg = {**load_config(), 'output_root': str(tmp_path / 'offline'),
           'cloud_config_path': str(tmp_path / '.env')}
    monkeypatch.setattr(cloud, 'HierarchicalKnowledge', lambda cfg: KnowledgeDouble())
    monkeypatch.setattr(cloud, 'build_teaching_context', lambda cfg: {'available': False})
    # These tests isolate reporting. Real default retrieval is exercised in
    # test_default_rag with verified reference assets and a recorded query.
    monkeypatch.setattr(cloud, 'attach_default_rag', lambda fold, row: copy.deepcopy(row))
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: pytest.fail('Online cloud test attempted network traffic'))
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Online cloud test attempted HTTP traffic'))
    return lambda provider: cloud.OnlineCloudReporter(cfg, lambda *a: None, provider=provider)


def test_real_judge_and_claim_id_reports_keep_three_domain_sources_separate(reporter_factory):
    provider = ProviderDouble()
    row, numeric = completed_path(), prediction()
    before = copy.deepcopy((row, numeric))
    report = reporter_factory(provider).report(row, numeric)
    assert report['status'] == 'verified' and report['final_class'] == 'Low'
    assert report['explanation_status'] == 'completed'
    assert report['scope'] == 'path_end' and report['data_cutoff_seconds'] == row['data_cutoff_seconds']
    assert report['reports']['vrms']['raw_llm_final'] == final()
    assert [call['role'] for call in provider.calls] == ['VRMSIndependentJudge', 'FatigueAgent', 'EmotionAgent', 'Supervisor']
    assert (row, numeric) == before  # Cloud Low does not rewrite local High.
    for name in ('fatigue', 'emotion', 'supervisor'):
        domain = report['reports'][name]
        assert domain['generation_status'] == 'cloud_verified'
        assert isinstance(domain['explanation'], str) and domain['measurement_claims']
        assert 'measurement_claim_ids' not in domain
    fatigue = provider.calls[1]['payload']
    emotion = provider.calls[2]['payload']
    assert {fact['source'] for fact in fatigue['tool_facts']} == {'fatigue.raw_spectrum'}
    assert {fact['source'] for fact in emotion['tool_facts']} == {'emotion.raw_workload'}
    for call in provider.calls[1:]:
        assert call['payload']['assessment_scope'] == 'path_end'
        assert call['messages'][1]['role'] == 'user'


@pytest.mark.parametrize('role', ['FatigueAgent', 'EmotionAgent', 'Supervisor'])
@pytest.mark.parametrize('damage', ['exception', 'invented_number', 'unknown_citation', 'unknown_fact', 'changed_claim'])
def test_bad_domain_explanation_is_partial_and_preserves_valid_vrms_decision(reporter_factory, role, damage):
    provider = ProviderDouble(fail_role=role, alteration=damage)
    report = reporter_factory(provider).report(completed_path(), prediction())
    name = {'FatigueAgent': 'fatigue', 'EmotionAgent': 'emotion', 'Supervisor': 'supervisor'}[role]
    assert report['status'] == 'verified' and report['final_class'] == 'Low'
    assert report['explanation_status'] == 'partial'
    assert report['reports'][name]['generation_status'] == 'fallback'
    assert '99.00' not in json.dumps(report, ensure_ascii=False)
    assert 'private-provider-diagnostic' not in json.dumps(report)
    assert report['reports']['vrms']['available'] is True


@pytest.mark.parametrize('stop_role', [None, 'VRMSIndependentJudge', 'FatigueAgent', 'EmotionAgent', 'Supervisor'])
def test_cancellation_stops_subsequent_roles_and_keeps_partial_usage(reporter_factory, stop_role):
    stopped = threading.Event()

    class StoppingProvider(ProviderDouble):
        def call(self, role, *args, **kwargs):
            result = super().call(role, *args, **kwargs)
            if role == stop_role:
                stopped.set()
            return result

    if stop_role is None:
        stopped.set()
    provider = StoppingProvider()
    report = reporter_factory(provider).report(completed_path(), prediction(), cancelled=stopped.is_set)
    roles = ['VRMSIndependentJudge', 'FatigueAgent', 'EmotionAgent', 'Supervisor']
    count = roles.index(stop_role) + 1 if stop_role else 0
    assert [call['role'] for call in provider.calls] == roles[:count]
    assert report['status'] == 'fallback' and report['generation_status'] == 'cancelled'
    assert report['final_class'] is None and report['explanation_status'] == 'cancelled'
    assert report['usage']['calls'] == count
    assert len(report['reports']) == count
    if count:
        assert report['reports']['vrms']['final_class'] == 'Low'


@pytest.mark.parametrize('mode', ['exception', 'extra_field', 'unknown_citation', 'invalid_class', 'unfinished'])
def test_failed_or_malformed_judgement_never_forces_a_local_class(reporter_factory, mode):
    provider = ProviderDouble(judge_mode=mode)
    report = reporter_factory(provider).report(completed_path(), prediction())
    assert report['status'] == 'fallback' and report['final_class'] is None
    assert report['reports']['vrms']['available'] is False
    assert report['reports']['vrms']['final_class'] is None
    assert 'private-provider-diagnostic' not in json.dumps(report)


@pytest.mark.parametrize('scope,complete', [('task', False), ('rest', True), ('rest', False)])
def test_unfinished_or_rest_segments_cannot_call_any_cloud_role(reporter_factory, scope, complete):
    provider = ProviderDouble()
    row = {**completed_path(), 'scope': scope, 'complete': complete}
    with pytest.raises(ValueError, match='completed task'):
        reporter_factory(provider).report(row, prediction())
    assert provider.calls == []


def test_missing_model_can_report_indicators_without_a_vrms_judge_call(reporter_factory):
    provider = ProviderDouble()
    report = reporter_factory(provider).report(completed_path(), None)
    assert report['status'] == 'fallback' and report['final_class'] is None
    assert [call['role'] for call in provider.calls] == ['FatigueAgent', 'EmotionAgent', 'Supervisor']
    assert report['reports']['fatigue']['generation_status'] == 'cloud_verified'
    assert report['reports']['emotion']['generation_status'] == 'cloud_verified'


def test_identity_labels_waveforms_and_fused_class_are_never_outbound(reporter_factory):
    provider = ProviderDouble()
    row, numeric = completed_path(), prediction()
    row.update(subject_id='identity-sentinel', target_label='label-sentinel', path_score=999432,
               waveform=[[999432.]])
    row['fatigue'].update(subject_id='identity-sentinel', target_label='label-sentinel', raw_values=[[999432.]])
    row['emotion'].update(subject_id='identity-sentinel', path_score=999432, raw_eeg=[[999432.]])
    numeric.update(subject_id='identity-sentinel', target_label='label-sentinel', waveform=[[999432.]])
    reporter_factory(provider).report(row, numeric)
    outbound = json.dumps(provider.calls, ensure_ascii=False)
    for sentinel in ('identity-sentinel', 'label-sentinel', '999432', 'measurements.raw_values', 'measurements.raw_eeg'):
        assert sentinel not in outbound
    judge_payload = json.loads(provider.calls[0]['messages'][1]['content'])
    wire = json.dumps(judge_payload)
    assert 'numeric_fusion' not in wire and 'raw_combined_score' not in wire and 'raw_combined_class' not in wire
    assert judge_payload['initial_evidence']['id'] == 'qsingle'


def test_known_indicator_slots_cannot_forward_raw_arrays(reporter_factory):
    provider = ProviderDouble()
    row = completed_path()
    row['fatigue']['theta_alpha'] = [[999432.]]
    row['emotion']['workload_index'] = [[999432.]]
    report = reporter_factory(provider).report(row, prediction())
    assert report['status'] == 'verified' and report['final_class'] == 'Low'
    assert '999432' not in json.dumps(provider.calls)
    assert report['explanation_status'] == 'partial'


@pytest.mark.parametrize('override,expected', [(None, 120.), (75, 75.)])
def test_path_report_timeout_is_independent_from_offline_provider(reporter_factory, monkeypatch, override, expected):
    cfg = {'api_timeout_seconds': 10}
    if override is not None:
        cfg['online_cloud_timeout_seconds'] = override
    original = dict(cfg)
    provider = ProviderDouble()
    seen = []
    monkeypatch.setattr(cloud, 'CloudProvider', lambda backend, settings, emit: seen.append(dict(settings)) or provider)
    reporter = cloud.OnlineCloudReporter(cfg, lambda *args: None)
    assert cfg == original
    assert seen[0]['api_timeout_seconds'] == expected
    assert reporter.report(completed_path(), prediction())['timeout_seconds'] == expected


@pytest.mark.parametrize('value', [True, None, 0, -1, float('nan'), float('inf'), 601, 'invalid'])
def test_bad_online_timeout_is_rejected_before_constructing_provider(reporter_factory, monkeypatch, value):
    monkeypatch.setattr(cloud, 'CloudProvider', lambda *args: pytest.fail('Invalid timeout reached provider'))
    with pytest.raises(ValueError, match='online_cloud_timeout_seconds'):
        cloud.OnlineCloudReporter({'api_timeout_seconds': 10, 'online_cloud_timeout_seconds': value}, lambda *args: None)


def test_timeout_is_reported_as_transport_failure_and_other_agents_remain_available(reporter_factory):
    provider = ProviderDouble(judge_mode='exception', fail_role='Supervisor', alteration='exception')
    result = reporter_factory(provider).report(completed_path(), prediction())
    assert result['status'] == 'fallback' and result['final_class'] is None
    assert result['reports']['fatigue']['available'] is True
    assert result['reports']['emotion']['available'] is True
    for name in ('vrms', 'supervisor'):
        item = result['reports'][name]
        assert '超时' in item['reason'] and '核验' not in item['reason']
        assert item['error_type'] == 'TimeoutError'
        assert item['timeout_seconds'] == 120
    assert 'private-provider-diagnostic' not in json.dumps(result)


def test_real_validation_failure_keeps_validation_code(reporter_factory):
    result = reporter_factory(ProviderDouble(fail_role='FatigueAgent', alteration='unknown_fact')).report(completed_path(), prediction())
    item = result['reports']['fatigue']
    assert item['error_type'] == 'ValueError'
    assert item['validation_code']
    assert '超时' not in item['reason']


@pytest.mark.parametrize('error_type,word', [('APITimeoutError', '超时'), ('AuthenticationError', '认证'),
                                          ('APIConnectionError', '连接'), ('RateLimitError', '限额')])
def test_historical_transport_failure_reason_is_safe_and_does_not_rewrite_saved_report(error_type, word):
    saved = {'available': False, 'error_type': error_type, 'reason': '云端解释未通过来源核验；保留本地指标'}
    original = dict(saved)
    assert word in cloud.failure_reason(saved)
    assert '核验' not in cloud.failure_reason(saved)
    assert saved == original


def test_none_reporter_pins_default_graph_once_and_reports_the_actual_mode(reporter_factory, monkeypatch):
    from eeg_agent import default_rag
    from tests.test_default_rag import recorded_prediction
    monkeypatch.setattr(default_rag, 'DEFAULT_MODE', 'graph')
    monkeypatch.setattr(default_rag, 'DEFAULT_JUDGE_PROFILE', 'matched_decisive_high_v1')
    monkeypatch.setattr(cloud, 'attach_default_rag', default_rag.attach)
    provider = ProviderDouble()
    reporter = reporter_factory(provider)
    row = recorded_prediction()
    row['fold'] = 1
    first = reporter.report(completed_path(), row)
    assert first['rag_mode'] == 'graph' and first['reports']['vrms']['rag_variant'] == 'graph_v1'
    pinned = copy.deepcopy(reporter.rag_policy)
    monkeypatch.setattr(default_rag, 'DEFAULT_MODE', 'regular')
    second = reporter.report(completed_path(), row)
    assert second['rag_policy'] == pinned and second['rag_mode'] == 'graph'
    assert len([call for call in provider.calls if call['role'] == 'VRMSIndependentJudge']) == 2


def test_local_rag_asset_failure_is_not_misreported_as_cloud_output_error(reporter_factory, monkeypatch, tmp_path):
    from eeg_agent import default_rag, graph_rag
    from tests.test_default_rag import recorded_prediction
    monkeypatch.setattr(cloud, 'attach_default_rag', default_rag.attach)
    provider = ProviderDouble()
    reporter = reporter_factory(provider)
    reporter.rag_policy = default_rag.resolve('graph')
    monkeypatch.setattr(graph_rag, 'GRAPH_MANIFEST', tmp_path / 'missing.json')
    row = recorded_prediction()
    row['fold'] = 1
    result = reporter.report(completed_path(), row)
    assert result['status'] == 'fallback' and result['final_class'] is None
    judgment = result['reports']['vrms']
    assert judgment['error_stage'] == 'local_rag_validation'
    assert judgment['validation_code'] == 'rag_assets_unavailable'
    assert '本地 RAG' in judgment['reason'] and '云端判读输出' not in judgment['reason']
    assert not any(call['role'] == 'VRMSIndependentJudge' for call in provider.calls)
