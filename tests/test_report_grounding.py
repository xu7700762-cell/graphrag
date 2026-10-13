"""Source selection must preserve exact facts and reject unsupported prose."""
import copy
import json

import pytest

from eeg_agent.brain import BrainSession, _language, _resolve_report_claims, _tool_facts


TOOLS = [{'tool': 'fatigue.raw_spectrum', 'result': {'available': True, 'measurements': {
    'task_value': 2.1124514776090777,
    'first_to_last': {'first_ordinal': 1, 'last_ordinal': 7,
        'first_value': 1.8585300723087905, 'last_value': 2.4323665636896363,
        'first_start_clock': '00:53.812', 'first_end_clock': '02:33.733',
        'last_start_clock': '33:19.370', 'last_end_clock': '35:31.802',
        'change': {'text': '上升30.88%'}}, 'unavailable': None}}}]
TEXT = ('疲劳指标完整路径汇总为2.112451；第1条（00:53.812–02:33.733）为1.858530，'
        '第7条（33:19.370–35:31.802）为2.432367，首末上升30.88%。')


def reply(ids, text=TEXT):
    return {'explanation': text, 'citations': [], 'measurement_claim_ids': ids}


def test_report_ids_resolve_exact_times_values_and_ordinals_without_changing_tools():
    before = copy.deepcopy(TOOLS)
    value = _resolve_report_claims(reply([f'F{i}' for i in range(len(_tool_facts(TOOLS)))]), TOOLS)
    assert _language(value, [], TOOLS)[0] == TEXT
    assert value['measurement_claims'] == _tool_facts(TOOLS)
    assert value['measurement_claims'][0]['value'] == 2.1124514776090777
    assert TOOLS == before


@pytest.mark.parametrize('ids', [['unknown'], ['F999'], [None], [True], 'F0'])
def test_unknown_or_null_fact_ids_are_never_accepted(ids):
    with pytest.raises(ValueError, match='unknown measurement source'):
        _resolve_report_claims(reply(ids), TOOLS)


def test_valid_ids_do_not_allow_fabricated_values_or_missing_time_sources():
    all_ids = [f'F{i}' for i in range(len(_tool_facts(TOOLS)))]
    with pytest.raises(ValueError, match='numerical statement lacks'):
        _language(_resolve_report_claims(reply(all_ids, TEXT.replace('2.112451', '9.123456')), TOOLS), [], TOOLS)
    with pytest.raises(ValueError, match='numerical statement lacks'):
        _language(_resolve_report_claims(reply(['F0']), TOOLS), [], TOOLS)


def test_ids_cannot_reference_unavailable_class_or_create_a_class_from_numeric_evidence():
    tools = [{'tool': 'vrms.raw_model', 'result': {'measurements': {
        'segments': [{'ordinal': 1, 'predicted_class': None}], 'first_high': None}}}]
    facts = _tool_facts(tools)
    assert facts == [{'source': 'vrms.raw_model', 'path': 'measurements.segments.0.ordinal', 'value': 1}]
    with pytest.raises(ValueError, match='lacks cloud judgment tool source'):
        _language(_resolve_report_claims(reply(['F0'], '最终分类为High。'), tools), [], tools)


def test_missing_first_high_time_is_not_bound_to_the_next_incomplete_path(tmp_path):
    session = object.__new__(BrainSession)
    session.recording_id, session.turn, session.directory = 'subject-08', 1, tmp_path
    tools = [{'tool': 'vrms.raw_model', 'result': {'available': True, 'measurements': {
        'segments': [{'ordinal': 8, 'predicted_class': None}], 'first_high': None}}}]
    text = '所有路径云端分类未完成，无首条高类路径区间；第8条因缺少结束标记未分类。'
    value = _resolve_report_claims(reply(['F0'], text), tools)
    assert session._checked_language(value, [], tools, 'vrms')[0] == text


def test_selected_segment_class_includes_its_own_ordinal_and_time_interval_only():
    tools = [{'tool': 'vrms.raw_model', 'result': {'measurements': {'segments': [
        {'ordinal': 1, 'start_clock': '00:53.812', 'end_clock': '02:33.733', 'predicted_class': 'High'},
        {'ordinal': 2, 'start_clock': '05:48.606', 'end_clock': '08:01.370', 'predicted_class': 'Low'}]}}}]
    facts = _tool_facts(tools)
    key = next(f'F{i}' for i, f in enumerate(facts) if f['path'] == 'measurements.segments.0.predicted_class')
    value = _resolve_report_claims(reply([key], '第1条路径00:53.812–02:33.733，VRMS最终判为High。'), tools)
    assert {f['path'] for f in value['measurement_claims']} == {
        'measurements.segments.0.' + k for k in ('predicted_class', 'ordinal', 'start_clock', 'end_clock')}
    assert _language(value, [], tools)
    with pytest.raises(ValueError, match='numerical statement lacks'):
        _language({**value, 'explanation': value['explanation'].replace('05:48.606', '10:48.606').replace('00:53.812', '05:48.606')}, [], tools)


def test_selected_endpoint_value_resolves_existing_endpoint_identity_not_array_position():
    facts = _tool_facts(TOOLS)
    key = next(f'F{i}' for i, f in enumerate(facts) if f['path'] == 'measurements.first_to_last.first_value')
    value = _resolve_report_claims(reply([key], '第1条00:53.812–02:33.733，指标1.858530。'), TOOLS)
    assert _language(value, [], TOOLS)
    assert all(f['path'].startswith('measurements.first_to_last.first_') for f in value['measurement_claims'])


def test_recording_counts_are_bound_together_without_confusing_total_and_completed(tmp_path):
    session = object.__new__(BrainSession)
    session.recording_id, session.turn, session.directory = 'subject-08', 1, tmp_path
    tools = [{'tool': 'vrms.raw_recording', 'result': {'available': True, 'measurements': {
        'task_count': 8, 'complete_task_count': 7, 'incomplete_task_count': 1}}}]
    value = _resolve_report_claims(reply(['F1'], '原始记录8个任务片段，7条完整、1条不完整，共完成7条路径。'), tools)
    assert {f['path'] for f in value['measurement_claims']} == {f['path'] for f in _tool_facts(tools)}
    assert session._checked_language(value, [], tools, 'vrms')
    with pytest.raises(ValueError, match='completed path count conflicts'):
        session._checked_language({**value, 'explanation': value['explanation'].replace('完成7条', '完成8条')}, [], tools, 'vrms')


@pytest.mark.parametrize('text,valid', [
    ('VRMS整路径判定为High的是第1、2条，Low的是第3条。首条High为第1条。', True),
    ('第1、2条为High，第3条为Low。', True),
    ('VRMS整路径判定为High的是第1、3条，Low的是第2条。', False),
    ('第1、2条为Low，第3条为High。', False)])
def test_class_binding_supports_both_word_orders_without_crossing_clause_boundaries(tmp_path, text, valid):
    session = object.__new__(BrainSession)
    session.recording_id, session.turn, session.directory = 'subject-08', 1, tmp_path
    tools = [{'tool': 'vrms.raw_model', 'result': {'available': True, 'measurements': {
        'segments': [{'ordinal': i, 'predicted_class': cls} for i, cls in enumerate(('High', 'High', 'Low'), 1)],
        'first_high': {'ordinal': 1}}}}]
    value = _resolve_report_claims(reply([f'F{i}' for i in range(len(_tool_facts(tools)))], text), tools)
    if valid:
        assert session._checked_language(value, [], tools, 'vrms')[0] == text
    else:
        with pytest.raises(ValueError, match='classification conflicts'):
            session._checked_language(value, [], tools, 'vrms')


def test_report_gets_one_source_correction_and_still_uses_strict_validation(tmp_path):
    session = object.__new__(BrainSession)
    session.recording_id, session.turn, session.directory = 'subject-08', 1, tmp_path
    events, payloads = [], []
    session.emit = lambda kind, payload: events.append((kind, payload))
    def call(role, payload, instruction):
        payloads.append(copy.deepcopy(payload))
        return reply(['F0'] if len(payloads) == 1 else [f'F{i}' for i in range(len(_tool_facts(TOOLS)))])
    session._call = call
    assert session._report('FatigueAgent', {}, '', [], TOOLS, 'fatigue')[0] == TEXT
    assert len(payloads) == 2
    assert payloads[1]['validation_feedback']['validation_code'] == 'number_without_source'
    assert events[0][0] == 'report_repair'
    assert (tmp_path / 'unverified_fatigue_1.json').exists()


def test_failed_source_correction_never_bypasses_validation_and_api_errors_are_not_retried(tmp_path):
    session = object.__new__(BrainSession)
    session.recording_id, session.turn, session.directory = 'subject-08', 1, tmp_path
    session.emit = lambda *args: None
    calls = []
    def invalid(*args):
        calls.append(1)
        return reply(['F0'])
    session._call = invalid
    with pytest.raises(ValueError, match='numerical statement lacks'):
        session._report('FatigueAgent', {}, '', [], TOOLS, 'fatigue')
    assert len(calls) == 2
    def failed(*args):
        calls.append(1)
        raise RuntimeError('API failed')
    session._call = failed
    with pytest.raises(RuntimeError, match='API failed'):
        session._report('FatigueAgent', {}, '', [], TOOLS, 'fatigue')
    assert len(calls) == 3


def test_report_wire_format_contains_ids_but_preserves_original_tool_payload():
    session = object.__new__(BrainSession)
    session.recording_id = None
    sent = []
    class Provider:
        def call(self, role, payload, messages=None):
            sent.append((payload, messages))
            return reply(['F0'], '疲劳指标为2.112451。')
    session.provider = Provider()
    payload = {'tool_results': TOOLS, 'tool_facts': _tool_facts(TOOLS)}
    session._call('FatigueAgent', payload, '')
    assert 'id' not in sent[0][0]['tool_facts'][0]
    wire = json.loads(sent[0][1][1]['content'])
    assert wire['tool_facts'][0]['id'] == 'F0'
    assert 'measurement_claim_ids' in sent[0][1][0]['content']
