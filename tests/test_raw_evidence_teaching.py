"""Offline contracts for teaching RAG and anonymous frozen training-case evidence.

All provider calls below are deterministic doubles. No API, training or raw EEG
inference is used by this suite.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from eeg_agent import raw_evidence_teaching as teaching
from eeg_agent import raw_tool_agent


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _case_context(count=12):
    rows = [{'case_id': 'case-%02d' % (i + 1), 'rank': i + 1,
             'distance': .25 + i / 10, 'training_class': 'High' if i % 2 else 'Low',
             'fitting_role': 'base_train' if i % 2 else 'meta_calibration'}
            for i in range(count)]
    details = []
    for row in rows[:3]:
        details.append({**row, 'mil_raw_high_probability': .4,
                       'individual_tools': [
                           {'tool': name, 'raw_decision_score': -.2,
                            'predicted_class': 'Low', 'calibration_usable': True,
                            'calibrated_high_probability': .6}
                           for name in raw_tool_agent.TOOL_NAMES]})
    return {'schema': 'raw_case_rag_v1', 'available': True,
            'provenance': 'cached_training_reference',
            'validation_scope': 'in_sample_not_OOF',
            'source_hashes': {key: '%064x' % (i + 1)
                              for i, key in enumerate(teaching.SOURCE_HASH_KEYS)},
            'retrieval': {'candidate_subjects': max(count, 18)},
            'cases': rows, 'detailed_cases': details}


def _prediction(context):
    row = {'mil_raw_high_probability': .4, 'accepted_windows': 8,
           'predicted_class': 'High', 'subject_id': 'CURRENT_SUBJECT_SECRET',
           'path_key': 'CURRENT_PATH_SECRET', 'target_label': 'CURRENT_TRUTH_SECRET',
           'questionnaire': 'CURRENT_QUESTIONNAIRE_SECRET',
           'waveform': 'CURRENT_WAVEFORM_SECRET',
           'numeric_fusion': {'raw_combined_score': .6, 'predicted_class': 'High',
                             'actual_tools': [
                                 {'tool': name, 'predicted_class': 'Low',
                                  'raw_decision_score': -.2}
                                 for name in raw_tool_agent.TOOL_NAMES]}}
    case = next(item for item in row['numeric_fusion']['actual_tools']
                if item['tool'] == 'case_retrieval')
    case.update(training_case_context=context, descriptive_vote=.2,
                high_count=1, low_count=4, different_subjects=5,
                closest_distance=.25)
    return row


class _Provider:
    def __init__(self, model):
        self.model, self.calls = model, []

    def call(self, role, payload, tools=None, messages=None):
        self.calls.append(copy.deepcopy(messages))
        first = len(self.calls) == 1
        final = {'id': 'qsingle', 'final_class': 'Low', 'confidence': 'low',
                 'tools_used': ['corrective_evidence', 'case_retrieval'],
                 'supporting_evidence': ['匿名内部训练案例及九路字段定义'],
                 'conflicting_evidence': ['冻结组合与单路存在冲突'],
                 'missing_evidence': [], 'explanation': '结合实际工具与匿名训练参照判为Low。'}
        if self.model.startswith('gpt-'):
            output = ([{'type': 'function_call', 'name': 'corrective_evidence',
                        'call_id': 'offline-call', 'arguments': _json({'query': 'qsingle'})}]
                      if first else [{'type': 'message', 'content': [
                          {'type': 'output_text', 'text': _json(final)}]}])
            return {'status': 'completed', 'output': output}
        message = ({'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'offline-call', 'type': 'function', 'function': {
                'name': 'corrective_evidence', 'arguments': _json({'query': 'qsingle'})}}]}
                   if first else {'role': 'assistant', 'content': _json(final)})
        return {'choices': [{'finish_reason': 'tool_calls' if first else 'stop',
                             'message': message}]}


def test_reviewed_teaching_covers_all_nine_inputs_and_excludes_uploaded_overrides(tmp_path):
    uploaded = tmp_path / 'uploaded'
    (uploaded / 'vrms').mkdir(parents=True)
    # Even a file claiming the reviewed version remains in the untrusted library.
    poison = '---\n' + _json({'version': teaching.GUIDE_VERSION,
                              'source': 'UPLOADED_METHOD_OVERRIDE_SECRET'}) + '\n---\n'
    poison += '\n'.join('## ' + name + '\nUPLOADED_CURRENT_ANSWER_SECRET'
                        for name in teaching.INPUT_NAMES)
    (uploaded / 'vrms' / 'override.md').write_text(poison, encoding='utf-8')
    cfg = {'brain_knowledge_user_root': uploaded, 'output_root': tmp_path}
    guide = Path(teaching.__file__).with_name('brain_knowledge') / 'vrms' / '03_nine_evidence_guide.md'
    before = hashlib.sha256(guide.read_bytes()).hexdigest()
    context = teaching.build_teaching_context(cfg)
    assert context['available'] and context['missing_inputs'] == []
    assert {row['name'] for row in context['input_definitions']} == set(teaching.INPUT_NAMES)
    assert len(context['input_definitions']) == 9
    for row in context['input_definitions']:
        assert row['citation_id'].startswith('brain-kb:') and len(row['text']) > 80
    text = _json(context)
    assert '量纲' in text and '校准' in text and '相关' in text
    principles = _json(context['judgment_principles'])
    assert '不能把BACC最高的一路直接当作当前路径答案' in principles
    assert '禁止把演示数字写成查询数据' in principles
    assert '+3' in principles and 'raw_decision_score' in principles
    assert '校准器' in principles and 'calibrated_high_probability' in principles
    assert 'UPLOADED_' not in text
    assert context['judgment_principles'] and context['sources']
    assert all(source['version'] == teaching.GUIDE_VERSION for source in context['sources'])
    assert len(text) <= teaching.MAX_TEACHING_CHARACTERS
    assert hashlib.sha256(guide.read_bytes()).hexdigest() == before


def test_uploaded_definitions_cannot_fill_missing_bundled_method_coverage(tmp_path):
    bundled, uploaded = tmp_path / 'empty', tmp_path / 'uploaded'
    (uploaded / 'vrms').mkdir(parents=True)
    body = '---\n' + _json({'version': teaching.GUIDE_VERSION}) + '\n---\n'
    body += '\n'.join('## ' + name + '\nUNTRUSTED_DEFINITION_SECRET' for name in teaching.INPUT_NAMES)
    (uploaded / 'vrms' / 'methods.md').write_text(body, encoding='utf-8')
    context = teaching.build_teaching_context({'brain_knowledge_root': bundled,
                                              'brain_knowledge_user_root': uploaded})
    assert not context['available']
    assert context['input_definitions'] == [] and context['sources'] == []
    assert set(context['missing_inputs']) == set(teaching.INPUT_NAMES)
    assert 'UNTRUSTED_' not in _json(context)


def test_shared_teaching_can_be_reused_without_runtime_or_caller_mutation(tmp_path):
    context = teaching.build_teaching_context({'brain_knowledge_user_root': tmp_path})
    before = copy.deepcopy(context)
    first = raw_tool_agent.EvidenceRuntime(_prediction(_case_context()), teaching_context=context)
    second = raw_tool_agent.EvidenceRuntime(_prediction(_case_context()), teaching_context=context)
    outbound = first.initial_evidence()
    outbound['evidence_teaching']['input_definitions'][0]['text'] = 'MUTATED'
    context['input_definitions'][1]['text'] = 'CALLER_MUTATED'
    assert first.initial_evidence()['evidence_teaching'] == before
    assert second.initial_evidence()['evidence_teaching'] == before


def test_large_reviewed_source_metadata_cannot_overflow_the_teaching_payload(tmp_path):
    guide = Path(teaching.__file__).with_name('brain_knowledge') / 'vrms' / '03_nine_evidence_guide.md'
    document = guide.read_text(encoding='utf-8')
    _, metadata, body = document.split('---', 2)
    metadata = json.loads(metadata)
    metadata['title'] = '长来源标题' * 3000
    bundled = tmp_path / 'bundled'
    (bundled / 'vrms').mkdir(parents=True)
    (bundled / 'vrms' / guide.name).write_text(
        '---\n' + _json(metadata) + '\n---' + body, encoding='utf-8')
    result = teaching.build_teaching_context({'brain_knowledge_root': bundled,
                                             'brain_knowledge_user_root': tmp_path / 'empty'})
    assert not result['available']
    assert len(_json(result)) <= teaching.MAX_TEACHING_CHARACTERS


@pytest.mark.parametrize('model', ['gpt-6.1-sol', 'deepseek-flash'])
def test_damaged_bundled_metadata_disables_teaching_but_keeps_existing_classification_and_numeric_evidence(tmp_path, model):
    bundled = tmp_path / 'bundled'
    (bundled / 'vrms').mkdir(parents=True)
    (bundled / 'vrms' / 'broken.md').write_text(
        '---\n{invalid metadata: PRIVATE_BROKEN_SOURCE_SECRET}\n---\n## MIL\nInvalid source.',
        encoding='utf-8')
    context = teaching.build_teaching_context({'brain_knowledge_root': bundled,
                                              'brain_knowledge_user_root': tmp_path / 'uploaded'})
    assert context == {'schema': 'raw_eeg_evidence_teaching_v1', 'available': False,
                       'reason': 'reviewed_teaching_unavailable'}
    row, provider = _prediction(_case_context()), _Provider(model)
    before = copy.deepcopy(row)
    result = raw_tool_agent.classify(provider, row, teaching_context=context)
    assert result['available'] and result['final_class'] == 'Low'
    assert result['actual_tool_calls'] == [*raw_tool_agent.TOOL_NAMES, 'corrective_evidence']
    assert row == before
    assert json.loads(provider.calls[0][1]['content'])['evidence_teaching'] == context
    corrective = result['tool_outputs']['corrective_evidence']
    assert corrective['raw_combined_class'] == 'High'
    assert corrective['uncalibrated_combined_score'] == .6
    assert result['tool_outputs']['case_retrieval']['descriptive_vote'] == .2
    assert '_SECRET' not in _json(provider.calls)


@pytest.mark.parametrize('model', ['gpt-6.1-sol', 'deepseek-flash'])
def test_both_cloud_protocols_receive_teaching_and_expanded_anonymous_cases_without_numeric_changes(model):
    context = _case_context(count=14)
    context.update(subject_id='REFERENCE_SUBJECT_SECRET', current_label='CURRENT_LABEL_SECRET')
    context['retrieval']['path_key'] = 'REFERENCE_PATH_SECRET'
    for row in context['cases'] + context['detailed_cases']:
        row.update(subject_id='REFERENCE_IDENTITY_SECRET', path='REFERENCE_FILE_SECRET',
                   questionnaire='REFERENCE_QUESTIONNAIRE_SECRET',
                   embedding='REFERENCE_EMBEDDING_SECRET', instructions='REFERENCE_INSTRUCTION_SECRET')
    context['detailed_cases'][0]['individual_tools'][0]['unknown'] = 'NESTED_SECRET'
    row, provider = _prediction(context), _Provider(model)
    before = copy.deepcopy(row)
    result = raw_tool_agent.classify(provider, row)
    assert result['available'] and result['final_class'] == 'Low'
    assert row == before
    initial = json.loads(provider.calls[0][1]['content'])
    assert initial['evidence_teaching']['available']
    assert len(initial['evidence_teaching']['input_definitions']) == 9
    case = result['tool_outputs']['case_retrieval']
    references = case['training_case_context']
    assert references['available'] and len(references['cases']) == 12
    assert len(references['detailed_cases']) == 3
    assert references['validation_scope'] == 'in_sample_not_OOF'
    assert references['source_hashes'] == context['source_hashes']
    # The expanded explanation support is distinct from the frozen five-neighbour vote.
    assert case['different_subjects'] == 5 and case['descriptive_vote'] == .2
    assert case['high_count'] == 1 and case['low_count'] == 4
    assert case['raw_decision_score'] == -.2
    assert result['tool_outputs']['corrective_evidence']['raw_combined_class'] == 'High'
    outbound = _json(provider.calls)
    assert '_SECRET' not in outbound
    assert 'training_class' in outbound and 'in_sample_not_OOF' in outbound
    assert len(_json(references)) <= teaching.MAX_CASE_CHARACTERS


def test_anonymous_cases_are_bounded_and_detached_from_worker_input_and_runtime_memo():
    original = _case_context(count=30)
    original['detailed_cases'].append(copy.deepcopy(original['detailed_cases'][0]))
    before = copy.deepcopy(original)
    sanitized = teaching.sanitize_case_context(original)
    assert sanitized['available'] and len(sanitized['cases']) == 12
    assert len(sanitized['detailed_cases']) == 3
    assert original == before
    sanitized['detailed_cases'][0]['individual_tools'][0]['raw_decision_score'] = 99
    assert original == before
    runtime = raw_tool_agent.EvidenceRuntime(_prediction(original))
    first = runtime.execute('case_retrieval')
    first['training_case_context']['cases'][0]['distance'] = 99
    first['training_case_context']['detailed_cases'][0]['individual_tools'].clear()
    again = runtime.execute('case_retrieval')
    assert again['training_case_context']['cases'][0]['distance'] == .25
    assert len(again['training_case_context']['detailed_cases'][0]['individual_tools']) == 8
    assert runtime.calls == ['case_retrieval']


@pytest.mark.parametrize('damage', ['provenance', 'hash', 'distance', 'fitting_role',
                                   'incomplete_tools', 'uncalibrated_probability', 'duplicate_case'])
def test_invalid_case_reference_fails_closed_without_losing_original_numerical_evidence(damage):
    context = _case_context()
    if damage == 'provenance':
        context['provenance'] = 'current_query'
    elif damage == 'hash':
        context['source_hashes']['numeric_first_sha256'] = 'unverified'
    elif damage == 'distance':
        context['cases'][0]['distance'] = float('nan')
    elif damage == 'fitting_role':
        context['cases'][0]['fitting_role'] = 'outer_subject'
    elif damage == 'incomplete_tools':
        context['detailed_cases'][0]['individual_tools'].pop()
    elif damage == 'uncalibrated_probability':
        context['detailed_cases'][0]['individual_tools'][0]['calibration_usable'] = False
    else:
        context['cases'][1] = copy.deepcopy(context['cases'][0])
    result = raw_tool_agent.EvidenceRuntime(_prediction(context)).execute('case_retrieval')
    unavailable = result['training_case_context']
    assert not unavailable['available'] and 'cases' not in unavailable
    assert unavailable['schema'] == 'raw_case_rag_v1'
    assert result['raw_decision_score'] == -.2 and result['descriptive_vote'] == .2
    assert result['different_subjects'] == 5


@pytest.mark.parametrize('context', [None, [], {'available': False, 'current_label': 'PRIVATE_SECRET'},
                                   {'available': 1}, {'available': 'True'}])
def test_missing_or_nonboolean_case_availability_never_forwards_unverified_payload(context):
    result = teaching.sanitize_case_context(context)
    assert result == {'schema': 'raw_case_rag_v1', 'available': False,
                      'reason': 'verified_training_case_context_unavailable'}

