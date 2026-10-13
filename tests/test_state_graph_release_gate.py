"""Promotion must audit new semantic requests, not reuse old accuracy reports."""
import copy
import hashlib
import json

import pytest

from scripts import audit_state_graph_rag as audit
from eeg_agent import raw_independent_judge as judge, state_graph_rag as semantic
from tests.test_state_graph_rag import measured


def evaluation_fixture(high=73, correct=112):
    # Synthetic test records only; never persisted as a real evaluation.
    observation_document = measured()
    state = semantic.explain({'eeg_observations': observation_document}, semantic.load_assets())
    context = {'schema': semantic.SCHEMA, 'variant': semantic.VARIANT, 'available': True,
               'source_hashes': {'semantic_manifest_sha256': semantic.MANIFEST_SHA256},
               'state_evidence': state, 'review_guidance': {'citation_id': 'state-evidence-review'}}
    payload = {'initial_evidence': {'policy_heldout_case_context': context,
                                   'signal_quality': {'accepted_windows': observation_document['accepted_windows']}},
               'corrective_evidence': {'actual_individual_evidence': [{'tool': name} for name in judge.TOOLS]}}
    rows = []
    # Spread errors across both classes so the ordinary test meets BACC too.
    misses = 146 - correct
    high_misses, low_misses = misses // 2, misses - misses // 2
    for index, query_id in enumerate(audit.QUERY_IDS):
        truth = 'High' if index < high else 'Low'
        wrong = index < high_misses if truth == 'High' else index - high < low_misses
        predicted = ('Low' if truth == 'High' else 'High') if wrong else truth
        rows.append({'query_id': query_id, 'response_id': f'test-response-{index}',
                     'request_payload': copy.deepcopy(payload), 'request_sha256': semantic.digest(payload),
                     'true_class': truth,
                     'raw_llm_final': {'final_class': predicted, 'confidence': 'low',
                                       'explanation': 'Synthetic test fixture.', 'supporting_tools': ['MIL'],
                                       'counterevidence_tools': [],
                                       'rag_citations': [state['supporting_paths'][0]['citation_id']]}})
    evaluation = {'schema': 'semantic_graph_evaluation_v1', 'semantic_schema': semantic.SCHEMA,
                  'variant': semantic.VARIANT, 'semantic_manifest_sha256': semantic.MANIFEST_SHA256,
                  'numeric_source_sha256': audit.NUMERIC_SHA256, 'judge_profile': 'semantic_state_high_v1',
                  'new_semantic_judgments': True, 'model_calls': 146, 'records': rows,
                  'rag_policy': semantic.asset_policy()}
    engineering = {'schema': 'semantic_graph_engineering_audit_v1',
                   'semantic_manifest_sha256': semantic.MANIFEST_SHA256,
                   'numeric_source_sha256': audit.NUMERIC_SHA256,
                   'engineering_passed': True, 'paths_checked': 146, 'paths_passed': 146}
    return evaluation, engineering


def test_engineering_only_never_promotes_or_computes_accuracy():
    evaluation, engineering = evaluation_fixture()
    result = audit.release_gate(None, engineering)
    assert result['promotion_status'] == 'not_evaluated'
    assert result['release_ready'] is False and result['metrics'] is None


def test_146_new_audited_responses_and_both_metric_thresholds_are_eligible():
    evaluation, engineering = evaluation_fixture()
    result = audit.release_gate(evaluation, engineering)
    assert result['release_ready'] is True and result['promotion_status'] == 'eligible'
    assert result['metrics']['correct'] == 112
    assert result['metrics']['bacc'] == 112 / 146
    assert not result['reasons']


@pytest.mark.parametrize('failure', ['old_variant', 'partial', 'label', 'summary', 'missing_citations',
                                    'wrong_manifest', 'old_response', 'engineering', 'no_capture',
                                    'no_observations', 'no_semantic_citation', 'wrong_code_pin', 'forged_path'])
def test_fails_closed_without_complete_new_version_or_isolation_evidence(failure):
    evaluation, engineering = evaluation_fixture()
    row = evaluation['records'][0]
    if failure == 'old_variant':
        evaluation['variant'] = 'graph_v3'
    elif failure == 'partial':
        evaluation['records'].pop()
    elif failure == 'label':
        row['request_payload']['true_class'] = 'High'
    elif failure == 'summary':
        row['request_payload']['initial_evidence']['policy_heldout_case_context']['class_descriptions'] = {}
    elif failure == 'missing_citations':
        del row['raw_llm_final']['rag_citations']
    elif failure == 'wrong_manifest':
        evaluation['semantic_manifest_sha256'] = 'wrong'
    elif failure == 'old_response':
        evaluation['new_semantic_judgments'] = False
    elif failure == 'engineering':
        engineering['paths_passed'] = 145
    elif failure == 'no_capture':
        del row['request_payload']
    elif failure == 'no_observations':
        row['request_payload']['initial_evidence']['policy_heldout_case_context']['state_evidence'] = semantic._empty('actual_eeg_observations_unavailable')
    elif failure == 'no_semantic_citation':
        row['raw_llm_final']['rag_citations'] = ['state-evidence-review']
    elif failure == 'wrong_code_pin':
        evaluation['rag_policy']['semantic_code_sha256'] = 'old-implementation'
    elif failure == 'forged_path':
        path = row['request_payload']['initial_evidence']['policy_heldout_case_context']['state_evidence']['supporting_paths'][0]
        path['claims'][0]['text'] = 'A forged unsupported medical assertion.'
        path['content_sha256'] = semantic._content(path)
        row['request_sha256'] = semantic.digest(row['request_payload'])
    result = audit.release_gate(evaluation, engineering)
    assert result['release_ready'] is False and result['promotion_status'] == 'rejected'


def test_acc_cannot_hide_low_bacc_or_a_supplied_fake_metric():
    evaluation, engineering = evaluation_fixture(high=56)
    for index, row in enumerate(evaluation['records']):
        row['raw_llm_final']['final_class'] = 'Low' if index < 34 else row['true_class']
    evaluation['bacc'] = 1.0
    result = audit.release_gate(evaluation, engineering)
    assert result['metrics']['correct'] == 112
    assert result['metrics']['bacc'] < audit.MIN_BACC
    assert result['release_ready'] is False


def test_below_112_correct_cannot_promote_despite_other_complete_audits():
    evaluation, engineering = evaluation_fixture(correct=111)
    result = audit.release_gate(evaluation, engineering)
    assert result['metrics']['correct'] == 111
    assert result['release_ready'] is False
    assert 'correct_below_112_of_146' in result['reasons']


def test_actual_serialized_judge_hash_is_distinct_from_canonical_capture():
    row = {'request_payload': {'key': 'value'}, 'request_sha256': semantic.digest({'key': 'value'})}
    assert audit.captured_payload(row) == {'key': 'value'}
    serialized = json.dumps({'key': 'value'})
    row = {'request_json': serialized,
           'model_payload_sha256': hashlib.sha256(serialized.encode()).hexdigest()}
    assert audit.captured_payload(row) == {'key': 'value'}
    # Real request serialization may preserve insertion order and whitespace.
    serialized = '{ "key" : "value" }'
    row['request_json'] = serialized
    with pytest.raises(ValueError, match='serialized'):
        audit.captured_payload(row)
    row['model_payload_sha256'] = hashlib.sha256(serialized.encode()).hexdigest()
    assert audit.captured_payload(row) == {'key': 'value'}
    row['request_payload'] = {'key': 'different'}
    with pytest.raises(ValueError, match='differs'):
        audit.captured_payload(row)
