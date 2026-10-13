"""Boundary facts, conservative coverage, literal fallback and provenance."""
import copy
import json

import numpy as np
import pytest

from eeg_agent import graph_rag, graph_rag_v3 as graph3, regular_rag, raw_independent_judge
from tests.test_graph_rag import PROVENANCE, bank_with_scores, prediction, tools


def calibrated_bank(calibrations=None, scores=None, labels=None):
    bank = bank_with_scores(scores or [[-1.] * 8 for _ in range(8)], labels)
    calibrations = calibrations or [[.9] + [None] * 7 for _ in bank['cases']]
    for case, probabilities in zip(bank['cases'], calibrations):
        raw = [row['raw_decision_score'] for row in case['individual_tools']]
        case['individual_tools'] = tools(raw, probabilities)
    labels = np.asarray([case['reference_class'] == 'High' for case in bank['cases']])
    for index, name in enumerate(regular_rag.FAMILIES):
        probabilities = [case['individual_tools'][index]['calibrated_high_probability'] for case in bank['cases']]
        if all(value is None for value in probabilities):
            continue
        high = np.asarray(probabilities) >= .5
        bank['no_rest_policy_confusion']['rows'].append({
            'tool': name, 'boundary': 'calibrated_probability_0.5',
            'paths': len(bank['cases']), 'subjects': len(bank['split']['policy_validation']),
            'true_Low_output_Low': int((~labels & ~high).sum()),
            'true_Low_output_High': int((~labels & high).sum()),
            'true_High_output_Low': int((labels & ~high).sum()),
            'true_High_output_High': int((labels & high).sum())})
    return regular_rag.validate_bank(bank, bank['outer_subject'])


def assets(bank):
    source = (graph_rag.GRAPH_MANIFEST.parent / 'method_source.md').read_text(encoding='utf-8')
    return {'fold_graph': graph3.build_fold_graph(bank, PROVENANCE),
            'method_graph': graph3.build_method_graph(source), 'method_source_text': source,
            'graph_manifest_sha256': 'd' * 64}


def conflict_query(all_tools=False):
    return prediction(scores=[-1] * 8, calibrations=[.9] * 8 if all_tools else [.9] + [None] * 7)


def test_no_conflict_is_literal_ordinary_even_with_shared_feature_disagreement():
    bank, query = calibrated_bank(), prediction()
    expected = regular_rag.context_from_bank(query, bank, PROVENANCE, 'percentile_group')
    context = graph3.context_from_bank(query, bank, PROVENANCE, assets(bank))
    assert json.dumps(context, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)
    assert 'graph_evidence' not in context
    assert graph3.candidate_diagnostics(query, bank, PROVENANCE, assets(bank))['literal_regular']


def test_only_two_matching_subjects_fails_leave_subject_out_coverage():
    calibration = [[.9 if index < 4 else .1] + [None] * 7 for index in range(8)]
    bank, query = calibrated_bank(calibration), conflict_query()
    expected = regular_rag.context_from_bank(query, bank, PROVENANCE, 'percentile_group')
    context = graph3.context_from_bank(query, bank, PROVENANCE, assets(bank))
    assert context == expected
    diagnostic = graph3.candidate_diagnostics(query, bank, PROVENANCE, assets(bank))
    assert diagnostic['fallback_reason'] == 'no_subject_deletion_supported_boundary_condition'


def test_counted_boundary_errors_and_constant_direction_are_exact():
    bank, query = calibrated_bank(), conflict_query()
    ordinary = regular_rag.context_from_bank(query, bank, PROVENANCE, 'percentile_group')
    context = graph3.context_from_bank(query, bank, PROVENANCE, assets(bank))
    assert context['cases'][:len(ordinary['cases'])] == ordinary['cases']
    package = context['graph_evidence']['packages'][0]
    counts = package['conditional_evidence']
    assert (counts['paths'], counts['subjects'], counts['Low'], counts['High']) == (8, 4, 4, 4)
    assert counts['raw_correct'] == counts['calibrated_correct'] == 4
    assert len(counts['by_subject']) == 4
    assert all(row['paths'] == 2 and row['raw_correct'] == row['calibrated_correct'] == 1
               for row in counts['by_subject'])
    assert all(row['paths'] == 6 and row['subjects'] == 3 and row['raw_correct'] == row['calibrated_correct'] == 3
               for row in counts['leave_subject_out'])
    assert package['whole_bank_calibration']['constant_direction'] == 'High'
    assert package['whole_bank_calibration']['probability_range'] == [.9, .9]
    assert package['example_case_ids']['Low'] and package['example_case_ids']['High']
    assert package['source_sha256'] == PROVENANCE['current_raw_policy_bank_sha256']
    assert 'FEMBA' in package['shared_sources']


def test_opposite_transition_and_missing_calibration_never_match():
    bank = calibrated_bank([[.1] + [None] * 7 for _ in range(8)], [[1.] * 8 for _ in range(8)])
    query = conflict_query()
    assert graph3.context_from_bank(query, bank, PROVENANCE, assets(bank)) == regular_rag.context_from_bank(
        query, bank, PROVENANCE, 'percentile_group')
    missing = bank_with_scores([[-1.] * 8 for _ in range(8)])
    assert graph3.context_from_bank(query, missing, PROVENANCE, assets(missing)) == regular_rag.context_from_bank(
        query, missing, PROVENANCE, 'percentile_group')


def test_query_truth_mil_and_hidden_fusion_cannot_route_or_rank_graph():
    bank, query = calibrated_bank(), conflict_query()
    original = graph3.context_from_bank(query, bank, PROVENANCE, assets(bank))
    changed = copy.deepcopy(query)
    changed.update(mil_raw_high_probability=.99, true_class='High', subject=99, previous_answer='Low')
    changed['numeric_fusion'].update(raw_combined_score=.99, predicted_class='Low')
    second = graph3.context_from_bank(changed, bank, PROVENANCE, assets(bank))
    assert original['cases'] == second['cases']
    assert original['graph_evidence'] == second['graph_evidence']
    assert graph3.candidate_diagnostics(query, bank, PROVENANCE, assets(bank)) == graph3.candidate_diagnostics(
        changed, bank, PROVENANCE, assets(bank))
    payload = raw_independent_judge.build_payload(changed, teaching_context={'available': False}, rag_context=second)
    assert not raw_independent_judge._hidden(payload)


def test_complete_package_budget_and_all_numeric_facts_preserved():
    bank, query = calibrated_bank([[.9] * 8 for _ in range(8)]), conflict_query(True)
    before = copy.deepcopy(query)
    ordinary = regular_rag.context_from_bank(query, bank, PROVENANCE, 'percentile_group')
    context = graph3.context_from_bank(query, bank, PROVENANCE, assets(bank))
    assert query == before
    for key in ('query_reading', 'class_descriptions', 'direction_error_counts', 'group_sign_pattern_reference_counts'):
        assert context[key] == ordinary[key]
    packages = context['graph_evidence']['packages']
    assert 1 <= len(packages) <= 3
    assert len(json.dumps(context['graph_evidence'], ensure_ascii=False)) <= 4000
    assert all({'transition', 'conditional_evidence', 'whole_bank_calibration', 'example_case_ids', 'source_sha256'} <= set(row)
               for row in packages)
    assert len(context['cases']) <= 6
    assert context['cases'][:len(ordinary['cases'])] == ordinary['cases']
    payload = raw_independent_judge.build_payload(query, teaching_context={'available': False}, rag_context=context)
    assert {row['citation_id'] for row in packages} <= raw_independent_judge._citation_ids(payload)


def test_supplemental_examples_are_real_transition_members_and_keep_every_original_case():
    query = conflict_query()
    initial = calibrated_bank()
    core = regular_rag.context_from_bank(query, initial, PROVENANCE, 'percentile_group')['cases']
    core_mil = {row['raw_values'][0] for row in core}
    calibration = [[.1 if case['mil_raw_high_probability'] in core_mil else .9] + [None] * 7
                   for case in initial['cases']]
    bank = calibrated_bank(calibration)
    ordinary = regular_rag.context_from_bank(query, bank, PROVENANCE, 'percentile_group')
    context = graph3.context_from_bank(query, bank, PROVENANCE, assets(bank))
    assert len(ordinary['cases']) == 4 and len(context['cases']) == 6
    assert context['cases'][:4] == ordinary['cases']
    additional = context['cases'][4:]
    assert all(row['raw_values'][1] < 0 and row['calibrated_high_probabilities'][1] >= .5 for row in additional)
    assert {row['reference_class'] for row in additional} == {'Low', 'High'}
    assert all(row['selection_reason'] == 'actual_same_tool_boundary_state_with_reference_subject_deletion_coverage'
               for row in additional)
    assert len({row['case_id'] for row in context['cases']}) == 6
    diagnostic = graph3.candidate_diagnostics(query, bank, PROVENANCE, assets(bank))
    assert len(diagnostic['supplemental_offsets']) == 2
    assert all(bank['cases'][index]['individual_tools'][0]['calibrated_high_probability'] == .9
               for index in diagnostic['supplemental_offsets'])


def test_graph_tampering_and_fold_contamination_are_rejected():
    bank, query = calibrated_bank(), conflict_query()
    for field in ('outer_subject', 'bank_sha256'):
        wrong = assets(bank)
        wrong['fold_graph'][field] = 99 if field == 'outer_subject' else 'f' * 64
        with pytest.raises(ValueError, match='same-fold'):
            graph3.context_from_bank(query, bank, PROVENANCE, wrong)
    wrong = assets(bank)
    wrong['fold_graph']['nodes'][-1]['conditional_evidence']['raw_correct'] = 999
    with pytest.raises(ValueError, match='same-fold'):
        graph3.context_from_bank(query, bank, PROVENANCE, wrong)
    contaminated = copy.deepcopy(bank)
    contaminated['split']['policy_validation'].append(1)
    with pytest.raises(ValueError, match='isolation'):
        graph3.build_fold_graph(contaminated, PROVENANCE)


def test_corrupt_asset_is_explicitly_unavailable_even_without_conflict(monkeypatch, tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(graph3, 'GRAPH_MANIFEST', manifest)
    assert graph3.retrieve(1, prediction())['available'] is False


def test_nonfinite_calibration_is_rejected_before_fallback():
    bank, query = calibrated_bank(), conflict_query()
    query['numeric_fusion']['actual_tools'][0]['calibrated_high_probability'] = float('nan')
    with pytest.raises(ValueError, match='calibration'):
        graph3.context_from_bank(query, bank, PROVENANCE, assets(bank))


def test_unavailable_reference_calibration_does_not_hide_other_supported_tool():
    bank, query = calibrated_bank(), conflict_query()
    query['numeric_fusion']['actual_tools'][4]['calibration_usable'] = True
    query['numeric_fusion']['actual_tools'][4]['calibrated_high_probability'] = .9
    context = graph3.context_from_bank(query, bank, PROVENANCE, assets(bank))
    assert context['available'] is True
    assert [row['tool'] for row in context['graph_evidence']['packages']] == ['femba_features']
    assert graph3.candidate_diagnostics(query, bank, PROVENANCE, assets(bank))['raw_calibration_conflict_tools'] == [
        'femba_features', 'femba_temporal']


def test_partial_reference_calibration_rejects_the_frozen_bank_protocol():
    bank = calibrated_bank()
    bank['cases'][0]['individual_tools'][0].update(calibration_usable=False, calibrated_high_probability=None)
    with pytest.raises(ValueError, match='availability differs'):
        graph3.build_fold_graph(bank, PROVENANCE)


def test_all_24_assets_are_recomputed_with_current_source_and_label_isolation():
    manifest = graph3._read_verified(graph3.REFERENCE_MANIFEST, graph3.REFERENCE_MANIFEST_SHA256)
    assert len(manifest['files']) == 24
    for filename, expected in manifest['files'].items():
        entry = graph3._read_verified(graph3.REFERENCE_MANIFEST.parent / filename, expected)
        built = graph3._load_graph_assets(entry['bank']['outer_subject'])
        assert built['fold_graph'] == graph3.build_fold_graph(entry['bank'], entry['provenance'])
        assert built['method_graph'] == graph3.build_method_graph(built['method_source_text'])
        assert all(edge['source_sha256'] == entry['provenance']['current_raw_policy_bank_sha256']
                   for edge in built['fold_graph']['edges'])
        assert all(node.get('reference_subject') != entry['bank']['outer_subject']
                   for node in built['fold_graph']['nodes'] if node['kind'] == 'reference_case')
