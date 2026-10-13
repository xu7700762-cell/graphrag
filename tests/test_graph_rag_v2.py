"""Offline candidate tests: transition identity, isolation and preserved evidence."""
import copy
import json

import numpy as np
import pytest

from eeg_agent import decisive_rag, graph_rag, graph_rag_v2, raw_independent_judge
from tests.test_graph_rag import PROVENANCE, bank_with_scores, prediction, tools


def calibrated_bank(all_tools=False):
    scores = [[-1.] * 8 for _ in range(8)]
    scores[4][0] = 1.  # Opposite High -> Low transition is not Low -> High.
    bank = bank_with_scores(scores)
    for index, case in enumerate(bank['cases']):
        calibration = [.9 if index < 4 else .1] + [None] * 7
        calibration[4] = .9 if index in (0, 1, 4, 5) else .1
        if all_tools:
            calibration = [.9] * 8
        case['individual_tools'] = tools(scores[index], calibration)
    truth = np.asarray([case['reference_class'] == 'High' for case in bank['cases']])
    for column, name in enumerate(decisive_rag.FAMILIES):
        values = [case['individual_tools'][column]['calibrated_high_probability']
                  for case in bank['cases']]
        if all(value is None for value in values):
            continue
        high = np.asarray(values) >= .5
        bank['no_rest_policy_confusion']['rows'].append({
            'tool': name, 'boundary': 'calibrated_probability_0.5', 'paths': 8, 'subjects': 4,
            'true_Low_output_Low': int((~truth & ~high).sum()),
            'true_Low_output_High': int((~truth & high).sum()),
            'true_High_output_Low': int((truth & ~high).sum()),
            'true_High_output_High': int((truth & high).sum())})
    return decisive_rag.validate_bank(bank, 1)


def assets(bank):
    source = (graph_rag.GRAPH_MANIFEST.parent / 'method_source.md').read_text(encoding='utf-8')
    return {'fold_graph': graph_rag_v2.build_fold_graph(bank, PROVENANCE),
            'method_graph': graph_rag_v2.build_method_graph(source),
            'method_source_text': source, 'graph_manifest_sha256': 'd' * 64}


def test_source_disagreement_keeps_exact_ordinary_cases_but_explains_dependency():
    bank, query = bank_with_scores(), prediction()
    ordinary = decisive_rag.context_from_bank(query, bank, PROVENANCE, 'percentile_group')
    context = graph_rag_v2.context_from_bank(query, bank, PROVENANCE, assets(bank))
    assert context['cases'] == ordinary['cases']
    assert context['retrieval'] == ordinary['retrieval']
    assert context['graph_evidence']['trigger_groups'] == []
    assert context['graph_evidence']['fallback_reason'] == 'no_raw_calibration_conflict'
    assert any(path['citation_id'] == 'graph-dependency-FEMBA'
               for path in context['graph_evidence']['paths'])


def test_same_tools_and_direction_are_required_once_per_group():
    bank = calibrated_bank()
    query = prediction(scores=[-1] * 8, calibrations=[.9, None, None, None, .9, None, None, None])
    raw, _, calibrated, _ = decisive_rag._values(query['mil_raw_high_probability'],
                                                 query['numeric_fusion']['actual_tools'])
    graph = graph_rag_v2.build_fold_graph(bank, PROVENANCE)
    chosen, matched, fallback = graph_rag_v2._select(
        bank['cases'], graph, raw, calibrated, np.arange(8) / 100, ['FEMBA'])
    assert matched[0] == matched[1] == ['FEMBA']
    assert all(matched[i] == [] for i in range(2, 8))
    assert chosen[:2] == [0, 1] and fallback is None
    context = graph_rag_v2.context_from_bank(query, bank, PROVENANCE, assets(bank))
    assert any(case['selection_reason'] == 'unmatched_reference_class_comparison'
               for case in context['cases'])
    route = next(path for path in context['graph_evidence']['paths']
                 if path['citation_id'].startswith('graph-match-'))
    assert route['nodes'][1] == 'boundary-FEMBA-0:01-4:01'
    assert route['relations'] == ['has_raw_calibration_boundary_pattern'] * 2
    assert route['nodes'][1] in {node['id'] for node in graph['nodes']}
    assert route['edge_source_sha256'][0] == graph_rag_v2._digest(
        {'raw': raw[1:], 'calibrated': calibrated[1:]})


def test_query_subset_can_match_reference_with_more_actual_conflicts():
    bank = calibrated_bank()
    query = prediction(scores=[-1] * 8, calibrations=[.9] + [None] * 7)
    raw, _, calibrated, _ = decisive_rag._values(query['mil_raw_high_probability'],
                                                 query['numeric_fusion']['actual_tools'])
    _, matched, _ = graph_rag_v2._select(bank['cases'], assets(bank)['fold_graph'], raw,
                                       calibrated, np.arange(8) / 100, ['FEMBA'])
    assert matched[0] == matched[2] == ['FEMBA']
    assert matched[4] == []  # Raw High -> calibrated Low is a different transition.


def test_missing_calibration_never_matches_and_nonfinite_is_rejected():
    bank = bank_with_scores([[-1.] * 8 for _ in range(8)])
    query = prediction(scores=[-1] * 8, calibrations=[.9] + [None] * 7)
    ordinary = decisive_rag.context_from_bank(query, bank, PROVENANCE, 'percentile_group')
    context = graph_rag_v2.context_from_bank(query, bank, PROVENANCE, assets(bank))
    assert context['cases'] == ordinary['cases']
    assert context['graph_evidence']['fallback_reason'] == 'no_matching_boundary_transition_pattern'
    query['numeric_fusion']['actual_tools'][0]['calibrated_high_probability'] = float('nan')
    with pytest.raises(ValueError, match='calibration'):
        graph_rag_v2.context_from_bank(query, bank, PROVENANCE, assets(bank))


def test_mil_and_hidden_answers_cannot_route_or_change_cases():
    bank = calibrated_bank()
    query = prediction(scores=[-1] * 8, calibrations=[.9] + [None] * 7, mil=.01)
    before = graph_rag_v2.context_from_bank(query, bank, PROVENANCE, assets(bank))
    other = copy.deepcopy(query)
    other.update(mil_raw_high_probability=.99, true_class='High', subject=99,
                 previous_answer='Low')
    other['numeric_fusion'].update(raw_combined_score=.1, predicted_class='Low')
    after = graph_rag_v2.context_from_bank(other, bank, PROVENANCE, assets(bank))
    assert before['cases'] == after['cases']
    assert before['graph_evidence'] == after['graph_evidence']
    payload = raw_independent_judge.build_payload(other, teaching_context={'available': False},
                                                rag_context=after)
    assert not raw_independent_judge._hidden(payload)


def test_budget_preserves_actual_two_hop_routes_and_numeric_evidence():
    bank = calibrated_bank(all_tools=True)
    query = prediction(scores=[-1] * 8, calibrations=[.9] * 8)
    query['numeric_fusion']['tool_validation_profiles'] = {
        name: {'raw_policy': {'paths': 8, 'bacc': .5},
               'calibrated_policy': {'paths': 8, 'bacc': .5}} for name in decisive_rag.FAMILIES}
    before = copy.deepcopy(query)
    ordinary = decisive_rag.context_from_bank(query, bank, PROVENANCE, 'percentile_group')
    context = graph_rag_v2.context_from_bank(query, bank, PROVENANCE, assets(bank))
    assert query == before
    for key in ('query_reading', 'class_descriptions', 'direction_error_counts',
                'group_sign_pattern_reference_counts'):
        assert context[key] == ordinary[key]
    evidence = context['graph_evidence']
    assert len(json.dumps(evidence, ensure_ascii=False)) <= 4000
    assert evidence['truncated_paths'] > 0
    routes = [path for path in evidence['paths'] if path['citation_id'].startswith('graph-match-')]
    assert {route['reference_class'] for route in routes} == {'Low', 'High'}
    assert all(len(path['relations']) <= 2 for path in evidence['paths'])
    payload = raw_independent_judge.build_payload(query, teaching_context={'available': False},
                                                rag_context=context)
    assert {path['citation_id'] for path in evidence['paths']} <= raw_independent_judge._citation_ids(payload)


def test_fold_and_transition_tampering_are_rejected():
    bank = calibrated_bank()
    query = prediction(scores=[-1] * 8, calibrations=[.9] + [None] * 7)
    wrong = assets(bank)
    wrong['fold_graph']['outer_subject'] = 2
    with pytest.raises(ValueError, match='same-fold'):
        graph_rag_v2.context_from_bank(query, bank, PROVENANCE, wrong)
    wrong = assets(bank)
    edge = next(edge for edge in wrong['fold_graph']['edges']
                if edge['relation'] == 'has_raw_calibration_boundary_pattern')
    edge['source_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='same-fold'):
        graph_rag_v2.context_from_bank(query, bank, PROVENANCE, wrong)


def test_corrupt_candidate_manifest_is_unavailable(monkeypatch, tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(graph_rag_v2, 'GRAPH_MANIFEST', manifest)
    assert graph_rag_v2.retrieve(1, prediction())['available'] is False


def test_pinned_v2_assets_rebuild_from_all_24_frozen_reference_banks():
    manifest = graph_rag_v2._read_verified(graph_rag_v2.REFERENCE_MANIFEST,
                                         graph_rag_v2.REFERENCE_MANIFEST_SHA256)
    assert len(manifest['files']) == 24
    for filename, expected in manifest['files'].items():
        entry = graph_rag_v2._read_verified(graph_rag_v2.REFERENCE_MANIFEST.parent / filename, expected)
        built = graph_rag_v2._load_graph_assets(entry['bank']['outer_subject'])
        assert built['fold_graph'] == graph_rag_v2.build_fold_graph(entry['bank'], entry['provenance'])
        assert built['method_graph'] == graph_rag_v2.build_method_graph(built['method_source_text'])
        assert all(edge['source_sha256'] == entry['provenance']['current_raw_policy_bank_sha256']
                   for edge in built['fold_graph']['edges'])
        for node in built['fold_graph']['nodes']:
            if node['kind'] == 'raw_calibration_boundary_pattern':
                assert node['tool_transitions']
                assert all(state['raw_direction'] != state['calibrated_direction']
                           for state in node['tool_transitions'])
