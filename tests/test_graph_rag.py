"""Truth-blind graph selection, real provenance checks and numerical invariants."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from eeg_agent import decisive_rag, graph_rag, raw_independent_judge


def tools(scores, calibrations=None):
    calibrations = calibrations or [None] * 8
    return [{'tool': name, 'raw_decision_score': float(value),
             'predicted_class': 'High' if value >= 0 else 'Low',
             'calibration_usable': probability is not None,
             'calibrated_high_probability': probability}
            for name, value, probability in zip(decisive_rag.FAMILIES, scores, calibrations)]


def prediction(scores=None, calibrations=None, mil=.4):
    return {'accepted_windows': 3, 'mil_raw_high_probability': mil,
            'numeric_fusion': {'reference_available': False, 'raw_combined_score': .6,
                              'predicted_class': 'High', 'actual_tools': tools(
                                  [1, 1, 1, 1, -1, 1, 1, 1] if scores is None else scores, calibrations),
                              'tool_validation_profiles': {}}}


def bank_with_scores(scores=None, labels=None):
    query = np.array([1, 1, 1, 1, -1, 1, 1, 1], float)
    if scores is None:
        scores = [query.copy() for _ in range(8)]
        scores[2][0] = -1
        scores[5][4] = 1
        scores[6][0] = -1
        scores[7][3] = -1
    labels = labels or ['Low', 'High'] * 4
    cases, vectors = [], []
    for index, (values, label) in enumerate(zip(scores, labels)):
        p = .2 + index * .08
        actual = tools(values)
        vector = decisive_rag._vector(p, actual)
        vectors.append(vector)
        cases.append({'reference_index': index, 'reference_subject': 2 + index // 2,
                      'reference_class': label, 'mil_raw_high_probability': p,
                      'individual_tools': actual, 'retrieval_vector': vector.tolist()})
    vectors = np.asarray(vectors)
    subjects = sorted({row['reference_subject'] for row in cases})
    truth = np.asarray(labels) == 'High'
    rows = []
    for column, name in enumerate(decisive_rag.FIELDS):
        high = vectors[:, column] >= 0
        rows.append({'tool': name, 'boundary': 'raw_probability_0.5' if column == 0 else 'raw_score_zero',
                     'paths': len(cases), 'subjects': len(subjects),
                     'true_Low_output_Low': int((~truth & ~high).sum()),
                     'true_Low_output_High': int((~truth & high).sum()),
                     'true_High_output_Low': int((truth & ~high).sum()),
                     'true_High_output_High': int((truth & high).sum())})
    return {'schema': 'raw_policy_current_raw_v3', 'outer_subject': 1,
            'split': {'outer_subject': 1, 'base_train': [10], 'meta_calibration': [11],
                      'policy_validation': subjects}, 'cases': cases,
            'vector_median': np.median(vectors, axis=0).tolist(),
            'vector_scale': np.maximum(np.percentile(vectors, 75, axis=0) -
                                       np.percentile(vectors, 25, axis=0), .1).tolist(),
            'no_rest_policy_confusion': {'paths': len(cases), 'subjects': len(subjects),
                                        'true_Low_paths': int((~truth).sum()),
                                        'true_High_paths': int(truth.sum()), 'rows': rows},
            'source_hashes': {'current_raw_worker_result': 'c' * 64}}


PROVENANCE = {'current_raw_policy_bank_sha256': 'a' * 64,
              'current_raw_policy_manifest_sha256': 'b' * 64}


def assets(bank):
    text = (graph_rag.GRAPH_MANIFEST.parent / 'method_source.md').read_text(encoding='utf-8')
    return {'fold_graph': graph_rag.build_fold_graph(bank, PROVENANCE),
            'method_graph': graph_rag.build_method_graph(text), 'method_source_text': text,
            'graph_manifest_sha256': 'd' * 64}


def test_joint_pattern_pair_precedes_distance_and_covers_subjects():
    bank = bank_with_scores()
    graph = graph_rag.build_fold_graph(bank, PROVENANCE)
    query = [.4, 1, 1, 1, 1, -1, 1, 1, 1]
    chosen, matched, fallback = graph_rag._select(bank['cases'], graph, query,
                                                np.arange(1, 9) / 100, ['FEMBA'])
    assert chosen == [0, 1, 4, 3, 6]
    assert matched[4] == ['FEMBA'] and matched[3] == ['FEMBA']
    assert fallback is None
    assert len({bank['cases'][i]['reference_subject'] for i in chosen}) == 4


def test_mil_changes_never_change_graph_selection_or_conflict_routing():
    bank = bank_with_scores()
    first, second = prediction(mil=.01), prediction(mil=.99)
    one = graph_rag.context_from_bank(first, bank, PROVENANCE, assets(bank))
    two = graph_rag.context_from_bank(second, bank, PROVENANCE, assets(bank))
    assert one['cases'] == two['cases']
    assert one['graph_evidence'] == two['graph_evidence']


def test_graph_really_uses_method_adjacency_and_fold_pattern_edges():
    bank = bank_with_scores()
    built = assets(bank)
    qraw, _, qcal, _ = decisive_rag._values(.4, tools([1, 1, 1, 1, -1, 1, 1, 1]))
    groups, _, sources, _ = graph_rag._triggers(qraw, qcal, built['method_graph'])
    assert groups == ['FEMBA', 'training_case_retrieval'] and sources == ['FEMBA']
    removed = copy.deepcopy(built['method_graph'])
    removed['edges'] = [edge for edge in removed['edges'] if edge['relation'] != 'shares_feature_source']
    assert graph_rag._triggers(qraw, qcal, removed)[0] == []
    built['method_graph'] = removed
    with pytest.raises(ValueError, match='method graph'):
        graph_rag.context_from_bank(prediction(), bank, PROVENANCE, built)


def test_no_conflict_and_no_matching_pattern_keep_original_cases():
    bank = bank_with_scores()
    quiet = prediction(scores=[1] * 8)
    base = decisive_rag.context_from_bank(quiet, bank, PROVENANCE, 'percentile_group')
    graph = graph_rag.context_from_bank(quiet, bank, PROVENANCE, assets(bank))
    assert graph['cases'] == base['cases']
    assert graph['graph_evidence']['fallback_reason'] == 'no_conflict_trigger'
    scores = [[-1, 1, 1, 1, 1, 1, 1, 1] for _ in range(8)]
    unmatched_bank = bank_with_scores(scores)
    conflict = prediction(scores=[1] * 8, calibrations=[.1] + [None] * 7)
    base = decisive_rag.context_from_bank(conflict, unmatched_bank, PROVENANCE, 'percentile_group')
    graph = graph_rag.context_from_bank(conflict, unmatched_bank, PROVENANCE, assets(unmatched_bank))
    assert graph['cases'] == base['cases']
    assert graph['graph_evidence']['fallback_reason'] == 'no_matching_trigger_pattern'


def test_graph_budget_keeps_pattern_route_for_both_classes_and_exact_statistics():
    bank = bank_with_scores()
    query = prediction(calibrations=[.1, .1, .1, .1, .9, .1, .1, .1])
    query['numeric_fusion']['tool_validation_profiles'] = {
        name: {'policy': {'paths': 8, 'bacc': .5}} for name in decisive_rag.FAMILIES}
    before = copy.deepcopy(query)
    graph = graph_rag.context_from_bank(query, bank, PROVENANCE, assets(bank))
    base = decisive_rag.context_from_bank(query, bank, PROVENANCE, 'percentile_group')
    assert query == before
    assert len(json.dumps(graph['graph_evidence'], ensure_ascii=False)) <= 4000
    matches = [path for path in graph['graph_evidence']['paths'] if path['citation_id'].startswith('graph-match-')]
    assert {path['reference_class'] for path in matches} == {'Low', 'High'}
    for key in ('query_reading', 'class_descriptions', 'direction_error_counts', 'group_sign_pattern_reference_counts'):
        assert graph[key] == base[key]
    assert all(len(path['relations']) <= 2 for path in graph['graph_evidence']['paths'])
    assert graph['graph_evidence']['truncated_paths'] > 0


def test_validation_paths_only_link_actual_available_scopes():
    bank = bank_with_scores()
    query = prediction(calibrations=[.1] + [None] * 7)
    profile = {'policy': {'paths': 8, 'acc': .625, 'bacc': .5},
               'policy_correction': {'corrected': 1, 'damaged': 2, 'net': -1}}
    query['numeric_fusion']['tool_validation_profiles']['femba_features'] = profile
    graph = graph_rag.context_from_bank(query, bank, PROVENANCE, assets(bank))
    linked = next(path for path in graph['graph_evidence']['paths']
                  if path['citation_id'] == 'graph-validation-femba_features')
    assert linked['supplied_validation_scopes'] == ['policy', 'policy_correction']
    assert linked['source_sha256'] == graph_rag._digest(profile)
    assert 'raw_policy' not in linked['supplied_validation_scopes']


def test_same_fold_graph_and_subject_role_tampering_are_rejected():
    bank = bank_with_scores()
    wrong = assets(bank)
    wrong['fold_graph']['outer_subject'] = 2
    with pytest.raises(ValueError, match='same-fold'):
        graph_rag.context_from_bank(prediction(), bank, PROVENANCE, wrong)
    contaminated = copy.deepcopy(bank)
    contaminated['split']['policy_validation'].append(1)
    with pytest.raises(ValueError, match='isolation'):
        graph_rag.build_fold_graph(contaminated, PROVENANCE)


def test_single_available_class_case_is_not_duplicated():
    bank = bank_with_scores(labels=['Low'] + ['High'] * 7)
    query = prediction()
    context = graph_rag.context_from_bank(query, bank, PROVENANCE, assets(bank))
    assert len(context['cases']) <= 6
    assert len({case['case_id'] for case in context['cases']}) == len(context['cases'])
    assert sum(case['reference_class'] == 'Low' for case in context['cases']) == 1


def test_pinned_deployed_assets_all_folds_have_exact_case_graphs_and_sources():
    manifest = graph_rag._read_verified(graph_rag.REFERENCE_MANIFEST,
                                        graph_rag.REFERENCE_MANIFEST_SHA256)
    assert len(manifest['files']) == 24
    for name, sha in manifest['files'].items():
        entry = graph_rag._read_verified(graph_rag.REFERENCE_MANIFEST.parent / name, sha)
        fold = entry['bank']['outer_subject']
        built = graph_rag._load_graph_assets(fold)
        assert built['fold_graph'] == graph_rag.build_fold_graph(entry['bank'], entry['provenance'])
        method = built['method_graph']
        femba = {edge['from'] for edge in method['edges']
                 if edge['to'] == 'source-FEMBA' and edge['relation'] == 'shares_feature_source'}
        assert {'tool-MIL', 'tool-case_retrieval'} <= femba
        assert all(len(edge['source_sha256']) == 64 for edge in method['edges'] + built['fold_graph']['edges'])
        assert all(node.get('source_excerpt') for node in method['nodes'] if node['kind'] == 'method_definition')


def test_bad_manifest_is_unavailable_and_outbound_has_no_query_label(monkeypatch, tmp_path):
    bank = bank_with_scores()
    built = assets(bank)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(graph_rag, 'GRAPH_MANIFEST', manifest)
    assert graph_rag.retrieve(1, prediction())['available'] is False
    query = prediction()
    context = graph_rag.context_from_bank(query, bank, PROVENANCE, built)
    payload = raw_independent_judge.build_payload(query, teaching_context={'available': False}, rag_context=context)
    assert not raw_independent_judge._hidden(payload)
    assert set(path['citation_id'] for path in context['graph_evidence']['paths']) <= raw_independent_judge._citation_ids(payload)
