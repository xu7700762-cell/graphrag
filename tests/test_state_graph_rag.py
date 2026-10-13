"""Truth-blind semantic retrieval, provenance and the outbound six-field request."""
import copy
import json

import numpy as np
import pytest

from eeg_agent import state_graph_rag as graph, default_rag, raw_independent_judge as judge
from eeg_agent.semantic_observations import observations_from_windows
from tests.test_default_rag import recorded_prediction
from tests.test_raw_independent_judge import Provider, final


def measured():
    time = np.arange(1280) / 256.
    windows = [np.tile(np.sin(2*np.pi*6*time) + amplitude*np.sin(2*np.pi*10*time) +
                       .7*np.sin(2*np.pi*20*time), (30, 1)).astype(np.float32)
               for amplitude in (2., 2., 1., 1., .5, .5)]
    return observations_from_windows(np.stack(windows), True)


def synthetic_graph(counter=True):
    source = {'id': 'paper-a', 'title': 'Synthetic provenance fixture', 'year': 2020,
              'doi': '10.0000/fixture', 'url': 'https://example.invalid/fixture',
              'locator': 'fixture section', 'excerpt': 'Association fixture, not a real paper.',
              'evidence_level': 'medium', 'review_status': 'reviewed', 'source_layer': 'paper_evidence'}
    source['source_sha256'] = graph.digest(source)
    nodes = [{'id': identity, 'kind': kind, 'label': identity, 'marker': marker,
              'source_ref': source['id'], 'evidence_level': 'medium', 'review_status': 'reviewed'}
             for identity, kind, marker in [('biomarker-alpha', 'biomarker', 'alpha'),
                                            ('mechanism-sensory-conflict', 'mechanism', None),
                                            ('state-vrms', 'state', None), ('paper-a', 'paper_evidence', None),
                                            ('counter-alpha', 'counter_evidence', None)]]
    edges = [{'from': origin, 'to': target, 'relation': relation,
              'source_ref': 'paper-a', 'evidence_level': 'medium', 'review_status': 'reviewed',
              'claim': 'Possible research association.', 'scope': 'research_not_individual_diagnosis',
              'observed_direction': 'any'}
             for origin, target, relation in [('biomarker-alpha', 'mechanism-sensory-conflict', 'associated_with'),
                                             ('mechanism-sensory-conflict', 'state-vrms', 'explains'),
                                             ('state-vrms', 'paper-a', 'documented_by')]]
    if counter:
        edges.append({'from': 'counter-alpha', 'to': 'biomarker-alpha', 'relation': 'contradicts',
                      'source_ref': 'paper-a', 'evidence_level': 'medium', 'review_status': 'reviewed',
                      'claim': 'A spectral observation alone does not establish a state.',
                      'scope': 'individual_inference_limitation'})
    sources = {'schema': 'eeg_state_sources_v1', 'sources': [source]}
    claims = {'schema': 'eeg_state_claims_v1', 'nodes': nodes, 'edges': edges}
    return sources, claims


def test_reviewed_paths_are_citable_layered_and_counterevidence_lowers_strength():
    sources, claims = synthetic_graph()
    built = graph.build_graph(sources, claims)
    before = copy.deepcopy(built)
    explanation = graph.explain({'eeg_observations': measured(), 'true_class': 'High'}, built)
    assert built == before
    assert explanation['evidence_status'] == 'available'
    assert explanation['evidence_strength'] == 'low'
    assert explanation['supporting_paths'] and explanation['contradictory_paths']
    path = explanation['supporting_paths'][0]
    assert 'mechanism-sensory-conflict' in path['nodes'] and 'state-vrms' in path['nodes']
    assert 'paper-a' in path['nodes'] and 'documented_by' in path['relations']
    assert path['sources'][0]['doi'] == '10.0000/fixture'
    assert path['observation_edge']['source_layer'] == 'model_evidence'
    assert path['independent_sources'] == ['10.0000/fixture']
    assert not judge._hidden(explanation)
    assert graph.explain({'eeg_observations': measured()}, built) == explanation
    assert all(all(row.get(key) for key in graph.META) for row in built['nodes'] + built['edges'])


def test_candidate_claims_and_unreviewed_sources_never_enter_graph():
    sources, claims = synthetic_graph()
    claims['nodes'][0]['review_status'] = 'candidate'
    built = graph.build_graph(sources, claims)
    assert 'biomarker-alpha' not in {row['id'] for row in built['nodes']}
    assert not any(row['from'] == 'biomarker-alpha' for row in built['edges'])
    assert graph.explain({'eeg_observations': measured()}, built)['evidence_status'] == 'unavailable'
    sources['sources'][0]['review_status'] = 'candidate'
    sources['sources'][0]['source_sha256'] = graph.digest({k: v for k, v in sources['sources'][0].items()
                                                         if k != 'source_sha256'})
    assert graph.build_graph(sources, claims)['nodes'] == []


def test_relations_sources_and_measurements_reject_tampering():
    sources, claims = synthetic_graph()
    built = graph.build_graph(sources, claims)
    broken = copy.deepcopy(built)
    broken['edges'][0]['claim'] = 'Tampered conclusion'
    with pytest.raises(ValueError, match='integrity'):
        graph.validate_graph(broken)
    sources['sources'][0]['excerpt'] = 'Tampered evidence'
    with pytest.raises(ValueError, match='hash'):
        graph.build_graph(sources, claims)
    invalid = measured(); invalid['items'][0]['value'] = 999
    with pytest.raises(ValueError):
        graph.explain({'eeg_observations': invalid}, built)


def test_missing_observations_never_inferred_from_classifier_scores():
    sources, claims = synthetic_graph()
    built = graph.build_graph(sources, claims)
    evidence = graph.explain(recorded_prediction(), built)
    assert evidence['evidence_status'] == 'unavailable'
    assert evidence['evidence_strength'] == 'low'
    assert not evidence['supporting_paths'] and not evidence['observations']
    assert evidence['fallback_reason'] == 'actual_eeg_observations_unavailable'


def test_rest_specific_claim_does_not_transfer_to_exposure_path_change():
    sources, claims = synthetic_graph(False)
    claims['edges'][0]['observation_baseline'] = 'rest'
    result = graph.explain({'eeg_observations': measured()}, graph.build_graph(sources, claims))
    assert result['evidence_status'] == 'unavailable'


def test_deployed_assets_rebuild_and_retrieve_without_modifying_numeric_or_cases():
    assets = graph.load_assets()
    assert assets['nodes'] and assets['edges']
    assert all(row['review_status'] == 'reviewed' for row in assets['nodes'] + assets['edges'])
    row = recorded_prediction(); row['eeg_observations'] = measured()
    before = copy.deepcopy(row)
    context = graph.retrieve(1, row)
    assert context['available'] and context['variant'] == graph.VARIANT
    base = graph.boundary.retrieve(1, row)
    assert context['cases'] == base['cases'] and row == before
    assert context['query_reading'] == base['query_reading']
    assert not any(key in context for key in ('class_descriptions', 'direction_error_counts',
                                              'group_sign_pattern_reference_counts'))
    assert not judge._hidden(context)


def test_complete_support_and_counter_packages_fit_budget_without_losing_observations():
    observed = measured()
    result = graph.explain({'eeg_observations': observed}, graph.load_assets())
    assert result['evidence_status'] == 'available'
    assert result['observations'] == observed['items']
    assert result['supporting_paths'] and result['contradictory_paths']
    assert any('method-raw-path' in path['nodes'] for path in result['supporting_paths'])
    assert any(claim['relation'] == 'solves' and claim['from'] == 'method-raw-path'
               for path in result['supporting_paths'] for claim in path['claims'])
    assert result['evidence_strength'] == 'low'
    assert len(json.dumps(result, ensure_ascii=False, allow_nan=False)) <= graph.MAX_CHARACTERS
    for path in result['supporting_paths'] + result['contradictory_paths']:
        assert path['content_sha256'] == graph._content(path)
        assert set(path['source_refs']) == {source['id'] for source in path['sources']}
        assert len(path['nodes']) == len(set(path['nodes']))
        assert all(all(row.get(key) for key in graph.META) for row in path['nodes_metadata'])


def test_tight_budget_keeps_actual_observations_with_explicit_unavailable_reason(monkeypatch):
    observed = measured()
    monkeypatch.setattr(graph, 'MAX_CHARACTERS', 6500)
    result = graph.explain({'eeg_observations': observed}, graph.load_assets())
    assert result['observations'] == observed['items'] and result['model_evidence']
    assert len(json.dumps(result, ensure_ascii=False)) <= graph.MAX_CHARACTERS
    assert result['evidence_status'] == 'unavailable'
    assert result['evidence_strength'] == 'low'
    assert result['fallback_reason'] == 'complete_semantic_package_exceeds_budget'
    assert result['truncation']['supporting_paths_omitted'] > 0


def test_actual_outbound_request_and_six_fields_are_truth_blind():
    row = recorded_prediction(); row['eeg_observations'] = measured()
    row['true_class'] = 'LEAK_TRUTH'; row['questionnaire'] = 'LEAK_SCORE'
    row['subject_id'] = 'LEAK_IDENTITY'; row['prior_report'] = 'LEAK_ANSWER'
    attached = default_rag.attach(1, row, default_rag.resolve('semantic_graph'))
    state = attached['current_raw_contrastive_context']['state_evidence']
    citation = state['supporting_paths'][0]['citation_id'] if state['supporting_paths'] else 'state-evidence-review'
    provider = Provider(final([citation]))
    result = judge.classify(provider, attached)
    assert result['available'] and result['judge_profile'] == 'semantic_state_high_v1'
    payload = json.loads(provider.calls[0]['messages'][1]['content'])
    assert not judge._hidden(payload)
    assert not any(secret in json.dumps(payload) for secret in ('LEAK_TRUTH', 'LEAK_SCORE', 'LEAK_IDENTITY', 'LEAK_ANSWER'))
    outbound = payload['initial_evidence']['policy_heldout_case_context']['state_evidence']
    assert outbound == state and set(result['raw_llm_final']) == set(judge.FIELDS)
    assert 'contradictory_paths' in provider.calls[0]['messages'][0]['content']


def test_broken_semantic_source_sends_no_classification(tmp_path, monkeypatch):
    bad = tmp_path / 'manifest.json'; bad.write_text('{}')
    monkeypatch.setattr(graph, 'MANIFEST', bad)
    attached = default_rag.attach(1, recorded_prediction(), default_rag.resolve('semantic_graph'))
    provider = Provider(final())
    result = judge.classify(provider, attached)
    assert not provider.calls and not result['available']
    assert result['validation_code'] == 'rag_assets_unavailable'


def test_session_projection_has_no_identity_truth_or_feedback_to_graph():
    sources, claims = synthetic_graph()
    state = graph.explain({'eeg_observations': measured()}, graph.build_graph(sources, claims))
    row = {'current_raw_contrastive_context': {'state_evidence': state},
           'true_class': 'High', 'subject_id': 'private', 'questionnaire': 99}
    before = copy.deepcopy(row)
    evidence = graph.session_evidence(row, {'available': True, 'final_class': 'Low', 'confidence': 'low'})
    assert row == before and not judge._hidden(evidence)
    assert evidence['prediction']['source'] == 'independent_llm_not_ground_truth'
    assert evidence['reuse_policy'] == 'session_audit_only_never_reference_bank'
    assert graph.session_evidence(row, {'available': False})['prediction'] is None
