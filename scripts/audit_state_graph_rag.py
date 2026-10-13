"""Audit 146 frozen numeric inputs without an LLM call or a new accuracy claim.

Run with ``python -m scripts.audit_state_graph_rag``.  Optional --evaluation
reads a NEW semantic evaluation (never historical GraphRAG predictions). Each
record must contain its captured request_payload, strict raw_llm_final,
response_id, and separately held true_class. The gate recomputes accuracy from
these records; neither the engineering replay nor old responses qualify.

Request capture has two explicit formats: request_payload + request_sha256
(canonical JSON SHA256 using semantic.digest), or request_json +
model_payload_sha256 (the exact user-message string and the judge's serialized
message SHA256). These are different hashes and must never be interchanged.
If both formats are supplied, the parsed objects and both hashes must agree.
--source can point to an identical source directory on another machine; the
Windows default is only the known local historical input, still hash-pinned.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from eeg_agent import raw_independent_judge as judge
from eeg_agent import state_graph_rag as semantic
from eeg_agent.semantic_observations import SCHEMA as OBSERVATION_SCHEMA, SCOPE as OBSERVATION_SCOPE, validate_observations

SOURCE = Path(r'D:\桌面\文献总\EEGAGENT\outputs\vrms_rag_probes\current_raw_context_all146_offline_source_20261011')
OUT = semantic.PROJECT / 'outputs/semantic_graphrag_v1_delivery_20261013/engineering_audit.json'
NUMERIC_SHA256 = 'bd7b5ff45d0fcd3093c5b6294dc8b867f832d482a3f1ad3380b62b8150e68c67'
SELECTION_SHA256 = '1b7ffad3f9cbe0287d2378611a0069d61b9609fd9a4df87181dc63b35a35ea6c'
QUERY_IDS = tuple(f'q{i:03d}' for i in range(1, 147))
MIN_CORRECT = 112
MIN_BACC = .7321428571428571
BANK_SUMMARIES = {'class_descriptions', 'direction_error_counts',
                  'group_sign_pattern_reference_counts', 'whole_bank_calibration',
                  'reference_Low_paths', 'reference_High_paths', 'raw_correct',
                  'calibrated_correct', 'by_subject', 'leave_subject_out'}
PRIVATE_KEYS = {'questionnaire', 'questionnaire_answers', 'true_label',
                'ground_truth', 'test_truth', 'patient_id', 'identity'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def found_keys(value, prohibited):
    result = set()
    if isinstance(value, dict):
        result.update(set(value) & prohibited)
        for child in value.values():
            result.update(found_keys(child, prohibited))
    elif isinstance(value, list):
        for child in value:
            result.update(found_keys(child, prohibited))
    return result


def context_checks(context, expected_manifest=semantic.MANIFEST_SHA256):
    state = context.get('state_evidence', {})
    required = {'supporting_paths', 'contradictory_paths', 'evidence_strength',
                'confidence_basis', 'evidence_status', 'fallback_reason'}
    paths = state.get('supporting_paths', []) + state.get('contradictory_paths', [])
    citation_integrity = all(
        path.get('citation_id') and path.get('content_sha256') == semantic._content(path)
        and set(path.get('source_refs', [])) == {source.get('id') for source in path.get('sources', [])}
        and all(source.get('title') and source.get('locator') and source.get('source_sha256')
                and (source.get('source_layer') == 'local_protocol' or
                     source.get('doi') and source.get('year'))
                for source in path.get('sources', []))
        for path in paths)
    status = state.get('evidence_status')
    evidence_valid = (
        required <= set(state) and status in ('available', 'unavailable')
        and state.get('evidence_strength') in semantic.LEVELS
        and (bool(state.get('fallback_reason')) if status == 'unavailable'
             else bool(state.get('supporting_paths')) and state.get('fallback_reason') is None))
    return {
        'semantic_version': context.get('schema') == semantic.SCHEMA and context.get('variant') == semantic.VARIANT,
        'retrieval_available': context.get('available') is True,
        'manifest_pinned': context.get('source_hashes', {}).get('semantic_manifest_sha256') == expected_manifest,
        'label_isolation': not judge._hidden(context) and not found_keys(context, PRIVATE_KEYS),
        'bank_summary_isolation': not found_keys(context, BANK_SUMMARIES),
        'semantic_evidence_contract': evidence_valid,
        'citation_integrity': citation_integrity,
    }


def captured_payload(row):
    """Separate canonical object hashing from the actual serialized request."""
    payload = row.get('request_payload')
    serialized = row.get('request_json')
    if serialized is not None:
        if not isinstance(serialized, str):
            raise ValueError('Invalid serialized model request')
        actual_hash = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        if row.get('model_payload_sha256') != actual_hash:
            raise ValueError('Actual serialized model payload hash differs')
        decoded = json.loads(serialized)
        if payload is not None and decoded != payload:
            raise ValueError('Parsed captured request differs from request_payload')
        payload = decoded
    elif row.get('model_payload_sha256') is not None:
        raise ValueError('Actual serialized model hash requires request_json')
    if not isinstance(payload, dict):
        raise ValueError('Captured request missing')
    canonical = row.get('request_sha256')
    if canonical is not None:
        if canonical != semantic.digest(payload):
            raise ValueError('Canonical request hash differs')
    elif serialized is None:
        raise ValueError('Canonical or actual serialized capture hash required')
    return payload


def release_gate(evaluation, engineering_audit, expected_manifest=semantic.MANIFEST_SHA256):
    """Pure, fail-closed promotion decision from a complete new captured run.

    This does not call a model, trust supplied ACC/BACC, or change the default.
    Offline true_class fields are only compared after auditing model requests.
    """
    if evaluation is None:
        return {'promotion_status': 'not_evaluated', 'release_ready': False,
                'reasons': ['146_new_semantic_judgments_not_provided'], 'metrics': None}
    reasons = []
    if not isinstance(evaluation, dict):
        return {'promotion_status': 'rejected', 'release_ready': False,
                'reasons': ['invalid_evaluation_document'], 'metrics': None}
    if (evaluation.get('schema') != 'semantic_graph_evaluation_v1'
            or evaluation.get('semantic_schema') != semantic.SCHEMA
            or evaluation.get('variant') != semantic.VARIANT
            or evaluation.get('semantic_manifest_sha256') != expected_manifest
            or evaluation.get('numeric_source_sha256') != NUMERIC_SHA256
            or evaluation.get('judge_profile') != 'semantic_state_high_v1'
            or evaluation.get('new_semantic_judgments') is not True):
        reasons.append('new_semantic_run_provenance_missing_or_wrong_version')
    reviewed_graph = None
    try:
        reviewed_graph = semantic.load_assets(expected_manifest)
        policy_pins = semantic.asset_policy()
        captured_policy = evaluation.get('rag_policy', {})
        if not isinstance(captured_policy, dict) or any(captured_policy.get(key) != value for key, value in policy_pins.items()):
            reasons.append('current_semantic_observation_judge_and_base_code_pins_required')
    except (OSError, ValueError, TypeError, KeyError):
        reasons.append('current_reviewed_assets_required')
    if (engineering_audit.get('schema') != 'semantic_graph_engineering_audit_v1'
            or engineering_audit.get('semantic_manifest_sha256') != expected_manifest
            or engineering_audit.get('numeric_source_sha256') != NUMERIC_SHA256
            or engineering_audit.get('engineering_passed') is not True
            or engineering_audit.get('paths_checked') != 146
            or engineering_audit.get('paths_passed') != 146):
        reasons.append('complete_current_engineering_audit_required')
    records = evaluation.get('records')
    if not isinstance(records, list) or len(records) != 146:
        reasons.append('exactly_146_new_records_required')
        records = records if isinstance(records, list) else []
    if {row.get('query_id') for row in records if isinstance(row, dict)} != set(QUERY_IDS):
        reasons.append('canonical_146_query_ids_required')
    response_ids = [row.get('response_id') for row in records if isinstance(row, dict)]
    if len(response_ids) != 146 or not all(isinstance(value, str) and value for value in response_ids) or len(set(response_ids)) != 146:
        reasons.append('146_distinct_captured_responses_required')
    if type(evaluation.get('model_calls')) is not int or evaluation['model_calls'] < 146:
        reasons.append('new_run_model_calls_missing')
    audited = []
    for row in records:
        if not isinstance(row, dict):
            reasons.append('invalid_evaluation_record')
            continue
        final = row.get('raw_llm_final')
        try:
            payload = captured_payload(row)
            if not isinstance(payload, dict) or judge._hidden(payload) or found_keys(payload, PRIVATE_KEYS):
                raise ValueError('Request label isolation missing')
            context = payload['initial_evidence']['policy_heldout_case_context']
            if not all(context_checks(context, expected_manifest).values()):
                raise ValueError('Semantic context/citation audit failed')
            state = context['state_evidence']
            items = state.get('observations', [])
            if state.get('evidence_status') != 'available' or not items:
                raise ValueError('Real semantic evaluation requires actual EEG observations and an applicable reviewed path')
            observation_document = {
                'schema': OBSERVATION_SCHEMA, 'scope': OBSERVATION_SCOPE,
                'accepted_windows': payload['initial_evidence']['signal_quality']['accepted_windows'],
                'resting_reference_available': False, 'whole_path_complete': True,
                'evidence_status': 'available', 'fallback_reason': None,
                'source_sha256': items[0]['source_sha256'], 'sample_sha256': items[0]['sample_sha256'],
                'items': items,
            }
            validate_observations(observation_document)
            if reviewed_graph is None or semantic.explain({'eeg_observations': observation_document}, reviewed_graph) != state:
                raise ValueError('Captured semantic paths differ from the current reviewed graph and actual observations')
            actual = payload['corrective_evidence']['actual_individual_evidence']
            if len(actual) != len(judge.TOOLS) or {entry.get('tool') for entry in actual} != set(judge.TOOLS):
                raise ValueError('Eight-tool request evidence missing')
            judge._validate_final(final, payload)
            if not final['rag_citations']:
                raise ValueError('Captured response citation check missing')
            semantic_ids = {path['citation_id'] for path in state['supporting_paths'] + state['contradictory_paths']}
            if not set(final['rag_citations']) & semantic_ids:
                raise ValueError('Response did not cite its semantic evidence')
            if row.get('true_class') not in ('High', 'Low'):
                raise ValueError('Offline evaluation truth missing')
            audited.append((row['true_class'], final['final_class']))
        except (KeyError, TypeError, ValueError, AttributeError):
            reasons.append('captured_request_response_label_or_citation_audit_failed')
    metrics = None
    if len(audited) == 146:
        correct = sum(truth == predicted for truth, predicted in audited)
        recalls = {}
        for category in ('High', 'Low'):
            own = [(truth, predicted) for truth, predicted in audited if truth == category]
            if not own:
                reasons.append('both_truth_classes_required_for_bacc')
                break
            recalls[category] = sum(truth == predicted for truth, predicted in own) / len(own)
        bacc = sum(recalls.values()) / 2 if len(recalls) == 2 else None
        metrics = {'n': 146, 'correct': correct, 'acc': correct / 146, 'bacc': bacc}
        if correct < MIN_CORRECT:
            reasons.append('correct_below_112_of_146')
        if bacc is None or bacc < MIN_BACC:
            reasons.append('bacc_below_frozen_regular_baseline')
    else:
        reasons.append('146_valid_new_judgments_required')
    reasons = sorted(set(reasons))
    return {'promotion_status': 'eligible' if not reasons else 'rejected',
            'release_ready': not reasons, 'reasons': reasons, 'metrics': metrics,
            'gate': {'n': 146, 'minimum_correct': MIN_CORRECT, 'minimum_bacc': MIN_BACC,
                     'requires_new_semantic_run': True, 'requires_captured_request_and_citation_audit': True}}


def audit(source=SOURCE):
    source = Path(source)
    numeric_path, selection_path = source / 'frozen_predictions.json', source / 'selection.json'
    if sha(numeric_path) != NUMERIC_SHA256 or sha(selection_path) != SELECTION_SHA256:
        raise ValueError('Canonical 146 frozen numeric source or fold mapping hash differs')
    predictions = json.loads(numeric_path.read_text(encoding='utf-8'))
    # Selection has historical labels/answers: only extract the fixed query ID
    # and fold mapping. No selection row, label, old judgment, or identity enters
    # the anonymous numeric input or any emitted model payload.
    fold_by_query = {row['query_id']: int(row['subject']) for row in
                     json.loads(selection_path.read_text(encoding='utf-8'))}
    if set(predictions) != set(QUERY_IDS) or set(fold_by_query) != set(QUERY_IDS):
        raise ValueError('Canonical source must have exactly q001 through q146')
    graph = semantic.load_assets()
    reference_manifest = semantic.boundary._read_verified(
        semantic.boundary.REFERENCE_MANIFEST, semantic.boundary.REFERENCE_MANIFEST_SHA256)
    rows = []
    for query_id in QUERY_IDS:
        raw = predictions[query_id]
        # Frozen numeric evidence contains learned classes; they are model
        # outputs, not biomarker observations. Do not copy old cloud_judgment.
        prediction = {key: copy.deepcopy(raw[key]) for key in
                      ('mil_raw_high_probability', 'accepted_windows', 'numeric_fusion')}
        fold = fold_by_query[query_id]
        reference_file = f'fold_{fold:02d}.json'
        reference = semantic.boundary._read_verified(
            semantic.boundary.REFERENCE_MANIFEST.parent / reference_file,
            reference_manifest['files'][reference_file])
        before = semantic.digest(prediction)
        context = semantic.retrieve(fold, prediction)
        base = semantic.boundary.retrieve(fold, prediction)
        checks = context_checks(context)
        checks.update(
            same_fold=reference['bank']['outer_subject'] == fold,
            query_excluded_from_reference=all(case['reference_subject'] != fold for case in reference['bank']['cases']),
            unchanged_v3_cases=context.get('cases') == base.get('cases'),
            unchanged_numeric_reading=context.get('query_reading') == base.get('query_reading'),
            input_immutable=before == semantic.digest(prediction),
        )
        state = context.get('state_evidence', {})
        checks['no_fabricated_observations'] = (
            state.get('evidence_status') == 'unavailable'
            and state.get('fallback_reason') == 'actual_eeg_observations_unavailable'
            and state.get('observations') == []
            and not state.get('supporting_paths') and not state.get('contradictory_paths'))
        payload_sha = None
        try:
            payload = judge.build_payload(prediction, rag_context=context)
            checks['outbound_payload_label_isolation'] = not judge._hidden(payload) and not found_keys(payload, PRIVATE_KEYS)
            checks['outbound_context_unchanged'] = payload['initial_evidence']['policy_heldout_case_context'] == context
            payload_sha = semantic.digest(payload)
        except (KeyError, TypeError, ValueError):
            checks['outbound_payload_label_isolation'] = False
            checks['outbound_context_unchanged'] = False
        rows.append({'query_id': query_id, 'passed': all(checks.values()), 'checks': checks,
                     'context_sha256': semantic.digest(context), 'outbound_payload_sha256': payload_sha,
                     'evidence_status': state.get('evidence_status'), 'fallback_reason': state.get('fallback_reason')})
    passed = sum(row['passed'] for row in rows)
    result = {'schema': 'semantic_graph_engineering_audit_v1',
              'created_at': datetime.now(timezone.utc).isoformat(),
              'semantic_schema': semantic.SCHEMA, 'variant': semantic.VARIANT,
              'semantic_manifest_sha256': semantic.MANIFEST_SHA256,
              'numeric_source_sha256': NUMERIC_SHA256,
              'selection_fold_mapping_sha256': SELECTION_SHA256,
              'reviewed_graph_sha256': semantic.digest(graph),
              'rag_policy': semantic.asset_policy(),
              'paths_checked': len(rows), 'paths_passed': passed,
              'engineering_passed': passed == 146,
              'model_calls': 0, 'new_accuracy_computed': False,
              'semantic_observations_available': 0,
              'scope': 'same_fold_engineering_only_on_historical_numeric_inputs',
              'limitations': ['historical numeric inputs contain no actual EEG observations',
                              'semantic evidence correctly stays unavailable; no new LLM judgment was produced',
                              'engineering success does not meet the 146 new semantic classification release gate'],
              'rows': rows}
    result.update(release_gate(None, result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--evaluation', type=Path, help='Audit a separately captured NEW semantic 146-row evaluation; never calls an LLM')
    args = parser.parse_args()
    report = audit(args.source)
    if args.evaluation:
        evaluation = json.loads(args.evaluation.read_text(encoding='utf-8'))
        report['evaluation_gate'] = release_gate(evaluation, report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('paths_checked', 'paths_passed', 'engineering_passed',
                                                 'model_calls', 'promotion_status', 'release_ready')}, ensure_ascii=False))
    print(str(args.out.resolve()))
    return 0 if report['engineering_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
