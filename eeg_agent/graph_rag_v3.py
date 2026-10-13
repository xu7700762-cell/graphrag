"""Offline boundary-applicability GraphRAG candidate, never a query classifier.

The fixed ordinary context is returned literally when no usable conflict exists.
Otherwise the graph adds counted same-fold boundary counterexamples, preserving
every ordinary demonstration in order. Subject deletion is a coverage check,
not a claim of reliability or a recommended query class.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from .config import PROJECT
from . import regular_rag as regular
from . import graph_rag as frozen_v1

SCHEMA = 'raw_eeg_graph_rag_v3'
VARIANT = 'graph_v3'
GRAPH_SCHEMA = 'eeg_boundary_applicability_graph_v3'
ALGORITHM_VERSION = 'same_fold_boundary_counterexamples_preserve_regular_v3'
MAX_GRAPH_CHARACTERS = 4000
MAX_PACKAGES = 3
MIN_MATCHED_PATHS = 4
MIN_MATCHED_SUBJECTS = 2
MIN_REMAINING_PATHS = 2
MIN_REMAINING_SUBJECTS = 2
REFERENCE_MANIFEST = frozen_v1.REFERENCE_MANIFEST
REFERENCE_MANIFEST_SHA256 = frozen_v1.REFERENCE_MANIFEST_SHA256
GRAPH_MANIFEST = PROJECT / 'outputs/rag_assets/eeg_evidence_graph_v3/manifest.json'
GRAPH_MANIFEST_SHA256 = '4e063d041e6f0b971fafc9462d42107a3f5bca005b23e6477e02990f9e03ff8f'

GUIDANCE = (
    'Boundary graph counts describe this same-fold no-rest reference population. '
    'A constant calibrated direction can include both true reference classes, '
    'so that direction supplies no reference-set class separation; the numeric '
    'probabilities can still differ. Compare actual raw/calibration conflicts '
    'and their displayed counterexamples with the unchanged ordinary cases. '
    'Deleting each reference subject checks remaining source coverage only, '
    'not predictive stability, confidence, or transfer to the query. '
    'Condition counts, class frequencies, graph degree and repeated paths are '
    'not a query prior, vote, probability or recommended class. Shared-source '
    'tools and paths from one subject remain dependent. Make your independent '
    'High/Low decision from all actual readings and opposing evidence.'
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _read_verified(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('GraphRAG v3 asset hash differs')
    return json.loads(data.decode('utf-8'))


def asset_policy():
    return {'graph_manifest_sha256': GRAPH_MANIFEST_SHA256,
            'graph_schema': SCHEMA, 'graph_variant': VARIANT}


def _state_id(tool, raw_high, calibrated_high):
    return f'boundary-{tool}-{int(raw_high)}{int(calibrated_high)}'


def _states(tools):
    raw, _, calibrated, _ = regular._values(.5, tools)
    return {name: (raw[index + 1] >= 0,
                   None if calibrated[index + 1] is None else calibrated[index + 1] >= .5)
            for index, name in enumerate(regular.FAMILIES)}


def _counts(cases, raw_high, calibrated_high):
    high = sum(case['reference_class'] == 'High' for case in cases)
    low = len(cases) - high
    return {'paths': len(cases),
            'subjects': len({case['reference_subject'] for case in cases}),
            'Low': low, 'High': high,
            'raw_correct': high if raw_high else low,
            'calibrated_correct': high if calibrated_high else low}


def _condition(cases, members, raw_high, calibrated_high):
    subjects = sorted({case['reference_subject'] for case in cases})
    selected = [cases[index] for index in members]
    result = _counts(selected, raw_high, calibrated_high)
    result['by_subject'] = []
    result['leave_subject_out'] = []
    for rank, identity in enumerate(subjects, 1):
        name = f'internal-subject-{rank:02d}'
        own = [case for case in selected if case['reference_subject'] == identity]
        if own:
            result['by_subject'].append({'subject_group': name,
                                         **_counts(own, raw_high, calibrated_high)})
        others = [case for case in selected if case['reference_subject'] != identity]
        result['leave_subject_out'].append({'removed_subject_group': name,
                                            **_counts(others, raw_high, calibrated_high)})
    result['coverage_supported'] = (
        result['paths'] >= MIN_MATCHED_PATHS and result['subjects'] >= MIN_MATCHED_SUBJECTS
        and all(row['paths'] >= MIN_REMAINING_PATHS and row['subjects'] >= MIN_REMAINING_SUBJECTS
                for row in result['leave_subject_out']))
    return result


def _calibration_profile(cases, tool):
    readings = [next(row for row in case['individual_tools'] if row['tool'] == tool)
                for case in cases]
    probabilities = [row['calibrated_high_probability'] for row in readings]
    if any(value is None for value in probabilities):
        return None
    high = sum(value >= .5 for value in probabilities)
    low = len(cases) - high
    return {'paths': len(cases), 'subjects': len({case['reference_subject'] for case in cases}),
            'High_outputs': high, 'Low_outputs': low,
            'constant_direction': 'High' if low == 0 else ('Low' if high == 0 else None),
            'probability_range': [min(probabilities), max(probabilities)],
            'reference_Low_paths': sum(case['reference_class'] == 'Low' for case in cases),
            'reference_High_paths': sum(case['reference_class'] == 'High' for case in cases)}


def build_method_graph(source_text):
    result = frozen_v1.build_method_graph(source_text)
    result['schema'], result['algorithm_version'] = GRAPH_SCHEMA, ALGORITHM_VERSION
    return result


def build_fold_graph(bank, provenance):
    regular.validate_bank(bank, bank['outer_subject'])
    cases, sha = bank['cases'], provenance['current_raw_policy_bank_sha256']
    states = [_states(case['individual_tools']) for case in cases]
    nodes = [{'id': f"fold-{bank['outer_subject']:02d}", 'kind': 'fold'}]
    edges = []
    for index, case in enumerate(cases):
        identity = f'reference-{index:03d}'
        nodes.append({'id': identity, 'kind': 'reference_case', 'bank_offset': index,
                      'reference_index': case['reference_index'],
                      'reference_subject': case['reference_subject'],
                      'reference_class': case['reference_class']})
        edges.append({'from': identity, 'to': f"fold-{bank['outer_subject']:02d}",
                      'relation': 'belongs_to_fold', 'source_sha256': sha})
    for tool in regular.FAMILIES:
        profile = _calibration_profile(cases, tool)
        if profile is not None:
            nodes.append({'id': 'calibration-' + tool, 'kind': 'calibrated_direction_profile',
                          'tool': tool, 'whole_bank_calibration': profile})
        for raw_high, calibrated_high in ((False, True), (True, False)):
            members = [index for index, state in enumerate(states)
                       if state[tool] == (raw_high, calibrated_high)]
            if not members:
                continue
            identity = _state_id(tool, raw_high, calibrated_high)
            nodes.append({'id': identity, 'kind': 'raw_calibration_conflict_condition', 'tool': tool,
                          'raw_direction': 'High' if raw_high else 'Low',
                          'calibrated_direction': 'High' if calibrated_high else 'Low',
                          'conditional_evidence': _condition(cases, members, raw_high, calibrated_high)})
            edges.append({'from': identity, 'to': 'calibration-' + tool,
                          'relation': 'bounded_by_observed_calibration_scope', 'source_sha256': sha})
            for index in members:
                edges.append({'from': identity, 'to': f'reference-{index:03d}',
                              'relation': 'has_actual_boundary_state', 'source_sha256': sha})
    return {'schema': GRAPH_SCHEMA, 'kind': 'same_fold_boundary_applicability_graph',
            'algorithm_version': ALGORITHM_VERSION, 'outer_subject': bank['outer_subject'],
            'bank_sha256': sha, 'nodes': nodes, 'edges': edges}


def _load_graph_assets(fold, expected_manifest_sha256=None):
    expected = expected_manifest_sha256 or GRAPH_MANIFEST_SHA256
    manifest = _read_verified(GRAPH_MANIFEST, expected)
    if (manifest.get('schema') != GRAPH_SCHEMA or manifest.get('algorithm_version') != ALGORITHM_VERSION
            or manifest.get('reference_manifest_sha256') != REFERENCE_MANIFEST_SHA256):
        raise ValueError('GraphRAG v3 manifest version or reference differs')
    filename = f'fold_{fold:02d}.json'
    graph = _read_verified(GRAPH_MANIFEST.parent / filename, manifest['files'][filename])
    method = _read_verified(GRAPH_MANIFEST.parent / 'method_graph.json', manifest['files']['method_graph.json'])
    source = (GRAPH_MANIFEST.parent / 'method_source.md').read_bytes()
    if (hashlib.sha256(source).hexdigest() != manifest['files']['method_source.md']
            or method != build_method_graph(source.decode('utf-8'))):
        raise ValueError('GraphRAG v3 reviewed method source differs')
    return {'fold_graph': graph, 'method_graph': method,
            'method_source_text': source.decode('utf-8'), 'graph_manifest_sha256': expected}


def _case_offsets(context, bank):
    subjects = sorted({case['reference_subject'] for case in bank['cases']})
    identities = {}
    for row in context['cases']:
        matches = [index for index, case in enumerate(bank['cases'])
                   if row['subject_group'] == f"internal-subject-{subjects.index(case['reference_subject']) + 1:02d}"
                   and row['reference_class'] == case['reference_class']
                   and row['raw_values'] == regular._values(case['mil_raw_high_probability'], case['individual_tools'])[0]
                   and row['calibrated_high_probabilities'] == regular._values(case['mil_raw_high_probability'], case['individual_tools'])[2]]
        if not matches:
            raise ValueError('Ordinary case does not belong to this reference bank')
        identities[row['case_id']] = matches[0]
    return identities


def _distance(prediction, bank):
    raw = np.asarray([regular._values(case['mil_raw_high_probability'], case['individual_tools'])[0][1:]
                      for case in bank['cases']])
    query = np.asarray(regular._values(prediction['mil_raw_high_probability'],
                                     prediction['numeric_fusion']['actual_tools'])[0][1:])
    percentiles = regular._percentiles(raw, np.vstack([raw, query]))
    deltas = percentiles[:-1] - percentiles[-1]
    grouped = np.asarray([np.sqrt(np.mean(deltas[:, list(indices)] ** 2, axis=1))
                          for indices in regular.GROUPS.values()]).T
    return np.sqrt(np.mean(grouped ** 2, axis=1)), grouped


def _build_context(prediction, bank, provenance, assets):
    regular.validate_bank(bank, bank['outer_subject'])
    if assets['fold_graph'] != build_fold_graph(bank, provenance):
        raise ValueError('GraphRAG v3 same-fold graph differs from reference evidence')
    if assets['method_graph'] != build_method_graph(assets['method_source_text']):
        raise ValueError('GraphRAG v3 method graph differs from reviewed source')
    ordinary = regular.context_from_bank(prediction, bank, provenance, 'percentile_group')
    state = _states(prediction['numeric_fusion']['actual_tools'])
    conflicts = [tool for tool in regular.FAMILIES if state[tool][1] is not None and state[tool][0] != state[tool][1]]
    diagnostics = {'actual_fold': bank['outer_subject'], 'source_bank_sha256': assets['fold_graph']['bank_sha256'],
                   'raw_calibration_conflict_tools': conflicts, 'eligible_tools': [], 'supplemental_offsets': [],
                   'graph_characters': 0, 'literal_regular': True, 'retained_ordinary_case_count': len(ordinary['cases']),
                   'fallback_reason': 'no_raw_calibration_conflict', 'packages': []}
    if not conflicts:
        return ordinary, diagnostics
    nodes = {node['id']: node for node in assets['fold_graph']['nodes']}
    eligible = [nodes[_state_id(tool, *state[tool])] for tool in conflicts
                if _state_id(tool, *state[tool]) in nodes
                and nodes[_state_id(tool, *state[tool])]['conditional_evidence']['coverage_supported']]
    eligible.sort(key=lambda node: (-node['conditional_evidence']['subjects'],
                                   -min(row['subjects'] for row in node['conditional_evidence']['leave_subject_out']),
                                   -node['conditional_evidence']['paths'], regular.FAMILIES.index(node['tool'])))
    diagnostics['eligible_tools'] = [node['tool'] for node in eligible]
    if not eligible:
        diagnostics['fallback_reason'] = 'no_subject_deletion_supported_boundary_condition'
        return ordinary, diagnostics
    result = copy.deepcopy(ordinary)
    identities = _case_offsets(result, bank)
    distance, grouped = _distance(prediction, bank)
    all_subjects = sorted({case['reference_subject'] for case in bank['cases']})
    evidence = {'schema': GRAPH_SCHEMA, 'algorithm_version': ALGORITHM_VERSION,
                'scope': 'same_fold_current_raw_no_rest_reference_only',
                'coverage_note': 'Subject deletion tests source coverage, not reliability or query confidence.',
                'packages': []}
    for node in eligible:
        if len(evidence['packages']) >= MAX_PACKAGES:
            break
        members = [int(edge['to'].split('-')[1]) for edge in assets['fold_graph']['edges']
                   if edge['from'] == node['id'] and edge['relation'] == 'has_actual_boundary_state']
        proposed_cases = copy.deepcopy(result['cases'])
        proposed_ids = dict(identities)
        added = []
        for label in ('Low', 'High'):
            if any(index in members and row['reference_class'] == label for row in proposed_cases
                   for index in [proposed_ids[row['case_id']]]):
                continue
            options = [index for index in members if bank['cases'][index]['reference_class'] == label
                       and index not in proposed_ids.values()]
            if not options or len(proposed_cases) >= 6:
                continue
            index = min(options, key=lambda i: (float(distance[i]), bank['cases'][i]['reference_index'], i))
            case = bank['cases'][index]
            values, _, calibrated, _ = regular._values(case['mil_raw_high_probability'], case['individual_tools'])
            identity = f'graph-v3-reference-case-{len(proposed_cases) + 1:02d}'
            proposed_cases.append({'case_id': identity,
                'subject_group': f"internal-subject-{all_subjects.index(case['reference_subject']) + 1:02d}",
                'reference_class': label, 'raw_values': values, 'calibrated_high_probabilities': calibrated,
                'group_percentile_distance': float(distance[index]),
                'source_group_distances': dict(zip(regular.GROUPS, grouped[index].tolist())),
                'selection_reason': 'actual_same_tool_boundary_state_with_reference_subject_deletion_coverage'})
            proposed_ids[identity] = index
            added.append(index)
        source_nodes = [edge['to'] for edge in assets['method_graph']['edges']
                        if edge['from'] == 'tool-' + node['tool'] and edge['relation'] == 'shares_feature_source']
        profile = nodes['calibration-' + node['tool']]['whole_bank_calibration']
        package = {'citation_id': 'graph-v3-boundary-' + node['tool'], 'tool': node['tool'],
                   'transition': {key: node[key] for key in ('raw_direction', 'calibrated_direction')},
                   'conditional_evidence': copy.deepcopy(node['conditional_evidence']),
                   'whole_bank_calibration': copy.deepcopy(profile),
                   'example_case_ids': {label: [row['case_id'] for row in proposed_cases
                        if proposed_ids[row['case_id']] in members and row['reference_class'] == label]
                                        for label in ('Low', 'High')},
                   'shared_sources': [source[len('source-'):] for source in source_nodes],
                   'relations': ['matches_observed_boundary_state', 'has_counted_support_and_counterexamples',
                                 'bounded_by_observed_calibration_scope', 'shares_feature_source'],
                   'source_sha256': assets['fold_graph']['bank_sha256'],
                   'method_source_sha256': assets['method_graph']['source_sha256']}
        proposed_evidence = copy.deepcopy(evidence)
        proposed_evidence['packages'].append(package)
        if len(json.dumps(proposed_evidence, ensure_ascii=False, allow_nan=False)) > MAX_GRAPH_CHARACTERS:
            continue  # Remove only a complete optional package and its uncommitted cases.
        result['cases'], identities, evidence = proposed_cases, proposed_ids, proposed_evidence
        diagnostics['supplemental_offsets'].extend(added)
    if not evidence['packages']:
        diagnostics['fallback_reason'] = 'no_complete_boundary_package_fits_budget'
        return ordinary, diagnostics
    result.update(schema=SCHEMA, variant=VARIANT, graph_evidence=evidence)
    result['source_hashes'].update(graph_manifest_sha256=assets.get('graph_manifest_sha256'),
                                  graph_method_source_sha256=assets['method_graph']['source_sha256'])
    result['retrieval'].update(returned_paths=len(result['cases']),
                              returned_subjects=len({row['subject_group'] for row in result['cases']}),
                              graph_supplement_rule='preserve_all_ordinary_cases_then_only_supported_actual_boundary_members_max6')
    result['review_guidance'] = {'citation_id': 'graph-v3-boundary-review', 'text': regular.GUIDANCE + ' ' + GUIDANCE}
    diagnostics.update(literal_regular=False, graph_characters=len(json.dumps(evidence, ensure_ascii=False)),
                       displayed_tools=[package['tool'] for package in evidence['packages']],
                       fallback_reason=None, packages=copy.deepcopy(evidence['packages']))
    return result, diagnostics


def context_from_bank(prediction, bank, provenance, graph_assets_or_fold=None):
    assets = (graph_assets_or_fold if isinstance(graph_assets_or_fold, dict) else
              _load_graph_assets(bank['outer_subject'] if graph_assets_or_fold is None else graph_assets_or_fold))
    return _build_context(prediction, bank, provenance, assets)[0]


def candidate_diagnostics(prediction, bank, provenance, graph_assets_or_fold=None):
    """Local truth-blind audit metadata; never inserted into a model context."""
    assets = (graph_assets_or_fold if isinstance(graph_assets_or_fold, dict) else
              _load_graph_assets(bank['outer_subject'] if graph_assets_or_fold is None else graph_assets_or_fold))
    return _build_context(prediction, bank, provenance, assets)[1]


def retrieve(fold, prediction, graph_manifest_sha256=None):
    try:
        if type(fold) is not int or fold < 1:
            raise ValueError('Verified fold required')
        manifest = _read_verified(REFERENCE_MANIFEST, REFERENCE_MANIFEST_SHA256)
        if manifest['schema'] != 'default_rag_reference_assets_v1':
            raise ValueError('Wrong reference manifest')
        filename = f'fold_{fold:02d}.json'
        entry = _read_verified(REFERENCE_MANIFEST.parent / filename, manifest['files'][filename])
        if entry['bank']['outer_subject'] != fold:
            raise ValueError('Wrong reference fold')
        return context_from_bank(prediction, entry['bank'], entry['provenance'],
                                 _load_graph_assets(fold, graph_manifest_sha256))
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return {'schema': SCHEMA, 'variant': VARIANT, 'available': False,
                'reason': 'verified_graph_rag_v3_reference_unavailable'}
