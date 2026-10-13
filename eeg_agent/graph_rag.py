"""Deterministic, same-fold EEG evidence GraphRAG; never a query classifier.

The eight raw-tool percentile distances are unchanged. A reviewed method graph
routes conflicts to four retrieval-group sign-pattern nodes, which select
label-balanced reference cases. MIL is displayed but never used for selection.
"""
from __future__ import annotations

import copy
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from .config import PROJECT
from . import decisive_rag

SCHEMA = 'raw_eeg_graph_rag_v1'
VARIANT = 'graph_v1'
GRAPH_SCHEMA = 'eeg_evidence_graph_v1'
ALGORITHM_VERSION = 'same_fold_pattern_two_hop_v1'
MAX_GRAPH_CHARACTERS = 4000
REFERENCE_MANIFEST = PROJECT / 'outputs/rag_assets/current_raw_policy_v3/manifest.json'
REFERENCE_MANIFEST_SHA256 = '8e7c75aab8a1dc7957c555faa287ca8b6dff57bf09843f07c899246207b45ea1'
GRAPH_MANIFEST = PROJECT / 'outputs/rag_assets/eeg_evidence_graph_v1/manifest.json'
GRAPH_MANIFEST_SHA256 = '993c1717693fc42060338d3f5f033c78108567c62abf10865ebd392ba68a8516'

# Semantic dependencies overlap. Retrieval GROUPS intentionally remain frozen.
SEMANTIC_SOURCES = {
    'FEMBA': ('MIL', 'femba_features', 'femba_mean', 'femba_temporal', 'case_retrieval'),
    'spectral': ('spectral', 'spectral_compact', 'filterbank_csp'),
    'spatial': ('spatial_covariance', 'filterbank_csp'),
}
TOOL_ORDER = decisive_rag.FIELDS


def asset_policy():
    return {'graph_manifest_sha256': GRAPH_MANIFEST_SHA256,
            'graph_schema': SCHEMA, 'graph_variant': VARIANT}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _read_verified(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('GraphRAG asset hash differs')
    return json.loads(data.decode('utf-8'))


def _pattern_id(group, values):
    return 'pattern-' + group + '-' + ''.join('1' if value else '0' for value in values)


def _reviewed_sections(source_text):
    sections, heading, lines = {}, None, []
    for line in source_text.splitlines():
        if line.startswith('## '):
            if heading is not None:
                sections[heading] = '\n'.join(lines).strip()
            heading, lines = line[3:], []
        elif heading is not None:
            lines.append(line)
    if heading is not None:
        sections[heading] = '\n'.join(lines).strip()
    return sections


def _excerpt(text):
    # A complete source sentence, never an invented model interpretation.
    first, punctuation, _ = text.partition('。')
    return first + punctuation


def build_method_graph(source_text):
    """Build only reviewed dependencies, each bound to a frozen source document."""
    sha = hashlib.sha256(source_text.encode('utf-8')).hexdigest()
    if '## 证据相关性：' not in source_text or any(name not in source_text for name in TOOL_ORDER):
        raise ValueError('Reviewed nine-tool source is incomplete')
    sections = _reviewed_sections(source_text)
    dependency_section = '证据相关性：九路共享信息与重复计数'
    nodes = [{'id': 'tool-' + name, 'kind': 'tool', 'tool': name} for name in TOOL_ORDER]
    edges = []
    for source, tools in SEMANTIC_SOURCES.items():
        nodes.append({'id': 'source-' + source, 'kind': 'feature_source', 'source': source,
                      'source_section': dependency_section,
                      'source_excerpt': _excerpt(sections[dependency_section]), 'source_sha256': sha})
        for tool in tools:
            section = ('filterbank_csp：滤波器组空间模式'
                       if source == 'spectral' and tool == 'filterbank_csp' else dependency_section)
            edges.append({'from': 'tool-' + tool, 'to': 'source-' + source,
                          'relation': 'shares_feature_source', 'source_sha256': sha,
                          'source_section': section})
    for tool in TOOL_ORDER:
        section = next(heading for heading in sections if heading.startswith(tool + '：'))
        nodes.append({'id': 'definition-' + tool, 'kind': 'method_definition', 'tool': tool,
                      'source_section': section, 'source_excerpt': _excerpt(sections[section]),
                      'source_sha256': sha})
        edges.append({'from': 'tool-' + tool, 'to': 'definition-' + tool,
                      'relation': 'defined_by', 'source_sha256': sha,
                      'source_section': section})
    nodes.append({'id': 'protocol-current-no-rest', 'kind': 'protocol',
                  'scope': 'whole_path_current_raw_no_rest'})
    for tool in TOOL_ORDER:
        edges.append({'from': 'tool-' + tool, 'to': 'protocol-current-no-rest',
                      'relation': 'applies_protocol', 'source_sha256': sha,
                      'source_section': '案例来源与限制：方法RAG和留出参考案例RAG'})
    return {'schema': GRAPH_SCHEMA, 'kind': 'reviewed_method_graph',
            'algorithm_version': ALGORITHM_VERSION, 'source_sha256': sha,
            'nodes': nodes, 'edges': edges}


def build_fold_graph(bank, provenance):
    """No cross-fold cases, fitted weights, query labels or class aggregates."""
    fold = bank['outer_subject']
    decisive_rag.validate_bank(bank, fold)
    sha = provenance['current_raw_policy_bank_sha256']
    subjects = sorted({case['reference_subject'] for case in bank['cases']})
    nodes = [{'id': f'fold-{fold:02d}', 'kind': 'fold', 'outer_subject': fold}]
    nodes += [{'id': f'subject-{subject:02d}', 'kind': 'reference_subject',
               'reference_subject': subject} for subject in subjects]
    edges, patterns = [], {}
    for offset, case in enumerate(bank['cases']):
        raw = decisive_rag._values(case['mil_raw_high_probability'], case['individual_tools'])[0][1:]
        bits = np.asarray(raw) >= 0
        identity = f'reference-{offset:03d}'
        nodes.append({'id': identity, 'kind': 'reference_case', 'bank_offset': offset,
                      'reference_index': case['reference_index'],
                      'reference_subject': case['reference_subject'],
                      'reference_class': case['reference_class']})
        for target, relation in ((f"subject-{case['reference_subject']:02d}", 'belongs_to_subject'),
                                 (f'fold-{fold:02d}', 'belongs_to_fold')):
            edges.append({'from': identity, 'to': target, 'relation': relation, 'source_sha256': sha})
        for group, indices in decisive_rag.GROUPS.items():
            values = bits[list(indices)].tolist()
            pattern = _pattern_id(group, values)
            patterns[pattern] = {'id': pattern, 'kind': 'raw_sign_pattern',
                                 'retrieval_group': group, 'signs': values}
            edges.append({'from': pattern, 'to': identity, 'relation': 'has_raw_sign_pattern',
                          'source_sha256': sha})
    nodes.extend(patterns[key] for key in sorted(patterns))
    return {'schema': GRAPH_SCHEMA, 'kind': 'same_fold_reference_graph',
            'algorithm_version': ALGORITHM_VERSION, 'outer_subject': fold,
            'bank_sha256': sha, 'nodes': nodes, 'edges': edges}


def _load_graph_assets(fold, expected_manifest_sha256=None):
    expected = expected_manifest_sha256 or GRAPH_MANIFEST_SHA256
    manifest = _read_verified(GRAPH_MANIFEST, expected)
    if (manifest.get('schema') != GRAPH_SCHEMA or manifest.get('algorithm_version') != ALGORITHM_VERSION
            or manifest.get('reference_manifest_sha256') != REFERENCE_MANIFEST_SHA256):
        raise ValueError('GraphRAG manifest version or reference source differs')
    filename = f'fold_{fold:02d}.json'
    graph = _read_verified(GRAPH_MANIFEST.parent / filename, manifest['files'][filename])
    method = _read_verified(GRAPH_MANIFEST.parent / 'method_graph.json', manifest['files']['method_graph.json'])
    source_data = (GRAPH_MANIFEST.parent / 'method_source.md').read_bytes()
    if (hashlib.sha256(source_data).hexdigest() != manifest['files']['method_source.md']
            or method != build_method_graph(source_data.decode('utf-8'))):
        raise ValueError('GraphRAG reviewed method graph differs from source')
    return {'fold_graph': graph, 'method_graph': method,
            'method_source_text': source_data.decode('utf-8'),
            'graph_manifest_sha256': expected}


def _triggers(qraw, qcal, method):
    signs = dict(zip(TOOL_ORDER, [qraw[0] >= .5] + [score >= 0 for score in qraw[1:]]))
    conflict_tools = {tool for i, tool in enumerate(TOOL_ORDER[1:], 1)
                      if qcal[i] is not None and signs[tool] != (qcal[i] >= .5)}
    calibration_conflicts = sorted(conflict_tools, key=TOOL_ORDER.index)
    shared_sources = []
    # MIL is intentionally excluded from conflict routing used for selection.
    members_by_source = {}
    for edge in method['edges']:
        if edge['relation'] == 'shares_feature_source':
            members_by_source.setdefault(edge['to'], []).append(edge['from'][len('tool-'):])
    for source_node, members in members_by_source.items():
        eligible = [tool for tool in members if tool != 'MIL']
        if len({signs[tool] for tool in eligible}) > 1:
            shared_sources.append(source_node[len('source-'):])
            conflict_tools.update(eligible)
    groups = [group for group, indices in decisive_rag.GROUPS.items()
              if any(decisive_rag.FAMILIES[i] in conflict_tools for i in indices)]
    return groups, sorted(conflict_tools, key=TOOL_ORDER.index), shared_sources, calibration_conflicts


def _select(refs, graph, qraw, distance, triggers):
    order = sorted(range(len(refs)), key=lambda i: (float(distance[i]), refs[i]['reference_index'], i))
    anchors = [next(i for i in order if refs[i]['reference_class'] == label) for label in ('Low', 'High')]
    matched = {i: [] for i in range(len(refs))}
    for group in triggers:
        indices = decisive_rag.GROUPS[group]
        pattern = _pattern_id(group, [qraw[i + 1] >= 0 for i in indices])
        for edge in graph['edges']:
            if edge['relation'] == 'has_raw_sign_pattern' and edge['from'] == pattern:
                matched[int(edge['to'].split('-')[1])].append(group)
    if not any(matched.values()):
        return None, matched, 'no_matching_trigger_pattern'
    options = [[i for i in order if refs[i]['reference_class'] == label and i not in anchors]
               for label in ('Low', 'High')]
    options = [items or [None] for items in options]
    pair = min(product(*options), key=lambda pair: (
        -sum(len(matched[i]) for i in pair if i is not None),
        -len({refs[i]['reference_subject'] for i in anchors + [j for j in pair if j is not None]}),
        sum(float(distance[i]) for i in pair if i is not None),
        tuple(refs[i]['reference_index'] if i is not None else -1 for i in pair)))
    chosen = anchors + [i for i in pair if i is not None]
    subjects = {case['reference_subject'] for case in refs}
    for index in order:
        present = {refs[i]['reference_subject'] for i in chosen}
        if len(chosen) >= 6 or present == subjects:
            break
        if refs[index]['reference_subject'] not in present:
            chosen.append(index)
    return chosen, matched, None


def _graph_context(method, graph, triggers, tools, sources, calibration_conflicts,
                   cases, chosen, matched, prediction, fallback):
    sha = method['source_sha256']
    mandatory, optional, extra_matches = [], [], []
    by_id = {node['id']: node for node in method['nodes']}
    actual_by_tool = {tool['tool']: tool for tool in prediction['numeric_fusion']['actual_tools']}
    eight_raw = [actual_by_tool[tool]['raw_decision_score'] for tool in decisive_rag.FAMILIES]
    query_reading_sha = _digest(eight_raw)
    if chosen is not None:
        for row, index in zip(cases, chosen):
            for group in matched[index]:
                pattern = _pattern_id(group, [eight_raw[i] >= 0 for i in decisive_rag.GROUPS[group]])
                path = {'citation_id': f"graph-match-{row['case_id']}-{group}",
                        'nodes': ['anonymous-query', pattern, row['case_id']],
                        'relations': ['has_raw_sign_pattern', 'has_raw_sign_pattern'],
                        'source_sha256': graph['bank_sha256'],
                        'edge_source_sha256': [query_reading_sha, graph['bank_sha256']],
                        'scope': 'same_fold_reference_pattern_not_query_label'}
                target = mandatory if not any(p['reference_class'] == row['reference_class']
                                              for p in mandatory) else extra_matches
                path['reference_class'] = row['reference_class']
                target.append(path)
    # Existing raw/calibrated numeric readings remain mandatory outside this graph.
    for tool in calibration_conflicts:
        node = by_id['definition-' + tool]
        optional.append({'citation_id': 'graph-boundary-' + tool,
                        'nodes': [tool, node['id']], 'relations': ['defined_by'],
                        'source_sha256': sha, 'source_section': node['source_section'],
                        'source_excerpt': node['source_excerpt'],
                        'scope': 'actual_raw_calibrated_direction_conflict'})
    for source in sources:
        members = [edge['from'][len('tool-'):] for edge in method['edges']
                   if edge['relation'] == 'shares_feature_source' and edge['to'] == 'source-' + source
                   and edge['from'] != 'tool-MIL']
        signs = {tool: actual_by_tool[tool]['raw_decision_score'] >= 0 for tool in members}
        first = next(tool for tool in members if signs[tool])
        second = next(tool for tool in members if not signs[tool])
        node = by_id['source-' + source]
        section_edges = [edge for edge in method['edges']
                         if edge['relation'] == 'shares_feature_source' and edge['to'] == node['id']
                         and edge['from'] in ('tool-' + first, 'tool-' + second)]
        sections = list(dict.fromkeys(edge['source_section'] for edge in section_edges))
        excerpt = node['source_excerpt']
        if 'filterbank_csp：滤波器组空间模式' in sections:
            excerpt += by_id['definition-filterbank_csp']['source_excerpt']
        optional.append({'citation_id': f'graph-dependency-{source}',
                         'nodes': [first, node['id'], second],
                         'relations': ['shares_feature_source', 'inverse_shares_feature_source'],
                         'source_sha256': sha, 'source_sections': sections,
                         'source_excerpt': excerpt,
                         'scope': 'shared_input_not_independent_votes'})
    profiles = prediction['numeric_fusion'].get('tool_validation_profiles', {})
    for tool in tools:
        profile = profiles.get(tool, {})
        visible = {key: copy.deepcopy(profile[key]) for key in
                   ('policy', 'raw_policy', 'calibrated_policy', 'policy_correction') if key in profile}
        if visible:
            optional.append({'citation_id': 'graph-validation-' + tool,
                             'nodes': [tool, 'validation-' + tool], 'relations': ['has_validation'],
                             'source_sha256': _digest(visible), 'supplied_validation_scopes': list(visible),
                             'scope': 'existing_actual_profile_only_no_boundary_metrics_inferred'})
    # Keep selected-case graph routes first, then remove optional whole paths.
    optional.extend(extra_matches)
    paths = mandatory + optional
    result = {'schema': GRAPH_SCHEMA, 'algorithm_version': ALGORITHM_VERSION, 'max_hops': 2,
              'trigger_groups': triggers, 'fallback_reason': fallback, 'paths': paths,
              'raw_calibration_conflict_tools': calibration_conflicts,
              'method_source_sha256': sha, 'reference_graph_bank_sha256': graph['bank_sha256'],
              'selection_note': 'Pattern matches select demonstrations, never a query class or confidence.',
              'validation_note': 'Only actual supplied profile scopes exist; absent raw_policy or calibrated_policy is unknown.',
              'truncated_paths': 0}
    while len(paths) > len(mandatory) and len(json.dumps(result, ensure_ascii=False, allow_nan=False)) > MAX_GRAPH_CHARACTERS:
        paths.pop()
        result['truncated_paths'] += 1
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False)) > MAX_GRAPH_CHARACTERS:
        raise ValueError('Mandatory graph metadata exceeds budget')
    return result


def context_from_bank(prediction, bank, provenance, graph_assets_or_fold=None):
    """Attach a validated graph context without changing original model readings."""
    fold = bank['outer_subject']
    decisive_rag.validate_bank(bank, fold)
    assets = (graph_assets_or_fold if isinstance(graph_assets_or_fold, dict) else
              _load_graph_assets(fold if graph_assets_or_fold is None else graph_assets_or_fold))
    graph, method = assets['fold_graph'], assets['method_graph']
    if graph != build_fold_graph(bank, provenance):
        raise ValueError('Reference graph differs from this same-fold bank')
    if method != build_method_graph(assets['method_source_text']):
        raise ValueError('Wrong reviewed method graph')
    result = decisive_rag.context_from_bank(prediction, bank, provenance, 'percentile_group')
    qraw, _, qcal, _ = decisive_rag._values(prediction['mil_raw_high_probability'],
                                         prediction['numeric_fusion']['actual_tools'])
    triggers, tools, sources, calibration_conflicts = _triggers(qraw, qcal, method)
    raw = np.asarray([decisive_rag._values(case['mil_raw_high_probability'], case['individual_tools'])[0][1:]
                      for case in bank['cases']])
    percentiles = decisive_rag._percentiles(raw, np.vstack([raw, qraw[1:]]))
    deltas = percentiles[:-1] - percentiles[-1]
    group_distance = np.asarray([np.sqrt(np.mean(deltas[:, list(indices)] ** 2, axis=1))
                                 for indices in decisive_rag.GROUPS.values()]).T
    distance = np.sqrt(np.mean(group_distance ** 2, axis=1))
    if triggers:
        chosen, matched, fallback = _select(bank['cases'], graph, qraw, distance, triggers)
    else:
        chosen, matched, fallback = None, {}, 'no_conflict_trigger'
    if chosen is not None:
        subjects = sorted({case['reference_subject'] for case in bank['cases']})
        cases = []
        for rank, index in enumerate(chosen, 1):
            case = bank['cases'][index]
            values, _, calibrated, _ = decisive_rag._values(case['mil_raw_high_probability'], case['individual_tools'])
            reason = 'nearest_reference_class_anchor' if rank <= 2 else ('trigger_pattern_joint_selection' if rank <= 4 else 'missing_reference_subject_coverage')
            cases.append({'case_id': f'graph-reference-case-{rank:02d}',
                          'subject_group': f"internal-subject-{subjects.index(case['reference_subject']) + 1:02d}",
                          'reference_class': case['reference_class'], 'raw_values': values,
                          'calibrated_high_probabilities': calibrated,
                          'group_percentile_distance': float(distance[index]),
                          'source_group_distances': dict(zip(decisive_rag.GROUPS, group_distance[index].tolist())),
                          'selection_reason': reason, 'matched_trigger_groups': matched[index]})
        result['cases'] = cases
        result['retrieval'].update(selection_rule='nearest_each_class_then_joint_pattern_match_subjects_distance_index_then_missing_subject_coverage_max6',
                                   returned_paths=len(cases), returned_subjects=len({case['subject_group'] for case in cases}))
    result['schema'], result['variant'] = SCHEMA, VARIANT
    result['graph_evidence'] = _graph_context(method, graph, triggers, tools, sources, calibration_conflicts,
                                             result['cases'], chosen, matched, prediction, fallback)
    result['source_hashes'].update(graph_manifest_sha256=assets.get('graph_manifest_sha256'),
                                  graph_method_source_sha256=method['source_sha256'])
    result['review_guidance'] = {'citation_id': 'graph-eeg-comparison', 'text':
        decisive_rag.GUIDANCE + ' Graph paths explain shared sources, field boundaries and same-fold pattern matches. '
        'A pattern match, graph degree or selected label frequency is not a query probability or vote. '
        'Only explicitly supplied validation scopes are available; do not infer raw/calibrated boundary metrics.'}
    return result


def retrieve(fold, prediction, graph_manifest_sha256=None):
    """Load explicitly pinned bank and graph assets; corruption never falls back."""
    try:
        if type(fold) is not int or fold < 1:
            raise ValueError('A verified model fold is required')
        manifest = _read_verified(REFERENCE_MANIFEST, REFERENCE_MANIFEST_SHA256)
        if manifest['schema'] != 'default_rag_reference_assets_v1':
            raise ValueError('Wrong source reference manifest')
        filename = f'fold_{fold:02d}.json'
        entry = _read_verified(REFERENCE_MANIFEST.parent / filename, manifest['files'][filename])
        if entry['bank']['outer_subject'] != fold:
            raise ValueError('Wrong reference fold')
        assets = _load_graph_assets(fold, graph_manifest_sha256)
        return context_from_bank(prediction, entry['bank'], entry['provenance'], assets)
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return {'schema': SCHEMA, 'variant': VARIANT, 'available': False,
                'reason': 'verified_graph_rag_reference_unavailable'}
