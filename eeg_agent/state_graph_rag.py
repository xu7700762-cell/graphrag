"""Reviewed VRMS semantic evidence over the frozen same-fold v3 retriever.

Associations explain observations; they never classify a query or diagnose VRMS.
Only reviewed, hash-pinned claims and actual anonymous measurements enter context.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .config import PROJECT
from . import graph_rag_v3 as boundary

SCHEMA = 'raw_eeg_state_graph_v1'
VARIANT = 'state_reasoning_v1'
VERSION = 'eeg_state_graph_v1'
MANIFEST = PROJECT / 'outputs/rag_assets/eeg_state_graph_v1/manifest.json'
GRAPH_MANIFEST = MANIFEST
MANIFEST_SHA256 = '69dbb9fdd6dc933c5a0e0d635a123f24eaa0114d34a3245bc62a0de284f1a55a'
KNOWLEDGE = PROJECT / 'knowledge/graph'
LEVELS = {'high': 3, 'medium': 2, 'low': 1}
KINDS = {'observation', 'biomarker', 'mechanism', 'state', 'method',
         'paper_evidence', 'model_evidence', 'counter_evidence'}
RELATIONS = {'observed_as', 'associated_with', 'explains', 'supports',
             'contradicts', 'solves', 'documented_by'}
LAYERS = {'domain_knowledge', 'paper_evidence', 'model_evidence',
          'counter_evidence', 'local_protocol'}
META = ('source_layer', 'evidence_level', 'source_ref', 'source_sha256',
        'content_sha256', 'version')
MAX_PATHS = 6
MAX_CHARACTERS = 24000
# These hash-reviewed MVP nodes describe scope, not a direct opposing finding.
# New counterclaims default to material until their role has been reviewed.
SCOPE_LIMIT_COUNTER_NODES = {
    'counter-alpha-direction', 'counter-theta-direction', 'counter-beta-regional',
    'counter-eeg-symptoms', 'counter-no-baseline', 'counter-autonomic-modality',
}
GUIDANCE = (
    'Semantic paths explain VRMS research associations, not diagnosis or High/Low rules. '
    'Path thirds are not a resting baseline; associations do not establish individual causes. '
    'Keep measurement, paper and counterclaim provenance separate; one paper or PSD is '
    'one source. Strength is qualitative, never probability or a graph vote. Unmeasured '
    'markers are not absent. State actual paths and one concrete scope limit. Generic '
    'direction/baseline/modality/EEG-to-questionnaire limits are not direct contradictions. '
    'Medium requires direct reviewed paper support; low requires missing evidence or a '
    'material counterclaim.'
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path, expected):
    if _sha(path) != expected:
        raise ValueError('Semantic asset hash differs')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _content(record):
    return digest({key: value for key, value in record.items() if key != 'content_sha256'})


def asset_policy():
    return {'semantic_manifest_sha256': MANIFEST_SHA256,
            'semantic_code_sha256': _sha(__file__),
            'observation_code_sha256': _sha(PROJECT / 'eeg_agent/semantic_observations.py'),
            'semantic_judge_code_sha256': _sha(PROJECT / 'eeg_agent/raw_independent_judge.py'),
            'semantic_schema': SCHEMA, 'semantic_variant': VARIANT,
            'graph_manifest_sha256': boundary.GRAPH_MANIFEST_SHA256,
            'graph_code_sha256': _sha(boundary.__file__),
            'graph_schema': boundary.SCHEMA, 'graph_variant': boundary.VARIANT}


def build_graph(sources_document, claims_document):
    """Build a deterministic reviewed graph; candidate claims remain inventory only."""
    if sources_document.get('schema') != 'eeg_state_sources_v1' or claims_document.get('schema') != 'eeg_state_claims_v1':
        raise ValueError('Wrong semantic source schema')
    sources = {}
    for source in sources_document['sources']:
        if source['id'] in sources or source['evidence_level'] not in LEVELS or source['source_layer'] not in LAYERS:
            raise ValueError('Invalid or duplicate source')
        if source['source_sha256'] != digest({k: v for k, v in source.items() if k != 'source_sha256'}):
            raise ValueError('Source content hash differs')
        if source['review_status'] == 'reviewed' and source['source_layer'] == 'paper_evidence':
            if not all(source.get(key) for key in ('doi', 'title', 'year', 'locator', 'excerpt')):
                raise ValueError('Reviewed paper lacks bibliographic evidence')
        sources[source['id']] = copy.deepcopy(source)

    def materialize(raw):
        source = sources[raw['source_ref']]
        if raw.get('review_status') != 'reviewed' or source.get('review_status') != 'reviewed':
            return None
        level = raw['evidence_level']
        if level not in LEVELS or LEVELS[level] > LEVELS[source['evidence_level']]:
            raise ValueError('Claim evidence exceeds reviewed source')
        row = copy.deepcopy(raw)
        layer = ('counter_evidence' if raw.get('kind') == 'counter_evidence' or raw.get('relation') == 'contradicts'
                 else 'paper_evidence' if raw.get('kind') == 'paper_evidence' or raw.get('relation') == 'documented_by'
                 else 'local_protocol' if source['source_layer'] == 'local_protocol' else 'domain_knowledge')
        row.update(source_layer=layer, source_sha256=source['source_sha256'], version=VERSION)
        row['content_sha256'] = _content(row)
        return row

    nodes, edges = [], []
    seen = set()
    for raw in claims_document['nodes']:
        if raw['kind'] not in KINDS or raw['id'] in seen:
            raise ValueError('Invalid or duplicate node')
        seen.add(raw['id'])
        row = materialize(raw)
        if row:
            nodes.append(row)
    reviewed = {row['id'] for row in nodes}
    seen_edges = set()
    for raw in claims_document['edges']:
        if raw['relation'] not in RELATIONS or raw['from'] not in seen or raw['to'] not in seen:
            raise ValueError('Invalid semantic edge')
        key = (raw['from'], raw['to'], raw['relation'], raw['source_ref'])
        if key in seen_edges:
            raise ValueError('Duplicate semantic edge')
        seen_edges.add(key)
        row = materialize(raw)
        if row and row['from'] in reviewed and row['to'] in reviewed:
            if not row.get('claim') or not row.get('scope'):
                raise ValueError('Semantic claim or scope missing')
            edges.append(row)
    graph = {'schema': SCHEMA, 'version': VERSION,
             'nodes': sorted(nodes, key=lambda row: row['id']),
             'edges': sorted(edges, key=lambda row: (row['from'], row['to'], row['relation'], row['source_ref'])),
             'sources': [sources[key] for key in sorted(sources) if sources[key]['review_status'] == 'reviewed']}
    validate_graph(graph)
    return graph


def validate_graph(graph):
    if graph.get('schema') != SCHEMA or graph.get('version') != VERSION:
        raise ValueError('Wrong semantic graph version')
    ids = {node['id'] for node in graph['nodes']}
    sources = {source['id']: source for source in graph['sources']}
    if len(ids) != len(graph['nodes']):
        raise ValueError('Duplicate graph node')
    for row in graph['nodes'] + graph['edges']:
        if not all(row.get(key) for key in META) or row['evidence_level'] not in LEVELS:
            raise ValueError('Semantic provenance missing')
        source = sources[row['source_ref']]
        if (row['content_sha256'] != _content(row) or row['source_sha256'] != source['source_sha256']
                or row['source_layer'] not in LAYERS or row['review_status'] != 'reviewed'
                or LEVELS[row['evidence_level']] > LEVELS[source['evidence_level']]):
            raise ValueError('Semantic relation integrity differs')
        if 'kind' in row and row['kind'] not in KINDS:
            raise ValueError('Invalid graph node kind')
        if 'relation' in row and (row['relation'] not in RELATIONS or row['from'] not in ids or row['to'] not in ids):
            raise ValueError('Invalid graph endpoint')
    return graph


def load_assets(expected=None):
    manifest = _read(MANIFEST, expected or MANIFEST_SHA256)
    if (manifest['schema'] != SCHEMA or manifest['variant'] != VARIANT
            or manifest['base_graph_manifest_sha256'] != boundary.GRAPH_MANIFEST_SHA256
            or manifest['reference_manifest_sha256'] != boundary.REFERENCE_MANIFEST_SHA256):
        raise ValueError('Semantic base version differs')
    for name, sha in manifest['source_files'].items():
        path = (KNOWLEDGE / name).resolve()
        if KNOWLEDGE.resolve() not in path.parents or _sha(path) != sha:
            raise ValueError('Semantic reviewed source file differs')
    graph = _read(MANIFEST.parent / 'state_graph.json', manifest['files']['state_graph.json'])
    sources = json.loads((KNOWLEDGE / 'sources.json').read_text(encoding='utf-8'))
    for source in sources['sources']:
        for name, sha in source.get('local_file_sha256', {}).items():
            path = (PROJECT / name).resolve()
            if PROJECT.resolve() not in path.parents or _sha(path) != sha:
                raise ValueError('Semantic local protocol source differs')
    claims = json.loads((KNOWLEDGE / 'claims.json').read_text(encoding='utf-8'))
    if graph != build_graph(sources, claims):
        raise ValueError('Semantic graph differs from reviewed claims')
    return validate_graph(graph)


def _observations(prediction):
    from .semantic_observations import validate_observations
    return validate_observations(prediction.get('eeg_observations'))


def _empty(reason):
    return {'schema': SCHEMA, 'variant': VARIANT, 'supporting_paths': [],
            'contradictory_paths': [], 'observations': [], 'model_evidence': [],
            'evidence_strength': 'low', 'confidence_basis': [reason],
            'evidence_status': 'unavailable', 'fallback_reason': reason,
            'guidance': GUIDANCE}


def explain(prediction, graph):
    """Project actual observations to bounded literature paths without labels."""
    validate_graph(graph)
    observations = _observations(prediction)
    if not observations:
        return _empty('actual_eeg_observations_unavailable')
    nodes = {node['id']: node for node in graph['nodes']}
    sources = {source['id']: source for source in graph['sources']}
    adjacency = {}
    for edge in graph['edges']:
        adjacency.setdefault(edge['from'], []).append(edge)

    def path_record(observation, ids, edges, polarity):
        refs = sorted({row['source_ref'] for row in edges})
        level = min((row['evidence_level'] for row in edges), key=LEVELS.get)
        measured = {'from': observation['id'], 'to': 'biomarker-' + observation['biomarker'], 'relation': 'observed_as',
                    'source_layer': 'model_evidence', 'evidence_level': 'low',
                    'source_ref': 'actual-whole-path-eeg', 'source_sha256': observation['source_sha256'],
                    'version': VERSION, 'claim': 'Measured intra-path change; not a disease label.',
                    'scope': 'path_first_third_vs_path_last_third_no_rest'}
        measured['content_sha256'] = _content(measured)
        record = {'citation_id': 'state-path-' + digest([observation['id'], ids, refs, polarity])[:16],
                  'observation_id': observation['id'], 'nodes': ids,
                  'relations': ['observed_as'] + [edge['relation'] for edge in edges],
                  'observation_edge': measured,
                  'claims': [{'from': edge['from'], 'to': edge['to'], 'relation': edge['relation'],
                              'text': edge['claim'], 'scope': edge['scope'],
                              **({'material_counterevidence': True}
                                 if edge['relation'] == 'contradicts' and
                                 (edge.get('material_counterevidence') is True or
                                  edge['from'] not in SCOPE_LIMIT_COUNTER_NODES) else {}),
                              **{key: edge[key] for key in META}} for edge in edges],
                  'evidence_level': level, 'source_refs': refs,
                  'independent_sources': sorted({sources[ref].get('independence_id', sources[ref].get('independence_group', sources[ref].get('doi') or ref))
                                                 for ref in refs}),
                  'source_layers': sorted({edge['source_layer'] for edge in edges}),
                  'nodes_metadata': [{key: nodes[node][key] for key in ('id', 'kind', 'label') + META
                                      if key in nodes[node]} for node in ids if node in nodes],
                  'sources': [{key: source[key] for key in ('id', 'title', 'doi', 'year', 'locator', 'excerpt',
                                                         'source_layer', 'source_sha256') if key in source}
                              for source in (sources[ref] for ref in refs)],
                  'scope': 'research_association_not_query_class_or_diagnosis', 'polarity': polarity}
        record['content_sha256'] = _content(record)
        return record

    supporting, contradictory, model_nodes = [], [], []
    for observation in observations:
        starts = [node for node in nodes.values() if node['kind'] == 'biomarker'
                  and node.get('marker') == observation['biomarker']]
        for start in starts:
            model_node = {'id': observation['id'], 'kind': 'observation', 'label': observation['biomarker'],
                          'observation_id': observation['id'], 'source_layer': 'model_evidence',
                          'evidence_level': 'low', 'source_ref': 'actual-whole-path-eeg',
                          'source_sha256': observation['source_sha256'], 'version': VERSION}
            model_node['content_sha256'] = _content(model_node)
            model_nodes.append(model_node)
            model_reading = {'id': 'model-' + observation['id'], 'kind': 'model_evidence',
                             'observation_id': observation['id'], 'source_layer': 'model_evidence',
                             'evidence_level': 'low', 'source_ref': 'actual-whole-path-eeg',
                             'source_sha256': observation['source_sha256'], 'version': VERSION}
            model_reading['content_sha256'] = _content(model_reading)
            model_nodes.append(model_reading)
            stack = [(start['id'], [start['id']], [])]
            applicable = {start['id']}
            while stack:
                current, ids, edges = stack.pop()
                if len(edges) >= 3:
                    continue
                for edge in adjacency.get(current, []):
                    if edge['relation'] not in ('associated_with', 'explains', 'supports'):
                        continue
                    # This MVP measures EEG only; never infer an unobserved EDA/ECG mechanism.
                    if edge.get('requires_modalities') or nodes[edge['to']].get('requires_modalities'):
                        continue
                    if edge.get('observed_direction', 'any') not in ('any', observation['direction']):
                        continue
                    # No transfer of a rest-baseline finding onto intra-path change.
                    if edge.get('observation_baseline', 'any') not in ('any', observation['baseline']):
                        continue
                    target = edge['to']
                    if target in ids:
                        continue
                    applicable.add(target)
                    new_ids, new_edges = ids + [target], edges + [edge]
                    if nodes[target]['kind'] == 'state':
                        documents = {}
                        for vertex in new_ids:
                            for item in adjacency.get(vertex, []):
                                if item['relation'] == 'documented_by' and nodes[item['to']]['kind'] == 'paper_evidence':
                                    documents.setdefault((item['to'], item['source_ref']), item)
                        documents = list(documents.values())
                        methods = [item for item in graph['edges'] if item['to'] in new_ids
                                   and item['relation'] == 'solves' and nodes[item['from']]['kind'] == 'method']
                        # Documentation and reverse method links keep explicit
                        # endpoints, so branches are never presented as a causal chain.
                        supporting.append(path_record(observation, [observation['id']] + new_ids +
                                                      list(dict.fromkeys(item['to'] for item in documents)) +
                                                      list(dict.fromkeys(item['from'] for item in methods)),
                                                      new_edges + documents + methods, 'supporting'))
                    else:
                        stack.append((target, new_ids, new_edges))
            for edge in graph['edges']:
                if edge['relation'] == 'contradicts' and edge['to'] in applicable:
                    if edge.get('observed_direction', 'any') not in ('any', observation['direction']):
                        continue
                    contradictory.append(path_record(observation, [observation['id'], edge['from'], edge['to']],
                                                     [edge], 'limitation'))

    def rank(row):
        return (0 if has_material_counterevidence(row) else 1,
                -LEVELS[row['evidence_level']], -len(row['independent_sources']),
                0 if 'paper_evidence' in row['source_layers'] else 1, row['citation_id'])

    def has_material_counterevidence(row):
        # The reviewed MVP contains generic scope limits (directional,
        # baseline, modality, and EEG-to-questionnaire boundaries). They remain
        # visible as contradictory_paths but do not reduce the evidence level.
        # Explicitly material and unclassified counterclaims take precedence
        # over generic limitations when selecting a bounded context.
        return any(claim.get('material_counterevidence') is True
                   for claim in row.get('claims', []))

    def has_direct_paper_support(row):
        # Mechanism edges are hypotheses and are intentionally low. A direct
        # reviewed biomarker-to-paper edge can still support a medium evidence
        # grade for the observed association package.
        return any(claim.get('relation') == 'documented_by' and
                   claim.get('from') == row['observation_edge']['to'] and
                   claim.get('source_layer') == 'paper_evidence' and
                   LEVELS.get(claim.get('evidence_level'), 0) >= LEVELS['medium']
                   for claim in row.get('claims', []))
    supporting.sort(key=rank)
    contradictory.sort(key=rank)
    total_supporting, total_contradictory = len(supporting), len(contradictory)
    supporting, contradictory = supporting[:MAX_PATHS], contradictory[:MAX_PATHS]
    result = _empty('no_applicable_reviewed_state_path')
    result.update(observations=observations, model_evidence=model_nodes,
                  supporting_paths=supporting, contradictory_paths=contradictory,
                  truncation={'supporting_paths_omitted': total_supporting - len(supporting),
                              'contradictory_paths_omitted': total_contradictory - len(contradictory)})
    if supporting:
        material_limits = [row for row in contradictory if has_material_counterevidence(row)]
        direct_paper = any(has_direct_paper_support(row) for row in supporting)
        # Research associations are capped at medium. Generic scope limits stay
        # visible, while only a directly applicable reviewed counterclaim lowers
        # the grade to low.
        strength = 'low' if material_limits or not direct_paper else 'medium'
        basis = ['actual_intra_path_observation', 'direct_reviewed_biomarker_paper_evidence'
                 if direct_paper else 'reviewed_research_associations']
        if material_limits:
            basis.append('material_counterevidence_limits_state_inference')
        elif contradictory:
            basis.append('generic_scope_limits_do_not_override_direct_association')
        else:
            basis.append('association_is_not_individual_diagnosis')
        result.update(evidence_status='available', fallback_reason=None, evidence_strength=strength,
                      confidence_basis=basis)
    # Drop complete packages, retaining both polarities when they fit. Counts are
    # audit information only; observed measurements survive even if no path fits.
    while len(json.dumps(result, ensure_ascii=False, allow_nan=False)) > MAX_CHARACTERS:
        if len(supporting) > 1:
            supporting.pop()
            result['truncation']['supporting_paths_omitted'] += 1
        elif len(contradictory) > 1:
            contradictory.pop()
            result['truncation']['contradictory_paths_omitted'] += 1
        elif supporting:
            supporting.pop()
            result['truncation']['supporting_paths_omitted'] += 1
            result.update(evidence_status='unavailable', evidence_strength='low',
                          fallback_reason='complete_semantic_package_exceeds_budget',
                          confidence_basis=['complete_semantic_package_exceeds_budget'])
        elif contradictory:
            contradictory.pop()
            result['truncation']['contradictory_paths_omitted'] += 1
        else:
            break
    return result


def retrieve(fold, prediction, semantic_manifest_sha256=None):
    try:
        graph = load_assets(semantic_manifest_sha256)
        context = boundary.retrieve(fold, prediction, boundary.GRAPH_MANIFEST_SHA256)
        if context.get('available') is not True:
            raise ValueError('Frozen same-fold v3 unavailable')
        state = explain(prediction, graph)
        context = copy.deepcopy(context)
        # Keep individual opposing demonstrations, not bank-wide category summaries.
        for key in ('class_descriptions', 'direction_error_counts', 'group_sign_pattern_reference_counts'):
            context.pop(key, None)
            context.get('reading_definitions', {}).pop(key, None)
        if 'graph_evidence' in context:
            evidence = context['graph_evidence']
            evidence['coverage_note'] = 'Same-fold support/coverage is not a query class or probability.'
            for package in evidence['packages']:
                package.pop('whole_bank_calibration', None)
                condition = package.get('conditional_evidence', {})
                package['conditional_evidence'] = {key: condition[key] for key in
                                                   ('paths', 'subjects', 'coverage_supported') if key in condition}
        context['review_guidance'] = {'citation_id': 'state-evidence-review', 'text': GUIDANCE}
        context.update(schema=SCHEMA, variant=VARIANT, state_evidence=state)
        context['source_hashes'].update(semantic_manifest_sha256=semantic_manifest_sha256 or MANIFEST_SHA256,
                                       semantic_graph_sha256=digest(graph))
        return context
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return {'schema': SCHEMA, 'variant': VARIANT, 'available': False,
                'reason': 'semantic_graph_pinned_assets_or_observations_invalid',
                'state_evidence': _empty('semantic_graph_pinned_assets_or_observations_invalid')}


def session_evidence(prediction, judgment=None):
    """Allowlisted audit record; never feeds the reusable graph or next query."""
    state = prediction.get('current_raw_contrastive_context', {}).get('state_evidence')
    if not isinstance(state, dict) or any(key not in state for key in
            ('observations', 'supporting_paths', 'contradictory_paths', 'evidence_strength',
             'evidence_status', 'confidence_basis')):
        return None
    record = {'schema': 'session_model_evidence_v1', 'source_layer': 'model_evidence',
              'version': VERSION, 'observed_eeg_changes': copy.deepcopy(state['observations']),
              'supporting_paths': copy.deepcopy(state['supporting_paths']),
              'contradictory_paths': copy.deepcopy(state['contradictory_paths']),
              'evidence_strength': state['evidence_strength'], 'evidence_status': state['evidence_status'],
              'confidence_basis': copy.deepcopy(state['confidence_basis']),
              'supporting_biomarkers': sorted({item['biomarker'] for item in state['observations']}),
              'related_mechanisms': sorted({node for path in state['supporting_paths'] for node in path['nodes']
                                            if node.startswith('mechanism-')}),
              'prediction': None, 'reuse_policy': 'session_audit_only_never_reference_bank'}
    if isinstance(judgment, dict) and judgment.get('available') is True:
        if judgment.get('final_class') in ('High', 'Low') and judgment.get('confidence') in LEVELS:
            record['prediction'] = {'final_class': judgment['final_class'], 'confidence': judgment['confidence'],
                                    'source': 'independent_llm_not_ground_truth'}
    record['content_sha256'] = _content(record)
    return record
