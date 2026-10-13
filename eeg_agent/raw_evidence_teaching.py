"""Reviewed numerical-method RAG and bounded anonymous training-case context.

This module never reads query labels, training tables, credentials or waveforms.
Training cases come only from the worker's verified frozen internal case bank.
"""
import copy
import json
import math
import re

from .domain_knowledge import HierarchicalKnowledge

INPUT_NAMES = ('MIL', 'femba_features', 'spectral', 'spatial_covariance',
               'case_retrieval', 'femba_mean', 'femba_temporal',
               'spectral_compact', 'filterbank_csp')
GUIDE_VERSION = 'raw-only-nine-evidence-2.0'
MAX_TEACHING_CHARACTERS = 8000
MAX_CASES, MAX_DETAILED_CASES = 12, 3
MAX_CASE_CHARACTERS = 9000
SOURCE_HASH_KEYS = ('numeric_first_sha256', 'numeric_second_sha256',
                    'numeric_selector_sha256', 'case_bank_sha256',
                    'case_bank_manifest_sha256')


def _build_teaching_context(cfg=None):
    """Cover each input explicitly; a single top-five query cannot cover nine.

    Only bundled, reviewed raw-only teaching is eligible. Uploaded documents
    cannot supply model instructions or labeled demonstrations through this path.
    """
    knowledge = HierarchicalKnowledge(cfg)
    inputs, principles, references = [], [], {}

    def add_reference(chunk):
        key = chunk['document_sha256']
        references[key] = {name: chunk[name] for name in
                           ('source', 'source_sha256', 'document_sha256', 'version', 'title')}

    for name in INPUT_NAMES:
        results = knowledge.search(name, domains=['vrms'], limit=5,
                                   method_only=True)['results']
        pattern = re.compile(r'(?<![a-z_])' + re.escape(name) + r'(?![a-z_])', re.I)
        chunk = next((item for item in results if item['version'] == GUIDE_VERSION
                      and pattern.search(item['section'])), None)
        if chunk:
            inputs.append({'name': name, 'citation_id': chunk['citation_id'], 'text': chunk['text']})
            add_reference(chunk)
    for section in ('独立综合判读流程', '证据判读演示',
                    'raw_decision_score', 'calibrated_high_probability', '原始类别与校准概率',
                    '冻结数值融合', '证据相关性', '判断步骤', '案例来源与限制'):
        results = knowledge.search(section, domains=['vrms'], limit=5,
                                   method_only=True)['results']
        chunk = next((item for item in results if item['version'] == GUIDE_VERSION
                      and section in item['section']), None)
        if chunk:
            principles.append({'citation_id': chunk['citation_id'], 'text': chunk['text']})
            add_reference(chunk)
    context = {'schema': 'raw_eeg_evidence_teaching_v1',
               'available': len(inputs) == len(INPUT_NAMES),
               'input_definitions': inputs, 'judgment_principles': principles,
               'sources': list(references.values()),
               'missing_inputs': [name for name in INPUT_NAMES if name not in {i['name'] for i in inputs}]}
    # Preserve complete per-input definitions before optional additional prose.
    while principles and len(json.dumps(context, ensure_ascii=False)) > MAX_TEACHING_CHARACTERS:
        principles.pop()
    if len(json.dumps(context, ensure_ascii=False)) > MAX_TEACHING_CHARACTERS:
        return {'schema': context['schema'], 'available': False,
                'reason': 'reviewed_teaching_exceeds_context_budget'}
    return context


def build_teaching_context(cfg=None):
    """Missing or damaged method evidence must not invalidate numerical work."""
    try:
        return _build_teaching_context(cfg)
    except (OSError, ValueError, TypeError, AttributeError):
        return {'schema': 'raw_eeg_evidence_teaching_v1', 'available': False,
                'reason': 'reviewed_teaching_unavailable'}


def _number(value, lower=0., upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Nonfinite case evidence')
    if value < lower or (upper is not None and value > upper):
        raise ValueError('Case evidence outside range')
    return value


def sanitize_case_context(value):
    """Allow only anonymous typed reference fields; never forward a whole bank."""
    unavailable = {'schema': 'raw_case_rag_v1', 'available': False,
                   'reason': 'verified_training_case_context_unavailable'}
    if not isinstance(value, dict) or value.get('available') is not True:
        return unavailable
    try:
        if (value.get('schema') != 'raw_case_rag_v1'
                or value.get('provenance') != 'cached_training_reference'
                or value.get('validation_scope') != 'in_sample_not_OOF'):
            raise ValueError('Unknown case provenance')
        hashes = {key: value.get('source_hashes', {}).get(key) for key in SOURCE_HASH_KEYS}
        if any(not isinstance(h, str) or not re.fullmatch(r'[a-f0-9]{64}', h) for h in hashes.values()):
            raise ValueError('Missing verified case hashes')

        def case_row(row):
            identity = row.get('case_id')
            rank = row.get('rank')
            if (not isinstance(identity, str) or not re.fullmatch(r'case-\d{2}', identity)
                    or type(rank) is not int or not 1 <= rank <= MAX_CASES
                    or row.get('training_class') not in ('High', 'Low')
                    or row.get('fitting_role') not in ('base_train', 'meta_calibration')):
                raise ValueError('Invalid anonymous training case')
            return {'case_id': identity, 'rank': rank,
                    'distance': _number(row['distance']),
                    'training_class': row['training_class'], 'fitting_role': row['fitting_role']}

        cases = [case_row(row) for row in value.get('cases', [])[:MAX_CASES]]
        if not cases or len({c['case_id'] for c in cases}) != len(cases):
            raise ValueError('Missing or repeated training cases')
        by_id = {row['case_id']: row for row in cases}
        details = []
        for row in value.get('detailed_cases', [])[:MAX_DETAILED_CASES]:
            detail = case_row(row)
            if by_id.get(detail['case_id']) != detail:
                raise ValueError('Detailed case is not a retrieved case')
            detail['mil_raw_high_probability'] = _number(row['mil_raw_high_probability'], upper=1.)
            tools = []
            for tool in row['individual_tools']:
                name = tool.get('tool')
                if name not in INPUT_NAMES[1:] or tool.get('predicted_class') not in ('High', 'Low'):
                    raise ValueError('Unknown case tool')
                score = tool.get('raw_decision_score')
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError('Nonfinite training tool score')
                usable = tool.get('calibration_usable')
                if type(usable) is not bool:
                    raise ValueError('Invalid case calibration status')
                probability = tool.get('calibrated_high_probability')
                if usable:
                    probability = _number(probability, upper=1.)
                elif probability is not None:
                    raise ValueError('Unavailable case calibration has a probability')
                tools.append({'tool': name, 'raw_decision_score': score,
                              'predicted_class': tool['predicted_class'],
                              'calibration_usable': usable,
                              'calibrated_high_probability': probability})
            if len(tools) != 8 or {t['tool'] for t in tools} != set(INPUT_NAMES[1:]):
                raise ValueError('Incomplete case input vector')
            detail.update(individual_tools=tools,
                          evidence_source='frozen_heads_on_cached_internal_training_features_no_rest')
            details.append(detail)
        retrieval = value.get('retrieval', {})
        candidate_subjects = retrieval.get('candidate_subjects')
        if type(candidate_subjects) is not int or not len(cases) <= candidate_subjects <= 10000:
            raise ValueError('Invalid internal subject support')
        result = {'schema': 'raw_case_rag_v1', 'available': True,
                  'provenance': 'cached_training_reference', 'validation_scope': 'in_sample_not_OOF',
                  'retrieval': {'candidate_subjects': candidate_subjects,
                                'returned_subjects': len(cases), 'requested_subjects': MAX_CASES,
                                'selection_rule': 'nearest_one_path_per_distinct_internal_subject',
                                'detail_rule': 'closest_two_plus_closest_opposite_training_class_if_available',
                                'distance_metric': 'training_scaler_root_mean_square_embedding_distance'},
                  'cases': cases, 'detailed_cases': details, 'source_hashes': hashes,
                  'limitations': ['These are internal training references, not query truth or independent validation.',
                                  'No resting reference is used. Case vectors reuse cached training features.',
                                  'Neighbour classes and distances are descriptive, not calibrated query probabilities.',
                                  'The expanded context does not change the frozen numerical neighbour vote.']}
        while details and len(json.dumps(result, ensure_ascii=False)) > MAX_CASE_CHARACTERS:
            details.pop()
        if len(json.dumps(result, ensure_ascii=False)) > MAX_CASE_CHARACTERS:
            raise ValueError('Case context exceeds budget')
        return copy.deepcopy(result)
    except (KeyError, TypeError, ValueError, AttributeError):
        return unavailable
