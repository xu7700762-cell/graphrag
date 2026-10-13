"""The production RAG policy shared by offline and path-end online judgement.

Promote future retrieval improvements here, together with their verified assets
and service checks. Historical experiments keep their explicit variant.
"""
import copy
import hashlib
import json
from pathlib import Path

from .config import PROJECT
from .regular_rag import SCHEMA, context_from_bank, validate_bank

DEFAULT_VARIANT = 'percentile_group'
REFERENCE_MANIFEST = PROJECT / 'outputs/rag_assets/current_raw_policy_v3/manifest.json'
REFERENCE_MANIFEST_SHA256 = '8e7c75aab8a1dc7957c555faa287ca8b6dff57bf09843f07c899246207b45ea1'
SNAPSHOT = PROJECT / 'outputs/rag_assets/regular_baseline_v1/manifest.json'
SNAPSHOT_SHA256 = 'a20f257bec5263449a84fa351f8eda1f31cfb427fe36676dce3b89e3c45a44b0'
DEFAULT_MODE = 'regular'
DEFAULT_JUDGE_PROFILE = 'legacy_production'


def _snapshot():
    manifest = _read_verified(SNAPSHOT, SNAPSHOT_SHA256)
    if manifest['schema'] != 'regular_rag_snapshot_v1':
        raise ValueError('Wrong ordinary RAG snapshot')
    if hashlib.sha256((PROJECT / 'eeg_agent/regular_rag.py').read_bytes()).hexdigest() != manifest['retriever_sha256']:
        raise ValueError('Frozen ordinary retriever changed')
    return manifest


def resolve(mode='default'):
    """Resolve once at session creation; this record is saved with the session."""
    if mode not in ('default', 'regular', 'graph'):
        raise ValueError('Unknown RAG mode')
    manifest = _snapshot()
    actual = DEFAULT_MODE if mode == 'default' else mode
    profile = DEFAULT_JUDGE_PROFILE if mode == 'default' else 'matched_decisive_high_v1'
    result = {'rag_mode': actual, 'requested_mode': mode, 'judge_profile': profile,
              'schema': SCHEMA, 'variant': DEFAULT_VARIANT,
              'reference_manifest_sha256': REFERENCE_MANIFEST_SHA256,
              'regular_snapshot_sha256': SNAPSHOT_SHA256,
              'retriever_sha256': manifest['retriever_sha256'],
              'judge_protocol_sha256': manifest['files']['judge_protocol.json'],
              'legacy_system_sha256': manifest['files']['legacy_system.json'],
              'teaching_sha256': manifest['files']['legacy_teaching.json'] if profile == 'legacy_production' else None}
    if actual == 'graph':
        from . import graph_rag
        result.update(graph_rag.asset_policy())
        result['graph_code_sha256'] = hashlib.sha256(Path(graph_rag.__file__).read_bytes()).hexdigest()
        result.update(schema=graph_rag.SCHEMA, variant=graph_rag.VARIANT)
    return result


def validate_policy(value):
    """Validate saved versions without resolving a potentially newer default."""
    if not isinstance(value, dict) or value.get('rag_mode') not in ('regular', 'graph'):
        raise ValueError('Missing resolved RAG policy')
    expected = resolve(value['rag_mode'])
    for key in ('reference_manifest_sha256', 'regular_snapshot_sha256', 'retriever_sha256',
                'judge_protocol_sha256', 'legacy_system_sha256', 'graph_manifest_sha256', 'graph_code_sha256'):
        if value.get(key) != expected.get(key):
            raise ValueError('Pinned RAG version differs: ' + key)
    if value.get('judge_profile') not in ('legacy_production', 'matched_decisive_high_v1'):
        raise ValueError('Unknown frozen judge profile')
    if value['judge_profile'] == 'legacy_production' and (value['rag_mode'] != 'regular' or
            value.get('teaching_sha256') != _snapshot()['files']['legacy_teaching.json']):
        raise ValueError('Invalid legacy teaching snapshot')
    return value


def judge_contract(value):
    validate_policy(value)
    manifest = _snapshot()
    return _read_verified(SNAPSHOT.parent / 'judge_protocol.json', manifest['files']['judge_protocol.json'])


def frozen_teaching(value):
    validate_policy(value)
    manifest = _snapshot()
    return _read_verified(SNAPSHOT.parent / 'legacy_teaching.json', manifest['files']['legacy_teaching.json'])


def legacy_system(value):
    validate_policy(value)
    manifest = _snapshot()
    return _read_verified(SNAPSHOT.parent / 'legacy_system.json', manifest['files']['legacy_system.json'])['system']


def catalog_fields():
    return {'vrms_case_rag': policy(), 'default_rag_mode': DEFAULT_MODE,
            'rag_modes': [{'id': 'default', 'label': '默认版本'},
                          {'id': 'regular', 'label': '普通 RAG（固定基线，GPT high）'},
                          {'id': 'graph', 'label': 'GraphRAG（实验，GPT high）'}]}


def policy():
    value = resolve('default')
    return {**value,
            'reference_manifest_sha256': REFERENCE_MANIFEST_SHA256,
            'entrypoint': 'eeg_agent.default_rag',
            'promotion_policy': 'new validated RAG improvements become the production default',
            'missing_reference_policy': 'explicitly unavailable; no legacy fallback'}


def _read_verified(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Default RAG reference asset hash differs')
    return json.loads(data.decode('utf-8'))


def retrieve(fold, prediction, rag_policy=None):
    """Return labelled internal references, never a query class or probability."""
    try:
        if rag_policy is not None:
            validate_policy(rag_policy)
            if rag_policy['rag_mode'] == 'graph':
                from .graph_rag import retrieve as graph_retrieve
                return graph_retrieve(fold, prediction, rag_policy['graph_manifest_sha256'])
        if type(fold) is not int or fold < 1:
            raise ValueError('A verified model fold is required')
        manifest = _read_verified(REFERENCE_MANIFEST, REFERENCE_MANIFEST_SHA256)
        if manifest['schema'] != 'default_rag_reference_assets_v1':
            raise ValueError('Wrong default RAG asset schema')
        filename = f'fold_{fold:02d}.json'
        entry = _read_verified(REFERENCE_MANIFEST.parent / filename, manifest['files'][filename])
        bank = entry['bank']
        validate_bank(bank, fold)
        return context_from_bank(prediction, bank, entry['provenance'], DEFAULT_VARIANT)
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        if isinstance(rag_policy, dict) and rag_policy.get('rag_mode') == 'graph':
            return {'schema': 'raw_eeg_graph_rag_v1', 'variant': 'graph_v1', 'available': False,
                    'reason': 'graph_rag_pinned_assets_unavailable'}
        return {'schema': SCHEMA, 'variant': DEFAULT_VARIANT, 'available': False,
                'reason': 'default_optimized_rag_reference_unavailable'}


def attach(fold, prediction, rag_policy=None):
    """Replace any older worker context; preserve every actual model reading."""
    result = copy.deepcopy(prediction)
    result['current_raw_contrastive_context'] = retrieve(fold, prediction, rag_policy)
    if rag_policy is not None:
        result['rag_policy'] = copy.deepcopy(rag_policy)
    return result
