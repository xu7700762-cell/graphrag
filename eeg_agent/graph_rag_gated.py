"""Offline, post-hoc Graph v1 conflict gate; never a query classifier.

Actual raw/calibrated direction conflict in any of the eight usable tools
routes to the unchanged Graph v1 context. Otherwise the unchanged fixed
ordinary context is returned. Both assets must verify before either route.
Strategy/version metadata stays local and is never appended to LLM evidence.
This candidate is not connected to the web/default and makes no API calls.
"""
from __future__ import annotations

from . import graph_rag as graph_v1
from . import regular_rag as regular

SCHEMA = 'raw_eeg_graph_rag_gated_v1'
VARIANT = 'graph_gated_v1'
ALGORITHM_VERSION = 'raw_calibration_conflict_gate_literal_graph_v1_or_regular_v1'
REFERENCE_MANIFEST = graph_v1.REFERENCE_MANIFEST
REFERENCE_MANIFEST_SHA256 = graph_v1.REFERENCE_MANIFEST_SHA256
GRAPH_MANIFEST = graph_v1.GRAPH_MANIFEST
GRAPH_MANIFEST_SHA256 = graph_v1.GRAPH_MANIFEST_SHA256


def asset_policy():
    """Local session/audit metadata; do not merge into either returned context."""
    return {'strategy_schema': SCHEMA, 'strategy_variant': VARIANT,
            'algorithm_version': ALGORITHM_VERSION,
            'reference_manifest_sha256': REFERENCE_MANIFEST_SHA256,
            'graph_manifest_sha256': GRAPH_MANIFEST_SHA256,
            'graph_branch_variant': graph_v1.VARIANT,
            'regular_branch_variant': 'percentile_group',
            'evaluation_scope': 'posthoc_historical_strategy_replay',
            'web_default_enabled': False}


def conflict_tools(prediction):
    """Use only actual eight-tool readings; MIL and query identity are excluded."""
    raw, _, calibrated, _ = regular._values(
        prediction['mil_raw_high_probability'], prediction['numeric_fusion']['actual_tools'])
    return [name for i, name in enumerate(regular.FAMILIES, 1)
            if calibrated[i] is not None and ((raw[i] >= 0) != (calibrated[i] >= .5))]


def retrieve(fold, prediction, graph_manifest_sha256=None):
    """Verify both frozen assets, then return a literal existing branch context."""
    try:
        if type(fold) is not int or fold < 1:
            raise ValueError('A verified fold is required')
        reference = graph_v1._read_verified(REFERENCE_MANIFEST, REFERENCE_MANIFEST_SHA256)
        if reference['schema'] != 'default_rag_reference_assets_v1':
            raise ValueError('Wrong source reference manifest')
        filename = f'fold_{fold:02d}.json'
        entry = graph_v1._read_verified(REFERENCE_MANIFEST.parent / filename, reference['files'][filename])
        bank, provenance = entry['bank'], entry['provenance']
        regular.validate_bank(bank, fold)
        # Graph v1 performs full graph/source verification and rebuild checks.
        # Do this even when the gate ultimately chooses ordinary evidence.
        graph_context = graph_v1.retrieve(fold, prediction, graph_manifest_sha256)
        if graph_context.get('available') is not True:
            raise ValueError('Frozen Graph v1 context is unavailable')
        if conflict_tools(prediction):
            return graph_context
        return regular.context_from_bank(prediction, bank, provenance, 'percentile_group')
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return {'schema': SCHEMA, 'variant': VARIANT, 'available': False,
                'reason': 'verified_graph_gated_reference_unavailable'}
