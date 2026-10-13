"""Meaningful route, asset-failure and invariance checks without providers."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eeg_agent import graph_rag_gated as gate, graph_rag as graph, regular_rag as regular


def prediction(raw=-1., calibrated=.2):
    return {'mil_raw_high_probability': .999,
            'numeric_fusion': {'reference_available': False,
                'actual_tools': [{'tool': name, 'raw_decision_score': raw,
                                  'calibration_usable': calibrated is not None,
                                  'calibrated_high_probability': calibrated}
                                 for name in regular.FAMILIES]}}


class GatedRagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = json.loads((gate.REFERENCE_MANIFEST.parent / 'fold_01.json').read_text('utf-8'))

    def test_all_eight_same_direction_keep_literal_ordinary(self):
        query = prediction()
        expected = regular.context_from_bank(query, self.entry['bank'], self.entry['provenance'], 'percentile_group')
        actual = gate.retrieve(1, query)
        self.assertEqual(actual, expected)
        self.assertNotIn('strategy_schema', actual)
        self.assertNotIn('graph_evidence', actual)

    def test_one_actual_conflict_keeps_literal_graph(self):
        query = prediction()
        query['numeric_fusion']['actual_tools'][0]['calibrated_high_probability'] = .8
        expected = graph.retrieve(1, query)
        self.assertTrue(expected['available'])
        self.assertEqual(gate.retrieve(1, query), expected)

    def test_mil_and_hidden_identifiers_do_not_route(self):
        query = prediction()
        before = gate.retrieve(1, query)
        query['mil_raw_high_probability'] = .001
        # MIL is displayed so it changes the reading, but never the branch.
        self.assertEqual(gate.conflict_tools(query), [])
        hidden = copy.deepcopy(query)
        hidden.update(id='qtruth', true_class='High', previous_answer={'final_class': 'Low'}, subject_id=999)
        self.assertEqual(gate.retrieve(1, hidden), gate.retrieve(1, query))
        self.assertEqual(before['schema'], gate.retrieve(1, query)['schema'])

    def test_calibration_unavailable_is_not_a_conflict(self):
        query = prediction(calibrated=None)
        self.assertEqual(gate.conflict_tools(query), [])
        self.assertEqual(gate.retrieve(1, query)['schema'], regular.SCHEMA)

    def test_boundaries_zero_and_half(self):
        self.assertEqual(gate.conflict_tools(prediction(raw=0, calibrated=.5)), [])
        self.assertEqual(len(gate.conflict_tools(prediction(raw=0, calibrated=.499))), 8)
        self.assertEqual(len(gate.conflict_tools(prediction(raw=-.001, calibrated=.5))), 8)

    def test_invalid_calibration_does_not_silently_route(self):
        query = prediction()
        query['numeric_fusion']['actual_tools'][0]['calibration_usable'] = False
        self.assertFalse(gate.retrieve(1, query)['available'])

    def test_graph_missing_rejects_even_no_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(graph, 'GRAPH_MANIFEST', Path(directory) / 'missing.json'):
                self.assertFalse(gate.retrieve(1, prediction())['available'])

    def test_graph_corrupt_rejects_even_no_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text('{}', encoding='utf-8')
            with patch.object(graph, 'GRAPH_MANIFEST', path):
                self.assertFalse(gate.retrieve(1, prediction())['available'])

    def test_reference_corrupt_rejects_before_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text('{}', encoding='utf-8')
            with patch.object(gate, 'REFERENCE_MANIFEST', path), patch.object(graph, 'retrieve') as retrieve:
                self.assertFalse(gate.retrieve(1, prediction())['available'])
                retrieve.assert_not_called()

    def test_wrong_fold_or_manifest_is_unavailable(self):
        self.assertFalse(gate.retrieve(True, prediction())['available'])
        self.assertFalse(gate.retrieve(999, prediction())['available'])
        self.assertFalse(gate.retrieve(1, prediction(), '0' * 64)['available'])

    def test_policy_metadata_stays_local(self):
        policy = gate.asset_policy()
        self.assertEqual(policy['strategy_variant'], 'graph_gated_v1')
        self.assertFalse(policy['web_default_enabled'])
        context = gate.retrieve(1, prediction())
        self.assertFalse(set(policy) & set(context))


if __name__ == '__main__':
    unittest.main()
