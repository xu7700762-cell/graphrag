"""Verify the deployed retriever and its actual cloud payload without an API."""
import copy
import json
from pathlib import Path

from eeg_agent import default_rag, raw_independent_judge, raw_model
from eeg_agent.raw_evidence_teaching import build_teaching_context
from tests.test_raw_independent_judge import Provider, final
from tests.test_online_cloud import reporter_factory

WORK = Path(r'D:\桌面\文献总\EEGAGENT')


def recorded_prediction():
    path = WORK / 'outputs/vrms_rag_probes/current_raw_context_all146_offline_source_20261011/frozen_predictions.json'
    return json.loads(path.read_text(encoding='utf-8'))['q001']


def test_deployed_context_matches_the_frozen_optimized_experiment():
    import importlib.util
    source = WORK / 'tmp/github_decisive_rag_20261011/eeg_agent/rag_experiment.py'
    # The experiment is loaded in its own package to avoid mixing app modules.
    package = type(__import__('sys'))('default_rag_replay')
    package.__path__ = [str(source.parent)]
    import sys
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + '.rag_experiment', source)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    inputs, banks, protocol, manifest = replay.load_replay(source.parents[1] / 'results')
    expected = replay.payload_for('q001', 'percentile_group', inputs, banks, manifest)
    expected = expected['initial_evidence']['policy_heldout_case_context']
    actual = default_rag.retrieve(1, recorded_prediction())
    assert actual['available'] and actual['variant'] == 'percentile_group'
    # The app retains historical femba_* names; public replay uses vrms_*.
    text = json.dumps(actual, ensure_ascii=False).replace('femba_features', 'vrms_features')
    text = text.replace('femba_mean', 'vrms_mean').replace('femba_temporal', 'vrms_temporal')
    text = text.replace('FEMBA', 'VRMSModel')
    normalized = json.loads(text)
    for field in ('cases', 'query_reading', 'retrieval', 'class_descriptions',
                  'direction_error_counts', 'group_sign_pattern_reference_counts'):
        assert normalized[field] == expected[field]


def test_default_replaces_old_context_and_reaches_real_judge_payload():
    row = recorded_prediction()
    row['current_raw_contrastive_context'] = {'available': False, 'schema': 'old'}
    before = copy.deepcopy(row)
    attached = raw_model.attach_default_rag(1, row)
    assert row == before and attached['numeric_fusion'] == row['numeric_fusion']
    context = attached['current_raw_contrastive_context']
    citation = context['cases'][0]['case_id']
    provider = Provider(final([citation]))
    result = raw_independent_judge.classify(provider, attached, {'available': False})
    assert result['available'] and result['rag_variant'] == 'percentile_group'
    outbound = json.loads(provider.calls[0]['messages'][1]['content'])
    assert outbound['initial_evidence']['policy_heldout_case_context'] == context
    text = json.dumps(outbound)
    assert 'raw_combined_class' not in text and 'raw_combined_score' not in text
    assert 'questionnaire' not in text and 'true_class' not in text


def test_online_path_end_uses_the_same_verified_default(monkeypatch, reporter_factory):
    from eeg_agent.online import cloud
    from tests.test_online_cloud import completed_path, ProviderDouble
    monkeypatch.setattr(cloud, 'attach_default_rag', default_rag.attach)
    row = recorded_prediction()
    row['fold'] = 1
    provider = ProviderDouble()
    report = reporter_factory(provider).report(completed_path(), row)
    assert report['reports']['vrms']['rag_variant'] == 'percentile_group'
    payload = json.loads(provider.calls[0]['messages'][1]['content'])
    assert payload['initial_evidence']['policy_heldout_case_context']['variant'] == 'percentile_group'


def test_damaged_reference_is_explicitly_unavailable_without_legacy_fallback(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(default_rag, 'REFERENCE_MANIFEST', manifest)
    result = default_rag.retrieve(1, recorded_prediction())
    assert result['available'] is False and result['variant'] == 'percentile_group'
    assert result['reason'] == 'default_optimized_rag_reference_unavailable'


def test_all_reference_folds_are_verified_and_methods_cover_every_input():
    data = default_rag._read_verified(default_rag.REFERENCE_MANIFEST,
                                      default_rag.REFERENCE_MANIFEST_SHA256)
    assert len(data['files']) == 24
    for name, expected in data['files'].items():
        entry = default_rag._read_verified(default_rag.REFERENCE_MANIFEST.parent / name, expected)
        default_rag.validate_bank(entry['bank'], int(name[5:7]))
    teaching = build_teaching_context()
    assert teaching['available'] and len(teaching['input_definitions']) == 9
    assert any('percentile_group' in item['text'] for item in teaching['judgment_principles'])
