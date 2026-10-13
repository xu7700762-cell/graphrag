import json
import math
from pathlib import Path

import numpy as np
import pytest

from eeg_agent.knowledge import KnowledgeIndex
from eeg_agent.tools import (BaselineCollector, TemporalAnalyzer, analysis_policy,
                             biomarkers, covariance, TOOL_CATALOG)


def known_signal(scale=1.):
    t = np.arange(1280) / 256.
    values = (2*np.sin(2*np.pi*2*t) + 2*np.sin(2*np.pi*6*t) +
              4*np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + np.sin(2*np.pi*36*t))
    return np.repeat((scale*values)[None, :], 30, axis=0)


def frozen_baseline(x):
    baseline = BaselineCollector()
    assert baseline.observe(x, {'valid': True}, 10, 15)
    assert baseline.observe(x, {'valid': True}, 15, 20)
    baseline.freeze(20)
    return baseline


def test_welch_power_ratios_and_baseline_changes():
    x = known_signal()
    baseline = frozen_baseline(x)
    result = biomarkers(x, baseline)
    assert result['available'] and result['baseline_available']
    power = result['regions']['frontal']['absolute_power_uv2']
    assert power['delta'] == pytest.approx(2., abs=1e-9)
    assert power['theta'] == pytest.approx(2., abs=1e-9)
    assert power['alpha'] == pytest.approx(8., abs=1e-9)
    assert power['beta'] == pytest.approx(.5, abs=1e-9)
    assert power['gamma_low'] == pytest.approx(.5, abs=1e-9)
    assert result['regions']['occipital']['ratios']['theta_alpha'] == pytest.approx(.25)
    assert sum(result['regions']['all']['relative_power'].values()) == pytest.approx(1.)
    scaled = biomarkers(x*2, baseline)
    assert scaled['baseline_change']['regions']['all']['alpha'] == pytest.approx(3.)
    assert result['baseline_change']['regions']['all']['alpha'] == pytest.approx(0.)
    # Public tools also accept the stream's samples-by-channels layout.
    assert biomarkers(x.T)['absolute_power_uv2'] == result['absolute_power_uv2']
    json.dumps(scaled, allow_nan=False)


def test_baseline_only_past_valid_complete_windows_and_first_freeze():
    baseline = BaselineCollector()
    x = known_signal()
    assert not baseline.observe(x, {'valid': False}, 0, 5)
    assert not baseline.observe(x, True, 5, 9)
    assert baseline.observe(x, True, 10, 15)
    assert baseline.observe(x, True, 15, 20)
    assert not baseline.observe(x, True, 18, 23)
    assert not baseline.freeze(15)['available']
    assert baseline.summary()['windows'] == 1
    assert not baseline.observe(x, True, 20, 25)
    assert baseline.freeze(100)['cutoff_seconds'] == 15
    assert not baseline.summary()['z_scores_available']
    complete = BaselineCollector()
    for i in range(13):
        complete.observe(x, True, i*5, (i+1)*5)
    summary = complete.freeze(65)
    assert summary['windows'] == 12 and summary['seconds'] == 60


def test_covariance_spd_distance_known_scaling_and_unsigned():
    x = known_signal()
    baseline = frozen_baseline(x)
    assert np.linalg.eigvalsh(baseline.covariance_matrix).min() > 0
    identity = covariance(x, baseline)
    doubled = covariance(x*2, baseline)
    assert identity['airm_distance'] == pytest.approx(0., abs=1e-10)
    assert doubled['airm_distance'] == pytest.approx(math.sqrt(30)*math.log(4), abs=1e-10)
    assert doubled['z_score'] is None
    assert 'class' not in doubled and 'state_shift' not in doubled
    assert covariance(x)['airm_distance'] is None
    json.dumps(doubled, allow_nan=False)


def test_invalid_contracts_and_invalid_denominator_are_explicit():
    for invalid in (np.ones((29, 1280)), np.ones((30, 1279)), np.full((30, 1280), np.nan)):
        assert not biomarkers(invalid)['available']
        assert not covariance(invalid)['available']
    zero = biomarkers(np.zeros((30, 1280)))
    assert zero['available']
    assert zero['regions']['all']['ratios']['theta_alpha'] is None
    assert all(v is None for v in zero['relative_power']['alpha'])
    assert not covariance(np.zeros((30, 1280)))['available']
    json.dumps(zero, allow_nan=False)


def test_temporal_real_time_slope_history_and_no_rewrite():
    temporal = TemporalAnalyzer(.6)
    first = temporal.observe(0, 5, .2, 'path-a')
    assert first['trend'] is None
    temporal.observe(5, 10, .3, 'path-a')
    result = temporal.observe(10, 15, .4, 'path-a')
    assert result['available']
    assert result['mean_30s'] == pytest.approx(.3)
    assert result['slope_probability_per_minute'] == pytest.approx(1.2)
    assert result['max_jump'] == pytest.approx(.1)
    assert result['coverage'] == 1.
    for i in range(3, 10):
        result = temporal.observe(i*5, (i+1)*5, .4, 'path-a')
    assert result['slots'] == 6 and result['analysis_duration_seconds'] == 30.
    assert len(first['history']) == 1 and first['current_probability'] == .2
    json.dumps(result, allow_nan=False)


def test_persistence_third_confirmation_invalid_gap_and_path_reset():
    temporal = TemporalAnalyzer(.6)
    temporal.observe(0, 5, .6, 'path-a')
    temporal.observe(5, 10, None, 'path-a')
    after_missing = temporal.observe(10, 15, .6, 'path-a')
    assert after_missing['max_jump'] is None and after_missing['consecutive_high_windows'] == 1
    assert not after_missing['events']
    temporal.observe(15, 20, .6, 'path-a')
    confirmed = temporal.observe(20, 25, .6, 'path-a')
    event = confirmed['events'][0]
    assert event['first_high_start_seconds'] == 10 and event['confirmation_seconds'] == 25
    assert not temporal.observe(25, 30, .7, 'path-a')['events']
    after_gap = temporal.observe(35, 40, .8, 'path-a')
    assert after_gap['valid_windows'] == 1 and after_gap['last_reset_reason'] == 'stream_gap'
    after_path = temporal.observe(40, 45, .8, 'path-b')
    assert after_path['valid_windows'] == 1 and after_path['last_reset_reason'] == 'path_switch'
    with pytest.raises(ValueError, match='out-of-order'):
        temporal.observe(39, 44, .8, 'path-b')


def test_policy_exact_frozen_boundaries_and_auxiliary_only():
    deep = {'available': True, 'probability_high': .6, 'threshold': .5}
    temporal = {'valid_windows': 3, 'mean_30s': .6, 'std_probability': .15, 'max_jump': .20}
    stable = analysis_policy('adaptive', deep, temporal)
    assert stable['tools'] == [] and stable['early_exit']
    assert not stable['changes_final_class']
    insufficient = analysis_policy('adaptive', deep, dict(temporal, valid_windows=2))
    assert insufficient['tools'] == ['BiomarkerCalculator']
    borderline = analysis_policy('adaptive', dict(deep, probability_high=.599), temporal)
    assert borderline['tools'] == ['BiomarkerCalculator']
    volatile = analysis_policy('adaptive', deep, dict(temporal, std_probability=.151))
    assert volatile['tools'] == ['BiomarkerCalculator']
    jumped = analysis_policy('adaptive', deep, dict(temporal, max_jump=.201))
    assert jumped['tools'] == ['BiomarkerCalculator', 'CovarianceAnalyzer']
    conflict = analysis_policy('adaptive', deep, dict(temporal, mean_30s=.49))
    assert conflict['tools'] == ['BiomarkerCalculator', 'CovarianceAnalyzer']
    assert analysis_policy('deep_only', deep, temporal)['tools'] == []
    assert len(analysis_policy('full_evidence', deep, temporal)['tools']) == 2
    assert len(analysis_policy('adaptive', deep, temporal, full_requested=True)['tools']) == 2
    assert analysis_policy('full_evidence', {'available': False}, temporal)['tools'] == []
    ids = [tool['id'] for tool in TOOL_CATALOG]
    assert len(set(ids)) == len(ids) and 'DeepVRMSDetector' in ids


def test_bm25_chinese_stable_citations_max_five_and_no_match(tmp_path):
    source = tmp_path / 'knowledge'
    source.mkdir()
    document = '资料版本：2.0\n证据类型：项目协议\n\n# 因果处理\n\n' + (
        '脑电因果滤波只处理已到达数据；频谱工具计算功率，EEGNet 模型分类。\n\n'*80)
    (source / 'protocol.md').write_text(document, encoding='utf-8')
    index = KnowledgeIndex({'knowledge_root': str(source), 'output_root': str(tmp_path / 'out')})
    summary = index.build()
    assert summary['chunks'] > 5
    result = index.search('脑电因果滤波 EEGNet', limit=99)
    assert result['status'] == 'ok' and len(result['results']) <= 5
    first = result['results'][0]
    assert first['citation_id'] == first['id']
    assert first['source'] == 'protocol.md' and first['section'] == '因果处理'
    assert first['version'] == '2.0' and len(first['source_sha256']) == 64
    assert all(len(chunk['text']) <= 600 for chunk in index.chunks)
    ids = [row['citation_id'] for row in result['results']]
    restored = KnowledgeIndex({'knowledge_root': str(source), 'output_root': str(tmp_path / 'out')})
    assert [row['citation_id'] for row in restored.search('脑电因果滤波 EEGNet')['results']] == ids
    assert index.search('xyzunmatchedterm')['status'] == 'insufficient'
    assert index.search('')['results'] == []
    json.dumps(result, allow_nan=False)
