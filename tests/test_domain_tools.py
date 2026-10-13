import json
import math

import pytest

from eeg_agent.domain_tools import execute_tool, toolkit_catalog


def spectral_evidence():
    return {'arrived': {'quality': {'valid': True},
                        'biomarkers': {'available': True, 'baseline_available': False,
                                       'regions': {'frontal': {'absolute_power_uv2':
                                                              {'delta': 2., 'theta': 4., 'alpha': 2., 'beta': 1.}}}}}}


def test_catalog_and_actual_spectral_formulas():
    catalog = toolkit_catalog()
    assert set(catalog) == {'vrms', 'fatigue', 'emotion'}
    ids = [spec['id'] for specs in catalog.values() for spec in specs]
    assert len(ids) == len(set(ids)) == 6
    result = execute_tool('fatigue.spectral_ratios', spectral_evidence())
    assert result['available']
    assert result['measurements']['frontal']['ratios'] == {'theta_alpha': 2., 'theta_beta': 4., 'beta_alpha': .5, 'slow_fast': 2.}
    assert result['classifier_available'] is False and result['predicted_class'] is None


def test_zero_denominators_and_rejected_quality_are_not_predictions():
    evidence = spectral_evidence()
    evidence['arrived']['biomarkers']['regions']['frontal']['absolute_power_uv2'].update(alpha=0., beta=0.)
    result = execute_tool('fatigue.spectral_ratios', evidence)
    assert not result['available']
    assert all(value is None for value in result['measurements']['frontal']['ratios'].values())
    evidence = spectral_evidence()
    evidence['arrived']['quality']['valid'] = False
    assert not execute_tool('fatigue.spectral_ratios', evidence)['available']
    assert not execute_tool('emotion.frontal_asymmetry', evidence)['available']
    evidence['arrived']['quality'] = {}
    assert not execute_tool('fatigue.spectral_ratios', evidence)['available']


def test_faa_requires_actual_f3_f4_and_has_explicit_sign():
    evidence = spectral_evidence()
    assert not execute_tool('emotion.frontal_asymmetry', evidence)['available']
    evidence['arrived']['biomarkers'].update(channels=['F3', 'Fz', 'F4'], absolute_power_uv2={'alpha': [2., 100., 4.]})
    result = execute_tool('emotion.frontal_asymmetry', evidence)
    assert result['available'] and result['measurements']['faa'] == pytest.approx(math.log(2.))
    assert result['measurements']['formula'] == 'ln(alpha_F4) - ln(alpha_F3)'
    assert result['classifier_available'] is False and result['predicted_class'] is None
    evidence['arrived']['biomarkers']['absolute_power_uv2']['alpha'][2] = float('nan')
    result = execute_tool('emotion.frontal_asymmetry', evidence)
    assert not result['available'] and 'F4.alpha' in result['missing']
    json.dumps(result, allow_nan=False)


def test_vrms_probability_is_not_fatigue_temporal_evidence():
    result = execute_tool('fatigue.temporal_evidence', {'arrived': {'quality': {'valid': True}, 'temporal': {'valid_windows': 6, 'slope': .3}}})
    assert not result['available'] and result['measurements'] == {}


def test_spectral_history_trend_is_arrived_and_time_grounded():
    evidence = {'source_time_seconds': 15., 'arrived': {'quality': {'valid': True}, 'fatigue_history': [
        {'source_time_seconds': 5., 'theta_alpha': 1.},
        {'source_time_seconds': 10., 'theta_alpha': 2.},
        {'source_time_seconds': 15., 'theta_alpha': 3.}]}}
    result = execute_tool('fatigue.temporal_evidence', evidence)
    assert result['available']
    assert result['measurements']['theta_alpha']['slope_ratio_per_minute'] == pytest.approx(12.)
    assert result['predicted_class'] is None
    evidence['source_time_seconds'] = 10.
    assert not execute_tool('fatigue.temporal_evidence', evidence)['available']
    evidence['source_time_seconds'] = 15.
    evidence['arrived']['fatigue_history'][1]['source_time_seconds'] = 5.
    assert not execute_tool('fatigue.temporal_evidence', evidence)['available']


def test_tools_exclude_raw_data_identity_and_unknown_calls():
    evidence = {'arrived': {'deep': {'available': True, 'probability_high': .7,
                                    'threshold': .6, 'logits': [1, 2], 'raw': [1, 2]},
                            'subject': 'private subject'}, 'paths': [], 'raw': [[999]]}
    result = execute_tool('emotion.read_evidence', evidence)
    assert not result['available'] and not result['classifier_available']
    assert result['measurements'] == {}
    encoded = json.dumps(result, allow_nan=False)
    assert 'private subject' not in encoded and 'logits' not in encoded and '999' not in encoded
    with pytest.raises(ValueError, match='Unregistered'):
        execute_tool('emotion.hidden_tool', evidence)




def test_retired_pipeline_is_unregistered_and_raw_model_requires_waveform_context():
    evidence = {'model': {'model_id': 'eegnet'}, 'arrived': {'deep': {'probability_high': .7}}}
    ids = {spec['id'] for specs in toolkit_catalog(include_recordings=True).values() for spec in specs}
    assert 'vrms.femba_pipeline' not in ids and 'vrms.raw_model' in ids
    with pytest.raises(ValueError, match='Unregistered'):
        execute_tool('vrms.femba_pipeline', evidence)
    with pytest.raises(ValueError, match='Raw recording/model context required'):
        execute_tool('vrms.raw_model', evidence)
