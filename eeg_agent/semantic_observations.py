"""Anonymous, measured spectral changes for state-graph explanations.

The opening third is an exposure-path comparison, never a resting baseline.
Frozen model features, weights, and numeric decisions are not changed here.
"""
import hashlib
import json
import copy
import math
from pathlib import Path

import numpy as np


SCHEMA = 'eeg_observations_v1'
SCOPE = 'whole_path_first_last_third_no_rest'
CHANGE_THRESHOLD = float(np.log(1.1))
BAND_COLUMNS = (('theta', 1, (4, 8)), ('alpha', 2, (8, 13)), ('beta', 3, (13, 30)))


class SpectralObservations:
    """Reuse accepted-window powers; aggregate prefixes without another FFT."""
    def __init__(self):
        self.count = 0
        self._prefix = [np.zeros(3, dtype=np.float64)]
        self._source_hash = hashlib.sha256()
        self._sample_hash = hashlib.sha256()
        self._failure = None

    def add(self, window, powers, quality_valid):
        if quality_valid is not True:
            self._failure = 'window_quality_unavailable'
            return
        samples = np.asarray(window, dtype='<f4')
        if samples.shape != (30, 1280) or not np.isfinite(samples).all():
            self._failure = 'window_contract_unavailable'
            return
        if powers is None:
            self._failure = 'measured_band_powers_unavailable'
            return
        bands = np.asarray(powers, dtype='<f8')
        if bands.shape != (30, 4) or not np.isfinite(bands).all():
            self._failure = 'measured_band_powers_unavailable'
            return
        selected = bands[:, [column for _, column, _ in BAND_COLUMNS]]
        # Frozen features floor zero power at 1e-10; that floor is not evidence.
        if np.any(selected <= 1e-10):
            self._failure = 'insufficient_measured_band_power'
            return
        self._prefix.append(self._prefix[-1] + np.log(selected).mean(axis=0))
        self._sample_hash.update(samples.tobytes(order='C'))
        self._source_hash.update(bands.tobytes(order='C'))
        self.count += 1

    def result(self, complete=True):
        result = {'schema': SCHEMA, 'scope': SCOPE, 'items': [],
                  'accepted_windows': self.count, 'evidence_status': 'unavailable',
                  'fallback_reason': self._failure or 'fewer_than_three_accepted_windows',
                  'whole_path_complete': complete is True,
                  'resting_reference_available': False}
        if self._failure or self.count < 3:
            return result
        if complete is not True:
            result['fallback_reason'] = 'whole_path_not_complete'
            return result
        third = self.count // 3
        first = self._prefix[third] / third
        last = (self._prefix[-1] - self._prefix[self.count - third]) / third
        source_hash, sample_hash = self._source_hash.hexdigest(), self._sample_hash.hexdigest()
        for index, (name, _, band) in enumerate(BAND_COLUMNS):
            change = float(last[index] - first[index])
            item = {'id': 'observation:' + name + ':path_log_ratio',
                    'metric': 'band_power_log_ratio', 'biomarker': name,
                    'value': change, 'unit': 'log_ratio',
                    'direction': 'increase' if change > CHANGE_THRESHOLD else
                                 'decrease' if change < -CHANGE_THRESHOLD else 'stable',
                    'baseline': 'path_first_third', 'comparison': 'path_last_third',
                    'band_hz': list(band), 'band_boundary': 'left_closed_right_open',
                    'spatial_scope': 'all_30_eeg_channels',
                    'aggregation': 'mean_channel_log_power_last_third_minus_first_third',
                    'baseline_window_range': [0, third],
                    'comparison_window_range': [self.count - third, self.count],
                    'direction_threshold_log_ratio': CHANGE_THRESHOLD,
                    'source_layer': 'model_evidence', 'evidence_level': 'low',
                    'source_ref': 'actual-whole-path-eeg', 'source_sha256': source_hash,
                    'sample_sha256': sample_hash, 'version': SCHEMA}
            item['content_sha256'] = hashlib.sha256(json.dumps(
                item, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
            result['items'].append(item)
        result.update(evidence_status='available', fallback_reason=None,
                      source_sha256=source_hash, sample_sha256=sample_hash)
        return result


def observations_from_windows(windows, quality_valid, complete=True):
    """Summarize existing raw-worker inputs when the worker has no PSD output."""
    measured = SpectralObservations()
    array = np.asarray(windows, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (30, 1280) or not np.isfinite(array).all():
        measured._failure = 'window_contract_unavailable'
        return measured.result(complete)
    if quality_valid is not True:
        measured._failure = 'window_quality_unavailable'
        return measured.result(complete)
    # The frozen half-open definitions are also used by cached online powers.
    from .online.features import spectral_features
    for window in array:
        _, powers = spectral_features(window)
        measured.add(window, powers, quality_valid=True)
    return measured.result(complete)


def append_session_evidence(directory, evidence):
    """Append an already allowlisted record under the caller's session lock."""
    if not isinstance(evidence, dict) or evidence.get('schema') != 'session_model_evidence_v1':
        raise ValueError('Invalid session model evidence schema')
    encoded = json.dumps(evidence, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with (Path(directory) / 'model_evidence.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(encoded + '\n')


def validate_observations(value):
    """Validate provenance and return only known measured item fields."""
    if value is None:
        return []
    if not isinstance(value, dict) or value.get('schema') != SCHEMA or value.get('scope') != SCOPE:
        raise ValueError('Invalid EEG observations schema')
    count = value.get('accepted_windows')
    if type(count) is not int or count < 0 or value.get('resting_reference_available') is not False:
        raise ValueError('Invalid EEG observation scope')
    if value.get('evidence_status') == 'unavailable':
        if value.get('items') != [] or not isinstance(value.get('fallback_reason'), str):
            raise ValueError('Unavailable EEG observations contain measurements')
        return []
    if (value.get('evidence_status') != 'available' or count < 3 or
            value.get('whole_path_complete') is not True or value.get('fallback_reason') is not None):
        raise ValueError('Incomplete EEG observations')
    keys = {'id', 'metric', 'biomarker', 'value', 'unit', 'direction', 'baseline',
            'comparison', 'band_hz', 'band_boundary', 'spatial_scope', 'aggregation',
            'baseline_window_range', 'comparison_window_range', 'direction_threshold_log_ratio',
            'source_layer', 'evidence_level', 'source_ref', 'source_sha256', 'sample_sha256',
            'version', 'content_sha256'}
    bands = {name: list(band) for name, _, band in BAND_COLUMNS}
    items = value.get('items')
    if not isinstance(items, list) or len(items) != 3:
        raise ValueError('Invalid EEG observation items')
    found = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError('Invalid EEG observation fields')
        name = item['biomarker']
        if name not in bands or name in found:
            raise ValueError('Invalid EEG biomarker')
        found.add(name)
        number = item['value']
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError('Invalid EEG measured change')
        direction = ('increase' if number > CHANGE_THRESHOLD else
                     'decrease' if number < -CHANGE_THRESHOLD else 'stable')
        expected = {'id': 'observation:' + name + ':path_log_ratio',
                    'metric': 'band_power_log_ratio', 'unit': 'log_ratio', 'direction': direction,
                    'baseline': 'path_first_third', 'comparison': 'path_last_third',
                    'band_hz': bands[name], 'band_boundary': 'left_closed_right_open',
                    'spatial_scope': 'all_30_eeg_channels',
                    'aggregation': 'mean_channel_log_power_last_third_minus_first_third',
                    'baseline_window_range': [0, count // 3],
                    'comparison_window_range': [count - count // 3, count],
                    'direction_threshold_log_ratio': CHANGE_THRESHOLD,
                    'source_layer': 'model_evidence', 'evidence_level': 'low',
                    'source_ref': 'actual-whole-path-eeg', 'version': SCHEMA}
        if any(item[key] != expected_value for key, expected_value in expected.items()):
            raise ValueError('EEG observation provenance or comparison differs')
        for key in ('source_sha256', 'sample_sha256', 'content_sha256'):
            hash_value = item[key]
            if (not isinstance(hash_value, str) or len(hash_value) != 64 or
                    any(character not in '0123456789abcdef' for character in hash_value)):
                raise ValueError('Invalid EEG observation hash')
        if (item['source_sha256'] != value.get('source_sha256') or
                item['sample_sha256'] != value.get('sample_sha256')):
            raise ValueError('EEG observation source hash differs')
        unhashed = {key: content for key, content in item.items() if key != 'content_sha256'}
        expected_hash = hashlib.sha256(json.dumps(
            unhashed, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
        if item['content_sha256'] != expected_hash:
            raise ValueError('EEG observation content hash differs')
    return copy.deepcopy(items)
