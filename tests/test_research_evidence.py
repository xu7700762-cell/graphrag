"""Signal provenance, measurement correctness and gap handling without neural inference."""
import hashlib
import json
import math

import numpy as np
import pytest

from eeg_agent.config import EEG_CHANNELS
from eeg_agent.domain_tools import execute_tool, toolkit_catalog
from eeg_agent.research_evidence import read_path_evidence


@pytest.fixture
def frozen_signal(tmp_path):
    cache = tmp_path / 'outputs/vrms_pilot/v2_gpt_flow_20261008/cache'
    cache.mkdir(parents=True)
    t = np.arange(1280) / 256.
    wave = np.sin(2 * np.pi * 6 * t) + 2 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)
    x = np.tile(wave, (3, 30, 1))
    x[:, EEG_CHANNELS.index('F4'), :] *= math.sqrt(2)
    np.save(cache / 'windows.npy', x.astype(np.float32))
    paths = [{'path_index': index, 'window_start': 0, 'window_end': 3, 'accepted_windows': 3,
              'path_start_sec': 0., 'path_end_sec': 15., 'subject_key': 'PRIVATE_SUBJECT',
              'label': 1, 'score': 88.} for index in range(146)]
    records = [{'window_index': i, 'path_index': 0, 'start_sample': i * 5120, 'end_sample': (i + 1) * 5120,
                'qc': {'finite': True, 'max_ptp_uv': 20., 'nonflat_channels': 30}} for i in range(3)]
    (cache / 'window_evidence.jsonl').write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
    (cache / 'evaluation_manifest.json').write_text(json.dumps(paths), encoding='utf-8')
    (cache / 'dataset_audit.json').write_text(json.dumps({'accepted_windows': 3, 'channel_order': list(EEG_CHANNELS)}), encoding='utf-8')
    (cache / 'eligibility_audit.json').write_text('{}', encoding='utf-8')
    registered = tmp_path / 'outputs/vrms_agent_loso/v4_tools_r1_20261008'
    registered.mkdir(parents=True)
    (registered / 'local_paths.json').write_text(json.dumps(paths), encoding='utf-8')
    legacy = tmp_path / 'outputs/vrms_agent_seed_retest/seed2028_20261008/tools1'
    legacy.mkdir(parents=True)
    (legacy / 'local_paths.json').write_text(json.dumps(paths), encoding='utf-8')
    preflight = tmp_path / 'outputs/vrms_model_ablation/v3_20261008'
    preflight.mkdir(parents=True)

    def update_hashes():
        (preflight / 'preflight.json').write_text(json.dumps({'status': 'passed', 'cache_sha256': {
            name: hashlib.sha256((cache / name).read_bytes()).hexdigest() for name in
            ('windows.npy', 'evaluation_manifest.json', 'dataset_audit.json', 'eligibility_audit.json')}}), encoding='utf-8')

    update_hashes()
    return tmp_path, cache, registered, paths, records, update_hashes


@pytest.mark.parametrize('seed', [2026, 2028])
def test_selected_signal_drives_all_specialist_tools_without_models_or_labels(frozen_signal, monkeypatch, seed):
    def forbidden(*args, **kwargs):
        pytest.fail('A tools-only measurement must never invoke a neural classifier')
    monkeypatch.setattr('eeg_agent.models.build_model', forbidden)
    monkeypatch.setattr('eeg_agent.models.ONNXPredictor.__init__', forbidden)
    monkeypatch.setattr('eeg_agent.vrms_runtime.assess', forbidden)
    root, _, _, _, _, _ = frozen_signal
    evidence = read_path_evidence('path-000', seed=seed, root=root)
    assert evidence['measurement_context']['analyzed_seconds'] == 15
    for domain in ('fatigue', 'emotion'):
        for spec in toolkit_catalog()[domain]:
            result = execute_tool(spec['id'], evidence)
            assert result['available'] and not result['classifier_required']
            assert result['measurements']['observation']['accepted_windows'] == 3
    ratios = execute_tool('fatigue.spectral_ratios', evidence)['measurements']['all']['ratios']
    assert ratios['theta_alpha'] == pytest.approx(.25)
    assert ratios['beta_alpha'] == pytest.approx(.25)
    faa = execute_tool('emotion.frontal_asymmetry', evidence)['measurements']['faa']
    assert faa == pytest.approx(math.log(2), abs=1e-7)
    temporal = execute_tool('emotion.temporal_stability', evidence)['measurements']
    assert temporal['alpha']['coefficient_of_variation'] == pytest.approx(0.)
    serialized = json.dumps(evidence, allow_nan=False)
    assert all(term not in serialized for term in ('PRIVATE_SUBJECT', 'subject_key', '"score"', '"label"', '"windows"', 'logits'))


def test_modified_signal_is_rejected_after_successful_first_read(frozen_signal):
    root, cache, _, _, _, _ = frozen_signal
    read_path_evidence('path-000', root=root)
    with (cache / 'windows.npy').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        read_path_evidence('path-000', root=root)


def test_rejected_interval_breaks_temporal_evidence_without_discarding_good_spectral_measurements(frozen_signal):
    root, cache, registered, paths, records, update_hashes = frozen_signal
    for path in paths:
        path['path_end_sec'] = 25.
    (cache / 'evaluation_manifest.json').write_text(json.dumps(paths), encoding='utf-8')
    (registered / 'local_paths.json').write_text(json.dumps(paths), encoding='utf-8')
    for i in (1, 2):
        records[i]['start_sample'] += 5120
        records[i]['end_sample'] += 5120
    (cache / 'window_evidence.jsonl').write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
    update_hashes()
    evidence = read_path_evidence('path-000', root=root)
    assert len(evidence['arrived']['fatigue_history']) == 2
    assert execute_tool('fatigue.spectral_ratios', evidence)['available']
    assert execute_tool('emotion.frontal_asymmetry', evidence)['available']
    assert not execute_tool('fatigue.temporal_evidence', evidence)['available']
    assert not execute_tool('emotion.temporal_stability', evidence)['available']


def test_wrong_channel_order_cannot_be_used_to_compute_faa(frozen_signal):
    root, cache, _, _, _, update_hashes = frozen_signal
    channels = list(EEG_CHANNELS)
    channels.reverse()
    (cache / 'dataset_audit.json').write_text(json.dumps({'accepted_windows': 3, 'channel_order': channels}), encoding='utf-8')
    update_hashes()
    with pytest.raises(ValueError, match='channel contract'):
        read_path_evidence('path-000', root=root)
