import json
from pathlib import Path
import numpy as np
import pytest
from eeg_agent.config import contract, digest, load_config, write_json
from eeg_agent.models import build_model, model_catalog, model_directory, ONNXPredictor, probabilities_from_logits
from eeg_agent.learning import (training_arrays, calibrate_temperature, select_threshold, metrics,
                                evaluate_records, export_onnx, load_cache)


def row(subject, label, count=4, duration=20, value=1):
    return {'subject': subject, 'key': '%02d:100' % subject, 'label': label, 'raw_complete': True,
            'start': 100, 'end': 100 + int(duration * 1024), 'duration': duration,
            'x0': np.full((count, 30, 1280), value, dtype=np.float32),
            'x1': np.full((count, 30, 1280), 99999, dtype=np.float32),
            'valid': np.ones(count, dtype=bool), 'valid1': np.ones(count, dtype=bool),
            'p1': np.ones(count, dtype=bool)}


@pytest.mark.parametrize('model_id,parameters', [('eegnet', 2866), ('eegconformer', 986626)])
def test_real_architectures_random_initialization_and_gradients(model_id, parameters):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model(model_id)
    assert sum(p.numel() for p in model.parameters()) == parameters
    x = torch.randn(2, 1, 30, 1280)
    logits = model(x)
    assert logits.shape == (2, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.manual_seed(42)
    fresh = build_model(model_id)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), fresh.parameters()))


def test_training_weights_and_normalization_use_only_valid_complete_P0():
    rows = [row(2, 0, count=2, value=1), row(3, 0, count=4, value=3), row(4, 1, count=3, value=5)]
    invalid = row(5, 1, value=10000)
    invalid['valid'][:] = False
    incomplete = row(6, 1, value=20000)
    incomplete['raw_complete'] = False
    rows.extend([invalid, incomplete])
    x, y, weights, mean, std, contributing = training_arrays(rows)
    assert len(x) == 9 and len(contributing) == 3
    assert np.allclose(mean, (2 * 1 + 4 * 3 + 3 * 5) / 9)
    assert np.isclose(weights[:2].sum(), weights[2:6].sum())
    assert np.isclose(weights[y == 0].sum(), weights[y == 1].sum())
    changed = [{**r, 'x1': r['x1'] * -100} for r in rows]
    other = training_arrays(changed)
    assert np.array_equal(x, other[0]) and np.array_equal(mean, other[3])
    assert np.all(std > 0) and np.isfinite(x).all()


def logits_for(probability):
    return np.tile([0., np.log(probability / (1 - probability))], (3, 1))


def test_calibration_minimum_and_path_nll():
    insufficient = [{'label': i % 2, 'logits': logits_for(.8)} for i in range(9)]
    assert calibrate_temperature(insufficient)['temperature'] == 1
    single_class = [{'label': 1, 'logits': logits_for(.8)} for _ in range(10)]
    assert calibrate_temperature(single_class)['status'] == 'unavailable'
    paths = [{'label': i % 2, 'logits': np.array([[0., 5.], [0., -3.], [0., 1.]])} for i in range(12)]
    fit = calibrate_temperature(paths)
    assert fit['status'] == 'fitted' and .05 <= fit['temperature'] <= 20
    assert fit['calibrated_NLL'] <= fit['uncalibrated_NLL'] + 1e-10
    p = np.array([probabilities_from_logits(r['logits'], fit['temperature']).mean() for r in paths])
    y = np.array([r['label'] for r in paths])
    expected = -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))
    assert np.isclose(fit['calibrated_NLL'], expected)


def test_threshold_ties_prefer_nearest_half_then_smaller():
    # Both all-high and all-low yield BACC=.5; .5 itself yields BACC=0.
    paths = [{'label': 1, 'logits': logits_for(.495)}, {'label': 0, 'logits': logits_for(.505)}]
    result = select_threshold(paths)
    assert result['threshold'] == .49
    assert len(result['search']) == 81 and result['search'][0]['threshold'] == .1 and result['search'][-1]['threshold'] == .9
    perfect = [{'label': 0, 'logits': logits_for(.2)}, {'label': 1, 'logits': logits_for(.8)}]
    assert select_threshold(perfect)['threshold'] == .5


def test_frozen_threshold_coverage_rejection_and_probability_metrics():
    cfg = load_config()
    rows = [row(1, 0, value=.3), row(9, 1, value=.45), row(12, 1, count=2, duration=20)]
    result = evaluate_records(rows, lambda x: x[:, 0, 0], cfg, threshold=.4)
    assert [p['predicted_class'] for p in result['paths']] == [0, 1, None]
    measures = result['metrics']
    assert measures['BACC'] == 1 and measures['AUROC'] == 1
    assert measures['accepted_paths'] == 2 and measures['rejected_paths'] == 1
    assert measures['class_acceptance']['1']['acceptance_rate'] == .5
    assert measures['rejection_reasons'] == {'insufficient': 1}
    assert np.isclose(measures['Brier'], (.3 ** 2 + (.45 - 1) ** 2) / 2)
    assert np.isclose(measures['ECE_10_equal_width'], (.3 + .55) / 2)
    assert sum(bin['count'] for bin in measures['reliability_bins']) == 2
    with pytest.raises(ValueError, match='P0 only'):
        evaluate_records(rows, lambda x: x[:, 0, 0], cfg, mode='fixed_p1')


def test_cache_keeps_scalar_path_boundaries(monkeypatch, tmp_path):
    import eeg_agent.learning as learning
    cfg = load_config()
    cfg['output_root'] = str(tmp_path)
    cfg['data_root'] = str(tmp_path)
    labels = tmp_path / 'data/labels/task_segments_with_path_scores.csv'
    labels.parent.mkdir(parents=True)
    labels.write_text('fixture', encoding='utf-8')
    item = {'subject': 8, 'start': 100, 'end': 20600, 'key': '08:100', 'label': 0, 'score': 10., 'duration': 20., 'raw_complete': True}
    cache = tmp_path / 'cache'
    cache.mkdir()
    arrays = {'x0': np.zeros((4, 30, 1280)), 'valid': np.ones(4, dtype=bool),
              'start': 100 + np.arange(4) * 5120, 'end': 100 + (np.arange(4) + 1) * 5120,
              'path': np.full(4, 100)}
    np.savez(cache / 'subject_08.npz', **arrays)
    monkeypatch.setattr(learning, 'SPLITS', {'validation': [8]})
    monkeypatch.setattr(learning, 'paths', lambda c: [item])
    from eeg_agent.config import file_hash
    write_json(cache / 'manifest.json', {'contract_hash': digest(contract(cfg)), 'label_sha256': file_hash(labels),
                                       'subjects': [{'subject': 8, 'paths': [item], 'cache_sha256': file_hash(cache / 'subject_08.npz')}]})
    records, _ = load_cache(cfg, 'validation')
    result = records[0]
    assert result['start'] == result['path_start'] == 100
    assert result['end'] == result['path_end'] == 20600
    assert np.array_equal(result['window_start'], arrays['start'])
    assert np.array_equal(result['window_end'], arrays['end'])


@pytest.mark.parametrize('model_id', ['eegnet', 'eegconformer'])
def test_onnx_and_frozen_prediction_contract(tmp_path, model_id):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = load_config()
    cfg['output_root'] = str(tmp_path)
    directory = model_directory(cfg, model_id)
    directory.mkdir(parents=True)
    model = build_model(model_id)
    torch.save({'state_dict': model.state_dict(), 'model_id': model_id, 'contract_hash': digest(contract(cfg))}, directory / 'checkpoint.pt')
    manifest = {'model_id': model_id, 'contract_hash': digest(contract(cfg)), 'allowed_profiles': ['P0'],
                'mean': [0.] * 30, 'std': [10.] * 30, 'temperature': 2., 'threshold': .4, 'software': {}}
    write_json(directory / 'contract.json', manifest)
    # Fixture verifies the export mechanism. Full training separately supplies
    # audited real CDT windows, whose provenance is in the experiment manifest.
    real_fixture = np.random.default_rng(42).normal(0, 10, (2, 30, 1280)).astype(np.float32)
    exported = export_onnx(cfg, directory, model_id=model_id, real_windows=real_fixture)
    assert exported['max_abs_error'] <= 1e-4
    predictor = ONNXPredictor(cfg, model_id=model_id)
    details = predictor.predict_details(real_fixture[0].T)
    assert details['model_id'] == model_id and predictor.profiles == ['P0']
    assert details['threshold'] == .4 and details['temperature'] == 2.
    assert np.isclose(details['probability_high'], predictor.predict_batch(real_fixture[:1])[0])
    catalog = model_catalog(cfg)
    entry = next(item for item in catalog if item['model_id'] == model_id)
    assert entry['available'] and entry['status'] == 'exported' and entry['device']['pi'] == 'not_tested'
    with pytest.raises(ValueError, match='Invalid model input'):
        predictor.predict(np.zeros((30, 1280)))
    wrong = {**cfg, 'warmup_seconds': cfg['warmup_seconds'] + 1}
    with pytest.raises(ValueError, match='contract mismatch'):
        ONNXPredictor(wrong, model_id=model_id)
    with pytest.raises(ValueError, match='Invalid checkpoint_id'):
        model_directory(cfg, model_id, '../escape')
    frozen = json.loads((directory / 'contract.json').read_text(encoding='utf-8'))
    tampered = {**frozen, 'temperature': 1.5}
    write_json(directory / 'contract.json', tampered)
    with pytest.raises(ValueError, match='metadata hash mismatch'):
        ONNXPredictor(cfg, model_id=model_id)
    assert not next(item for item in model_catalog(cfg) if item['model_id'] == model_id)['available']
    write_json(directory / 'contract.json', frozen)
    changed_policy = {**cfg, 'minimum_path_coverage': .9}
    with pytest.raises(ValueError, match='publication contract mismatch'):
        ONNXPredictor(changed_policy, model_id=model_id)
    with (directory / 'model.onnx').open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(ValueError, match='hash mismatch'):
        ONNXPredictor(cfg, model_id=model_id)
