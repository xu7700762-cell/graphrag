"""CPU scan, frozen-asset integrity, bounded prefixes and cache reuse contracts."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from eeg_agent.online.assets import Assets, sha
from eeg_agent.online.minimal_mamba import MambaBlock, ModelArgs, UPSTREAM_COMMIT
from eeg_agent.online.model import OnlineModel, OnlinePathModel, TemporalEncoder


def test_scan_matches_scalar_recurrence_and_resets_each_call():
    u = torch.tensor([[[2., 3.], [4., 5.]]])
    dt = torch.tensor([[[.1, .2], [.3, .4]]])
    A = torch.tensor([[-1.], [-2.]])
    B = torch.tensor([[[.7], [.9]]])
    C = torch.tensor([[[.5], [.8]]])
    D = torch.tensor([.2, .3])
    expected = np.zeros((1, 2, 2), np.float32)
    for channel in range(2):
        state = 0.
        for step in range(2):
            state = float(torch.exp(dt[0, step, channel] * A[channel, 0])) * state
            state += float(dt[0, step, channel] * B[0, step, 0] * u[0, step, channel])
            expected[0, step, channel] = state * float(C[0, step, 0]) + float(u[0, step, channel] * D[channel])
    actual = MambaBlock.selective_scan(u, dt, A, B, C, D)
    assert np.allclose(actual.numpy(), expected, atol=1e-7, rtol=1e-6)
    assert torch.equal(actual, MambaBlock.selective_scan(u, dt, A, B, C, D))


def test_depthwise_convolution_uses_only_prior_tokens():
    torch.manual_seed(27)
    block = MambaBlock(ModelArgs(d_model=8, expand=2, dt_rank=1)).eval()
    left = torch.randn(1, 8, 8)
    changed = left.clone(); changed[:, 4:] += 100
    with torch.inference_mode():
        assert torch.equal(block(left)[:, :4], block(changed)[:, :4])


def test_encoder_retains_layernorm_and_all_frozen_keys():
    encoder = TemporalEncoder()
    assert len(encoder.state_dict()) == 83
    assert sum(p.numel() for p in encoder.parameters()) == 28540155
    assert all(isinstance(norm, torch.nn.LayerNorm) and norm.bias is not None for norm in encoder.norm_layers)
    assert all(block.mamba_fwd is not block.mamba_rev for block in encoder.mamba_blocks)


def test_asset_tampering_and_path_escape_are_rejected(tmp_path):
    file = tmp_path / 'encoder.pt'; file.write_bytes(b'trusted')
    manifest = {'schema': 'online_cpu_seed2026_v1', 'seed': 2026, 'dtype': 'float32',
                'baseline_policy': 'disabled', 'folds': {},
                'encoder': {'file': 'encoder.pt', 'sha256': sha(file)}}
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(manifest))
    assets = Assets(path)
    assert assets.item(None, 'encoder')[0] == file
    file.write_bytes(b'changed-size')
    with pytest.raises(ValueError, match='hash differs'):
        assets.item(None, 'encoder')
    manifest['encoder']['file'] = '../outside.pt'; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='inside'):
        Assets(path).item(None, 'encoder')


def test_prefix_does_not_reencode_or_reextract_prior_windows(monkeypatch):
    module = importlib.import_module('eeg_agent.online.model')
    counts = {'encode': 0, 'extract': 0, 'numeric': 0}

    class Encoder:
        def encode_windows(self, x):
            counts['encode'] += 1
            return torch.ones(1, 525) * x.mean()

    class Numeric:
        hashes = {'numeric_first_sha256': 'a', 'numeric_second_sha256': 'b', 'numeric_selector_sha256': 'c'}

        def predict(self, cached, features, probability):
            counts['numeric'] += 1
            assert len(cached) == len(features) >= 3
            return {'predicted_class': 'Low', 'raw_combined_score': .4, 'actual_tools': []}

    def extract(window):
        counts['extract'] += 1
        return {'absolute': np.ones(240)}

    monkeypatch.setattr(module, 'cache_window', extract)
    owner = SimpleNamespace(lock=threading.RLock(), encoder=Encoder(), encoder_sha256='encoder')
    bundle = {'fold': 1, 'mean': torch.zeros(1, 30, 1), 'scale': torch.ones(1, 30, 1),
              'head': lambda x: x.mean(), 'numeric': Numeric(),
              'model_sha256': 'head', 'normalization_sha256': 'norm'}
    path = OnlinePathModel(owner, bundle)
    assert path.result()['available'] is False
    for index in range(3):
        result = path.add(np.ones((30, 1280), np.float32) * index)
        assert result['available'] is (index == 2)
        assert result['probability_calibrated'] is False
        assert result['accepted_windows'] == index + 1
    saved = path.result(); saved['accepted_windows'] = 999
    assert path.result()['accepted_windows'] == 3
    assert counts == {'encode': 3, 'extract': 3, 'numeric': 1}
    with pytest.raises(ValueError, match='finite complete'):
        path.add(np.ones((1280, 30)))
    bad = np.ones((30, 1280)); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite complete'):
        path.add(bad)
    assert counts['encode'] == 3


def test_actual_assets_are_cpu_only_and_do_not_replace_unknown_folds(monkeypatch):
    manifest = Path(__file__).resolve().parents[1] / 'outputs/online_assets/seed2026/manifest.json'
    if not manifest.exists():
        pytest.skip('Private frozen model bundle is not part of public source')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CPU runtime consulted CUDA'))
    owner = OnlineModel(manifest)
    assert owner._fold is None
    assert all(p.device.type == 'cpu' and p.dtype == torch.float32 and not p.requires_grad
               for p in owner.encoder.parameters())
    with pytest.raises(ValueError, match='No frozen held-out'):
        owner.start_path(22)
    path = owner.start_path(1)
    assert owner.start_path(1).bundle is path.bundle
    assert owner.assets.value['mamba_minimal_commit'] == UPSTREAM_COMMIT
    assert len(owner.assets.value['folds']) == 24
    assert path.result()['available'] is False


def test_shared_welch_preserves_original_feature_boundaries_without_fft(monkeypatch):
    from eeg_agent.online import features
    from eeg_agent.tools import _spectral
    rng = np.random.default_rng(81)
    window = rng.normal(size=(30, 1280)).astype(np.float32)
    original = features.spectral_features(window)
    frequencies, psd, _ = _spectral(window)
    monkeypatch.setattr(features.signal, 'welch', lambda *a, **k: pytest.fail('Shared PSD recomputed Welch'))
    shared = features.spectral_features(window, spectrum=(frequencies, psd))
    assert np.array_equal(original[0], shared[0])
    assert np.array_equal(original[1], shared[1])
    assert np.array_equal(shared[0], features.spectral_features(window, spectrum=(frequencies, psd.T))[0])


def test_shared_spectrum_preserves_actual_eight_tool_scores():
    from eeg_agent.tools import _spectral
    manifest = Path(__file__).resolve().parents[1] / 'outputs/online_assets/seed2026/manifest.json'
    if not manifest.exists():
        pytest.skip('Private frozen model bundle is not part of public source')
    owner = OnlineModel(manifest)
    left, right = owner.start_path(1), owner.start_path(1)
    rng = np.random.default_rng(92)
    for _ in range(3):
        window = rng.normal(size=(30, 1280)).astype(np.float32)
        frequency, psd, _ = _spectral(window)
        left.add(window)
        right.add_shared(window, frequency, psd)
    a, b = left.result(), right.result()
    assert a['mil_raw_high_probability'] == b['mil_raw_high_probability']
    assert a['numeric_fusion'] == b['numeric_fusion']


def test_evaluator_aligns_fixed_raw_csv_offset_without_loading_a_model(tmp_path, monkeypatch):
    from scripts import validate_online_model as evaluator
    folder = tmp_path / 'data/labels'; folder.mkdir(parents=True)
    csv = ('subject_id,start_sample_1024,end_sample_1024,is_complete,path_score_available,path_score,duration_sec\n'
           '1,100,51300,true,true,999,50\n'
           '1,60100,111300,true,true,999,50\n'
           '1,120100,171333,true,true,999,50\n')
    (folder / 'task_segments_with_path_scores.csv').write_text(csv)
    monkeypatch.setattr(evaluator, 'metadata', lambda path: {
        'actual_samples': 171300, 'n_channels': 1, 'channels': ['Trigger']})
    monkeypatch.setattr(evaluator.np, 'memmap', lambda *a, **k: np.zeros((171300, 1), np.float32))
    monkeypatch.setattr(evaluator, 'trigger_events', lambda *a: [])
    monkeypatch.setattr(evaluator, 'state_segments_from_events', lambda *a: (
        [{'start_sample': 80, 'end_sample': 51280, 'complete': True},
         {'start_sample': 60080, 'end_sample': 111280, 'complete': True},
         {'start_sample': 120080, 'end_sample': 160000, 'complete': False}], [], []))
    monkeypatch.setattr(evaluator, 'OnlineModel', lambda *a: pytest.fail('Boundary evaluator loaded a model'))
    selected, _, alignment = evaluator.eligible_boundaries(tmp_path)
    assert selected == {(1, 80, 51280), (1, 60080, 111280)}
    assert alignment['csv_candidates'] == 3
    assert alignment['offsets']['1']['offset_samples'] == -20
    assert len(alignment['excluded_csv_candidates']) == 1
