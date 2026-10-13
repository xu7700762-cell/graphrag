"""Frozen FEMBA + MIL and eight tools on CPU FP32, independent of offline code."""
import copy
import threading
import time

import numpy as np
import torch
from torch import nn

from .assets import Assets
from .features import cache_window
from .minimal_mamba import MambaBlock, ModelArgs, UPSTREAM_COMMIT
from .numeric import FrozenNumeric


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 35, kernel_size=(2, 16), stride=(2, 16))

    def forward(self, windows):
        value = self.proj(windows.unsqueeze(1))
        return value.reshape(value.shape[0], 525, 80).transpose(1, 2)


class BidirectionalMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba_fwd = MambaBlock(ModelArgs())
        self.mamba_rev = MambaBlock(ModelArgs())

    def forward(self, tokens):
        return self.mamba_fwd(tokens) + self.mamba_rev(tokens.flip(1)).flip(1)


class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, 80, 525))
        self.mamba_blocks = nn.ModuleList([BidirectionalMamba() for _ in range(4)])
        # The frozen EEG model uses LayerNorm, not mamba-minimal's RMSNorm.
        self.norm_layers = nn.ModuleList([nn.LayerNorm(525) for _ in range(4)])

    def forward_tokens(self, windows):
        if windows.ndim != 3 or tuple(windows.shape[1:]) != (30, 1280):
            raise ValueError('Encoder requires complete 30 x 1280 EEG windows')
        tokens = self.patch_embed(windows) + self.pos_embed
        for block, norm in zip(self.mamba_blocks, self.norm_layers):
            tokens = norm(tokens + block(tokens))
        return tokens

    def encode_windows(self, windows):
        return self.forward_tokens(windows).mean(dim=1)


class PathMIL(nn.Module):
    def __init__(self, input_dim=525):
        super().__init__()
        self.register_buffer('center', torch.zeros(input_dim))
        self.register_buffer('scale', torch.ones(input_dim))
        self.projection = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU())
        self.attention = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.classifier = nn.Linear(64, 1)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 525 or not len(features):
            raise ValueError('MIL requires a nonempty 525-dimensional window bag')
        z = self.projection(((features.float() - self.center) / self.scale).clamp(-20, 20))
        attention = self.attention(z).softmax(dim=0)
        return self.classifier((attention * z).sum(dim=0)).squeeze(-1)


def load_encoder_state(encoder, checkpoint):
    original = checkpoint.get('state_dict', checkpoint)
    state = {}
    for key, value in original.items():
        name = key[6:] if key.startswith('model.') else key
        name = name.replace('.mamba_fwd.core.', '.mamba_fwd.').replace('.mamba_rev.core.', '.mamba_rev.')
        if name in state:
            raise ValueError('Duplicate frozen encoder key after mapping')
        state[name] = value
    if len(state) != 83 or len(encoder.state_dict()) != 83:
        raise ValueError('Frozen encoder requires exactly 83 tensors')
    if any(not torch.is_tensor(t) or t.dtype != torch.float32 or not torch.isfinite(t).all()
           for t in state.values()):
        raise ValueError('Frozen encoder tensors must be finite FP32')
    encoder.load_state_dict(state, strict=True)
    if sum(t.numel() for t in encoder.parameters()) != 28540155:
        raise ValueError('Frozen encoder parameter count differs')
    encoder.eval().requires_grad_(False)


class OnlineModel:
    def __init__(self, manifest_path, threads=2):
        if type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError('CPU thread count must be an integer from one to four')
        torch.set_num_threads(threads)
        self.threads, self.lock = threads, threading.RLock()
        self.assets = Assets(manifest_path)
        source = self.assets.value.get('mamba_minimal_commit')
        if source != UPSTREAM_COMMIT:
            raise ValueError('Mamba implementation commit differs from the frozen bundle')
        path, self.encoder_sha256 = self.assets.item(None, 'encoder')
        self.encoder = TemporalEncoder().float().cpu()
        load_encoder_state(self.encoder, torch.load(path, map_location='cpu', weights_only=True))
        # Only one lazily loaded fold is retained by the owner. Existing paths
        # keep their own fold reference until the replay service releases them.
        self._fold = None

    def start_path(self, fold):
        if type(fold) is not int or str(fold) not in self.assets.value['folds']:
            raise ValueError('No frozen held-out model for this subject; no replacement is allowed')
        with self.lock:
            if self._fold is None or self._fold['fold'] != fold:
                head_file, head_hash = self.assets.item(fold, 'head')
                norm_file, norm_hash = self.assets.item(fold, 'normalization')
                state = torch.load(head_file, map_location='cpu', weights_only=True)
                normalization = torch.load(norm_file, map_location='cpu', weights_only=True)
                if (state.get('seed') != 2026 or state['input_dim'] != 525
                        or state['split']['outer_subject'] != fold
                        or normalization['heldout_subject'] != fold
                        or fold in normalization['training_subjects']):
                    raise ValueError('Wrong frozen head or normalization split')
                for role in ('base_train', 'head_fit', 'probability_calibration', 'policy_validation'):
                    if fold in state['split'].get(role, []):
                        raise ValueError('Query identity overlaps model fitting')
                head = PathMIL().float().cpu()
                head.load_state_dict(state['state_dict'], strict=True)
                head.eval().requires_grad_(False)
                mean, scale = normalization['mean'].float(), normalization['scale'].float()
                if (tuple(mean.shape) != (1, 30, 1) or tuple(scale.shape) != (1, 30, 1)
                        or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
                        or torch.any(scale <= 0)):
                    raise ValueError('Invalid frozen input normalization')
                if (not torch.isfinite(head.center).all() or not torch.isfinite(head.scale).all()
                        or torch.any(head.scale <= 0)):
                    raise ValueError('Invalid frozen MIL normalization')
                numeric = FrozenNumeric(fold, self.assets)
                self._fold = {'fold': fold, 'head': head, 'mean': mean, 'scale': scale,
                              'numeric': numeric, 'model_sha256': head_hash,
                              'normalization_sha256': norm_hash}
            return OnlinePathModel(self, self._fold)


class OnlinePathModel:
    def __init__(self, owner, bundle):
        self.owner, self.bundle = owner, bundle
        self.features, self.cached = [], []
        self._result = None
        self.encoding_seconds = []

    def add(self, window30x1280):
        return self._add(window30x1280)

    def add_shared(self, window30x1280, frequencies, psd):
        """Reuse the same window's Welch PSD, preserving each tool's band edges."""
        return self._add(window30x1280, spectrum=(frequencies, psd))

    def _add(self, window30x1280, spectrum=None):
        window = np.asarray(window30x1280, dtype=np.float32)
        if window.shape != (30, 1280) or not np.isfinite(window).all():
            raise ValueError('Online model requires a finite complete 30 x 1280 EEG window')
        if len(self.features) >= 10000:
            raise ValueError('Path exceeds the frozen input limit')
        with self.owner.lock, torch.inference_mode():
            start = time.perf_counter()
            normalized = ((torch.from_numpy(window.copy())[None] - self.bundle['mean'])
                          / self.bundle['scale']).clamp(-20, 20)
            encoded = self.owner.encoder.encode_windows(normalized)[0].numpy().copy()
            cached = cache_window(window) if spectrum is None else cache_window(window, spectrum=spectrum)
            if not np.isfinite(encoded).all():
                raise ValueError('CPU encoder returned nonfinite features')
            self.features.append(encoded)
            self.cached.append(cached)
            self.encoding_seconds.append(time.perf_counter() - start)
            self._result = None
            return self.result()

    def result(self):
        with self.owner.lock, torch.inference_mode():
            if self._result is None:
                count = len(self.features)
                result = {'available': count >= 3, 'accepted_windows': count, 'fold': self.bundle['fold'],
                          'fresh_encoded_windows': count, 'fresh_raw_encoding': True,
                          'backend': 'mamba_minimal_cpu_fp32', 'device': 'cpu', 'dtype': 'float32',
                          'probability_calibrated': False, 'score_is_calibrated': False,
                          'prefix_provisional': True, 'classification_scope': 'whole_path_prefix',
                          'mil_raw_high_probability': None, 'mil_class': None, 'threshold': .5,
                          'numeric_fusion': None, 'numeric_available': False,
                          'baseline_policy': 'disabled', 'encoder_sha256': self.owner.encoder_sha256,
                          'model_sha256': self.bundle['model_sha256'],
                          'normalization_sha256': self.bundle['normalization_sha256'],
                          **self.bundle['numeric'].hashes}
                if count:
                    features = np.asarray(self.features, np.float32)
                    probability = float(self.bundle['head'](torch.from_numpy(features)).sigmoid())
                    result.update(mil_raw_high_probability=probability,
                                  mil_class='High' if probability >= .5 else 'Low')
                    if count >= 3:
                        fusion = self.bundle['numeric'].predict(self.cached, features, probability)
                        result.update(numeric_fusion=fusion, numeric_available=True,
                                      predicted_class=fusion['predicted_class'],
                                      classification_layer='numeric_fusion',
                                      reason='provisional_prefix_not_instantaneous_symptom')
                if count < 3:
                    result['reason'] = 'fewer_than_three_accepted_windows; uncalibrated_MIL_preview_only'
                self._result = result
            return copy.deepcopy(self._result)
