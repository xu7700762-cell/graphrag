"""Two random-initialized architectures with individually frozen P0/P1 ONNX contracts.

PyTorch is imported only by build_model, so the Raspberry Pi runtime does not
need the training stack. No factory downloads or loads external checkpoints.
"""
import json
from pathlib import Path
import numpy as np
from .config import contract, digest, file_hash, profile_config

MODEL_IDS = ('eegnet', 'eegconformer')
ARCHITECTURES = {
    'eegnet': {'name': 'EEGNet', 'version': 'vrms-eegnet-2',
              'description': '8 temporal filters, depth multiplier 2, separable convolution; dropout 0.25',
              'source': 'https://doi.org/10.1088/1741-2552/aace8c'},
    'eegconformer': {'name': 'EEGConformer', 'version': 'vrms-conformer-1',
                     'description': 'embedding 40; depth 6; heads 10; CNN 30 channels; 79 tokens; head 3160-256-32-2',
                     'source': 'https://github.com/eeyhsong/EEG-Conformer',
                     'adaptation': '30-channel spatial convolution and binary head; supervised random initialization; no upstream training or test selection'},
}


def build_model(model_id='eegnet'):
    if model_id not in MODEL_IDS:
        raise ValueError('Unknown model_id: ' + str(model_id))
    import torch
    from torch import nn
    if model_id == 'eegnet':
        return nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (30, 1), groups=8, bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(.25),
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(.25), nn.Flatten(), nn.Linear(16 * 40, 2))

    class PatchEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.shallownet = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25)), nn.Conv2d(40, 40, (30, 1)),
                nn.BatchNorm2d(40), nn.ELU(), nn.AvgPool2d((1, 75), (1, 15)), nn.Dropout(.5))
            self.projection = nn.Conv2d(40, 40, (1, 1))

        def forward(self, x):
            x = self.projection(self.shallownet(x))
            return x.flatten(2).transpose(1, 2)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.keys = nn.Linear(40, 40)
            self.queries = nn.Linear(40, 40)
            self.values = nn.Linear(40, 40)
            self.att_drop = nn.Dropout(.5)
            self.projection = nn.Linear(40, 40)

        def forward(self, x):
            batch, tokens, _ = x.shape
            q = self.queries(x).reshape(batch, tokens, 10, 4).transpose(1, 2)
            k = self.keys(x).reshape(batch, tokens, 10, 4).transpose(1, 2)
            v = self.values(x).reshape(batch, tokens, 10, 4).transpose(1, 2)
            # Match the author's embedding-size scaling (sqrt(40)), not sqrt(4).
            a = self.att_drop(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (40 ** .5), dim=-1))
            y = torch.matmul(a, v).transpose(1, 2).reshape(batch, tokens, 40)
            return self.projection(y)

    class EncoderBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Sequential(nn.LayerNorm(40), Attention(), nn.Dropout(.5))
            self.feedforward = nn.Sequential(nn.LayerNorm(40), nn.Linear(40, 160), nn.GELU(),
                                             nn.Dropout(.5), nn.Linear(160, 40), nn.Dropout(.5))

        def forward(self, x):
            x = x + self.attention(x)
            return x + self.feedforward(x)

    return nn.Sequential(PatchEmbedding(), *[EncoderBlock() for _ in range(6)], nn.Flatten(),
                         nn.Linear(3160, 256), nn.ELU(), nn.Dropout(.5),
                         nn.Linear(256, 32), nn.ELU(), nn.Dropout(.3), nn.Linear(32, 2))


def probabilities_from_logits(logits, temperature=1.):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2 or not np.isfinite(logits).all():
        raise ValueError('Invalid binary logits')
    if not np.isfinite(temperature) or not .05 <= temperature <= 20:
        raise ValueError('Invalid frozen temperature')
    values = logits / temperature
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp[:, 1] / exp.sum(axis=1)


def publication_policy(cfg):
    return {'window_seconds': 5, 'minimum_valid_windows': cfg['minimum_path_windows'],
            'minimum_duration_coverage': cfg['minimum_path_coverage'],
            'aggregation': 'mean of valid-window temperature-scaled high probabilities',
            'class_rule': 'high when probability >= frozen threshold'}


def inference_contract(manifest):
    """Bind every inference parameter to the exported weights and processing.

    This detects accidental manifest/file drift, not malicious replacement of
    both a manifest and its checksum. Pi packaging preserves this manifest.
    """
    fields = ('model_id', 'checkpoint_id', 'architecture', 'contract', 'contract_hash',
              'sos_coefficients', 'allowed_profiles', 'mean', 'std', 'temperature', 'threshold',
              'publication_policy', 'onnx_sha256')
    profile=manifest['allowed_profiles'][0]
    return {'schema': 1, 'input': 'float32 batch x 1 x 30 x 1280, train-only normalized '+profile+' uV',
            'output': 'two logits ordered low/high; raw and temperature-scaled probability',
            **{key: manifest[key] for key in fields}}


def model_directory(cfg, model_id='eegnet', checkpoint_id=None):
    if model_id not in MODEL_IDS:
        raise ValueError('Unknown model_id: ' + str(model_id))
    root = Path(cfg['output_root']) / 'models' / model_id
    if checkpoint_id is not None:
        if not isinstance(checkpoint_id, str) or not checkpoint_id or checkpoint_id in ('.','..') or Path(checkpoint_id).name != checkpoint_id or any(c in checkpoint_id for c in '/\\:'):
            raise ValueError('Invalid checkpoint_id')
        return root / checkpoint_id
    profile=cfg.get('classification_profile','P0')
    default = root / ('eog_rls_seed42' if profile=='P1' else 'fixed_seed42')
    if (default / 'model.onnx').exists():
        return default
    candidates = sorted((p for p in root.glob('*') if p.is_dir() and (p / 'model.onnx').exists() and
                         ((p.name.startswith('eog_rls_')) == (profile=='P1'))),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else default


class ONNXPredictor:
    def __init__(self, cfg, model_dir=None, model_id='eegnet', checkpoint_id=None):
        import onnxruntime as ort
        self.directory = Path(model_dir) if model_dir else model_directory(cfg, model_id, checkpoint_id)
        self.manifest = json.loads((self.directory / 'contract.json').read_text(encoding='utf-8'))
        profiles=self.manifest.get('allowed_profiles')
        if profiles not in (['P0'],['P1']):
            raise ValueError('Model must have exactly one supported preprocessing profile')
        self.profile=profiles[0]; self.cfg=profile_config(cfg,self.profile)
        if self.manifest['contract_hash'] != digest(contract(self.cfg)):
            raise ValueError('Model preprocessing contract mismatch')
        if self.manifest.get('inference_contract_hash') != digest(inference_contract(self.manifest)):
            raise ValueError('Frozen inference metadata hash mismatch')
        if self.manifest['publication_policy'] != publication_policy(cfg):
            raise ValueError('Model path publication contract mismatch')
        if digest(self.manifest['contract']) != self.manifest['contract_hash']:
            raise ValueError('Model processing metadata hash mismatch')
        from .signal import filter_sos
        if not np.allclose(np.asarray(self.manifest['sos_coefficients']), filter_sos(), atol=1e-12, rtol=0):
            raise ValueError('Model numerical filter coefficients mismatch')
        self.model_id = self.manifest.get('model_id', model_id)
        if self.model_id not in MODEL_IDS or self.model_id != model_id:
            raise ValueError('Model identity mismatch')
        self.model_hash = file_hash(self.directory / 'model.onnx')
        if self.model_hash != self.manifest['onnx_sha256']:
            raise ValueError('ONNX hash mismatch')
        self.profiles = profiles
        self.threshold = float(self.manifest.get('threshold', .5))
        self.temperature = float(self.manifest.get('temperature', 1.))
        if not .1 <= self.threshold <= .9 or not .05 <= self.temperature <= 20:
            raise ValueError('Invalid frozen calibration')
        self.mean = np.asarray(self.manifest['mean'], dtype=np.float32)
        self.std = np.asarray(self.manifest['std'], dtype=np.float32)
        if self.mean.shape != (30,) or self.std.shape != (30,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError('Invalid training normalization')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.directory / 'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])

    def logits_batch(self, x):
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[1:] != (30, 1280) or not np.isfinite(x).all():
            raise ValueError('Invalid model input')
        if not len(x):
            return np.empty((0, 2), dtype=np.float32)
        results = []
        for offset in range(0, len(x), 64):
            normalized = (x[offset:offset + 64].astype(np.float32) - self.mean[None, :, None]) / self.std[None, :, None]
            results.append(self.session.run(None, {'eeg': normalized[:, None]})[0])
        logits = np.concatenate(results)
        if logits.shape != (len(x), 2) or not np.isfinite(logits).all():
            raise ValueError('Nonfinite or invalid model output')
        return logits

    def predict_batch(self, x):
        return probabilities_from_logits(self.logits_batch(x), self.temperature)

    def predict_details(self, x, profile=None):
        if profile is not None and profile!=self.profile:
            raise ValueError('Model input preprocessing profile mismatch')
        x = np.asarray(x)
        if x.shape != (1280, 30) or not np.isfinite(x).all():
            raise ValueError('Invalid model input')
        logits = self.logits_batch(x.T[None])
        return {'logits': logits[0].tolist(), 'raw_probability': float(probabilities_from_logits(logits)[0]),
                'probability_high': float(probabilities_from_logits(logits, self.temperature)[0]),
                'threshold': self.threshold, 'temperature': self.temperature, 'model_id': self.model_id,
                'model_hash': self.model_hash, 'checkpoint_id': self.directory.name, 'preprocessing_profile':self.profile}

    def predict(self, x):
        return self.predict_details(x)['probability_high']


def model_catalog(cfg):
    catalog = []
    for model_id in MODEL_IDS:
        root = Path(cfg['output_root']) / 'models' / model_id
        checkpoints = []
        for directory in sorted((p for p in root.glob('*') if p.is_dir()), key=lambda p: p.name):
            item = {'checkpoint_id': directory.name, 'name': directory.name, 'directory': str(directory),
                    'status': 'untrained', 'available': False, 'reason': None, 'contract_compatible': None,
                    'threshold': None, 'temperature': None, 'device': {'windows': 'not_tested', 'pi': 'not_tested'}}
            if (directory / 'checkpoint.pt').exists():
                item['status'] = 'trained'
            if (directory / 'contract.json').exists():
                try:
                    manifest = json.loads((directory / 'contract.json').read_text(encoding='utf-8'))
                    profiles=manifest.get('allowed_profiles')
                    profile=profiles[0] if profiles in (['P0'],['P1']) else None
                    item.update(threshold=manifest.get('threshold', .5), temperature=manifest.get('temperature', 1.),
                                preprocessing_profile=profile,
                                contract_compatible=bool(profile) and manifest.get('contract_hash') == digest(contract(profile_config(cfg,profile))))
                    if not item['contract_compatible'] or manifest.get('model_id') != model_id or manifest.get('publication_policy') != publication_policy(cfg):
                        item.update(status='incompatible', reason='模型与当前预处理合同不兼容')
                    elif (directory / 'model.onnx').exists():
                        if manifest.get('inference_contract_hash') != digest(inference_contract(manifest)):
                            item.update(status='error', reason='冻结推理元数据哈希校验失败')
                        elif file_hash(directory / 'model.onnx') != manifest.get('onnx_sha256'):
                            item.update(status='error', reason='ONNX文件哈希校验失败')
                        else:
                            item.update(status='exported', available=True)
                            item['device']['windows'] = 'verified' if manifest.get('onnx_validation', {}).get('status') == 'passed' else 'not_tested'
                    else:
                        item['reason'] = '尚未导出有效ONNX'
                except (ValueError, OSError, KeyError) as exc:
                    item.update(status='error', reason=str(exc))
            checkpoints.append(item)
        available = [c for c in checkpoints if c['available']]
        preferred_cfg=profile_config(cfg,'P1') if cfg.get('prefer_eog_models',False) else cfg
        default_id = model_directory(preferred_cfg, model_id).name
        chosen = next((c for c in available if c['checkpoint_id'] == default_id), available[-1] if available else None)
        catalog.append({'model_id': model_id, **ARCHITECTURES[model_id], 'checkpoints': checkpoints,
                        'status': chosen['status'] if chosen else (checkpoints[-1]['status'] if checkpoints else 'untrained'),
                        'available': bool(chosen), 'default_checkpoint_id': chosen['checkpoint_id'] if chosen else None,
                        'device': chosen['device'] if chosen else {'windows': 'not_tested', 'pi': 'not_tested'},
                        'reason': None if chosen else '未加载有效模型，预测不可用'})
    return catalog
