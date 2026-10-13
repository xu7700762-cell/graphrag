"""Package existing seed2026 D2 raw-worker assets for CPU; no fitting or data access."""
import argparse
import importlib
import json
import platform
import sys
import types
from pathlib import Path

import joblib
import sklearn
import torch

from eeg_agent.online.assets import sha
from eeg_agent.online.minimal_mamba import UPSTREAM_COMMIT
from eeg_agent.online.model import TemporalEncoder, load_encoder_state


def local_path(value):
    value = str(value)
    if sys.platform == 'win32' and value.startswith('/mnt/'):
        value = value[5].upper() + ':/' + value[7:]
    elif sys.platform != 'win32' and len(value) > 2 and value[1:3] == ':/':
        value = '/mnt/' + value[0].lower() + '/' + value[3:]
    return Path(value)


def prepare(root, destination):
    root, destination = local_path(root).resolve(), local_path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a fresh online asset directory; existing bundles are immutable')
    import_file = root / 'tmp/model_asset_import.json'
    spec = json.loads(import_file.read_text(encoding='utf-8'))
    model_root = root / 'outputs/vrms_model_ablation/v3_20261008'
    preflight = json.loads((model_root / 'preflight.json').read_text(encoding='utf-8'))
    audits = json.loads((model_root / 'model_results/fold_audit.json').read_text(encoding='utf-8'))
    heads = {row['outer_subject']: row['checkpoint_sha256'] for row in audits
             if row['method'] == 'femba_frozen_mil'}
    if spec['seed'] != 2026 or len(spec['folds']) != 24 or set(map(int, spec['folds'])) != set(heads):
        raise ValueError('Expected the actual 24 held-out seed2026 raw-worker folds')
    summaries = {}
    for kind, name in (('first', 'v4_tools_r1_20261008'), ('second', 'v5_tools_portable_20261008'),
                       ('selector', 'v6_selector_20261008')):
        summaries[kind] = json.loads((root / 'outputs/vrms_agent_loso' / name /
                                      'training_summary.json').read_text(encoding='utf-8'))
    # Trusted legacy pickle globals map to the isolated online CSP definition;
    # no legacy CUDA/experiment module is imported into the CPU runtime.
    for old in ('vrms_cloud.agent_tools_v1', 'vrms_cloud.agent_tools_v2'):
        sys.modules.setdefault('vrms_cloud', types.ModuleType('vrms_cloud'))
        sys.modules[old] = importlib.import_module('eeg_agent.online.features')
    manifest = {'schema': 'online_cpu_seed2026_v1', 'seed': 2026, 'dtype': 'float32',
                'baseline_policy': 'disabled', 'mamba_minimal_commit': UPSTREAM_COMMIT,
                'encoder_tensor_count': 83, 'encoder_parameters': 28540155,
                'source_import_sha256': sha(import_file), 'folds': {},
                'software': {'python': platform.python_version(), 'torch': torch.__version__,
                             'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__},
                'data_copied': False, 'training_performed': False}
    destination.mkdir(parents=True)

    def package(source, filename, kind, expected=None):
        source = local_path(source)
        original_hash = sha(source)
        if expected is not None and original_hash != expected:
            raise ValueError('Raw worker source asset differs from its frozen audit: ' + kind)
        target = destination / filename
        if kind in ('encoder', 'head', 'normalization'):
            original = torch.load(source, map_location='cpu', weights_only=False)
            if kind == 'encoder':
                state = original.get('state_dict', original)
                value = {'state_dict': {key: tensor for key, tensor in state.items()
                         if (key[6:] if key.startswith('model.') else key).startswith(
                             ('patch_embed', 'pos_embed', 'mamba_blocks', 'norm_layers'))}}
                load_encoder_state(TemporalEncoder(), value)
            elif kind == 'head':
                value = {key: original[key] for key in ('state_dict', 'input_dim', 'split')}
                value['seed'] = original.get('seed', 2026)
                if value['seed'] != 2026:
                    raise ValueError('Conflicting frozen head seed')
            else:
                value = {key: original[key] for key in
                         ('mean', 'scale', 'heldout_subject', 'training_subjects')}
            torch.save(value, target)
        elif kind == 'nested_audit':
            # Preserve exact bytes for the frozen independent validation report.
            target.write_bytes(source.read_bytes())
        else:
            joblib.dump(joblib.load(source), target)
        return {'file': filename, 'sha256': sha(target), 'source_sha256': original_hash}

    manifest['encoder'] = package(spec['encoder'], 'encoder.pt', 'encoder',
                                  preflight['pretrained_checkpoint_sha256'])
    for fold, entries in spec['folds'].items():
        number = int(fold)
        record = {}
        for kind in ('head', 'normalization', 'first', 'second', 'selector', 'nested_audit'):
            source = local_path(entries[kind])
            expected = (heads[number] if kind == 'head' else
                        preflight['cnn_checkpoint_sha256'][source.name] if kind == 'normalization' else
                        summaries[kind]['checkpoint_sha256'][source.name] if kind in summaries else None)
            suffix = '.pt' if kind in ('head', 'normalization') else '.json' if kind == 'nested_audit' else '.joblib'
            record[kind] = package(source, f'fold_{number:02d}_{kind}{suffix}', kind, expected)
        manifest['folds'][fold] = record
    target = destination / 'manifest.json'
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'manifest': str(target), 'manifest_sha256': sha(target), 'folds': len(manifest['folds']),
            'bytes': sum(p.stat().st_size for p in destination.iterdir() if p.is_file()),
            'training_performed': False, 'data_copied': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('D:/桌面/文献总/EEGAGENT'))
    parser.add_argument('--out', type=Path, default=Path('outputs/online_assets/seed2026'))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_root, args.out), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
