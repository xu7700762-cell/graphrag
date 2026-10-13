"""Freeze original offline source and a deterministic raw-window reference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def freeze(destination):
    from eeg_agent.config import load_config
    from eeg_agent.data import metadata
    from eeg_agent.recordings import preprocess, trigger_events, state_segments_from_events
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / 'offline_source_baseline.json'
    if manifest_path.exists():
        raise FileExistsError('Baseline already exists; use --verify')
    protected = list((PROJECT / 'eeg_agent').glob('*.py'))
    protected += [PROJECT / 'pyproject.toml', PROJECT / 'README.md']
    protected += list((PROJECT / 'eeg_agent' / 'web').glob('*'))
    hashes = {}
    for path in protected:
        if not path.is_file():
            continue
        relative = str(path.relative_to(PROJECT)).replace('\\', '/')
        hashes[relative] = sha(path)
        backup = destination / 'source_backup' / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    cfg = load_config()
    data = Path(cfg['data_root']) / 'data/raw/Acquisition 01.cdt'
    meta = metadata(data)
    raw = np.memmap(data, mode='r', dtype='<f4', shape=(meta['actual_samples'], meta['n_channels']))
    limit = min(len(raw), 200 * 1024)
    raw = raw[:limit]
    processed, names, compatible = preprocess(raw, meta['channels'], 1024)
    tasks, rests, warnings = state_segments_from_events(trigger_events(raw, meta['channels'].index('Trigger')), len(raw), 1024)
    windows, records = [], []
    for scope, segments in (('task', tasks), ('rest', rests)):
        for segment in segments:
            for index in range(int((segment['end_sample'] - segment['start_sample']) // 5120)):
                stamp = segment['start_sample'] + index * 5120
                start = int(np.ceil(stamp / 4))
                window = processed[:, start:start + 1280]
                valid = bool(window.shape == (30, 1280) and np.isfinite(window).all() and
                             np.ptp(window, axis=1).max() <= 2000 and np.count_nonzero(window.std(axis=1) >= .05) >= 28)
                records.append({'scope': scope, 'ordinal': segment['ordinal'], 'start_sample': stamp,
                                'valid': valid, 'sha256': hashlib.sha256(window.tobytes()).hexdigest()})
                windows.append(window)
    np.savez_compressed(destination / 'offline_windows_acquisition01_200s.npz', windows=np.asarray(windows), processed=processed)
    value = {'source_hashes': hashes, 'allowed_additive_changes': ['eeg_agent/service.py', 'eeg_agent/web/brain.html', 'pyproject.toml', 'README.md'],
             'reference': {'recording': 'Acquisition 01', 'raw_samples': limit, 'channels': names,
                           'model_compatible': compatible, 'tasks': tasks, 'rests': rests,
                           'windows': records, 'warnings': warnings}, 'api_calls': 0, 'training_started': False}
    manifest_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'baseline': str(manifest_path), 'protected_files': len(hashes), 'reference_windows': len(windows)}


def verify(destination):
    value = json.loads((destination / 'offline_source_baseline.json').read_text(encoding='utf-8'))
    changed = [name for name, digest in value['source_hashes'].items() if not (PROJECT / name).is_file() or sha(PROJECT / name) != digest]
    unauthorized = sorted(set(changed) - set(value['allowed_additive_changes']))
    return {'passed': not unauthorized, 'changed_additive_files': changed, 'changed_offline_files': unauthorized}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=PROJECT / 'outputs/online_implementation/20261011')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = verify(args.out) if args.verify else freeze(args.out)
    print(json.dumps(result, ensure_ascii=False))
    if result.get('passed') is False:
        raise SystemExit(1)
