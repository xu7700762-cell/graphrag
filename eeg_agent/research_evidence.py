"""Anonymous EEG measurements from the selected frozen research path, without classifiers."""
import hashlib
import json
import re
import threading
from pathlib import Path

import numpy as np

from .config import EEG_CHANNELS
from .tools import BANDS, biomarkers
from .vrms_runtime import WORKER_ROOT

_lock = threading.Lock()
_verified = {}


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _contract(root, seed):
    cache = root / 'outputs/vrms_pilot/v2_gpt_flow_20261008/cache'
    preflight_path = root / 'outputs/vrms_model_ablation/v3_20261008/preflight.json'
    manifest_path = (root / 'outputs/vrms_agent_seed_retest/seed2028_20261008/tools1/local_paths.json'
                     if seed == 2028 else root / 'outputs/vrms_agent_loso/v4_tools_r1_20261008/local_paths.json')
    files = [preflight_path, manifest_path, cache / 'window_evidence.jsonl'] + [cache / name for name in
             ('windows.npy', 'evaluation_manifest.json', 'dataset_audit.json', 'eligibility_audit.json')]
    signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files)
    key = (str(root), seed)
    with _lock:
        saved = _verified.get(key)
        if saved and saved[0] == signature:
            return saved[1]
        preflight = _read(preflight_path)
        if preflight.get('status') != 'passed':
            raise ValueError('research cache preflight unavailable')
        for name in ('windows.npy', 'evaluation_manifest.json', 'dataset_audit.json', 'eligibility_audit.json'):
            if _hash(cache / name) != preflight['cache_sha256'][name]:
                raise ValueError('research cache hash mismatch')
        paths, registered = _read(cache / 'evaluation_manifest.json'), _read(manifest_path)
        audit = _read(cache / 'dataset_audit.json')
        if len(paths) != 146 or registered != paths or audit.get('channel_order') != list(EEG_CHANNELS):
            raise ValueError('research path or channel contract mismatch')
        records = [json.loads(line) for line in (cache / 'window_evidence.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        windows = np.load(cache / 'windows.npy', mmap_mode='r', allow_pickle=False)
        if windows.shape[1:] != (30, 1280) or len(records) != audit['accepted_windows']:
            raise ValueError('research window contract mismatch')
        value = {'cache': cache, 'paths': paths, 'audit': audit, 'records': records,
                 'signal_sha256': preflight['cache_sha256']['windows.npy'],
                 'timing_sha256': _hash(cache / 'window_evidence.jsonl')}
        _verified[key] = (signature, value)
        return value


def read_path_evidence(path_id, seed=2026, root=None):
    """Recompute band powers locally; never return EEG arrays, identities, scores or labels."""
    if seed not in (2026, 2028) or not isinstance(path_id, str) or not re.fullmatch(r'path-\d{3}', path_id):
        raise ValueError('unknown research path')
    contract = _contract(Path(root) if root is not None else WORKER_ROOT, seed)
    index = int(path_id[5:])
    if index >= len(contract['paths']):
        raise ValueError('unknown research path')
    path = contract['paths'][index]
    start, end = path['window_start'], path['window_end']
    if (not 0 <= start < end <= contract['audit']['accepted_windows'] or
            end - start != path['accepted_windows']):
        raise ValueError('invalid research window range')
    windows = np.load(contract['cache'] / 'windows.npy', mmap_mode='r', allow_pickle=False)
    totals = {band: np.zeros(30, dtype=float) for band in BANDS}
    history, emotion_history = [], []
    previous_start = None
    relative_end = None
    for window_index in range(start, end):
        record = contract['records'][window_index]
        raw_start, raw_end = record['start_sample'], record['end_sample']
        q = record.get('qc') or {}
        x = np.asarray(windows[window_index])
        if (record['window_index'] != window_index or record['path_index'] != path['path_index'] or
                not path['path_start_sec'] * 1024 <= raw_start < raw_end <= path['path_end_sec'] * 1024 + 1 or
                abs(raw_end - raw_start - 5120) > 4 or q.get('finite') is not True or
                q.get('max_ptp_uv', 2001) > 2000 or q.get('nonflat_channels', 0) < 28 or
                not np.isfinite(x).all() or float(np.ptp(x, axis=1).max()) > 2000 or
                int(np.count_nonzero(np.std(x, axis=1) >= .05)) < 28):
            raise ValueError('invalid research window evidence')
        measured = biomarkers(x)
        if not measured.get('available'):
            raise ValueError('spectral measurement failed')
        for band in BANDS:
            totals[band] += np.asarray(measured['absolute_power_uv2'][band])
        # A rejected interval breaks the trend instead of joining both sides.
        if previous_start is not None and raw_start - previous_start != 5120:
            history, emotion_history = [], []
        previous_start = raw_start
        relative_end = raw_end / 1024. - path['path_start_sec']
        history.append({'source_time_seconds': relative_end, **measured['regions']['all']['ratios']})
        frontal = measured['regions']['frontal']['absolute_power_uv2']
        emotion_history.append({'source_time_seconds': relative_end, 'alpha': frontal['alpha'], 'beta': frontal['beta']})
    count = end - start
    mean_power = {band: values / count for band, values in totals.items()}
    regions = {}
    for name, channels in (('all', EEG_CHANNELS), ('frontal', ('F3', 'Fz', 'F4')), ('occipital', ('O1', 'Oz', 'O2'))):
        positions = [EEG_CHANNELS.index(channel) for channel in channels]
        regions[name] = {'absolute_power_uv2': {band: float(values[positions].mean()) for band, values in mean_power.items()}}
    duration = path['path_end_sec'] - path['path_start_sec']
    context = {'source': 'selected_research_path', 'path_id': path_id,
               'aggregation': 'accepted-window mean band powers; trends use the latest contiguous windows',
               'sample_rate_hz': 256, 'window_seconds': 5, 'accepted_windows': count,
               'analyzed_seconds': count * 5, 'path_duration_seconds': duration,
               'coverage': min(1., count * 5 / duration), 'preprocessing_profile': 'causal_P0',
               'signal_sha256': contract['signal_sha256'], 'timing_sha256': contract['timing_sha256']}
    return {'status': 'arrived', 'source_time_seconds': relative_end, 'measurement_context': context,
            'arrived': {'quality': {'valid': True}, 'fatigue_history': history, 'emotion_history': emotion_history,
                        'biomarkers': {'available': True, 'baseline_available': False, 'regions': regions,
                                       'channel_power_uv2': {channel: {'alpha': float(mean_power['alpha'][EEG_CHANNELS.index(channel)])}
                                                            for channel in ('F3', 'F4')}}},
            'reason': None}
