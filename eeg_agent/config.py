import hashlib
import json
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parents[1]
EEG_CHANNELS = ['Fp1','Fp2','F11','F7','F3','Fz','F4','F8','F12','FT11','FC3','FCz','FC4','FT12','T7','C3','Cz','C4','T8','CP3','CPz','CP4','P7','P3','Pz','P4','P8','O1','Oz','O2']
SPLITS = {'train': [2,3,4,5,6,7,10,11,14,15,16,18,21,23,24], 'validation': [8,13,20,25], 'test': [1,9,12,17,19]}

def load_config(path=None):
    path = Path(path) if path else PROJECT / 'configs/local.yaml'
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if cfg.get('window_seconds')!=5 or not isinstance(cfg.get('block_samples'),int) or cfg['block_samples']<1:
        raise ValueError('This model contract requires 5s windows and positive integer block_samples')
    if not 0<cfg['minimum_path_coverage']<=1 or cfg['minimum_path_windows']<3:
        raise ValueError('Invalid path publication policy')
    if cfg['warmup_seconds']<0 or cfg['calibration_min_seconds']<10 or cfg['calibration_max_seconds']<cfg['calibration_min_seconds']:
        raise ValueError('Invalid warmup/calibration settings')
    cfg['data_root'] = str(Path(cfg['data_root']).resolve())
    out = Path(cfg['output_root'])
    cfg['output_root'] = str(out if out.is_absolute() else PROJECT / out)
    if cfg.get('classification_profile','P0') not in ('P0','P1'):
        raise ValueError('Unknown classification preprocessing profile')
    return cfg

def profile_config(cfg, profile):
    if profile not in ('P0','P1'): raise ValueError('Unknown classification preprocessing profile')
    return {**cfg, 'classification_profile':profile}

def cache_path(cfg):
    return Path(cfg['output_root']) / ('cache_eog_rls_v1' if cfg.get('classification_profile','P0')=='P1' else 'cache')

def eog_policy(cfg):
    # Frozen engineering safety limits, independent of labels and test performance.
    return {'version':'causal-eog-rls-v1', 'method':'causal_reference_rls', 'adaptive_memory_seconds':10.,
            'predict_before_update':True, 'initialize_from_prior_baseline':True,
            'skip_warmup_and_raw_jumps':True, 'raw_jump_guard':'current and three preceding arrived raw samples',
            'skip_update_above_centered_eeg_uv':float(cfg['qc']['peak_to_peak_uv']),
            'ridge_fraction':.001, 'minimum_eog_std_uv':.1,
            'maximum_condition':1e4, 'maximum_coefficient':50., 'residual_correlation_max':.6,
            'removed_rms_ratio_max':1.5}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def contract(cfg):
    value = {'schema_version':2,'channels': EEG_CHANNELS, 'unit': 'uV', 'reference': '(M1+M2)/2', 'input_hz': 1024,
            'output_hz': 256, 'window_samples': 1280, 'filter': 'notch50Q30; butter4 0.5-45; elliptic AA wp100 ws128 ripple0.1dB stop80dB; causal SOS',
            'warmup_seconds': cfg['warmup_seconds'], 'calibration_min_seconds': cfg['calibration_min_seconds'],
            'calibration_max_seconds': cfg['calibration_max_seconds'], 'qc': cfg['qc'],
            'threshold_score': 30, 'equality': 'high', 'classification_profiles':['P0'],
            'window_boundary':'half-open 5120 raw samples; no gap or path mixing',
            'calibration': 'first pre-task arrived QC-valid complete windows only; frozen before task; no classifier adaptation',
            'normalization': 'training-only P0 channel mean/std'}
    if cfg.get('classification_profile','P0')=='P1':
        value.update(schema_version=3, classification_profiles=['P1'],
                     normalization='training-only causal-EOG-RLS-cleaned P1 channel mean/std',
                     eog_cleaning=eog_policy(cfg),
                     calibration='QC-valid arrived pre-task windows initialize VEOG/HEOG RLS; algorithm parameters frozen; predict then update on arrived unlabeled samples including rests; reset to initialization after gaps; unavailable calibration rejects P1 classification')
    return value

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
