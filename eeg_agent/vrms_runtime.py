"""Run the selected VRMS research path in its isolated worker environment."""
import copy
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

PIPELINE_ID = 'femba_mil_tools_gpt_seed2026'
PIPELINE_IDS = {seed: f'femba_mil_tools_gpt_seed{seed}' for seed in (2026, 2028)}
EXPECTED_MODEL = 'gpt-6.1-sol'
SUPPORTED_MODELS = frozenset((EXPECTED_MODEL, 'deepseek-flash', 'deepseek-pro', 'deepseek-chat', 'deepseek-reasoner'))
WORKER_ROOT = Path(r'D:\桌面\文献总\EEGAGENT')
WORKER_PYTHON = Path(__file__).resolve().parents[1] / '.venv' / 'Scripts' / 'python.exe'
WORKER_TIMEOUT = 180
CATALOG_TIMEOUT = 30
CATALOG_TTL = 60
_cache = None
_cache_time = 0.0
_cache_lock = threading.Lock()
_invalid_path = '请选择当前 VRMS 模型 目录中的有效匿名研究路径。'


def seed_for_pipeline(pipeline_id):
    for seed, registered in PIPELINE_IDS.items():
        if pipeline_id == registered:
            return seed
    raise ValueError('未知 VRMS 推理管线')


def _unavailable(reason, seed=2026):
    return {'pipeline_id': PIPELINE_IDS[seed], 'seed': seed, 'status': 'unavailable',
            'available': False, 'reason': reason}


def _worker(payload=None, catalog_only=False, seed=2026):
    if not WORKER_PYTHON.is_file() or not WORKER_ROOT.is_dir():
        return _unavailable('VRMS 模型 本地推理环境不可用，请检查研究项目和 Python 环境。', seed)
    command = [str(WORKER_PYTHON), '-B', '-m', 'vrms_cloud.web_inference']
    if catalog_only:
        command.extend(['--seed', str(seed)])
        command.append('--catalog')
    try:
        completed = subprocess.run(
            command, cwd=str(WORKER_ROOT),
            input=json.dumps(payload, ensure_ascii=False, allow_nan=False) if payload is not None else '',
            capture_output=True, text=True, encoding='utf-8',
            timeout=CATALOG_TIMEOUT if catalog_only else WORKER_TIMEOUT, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if completed.returncode != 0:
            return _unavailable('VRMS 模型 推理进程失败；请检查本地研究产物。', seed)
        if len(completed.stdout) > 1_000_000:
            return _unavailable('VRMS 模型 推理返回内容超出预期范围。', seed)
        value = json.loads(completed.stdout)
        if not isinstance(value, dict):
            raise ValueError('invalid worker object')
        return value
    except subprocess.TimeoutExpired:
        return _unavailable('VRMS 模型 推理超时，本轮结果未发布；请重试或检查接口。', seed)
    except Exception:
        # Process exceptions may carry stdin, provider errors or credentials.
        return _unavailable('VRMS 模型 推理返回不可用；请检查本地环境和接口配置。', seed)


def catalog(seed=2026):
    """Return an anonymous, bounded allowlist; never load provider credentials."""
    global _cache, _cache_time
    if seed not in PIPELINE_IDS:
        raise ValueError('未登记的模型配置')
    pipeline_id = PIPELINE_IDS[seed]
    with _cache_lock:
        if _cache is not None and seed in _cache and time.monotonic() - _cache_time[seed] < CATALOG_TTL:
            return copy.deepcopy(_cache[seed])
        value = _worker(catalog_only=True, seed=seed)
        paths = value.get('paths')
        if value.get('pipeline_id') != pipeline_id or not isinstance(paths, list):
            value = _unavailable('VRMS 模型 研究路径目录暂不可用。', seed)
        else:
            anonymous = []
            seen = set()
            for item in paths:
                if not isinstance(item, dict):
                    continue
                token = item.get('id')
                if not isinstance(token, str) or not re.fullmatch(r'path-\d{3}', token):
                    continue
                if not 0 <= int(token[5:]) < 146 or token in seen:
                    continue
                seen.add(token)
                anonymous.append({'id': token, 'title': item.get('title') or item.get('label') or token,
                                  'source': pipeline_id,
                                  **({'accepted_windows': item['accepted_windows']}
                                     if isinstance(item.get('accepted_windows'), int) else {})})
            if anonymous:
                value = {**{key: copy.deepcopy(value[key]) for key in
                            ('model_description', 'capabilities', 'boundary', 'historical_metrics', 'expected_models') if key in value},
                         'pipeline_id': pipeline_id, 'seed': seed, 'status': 'ready',
                         'available': True, 'paths': anonymous, 'expected_model': EXPECTED_MODEL}
                if isinstance(value.get('capabilities'), dict):
                    value['capabilities'].pop('gpt_ready', None)
            else:
                value = _unavailable('VRMS 模型 目录中没有可用匿名研究路径。', seed)
        if _cache is None:
            _cache, _cache_time = {}, {}
        _cache[seed], _cache_time[seed] = value, time.monotonic()
        return copy.deepcopy(value)


def validate_path(path_id, seed=2026):
    if not isinstance(path_id, str) or not re.fullmatch(r'path-\d{3}', path_id):
        raise ValueError(_invalid_path)
    if path_id not in {item['id'] for item in catalog(seed).get('paths', [])}:
        raise ValueError(_invalid_path)
    return path_id


def _provider(provider):
    if not isinstance(provider, dict):
        return None, '云端二次判断缺少已保存的接口凭据；本轮仅执行本地 MIL 和数值融合。'
    model = provider.get('model')
    if model not in SUPPORTED_MODELS:
        return None, '云端二次判断支持已登记的 CPA GPT 或 DeepSeek 型号；当前型号不匹配，本轮仅执行本地 MIL 和数值融合。'
    base, key = provider.get('base_url'), provider.get('api_key')
    if not isinstance(base, str) or not isinstance(key, str) or not key.strip():
        return None, '云端二次判断缺少接口地址或密钥；本轮仅执行本地 MIL 和数值融合。'
    try:
        url = urlsplit(base)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('invalid API URL')
    except Exception:
        return None, '云端接口地址无效；本轮仅执行本地 MIL 和数值融合。'
    return {'base_url': base, 'model': model, 'api_key': key}, None


def _redact(value, secret):
    if isinstance(value, dict):
        return {k: _redact(v, secret) for k, v in value.items()
                if str(k).lower() not in ('api_key', 'authorization', 'access_token')}
    if isinstance(value, list):
        return [_redact(v, secret) for v in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, '[redacted]')
    return value


def assess(path_id, provider=None, cloud=True, pipeline_id=PIPELINE_ID):
    """Fresh local inference, then the selected GPT model when explicitly ready."""
    try:
        seed = seed_for_pipeline(pipeline_id)
    except ValueError:
        return _unavailable('未知 VRMS 推理管线')
    try:
        path_id = validate_path(path_id, seed)
    except ValueError:
        return _unavailable(_invalid_path, seed)
    ready, missing = _provider(provider) if cloud else (None, None)
    payload = {'pipeline_id': pipeline_id, 'path_id': path_id, 'cloud': bool(cloud and ready)}
    if ready:
        payload['provider'] = ready
    value = _worker(payload, seed=seed)
    if value.get('available') is not True:
        return _unavailable(value.get('reason') if value.get('reason') in {
            'VRMS 模型 本地推理环境不可用，请检查研究项目和 Python 环境。',
            'VRMS 模型 推理进程失败；请检查本地研究产物。',
            'VRMS 模型 推理返回内容超出预期范围。',
            'VRMS 模型 推理超时，本轮结果未发布；请重试或检查接口。',
            'VRMS 模型 推理返回不可用；请检查本地环境和接口配置。'}
                            else 'VRMS 模型 研究产物校验或推理失败，本轮分类不可用。', seed)
    value = _redact(value, ready['api_key'] if ready else None)
    measurements = value.get('measurements')
    provenance = value.get('provenance', {})
    pipeline = measurements.get('pipeline', {}) if isinstance(measurements, dict) else {}
    if (not isinstance(measurements, dict) or
            any(not isinstance(measurements.get(key), dict) for key in ('mil', 'numeric_fusion', 'cloud_judgment')) or
            not isinstance(provenance, dict) or provenance.get('path_id', path_id) != path_id or
            not isinstance(pipeline, dict) or pipeline.get('id', pipeline_id) != pipeline_id or
            pipeline.get('seed', seed) != seed or
            not isinstance(value.get('limitations', []), list)):
        return _unavailable('VRMS 模型 推理返回内容不符合当前研究管线，本轮分类不可用。', seed)
    value.update(pipeline_id=pipeline_id, path_id=path_id, seed=seed)
    if missing:
        measurements = value.setdefault('measurements', {})
        judgment = measurements.setdefault('cloud_judgment', {})
        judgment.update(available=False, generation_status='fallback', final_class=None, reason=missing)
        value.setdefault('limitations', []).append(missing)
    return value
