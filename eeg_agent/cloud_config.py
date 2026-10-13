"""Project-local cloud configuration; the settings editor can request visible keys."""
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from .config import PROJECT

NAMES = {'base_url': 'EEG_API_BASE_URL', 'model_id': 'EEG_API_MODEL', 'api_key': 'EEG_API_KEY'}
PROTOCOL_NAME = 'EEG_API_PROTOCOL'
PROTOCOLS = ('auto', 'responses', 'chat_completions')
PROFILE_DEFAULTS = {
    'deepseek': {'base_url': 'https://api.deepseek.com', 'model_id': 'deepseek-flash', 'api_key': ''},
    'cpa_gpt': {'base_url': 'http://127.0.0.1:8317/v1', 'model_id': 'gpt-6.1-sol', 'api_key': ''},
    'relay': {'base_url': '', 'model_id': '', 'api_key': '', 'protocol': 'chat_completions'}}
LOCK = threading.RLock()


class ConfigError(ValueError):
    pass


def resolve_protocol(model, protocol='auto'):
    if not isinstance(protocol, str) or protocol not in PROTOCOLS:
        raise ConfigError('请求协议须为auto、responses或chat_completions')
    return ('responses' if model.startswith('gpt-') else 'chat_completions') if protocol == 'auto' else protocol


def _path(path):
    return Path(path) if path is not None else PROJECT / '.env'


def _profiles_path(path):
    return _path(path).with_name('cloud_profiles.json')


def _profile_state(path):
    """Read only this project's provider profiles, importing its legacy config."""
    current = read_settings(path)
    active = 'cpa_gpt' if current['EEG_API_MODEL'].startswith('gpt-') else 'deepseek'
    state = {'version': 1, 'active_profile': active,
             'profiles': {key: dict(value) for key, value in PROFILE_DEFAULTS.items()}}
    target = _profiles_path(path)
    if target.exists():
        try:
            saved = json.loads(target.read_text(encoding='utf-8'))
            if (not isinstance(saved, dict) or saved.get('active_profile') not in PROFILE_DEFAULTS or
                    not isinstance(saved.get('profiles'), dict)):
                raise ValueError()
            state['active_profile'] = saved['active_profile']
            for profile_id in PROFILE_DEFAULTS:
                profile = saved['profiles'].get(profile_id, {})
                if (not isinstance(profile, dict) or set(profile) - (set(NAMES) | {'protocol'}) or
                        any(not isinstance(v, str) for v in profile.values())):
                    raise ValueError()
                state['profiles'][profile_id].update(profile)
        except (ValueError, TypeError):
            raise ConfigError('项目云端方案配置文件格式无效')
    # The existing .env is still the authoritative active configuration. This
    # also preserves explicit local edits before a save/switch.
    if any(current[name] for name in NAMES.values()):
        state['profiles'][state['active_profile']].update({field: current[name] for field, name in NAMES.items()})
        if current.get(PROTOCOL_NAME):
            state['profiles'][state['active_profile']]['protocol'] = current[PROTOCOL_NAME]
    return state


def _file_values(path):
    values = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.lstrip().startswith('#') or '=' not in line:
                continue
            name, value = line.split('=', 1)
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                try:
                    value = json.loads(value)
                except ValueError:
                    value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            if name.strip() in (*NAMES.values(), PROTOCOL_NAME):
                values[name.strip()] = value
    return values


def read_settings(path=None):
    # An explicitly saved local value takes priority, without changing process env.
    with LOCK:
        values = _file_values(_path(path))
        settings = {name: values.get(name, os.environ.get(name, '')) for name in NAMES.values()}
        settings[PROTOCOL_NAME] = values.get(PROTOCOL_NAME, os.environ.get(PROTOCOL_NAME, 'auto'))
        return settings


def _text(value, field, limit):
    if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ConfigError(field + '格式无效')
    return value.strip()


def _url(value):
    value = _text(value, 'Base URL', 2048)
    try:
        parsed = urlsplit(value)
        _ = parsed.port
        valid = (parsed.scheme in ('http', 'https') and parsed.hostname and
                 parsed.username is None and parsed.password is None and
                 not parsed.query and not parsed.fragment and not any(c.isspace() for c in value))
    except ValueError:
        valid = False
    if not valid:
        raise ConfigError('Base URL须为有效的HTTP或HTTPS接口地址，不能包含账号、密码或查询参数')
    return value


def _public_profile(profile, sources):
    try:
        base = _url(profile['base_url'])
    except ConfigError:
        base = ''
    try:
        model = _text(profile['model_id'], 'Model ID', 256)
        key = _text(profile['api_key'], 'API Key', 4096)
    except ConfigError:
        model, key = '', ''
    protocol = profile.get('protocol', 'auto')
    resolve_protocol(model, protocol)
    return {'base_url': base, 'model_id': model, 'protocol': protocol, 'key_configured': bool(key),
            'configured': bool(base and model and key), 'sources': sources}


def public_settings(path=None, in_use=False, include_keys=False):
    with LOCK:
        local = _file_values(_path(path))
        values = read_settings(path)
        sources = {field: 'local_file' if name in local else 'environment' if os.environ.get(name) else 'missing'
                   for field, name in NAMES.items()}
        current = {field: values[name] for field, name in NAMES.items()}
        current['protocol'] = values[PROTOCOL_NAME]
        state = _profile_state(path)
        persisted = _profiles_path(path).exists()
        profiles = {}
        for profile_id, profile in state['profiles'].items():
            profile_sources = {field: 'local_profile' if persisted and value else
                               'default' if value else 'missing' for field, value in profile.items()}
            if profile_id == state['active_profile'] and any(values[name] for name in NAMES.values()):
                profile_sources = sources
            profiles[profile_id] = {'id': profile_id, **_public_profile(profile, profile_sources)}
            if include_keys:
                profiles[profile_id]['api_key'] = profile['api_key']
        return {**_public_profile(current, sources), 'in_use': bool(in_use),
                'active_profile': state['active_profile'], 'profiles': profiles}


def _atomic_text(target, text):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.cloud-', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_active(target, profile):
    lines = target.read_text(encoding='utf-8-sig').splitlines() if target.exists() else []
    # Preserve unrelated settings and comments, replacing duplicate cloud keys.
    lines = [line for line in lines if line.lstrip().startswith('#') or
             '=' not in line or line.split('=', 1)[0].strip() not in (*NAMES.values(), PROTOCOL_NAME)]
    lines.extend(name + '=' + json.dumps(profile[field], ensure_ascii=False) for field, name in NAMES.items())
    lines.append(PROTOCOL_NAME + '=' + json.dumps(profile.get('protocol', 'auto')))
    _atomic_text(target, '\n'.join(lines) + '\n')


def save_settings(payload, path=None):
    allowed = set(NAMES) | {'profile_id', 'active_profile', 'activate', 'protocol'}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ConfigError('仅接受项目云端方案、Base URL、Model ID和API Key配置')
    target = _path(path)
    with LOCK:
        state = _profile_state(path)
        activate = True
        if 'active_profile' in payload:
            if (set(payload) != {'active_profile'} or not isinstance(payload['active_profile'], str) or
                    payload['active_profile'] not in PROFILE_DEFAULTS):
                raise ConfigError('云端方案切换格式无效')
            profile_id = payload['active_profile']
            if not _public_profile(state['profiles'][profile_id], {})['configured']:
                raise ConfigError('所选云端方案尚未完整配置，请先保存接口、模型与密钥')
        else:
            if not {'base_url', 'model_id'} <= set(payload):
                raise ConfigError('请同时提供Base URL和Model ID')
            profile_id = payload.get('profile_id', state['active_profile'])
            if not isinstance(profile_id, str) or profile_id not in PROFILE_DEFAULTS:
                raise ConfigError('云端方案须为deepseek、cpa_gpt或relay')
            activate = payload.get('activate', True)
            if not isinstance(activate, bool):
                raise ConfigError('activate须为布尔值')
            base = _url(payload['base_url'])
            model = _text(payload['model_id'], 'Model ID', 256)
            if not model:
                raise ConfigError('Model ID不能为空')
            key = _text(payload.get('api_key', ''), 'API Key', 4096)
            protocol = payload.get('protocol', state['profiles'][profile_id].get('protocol', 'auto'))
            resolve_protocol(model, protocol)
            state['profiles'][profile_id].update({'base_url': base, 'model_id': model, 'protocol': protocol})
            # A blank key preserves only this profile's own saved key.
            if key:
                state['profiles'][profile_id]['api_key'] = key
            configured = _public_profile(state['profiles'][profile_id], {})['configured']
            if payload.get('profile_id') is not None and activate and not configured:
                raise ConfigError('该云端方案尚无独立API密钥，请填写后启用，或先保存而不启用')
            # Editing an already configured active profile also applies its
            # saved settings; inactive/unconfigured drafts can stay dormant.
            activate = activate or (profile_id == state['active_profile'] and configured)
        if activate:
            state['active_profile'] = profile_id
        profile_target = _profiles_path(target)
        previous = profile_target.read_text(encoding='utf-8') if profile_target.exists() else None
        _atomic_text(profile_target, json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        try:
            if activate:
                _write_active(target, state['profiles'][profile_id])
        except OSError:
            if previous is None:
                profile_target.unlink()
            else:
                _atomic_text(profile_target, previous)
            raise
        return public_settings(target)


def test_connection(path=None, payload=None):
    """One fixed text request, with no EEG context and no automatic retry."""
    import openai

    if payload is None:
        settings = read_settings(path)
        status = public_settings(path)
        if not status['configured']:
            raise ConfigError('请先保存完整的云端API配置')
    else:
        if (not isinstance(payload, dict) or set(payload) - (set(NAMES) | {'protocol'}) or
                not set(NAMES) <= set(payload)):
            raise ConfigError('连接测试需要Base URL、Model ID和API Key')
        base = _url(payload['base_url'])
        model = _text(payload['model_id'], 'Model ID', 256)
        key = _text(payload['api_key'], 'API Key', 4096)
        if not model or not key:
            raise ConfigError('请填写完整的模型ID和API密钥再测试')
        protocol = payload.get('protocol', 'auto')
        resolve_protocol(model, protocol)
        settings = {NAMES['base_url']: base, NAMES['model_id']: model, NAMES['api_key']: key,
                    PROTOCOL_NAME: protocol}
        status = {'model_id': model}
    begin = time.perf_counter()
    client = None
    ok, reason, note, finish_reason = False, None, None, None
    try:
        client = openai.OpenAI(base_url=settings[NAMES['base_url']], api_key=settings[NAMES['api_key']],
                               timeout=10.0, max_retries=0)
        if resolve_protocol(settings[NAMES['model_id']], settings[PROTOCOL_NAME]) == 'responses':
            options = dict(model=settings[NAMES['model_id']], input='This is an API connectivity test. Reply OK.',
                           max_output_tokens=64, store=False)
            if settings[NAMES['model_id']].startswith('gpt-'):
                options['reasoning'] = {'effort': 'low'}
            response = client.responses.create(**options)
            finish_reason = getattr(response, 'status', None)
            ok = bool(getattr(response, 'output_text', '').strip())
            if finish_reason == 'incomplete' and not ok:
                ok = True
                note = '接口已响应，测试输出达到上限；完整报告能力需实际任务验证'
            choice = None
        else:
            response = client.chat.completions.create(
                model=settings[NAMES['model_id']],
                messages=[{'role': 'user', 'content': 'This is an API connectivity test. Reply OK.'}],
                max_tokens=64)
            choice = response.choices[0] if response.choices else None
        if choice is not None:
            returned_reason = getattr(choice, 'finish_reason', None)
            if returned_reason in ('stop', 'length', 'tool_calls', 'content_filter', 'function_call'):
                finish_reason = returned_reason
            content = getattr(choice.message, 'content', None)
            reasoning = getattr(choice.message, 'reasoning_content', None)
            ok = isinstance(content, str) and bool(content.strip())
            if not ok and isinstance(reasoning, str) and reasoning.strip() and finish_reason == 'length':
                ok = True
                note = '接口已响应，测试输出达到上限；完整报告能力需实际任务验证'
        if not ok:
            reason = '接口未返回有效文本响应'
    except openai.AuthenticationError:
        reason = '认证失败，请检查API Key'
    except openai.PermissionDeniedError:
        reason = '接口拒绝访问，请检查账号权限'
    except openai.RateLimitError:
        reason = '调用限额或余额不足，请检查服务商账户'
    except openai.NotFoundError:
        reason = '接口地址或Model ID不存在'
    except openai.APITimeoutError:
        reason = '连接超时（10秒），请检查接口或网络'
    except openai.APIConnectionError:
        reason = '无法连接接口，请检查Base URL和网络'
    except openai.BadRequestError:
        reason = '接口不接受测试请求，请检查Model ID及OpenAI兼容性'
    except Exception:
        # Provider errors may contain keys or URLs: never echo them into UI/logs.
        reason = '接口调用失败，请检查云端服务配置'
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    return {'ok': ok, 'reason': reason, 'note': note, 'finish_reason': finish_reason,
            'model_id': status['model_id'],
            'latency_seconds': time.perf_counter() - begin}
