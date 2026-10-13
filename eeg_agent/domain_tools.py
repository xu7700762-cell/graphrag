"""Specialists use arrived scalar EEG measurements; classifiers are not invented."""
import copy
import math

VERSION = 'brain-domain-tools-2.0'
_CATALOG = {
    'vrms': [],
    'fatigue': [{'id': 'fatigue.spectral_ratios', 'name': '疲劳相关频谱测量',
                 'description': '计算theta/alpha、theta/beta、beta/alpha和慢快比，返回实际频谱指标。',
                 'input': 'quality-valid arrived.biomarkers.regions.*.absolute_power_uv2',
                 'output': '各区域频谱比值与可用的先行基线变化', 'capability': 'measurement'},
                {'id': 'fatigue.temporal_evidence', 'name': '频谱比值时间分析',
                 'description': '至少3个已到达频谱比值的时间趋势；VRMS概率趋势不能替代疲劳趋势。',
                 'input': 'arrived.fatigue_history: source_time_seconds + theta_alpha/theta_beta/slow_fast',
                 'output': '比值每分钟斜率、前后均值、变异系数与覆盖时间', 'capability': 'measurement'}],
    'emotion': [{'id': 'emotion.frontal_asymmetry', 'name': '额区alpha不对称测量',
                 'description': '计算ln(alpha_F4)-ln(alpha_F3)，需要真实左右通道功率。',
                 'input': 'arrived.biomarkers.channels + absolute_power_uv2.alpha',
                 'output': 'FAA标量及公式；不映射效价或具体情绪', 'capability': 'measurement'},
                {'id': 'emotion.read_evidence', 'name': '情绪证据完整性检查',
                 'description': '检查信号质量、F3/F4通道和FAA可用性，列出缺失测量。',
                 'input': 'arrived.quality, arrived.biomarkers',
                 'output': '测量可用性与缺失输入；无需分类模型', 'capability': 'evidence_check'},
                {'id': 'emotion.spectral_biomarkers', 'name': '情感相关频谱指标',
                 'description': '计算额区BAR（beta/alpha）、TBR（theta/beta）和TAR（theta/alpha），结合领域资料解释。',
                 'input': 'quality-valid arrived.biomarkers.regions.*.absolute_power_uv2',
                 'output': '实际频谱比值；不套用未提供的症状阈值', 'capability': 'measurement'},
                {'id': 'emotion.temporal_stability', 'name': '额区频带波动分析',
                 'description': '分析连续alpha/beta功率的变异系数、前后变化和时间趋势。',
                 'input': 'arrived.emotion_history: source_time_seconds + alpha/beta powers',
                 'output': '频带功率均值、变异系数与每分钟斜率', 'capability': 'measurement'}]
}


_RAW_CATALOG = {
    'vrms': [
        {'id': 'vrms.raw_recording', 'name': '原始事件与路径统计', 'description': '从原始波形内的触发通道统计完整与未结束路径，并返回时间区间。',
         'input': '原始 EEG 及内嵌事件通道', 'output': '路径数量、完整性、起止时间', 'capability': 'measurement', 'classifier_required': False},
        {'id': 'vrms.raw_model', 'name': '原始 EEG 与云端工具 Agent', 'description': '重新编码原始 EEG，执行冻结 MIL 与八路工具；所选 GPT 或 DeepSeek 实际调用工具后给出最终 High/Low 并定位首条高类路径。',
         'input': '质量合格的原始 EEG 五秒窗口与已配置 GPT 或 DeepSeek', 'output': 'MIL、数值融合、云端最终分类与真实工具调用记录', 'capability': 'raw_vrms_cloud_tool_agent'}],
    'fatigue': [{'id': 'fatigue.raw_spectrum', 'name': '整记录疲劳相关指标', 'description': '按事件边界计算完整路径及休息段的 θ/α，直接比较同类片段的首末值与最高值。',
                 'input': '原始 EEG 波形与内嵌任务事件', 'output': 'TAR、路径与休息段值、首末变化、最高片段及时间', 'capability': 'measurement'}],
    'emotion': [{'id': 'emotion.raw_workload', 'name': '整记录心理负荷与额区指标', 'description': '按事件边界计算路径和休息段的额区 θ/顶区 α 及 FAA，直接比较同类片段。',
                 'input': '原始 EEG，F3/Fz/F4 与 P3/Pz/P4 通道', 'output': 'EEG工作负荷指标、路径与休息段变化、FAA', 'capability': 'measurement'}]
}


def toolkit_catalog(include_recordings=False):
    catalog = {d: list(specs) for d, specs in _CATALOG.items()}
    if include_recordings:
        for domain, specs in _RAW_CATALOG.items():
            catalog[domain] += [{**spec, 'input_scope': 'raw_eeg_only'} for spec in specs]
    return copy.deepcopy({domain: [{**spec, 'version': VERSION, 'domain': domain,
                                   'classifier_available': spec.get('classifier_required', domain == 'vrms'),
                                   'classifier_required': spec.get('classifier_required', domain == 'vrms')} for spec in specs]
                          for domain, specs in catalog.items()})


def _number(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and (not positive or value > 1e-12) else None


def _ratio(a, b):
    a, b = _number(a), _number(b, positive=True)
    if a is None or b is None or a < 0:
        return None
    return _number(a / b)


def _fields(value, keys):
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if k in keys and
            (v is None or isinstance(v, (str, bool)) or _number(v) is not None)}


def _context(evidence):
    evidence = evidence if isinstance(evidence, dict) else {}
    arrived = evidence.get('arrived')
    return evidence, arrived if isinstance(arrived, dict) else {}


def _result(tool, available, reason, measurements=None, missing=None):
    domain = tool.split('.')[0]
    return {'tool': tool, 'domain': domain, 'version': VERSION, 'available': bool(available),
            'status': 'ok' if available else 'insufficient', 'reason': reason,
            'measurements': measurements or {}, 'missing': missing or [],
            'classifier_available': False, 'classifier_required': domain == 'vrms', 'predicted_class': None,
            'interpretation': '本地EEG测量与证据检查；未验证的疲劳或情绪标签不发布。'}


def _quality_blocked(arrived):
    quality = arrived.get('quality')
    if not isinstance(quality, dict) or quality.get('valid') is False or quality.get('usable') is False:
        return True
    return not (quality.get('valid') is True or quality.get('usable') is True)


def _spectral_ratios(evidence):
    _, arrived = _context(evidence)
    tool = 'fatigue.spectral_ratios'
    if _quality_blocked(arrived):
        return _result(tool, False, '缺少质量通过证据或当前窗口质量不合格。', missing=['quality-valid EEG measurements'])
    bio = arrived.get('biomarkers') or {}
    if not isinstance(bio, dict) or bio.get('available') is False:
        return _result(tool, False, '已到达的频谱测量不可用。', missing=['biomarkers'])
    regions = bio.get('regions') or {}
    measurements, missing = {}, []
    for name in ('all', 'frontal', 'occipital'):
        region = regions.get(name) if isinstance(regions, dict) else None
        if not isinstance(region, dict):
            continue
        raw = region.get('absolute_power_uv2') or {}
        powers = {band: _number(raw.get(band)) for band in ('delta', 'theta', 'alpha', 'beta')} if isinstance(raw, dict) else {}
        ratios = {'theta_alpha': _ratio(powers.get('theta'), powers.get('alpha')),
                  'theta_beta': _ratio(powers.get('theta'), powers.get('beta')),
                  'beta_alpha': _ratio(powers.get('beta'), powers.get('alpha')),
                  'slow_fast': _ratio(powers['delta'] + powers['theta'], powers['alpha'] + powers['beta'])
                               if len(powers) == 4 and all(v is not None and v >= 0 for v in powers.values()) else None}
        measurements[name] = {'ratios': ratios, 'absolute_power_uv2': {k: v for k, v in powers.items() if v is not None and v >= 0}}
        missing.extend(name + '.' + key for key, value in ratios.items() if value is None)
    available = any(value is not None for region in measurements.values() for value in region['ratios'].values())
    changes = bio.get('baseline_change')
    changes = changes.get('regions', changes) if isinstance(changes, dict) else None
    if bio.get('baseline_available') is True and isinstance(changes, dict):
        for name, region in measurements.items():
            region['baseline_change'] = _fields(changes.get(name), ('delta', 'theta', 'alpha', 'beta', 'gamma_low'))
    return _result(tool, available, '已计算各区域频谱比值，可结合领域资料解释指标模式。' if available else '需要已到达、质量合格的区域频带功率。',
                   measurements, missing if measurements else ['biomarkers.regions.*.absolute_power_uv2'])


def _fatigue_temporal(evidence):
    evidence, arrived = _context(evidence)
    tool = 'fatigue.temporal_evidence'
    if _quality_blocked(arrived):
        return _result(tool, False, '缺少质量通过证据或当前窗口质量不合格。', missing=['quality-valid EEG measurements'])
    history = arrived.get('fatigue_history')
    if not isinstance(history, list) or len(history) < 3:
        return _result(tool, False, '至少需要3个已到达、质量合格的连续频谱窗口。',
                       missing=['arrived.fatigue_history (at least 3 spectral measurements)'])
    history = history[-12:]
    now = _number(evidence.get('source_time_seconds'))
    previous = None
    rows = []
    for row in history:
        if not isinstance(row, dict) or row.get('valid') is False:
            return _result(tool, False, '频谱历史包含无效窗口，需重新收集连续证据。', missing=['valid spectral history'])
        stamp = _number(row.get('source_time_seconds'))
        if stamp is None or stamp < 0 or (previous is not None and stamp <= previous) or (now is not None and stamp > now):
            return _result(tool, False, '频谱时间必须递增且不能晚于已到达时间。', missing=['ordered arrived source intervals'])
        rows.append((stamp, row))
        previous = stamp
    measurements = {}
    for key in ('theta_alpha', 'theta_beta', 'slow_fast'):
        points = [(t, _number(row.get(key))) for t, row in rows]
        points = [(t, v) for t, v in points if v is not None and v >= 0]
        if len(points) < 3:
            continue
        mean_t = sum(t for t, _ in points) / len(points)
        mean_v = sum(v for _, v in points) / len(points)
        denominator = sum((t - mean_t) ** 2 for t, _ in points)
        slope = sum((t - mean_t) * (v - mean_v) for t, v in points) / denominator * 60.0
        spread = math.sqrt(sum((v - mean_v) ** 2 for _, v in points) / (len(points) - 1))
        edge = max(1, len(points) // 3)
        early = sum(v for _, v in points[:edge]) / edge
        late = sum(v for _, v in points[-edge:]) / edge
        measurements[key] = {'mean': mean_v, 'slope_ratio_per_minute': slope, 'windows': len(points),
                             'coefficient_of_variation': _ratio(spread, mean_v),
                             'early_mean': early, 'late_mean': late,
                             'relative_change_from_early': _number((late - early) / early) if early > 1e-12 else None,
                             'duration_seconds': points[-1][0] - points[0][0],
                             'trend': 'increasing' if slope > 1e-12 else 'decreasing' if slope < -1e-12 else 'stable'}
    return _result(tool, bool(measurements), '已计算连续频谱比值的趋势和前后变化。' if measurements else '每个比值至少需要3个有限数值。',
                   measurements, [] if measurements else ['at least 3 values per spectral ratio'])


def _frontal_asymmetry(evidence):
    _, arrived = _context(evidence)
    tool = 'emotion.frontal_asymmetry'
    if _quality_blocked(arrived):
        return _result(tool, False, '缺少质量通过证据或当前窗口质量不合格。', missing=['quality-valid EEG measurements'])
    bio = arrived.get('biomarkers') or {}
    if not isinstance(bio, dict) or bio.get('available') is False:
        return _result(tool, False, '已到达频谱不可用。', missing=['F3/F4 alpha powers'])
    alpha = (bio.get('absolute_power_uv2') or {}).get('alpha') if isinstance(bio.get('absolute_power_uv2'), dict) else None
    channels = bio.get('channels')
    powers = {}
    if isinstance(channels, list) and isinstance(alpha, list) and len(channels) == len(alpha):
        for name in ('F3', 'F4'):
            if channels.count(name) == 1:
                powers[name] = _number(alpha[channels.index(name)], positive=True)
    elif isinstance(bio.get('channel_power_uv2'), dict):
        for name in ('F3', 'F4'):
            channel = bio['channel_power_uv2'].get(name)
            powers[name] = _number(channel.get('alpha'), positive=True) if isinstance(channel, dict) else None
    missing = [name + '.alpha' for name in ('F3', 'F4') if powers.get(name) is None]
    if missing:
        return _result(tool, False, '额区区域平均不能还原F3/F4左右通道；需真实正值alpha功率。', missing=missing)
    value = math.log(powers['F4']) - math.log(powers['F3'])
    return _result(tool, True, '已计算F3/F4 alpha功率不对称及其方向。',
                   {'faa': value, 'formula': 'ln(alpha_F4) - ln(alpha_F3)',
                    'alpha_F3_uv2': powers['F3'], 'alpha_F4_uv2': powers['F4'], 'unit': 'dimensionless'})


def _execute_tool(tool_id, evidence):
    if tool_id == 'fatigue.spectral_ratios':
        return _spectral_ratios(evidence)
    if tool_id == 'fatigue.temporal_evidence':
        return _fatigue_temporal(evidence)
    if tool_id == 'emotion.frontal_asymmetry':
        return _frontal_asymmetry(evidence)
    if tool_id == 'emotion.read_evidence':
        result = _frontal_asymmetry(evidence)
        result['tool'] = tool_id
        result['reason'] += ' 当前采用EEG工具分析模式，无需情绪分类模型。'
        return result
    if tool_id == 'emotion.spectral_biomarkers':
        result = _spectral_ratios(evidence)
        result.update(tool=tool_id, domain='emotion')
        if result['available']:
            result['reason'] = '已计算BAR、TBR、TAR；领域资料用于解释其研究关联。'
        return result
    if tool_id == 'emotion.temporal_stability':
        context, arrived = _context(evidence)
        rows = arrived.get('emotion_history')
        converted = [{'source_time_seconds': row.get('source_time_seconds'), 'valid': row.get('valid'),
                      'theta_alpha': row.get('alpha'), 'theta_beta': row.get('beta')}
                     if isinstance(row, dict) else row for row in rows] if isinstance(rows, list) else []
        result = _fatigue_temporal({**context, 'arrived': {**arrived, 'fatigue_history': converted}})
        result.update(tool=tool_id, domain='emotion')
        result['measurements'] = {band: {**values, 'slope_power_uv2_per_minute': values['slope_ratio_per_minute']}
                                  for key, values in result['measurements'].items()
                                  for band in ('alpha' if key == 'theta_alpha' else 'beta',)}
        for values in result['measurements'].values():
            del values['slope_ratio_per_minute']
        result['reason'] = '已计算连续额区alpha/beta功率的变异与趋势。' if result['available'] else '额区频带时间测量不可用：' + result['reason']
        result['missing'] = [field.replace('arrived.fatigue_history', 'arrived.emotion_history') for field in result['missing']]
        return result
    raise ValueError('Unregistered specialist tool')


def execute_tool(tool_id, evidence):
    if any(tool_id == spec['id'] for specs in _RAW_CATALOG.values() for spec in specs):
        if not isinstance(evidence, dict) or 'recording' not in evidence or tool_id == 'vrms.raw_model':
            raise ValueError('Raw recording/model context required')
        from .recordings import recording_tool
        return recording_tool(tool_id, evidence)
    result = _execute_tool(tool_id, evidence)
    if result['available'] and tool_id.startswith(('fatigue.', 'emotion.')):
        context = evidence.get('measurement_context', {}) if isinstance(evidence, dict) else {}
        observed = _fields(context, ('source', 'path_id', 'aggregation', 'sample_rate_hz', 'window_seconds',
                                    'accepted_windows', 'analyzed_seconds', 'path_duration_seconds', 'coverage',
                                    'preprocessing_profile', 'signal_sha256', 'timing_sha256'))
        if observed:
            result['measurements']['observation'] = observed
    return result
