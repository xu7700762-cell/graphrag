"""Build a local, traceable EOG effect report from completed artifacts.

Run after the final RLS cache, both model exports and evaluations are ready.
No training, replay, credential loading, cloud request or raw EEG transfer is
performed. Use repeatable --task arguments to add completed replay evidence.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_agent.config import PROJECT, SPLITS, cache_path, contract, digest, file_hash, load_config, profile_config, write_json
from eeg_agent.learning import metrics
from eeg_agent.models import MODEL_IDS, inference_contract


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def distribution(values):
    data = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    return {'count': len(data), 'mean': float(data.mean()) if len(data) else None,
            'median': float(np.median(data)) if len(data) else None,
            'p10': float(np.percentile(data, 10)) if len(data) else None,
            'p90': float(np.percentile(data, 90)) if len(data) else None,
            'minimum': float(data.min()) if len(data) else None,
            'maximum': float(data.max()) if len(data) else None}


def paired_measurements(rows):
    """Compare the same processed window pairs; no neural-fidelity inference."""
    result = {'windows': len(rows)}
    for name, before_key, after_key in (
            ('frontal_max_abs_eeg_eog_correlation', 'pre_frontal_eog_correlation', 'post_frontal_eog_correlation'),
            ('maximum_channel_peak_to_peak_uv', 'pre_peak_to_peak_uv', 'post_peak_to_peak_uv')):
        pairs = [(row.get(before_key), row.get(after_key)) for row in rows]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None and np.isfinite([a, b]).all()]
        result[name] = {'before': distribution([a for a, _ in pairs]),
                        'after': distribution([b for _, b in pairs]),
                        'paired_after_minus_before': distribution([b-a for a, b in pairs]),
                        'fraction_decreased': float(np.mean([b < a for a, b in pairs])) if pairs else None}
    result['maximum_channel_removed_demeaned_rms_uv'] = distribution([row.get('removed_rms_uv') for row in rows])
    result['maximum_channel_removed_rms_ratio'] = distribution([row.get('removed_rms_ratio') for row in rows])
    return result


def collect_natural(cfg, evidence):
    cache = cache_path(cfg)
    manifest_path = cache / 'manifest.json'
    manifest = read_json(manifest_path)
    if manifest.get('contract_hash') != digest(contract(cfg)):
        raise ValueError('Final RLS cache preprocessing contract mismatch')
    evidence(manifest_path, 'final_P1_cache_manifest')
    per_subject, applied_rows, accepted_rows = [], [], []
    combined_reasons = Counter()
    seen = set()
    for entry in manifest['subjects']:
        sid = int(entry['subject'])
        if sid in seen: raise ValueError('Duplicate subject in cache manifest')
        seen.add(sid)
        archive_file = cache / ('subject_%02d.npz' % sid)
        if file_hash(archive_file) != entry['cache_sha256']:
            raise ValueError('Final subject cache hash mismatch')
        evidence(archive_file, 'P1_subject_cache')
        with np.load(archive_file) as archive:
            arrays = {key: archive[key] for key in ('valid', 'valid1', 'p1', 'window_start', 'window_end')}
        size = len(arrays['valid1'])
        if any(len(values) != size for values in arrays.values()):
            raise ValueError('Unequal cache metadata lengths')
        valid1 = np.asarray(arrays['valid1'], dtype=bool)
        if np.any(valid1 & ~np.asarray(arrays['p1'], dtype=bool)):
            raise ValueError('Uncalibrated P1 cache windows were accepted')
        eog_file = cache / ('eog_%02d.json' % sid)
        quality_file = cache / ('quality_%02d.json' % sid)
        rows = read_json(eog_file)
        quality_rows = read_json(quality_file)
        evidence(eog_file, 'P1_window_EOG_summaries')
        evidence(quality_file, 'P1_window_quality_summaries')
        by_interval = {(int(row['start_sample']), int(row['end_sample'])): row for row in rows}
        by_start = {int(row['start_sample']): row for row in quality_rows}
        if len(rows) != size or len(by_interval) != size or len(quality_rows) != size or len(by_start) != size:
            raise ValueError('P1 window summaries are incomplete or duplicated')
        applied, accepted, reasons, statuses = [], [], Counter(), Counter()
        for i in range(size):
            key = (int(arrays['window_start'][i]), int(arrays['window_end'][i]))
            row = by_interval[key]
            quality = by_start[key[0]]
            if bool(quality['valid']) != bool(valid1[i]) or bool(row.get('input_available')) != bool(valid1[i]):
                raise ValueError('P1 summary quality gate differs from cached valid1')
            statuses[row['status']] += 1
            if row.get('applied'):
                applied.append(row)
            if valid1[i]:
                accepted.append(row)
            else:
                reasons.update(quality.get('reasons') or ['unspecified_rejection'])
        combined_reasons.update(reasons)
        applied_rows.extend(applied)
        accepted_rows.extend(accepted)
        per_subject.append({'subject': sid, 'windows': size,
                            'P0_quality_valid_windows': int(np.asarray(arrays['valid'], dtype=bool).sum()),
                            'P1_accepted_windows': int(valid1.sum()), 'P1_rejected_windows': int(size-valid1.sum()),
                            'P1_acceptance_rate': float(valid1.mean()) if size else None,
                            'calibration': entry['calibration'],
                            'status_counts': dict(statuses), 'rejection_reason_counts': dict(reasons),
                            'paired_all_applied_windows': paired_measurements(applied),
                            'paired_accepted_P1_windows': paired_measurements(accepted),
                            'cache_sha256': entry['cache_sha256'], 'raw_sha256': entry['raw_sha256']})
    expected_subjects = {sid for values in SPLITS.values() for sid in values}
    if seen != expected_subjects:
        raise ValueError('Final RLS cache does not contain the complete assigned subject set')
    failed = [row for row in per_subject if not row['calibration']['available']]
    totals = {key: sum(row[key] for row in per_subject)
              for key in ('windows', 'P0_quality_valid_windows', 'P1_accepted_windows', 'P1_rejected_windows')}
    return {'subjects': len(per_subject), 'totals': totals,
            'calibration_available_subjects': len(per_subject)-len(failed),
            'calibration_failed_subjects': [{'subject': row['subject'], **row['calibration']} for row in failed],
            'subjects_without_accepted_P1_windows': [row['subject'] for row in per_subject if not row['P1_accepted_windows']],
            'rejection_reason_counts': dict(combined_reasons),
            'reason_count_note': 'One rejected window can have multiple reasons; reason counts need not sum to window count.',
            'paired_all_applied_windows': paired_measurements(applied_rows),
            'paired_accepted_P1_windows': paired_measurements(accepted_rows),
            'per_subject': sorted(per_subject, key=lambda row: row['subject']),
            'measurement_note': 'Pairs use identical arrived windows. Correlation decrease and amplitude removal are measurements, not proof of neural fidelity.'}, manifest


def numeric_delta(after, before):
    return float(after-before) if after is not None and before is not None else None


def compare_models(cfg, manifest, evidence):
    output = Path(cfg['output_root'])
    comparisons = []
    for model_id in MODEL_IDS:
        artifacts = {}
        for profile, checkpoint in (('P0', 'fixed_seed42'), ('P1', 'eog_rls_seed42')):
            evaluation_file = output / 'reports' / ('evaluation_%s_%s_test.json' % (model_id, checkpoint))
            directory = output / 'models' / model_id / checkpoint
            model_file = directory / 'contract.json'
            evaluation = read_json(evaluation_file)
            model = read_json(model_file)
            if (model['allowed_profiles'] != [profile] or model['model_id'] != model_id or
                    model['contract_hash'] != digest(contract(profile_config(cfg, profile)))):
                raise ValueError('Comparison model identity/profile/contract mismatch')
            if (model.get('inference_contract_hash') != digest(inference_contract(model)) or
                    file_hash(directory / 'model.onnx') != model['onnx_sha256'] or
                    file_hash(directory / 'checkpoint.pt') != model['checkpoint_sha256']):
                raise ValueError('Comparison model file or inference contract hash mismatch')
            if (evaluation.get('model_hash') != model['onnx_sha256'] or
                    evaluation.get('contract_hash') != model['contract_hash'] or evaluation.get('split') != 'test'):
                raise ValueError('Comparison evaluation does not match model/preprocessing')
            if profile == 'P1' and model.get('cache_manifest_hash') != digest(manifest):
                raise ValueError('P1 model was not trained on the final cache manifest')
            if set(model['normalization_fit_subjects']) & set(SPLITS['test']):
                raise ValueError('Test subjects appeared in model normalization fitting')
            threshold = float(model['threshold'])
            if float(evaluation.get('threshold', evaluation['metrics']['threshold'])) != threshold:
                raise ValueError('Comparison evaluation threshold differs from frozen model')
            records = evaluation['paths']
            by_key = {row['key']: row for row in records}
            if len(records) != 23 or len(by_key) != 23:
                raise ValueError('Fixed test comparison requires all 23 paths in its denominator')
            artifacts[profile] = {'records': records, 'by_key': by_key, 'threshold': threshold,
                                  'all_paths': metrics(records, threshold),
                                  'model': {'checkpoint_id': checkpoint, 'threshold': threshold,
                                            'temperature': model['temperature'], 'selected_epoch': model['selected_epoch'],
                                            'parameter_count': model['parameter_count'],
                                            'training_seconds': model['training_seconds'],
                                            'assigned_training_subjects': model['assigned_training_subjects'],
                                            'normalization_fit_subjects': model['normalization_fit_subjects'],
                                            'validation_subjects': model['validation_subjects'],
                                            'contributing_training_paths': model['contributing_training_paths'],
                                            'calibration': model['calibration'], 'threshold_selection_status': model['threshold_selection']['status'],
                                            'onnx_validation': model['onnx_validation'],
                                            'onnx_sha256': model['onnx_sha256'], 'inference_contract_hash': model['inference_contract_hash'],
                                            'inference_p95_seconds': evaluation.get('inference_p95_seconds')}}
            for path, role in ((evaluation_file, profile+'_test_evaluation'), (model_file, profile+'_model_contract'),
                               (directory / 'model.onnx', profile+'_model_ONNX'), (directory / 'checkpoint.pt', profile+'_model_checkpoint'),
                               (directory / 'training_history.json', profile+'_training_history')):
                evidence(path, role)
        p0, p1 = artifacts['P0'], artifacts['P1']
        if set(p0['by_key']) != set(p1['by_key']):
            raise ValueError('P0/P1 test path keys differ')
        if any(p0['by_key'][key]['label'] != p1['by_key'][key]['label'] for key in p0['by_key']):
            raise ValueError('P0/P1 test labels differ')
        accepted0 = {row['key'] for row in p0['records'] if row['status'] == 'accepted'}
        accepted1 = {row['key'] for row in p1['records'] if row['status'] == 'accepted'}
        common = sorted(accepted0 & accepted1)
        common0 = metrics([p0['by_key'][key] for key in common], p0['threshold'])
        common1 = metrics([p1['by_key'][key] for key in common], p1['threshold'])
        common_predictions = []
        for key in common:
            before, after = p0['by_key'][key], p1['by_key'][key]
            common_predictions.append({'key': key, 'label': before['label'],
                                       'P0_probability': before['probability_high'], 'P1_probability': after['probability_high'],
                                       'P0_class': int(before['probability_high'] >= p0['threshold']),
                                       'P1_class': int(after['probability_high'] >= p1['threshold'])})
        comparisons.append({'model_id': model_id, 'test_path_denominator': 23,
                            'P0': {'model': p0['model'], 'all_paths': p0['all_paths'], 'common_accepted_paths': common0},
                            'P1': {'model': p1['model'], 'all_paths': p1['all_paths'], 'common_accepted_paths': common1},
                            'common_accepted_count': len(common), 'common_accepted_keys': common,
                            'accepted_only_P0_keys': sorted(accepted0-accepted1), 'accepted_only_P1_keys': sorted(accepted1-accepted0),
                            'common_accepted_metric_deltas_P1_minus_P0':
                                {key: numeric_delta(common1[key], common0[key]) for key in ('BACC', 'AUROC', 'Macro_F1', 'Brier', 'ECE_10_equal_width')},
                            'common_accepted_predictions': common_predictions,
                            'interpretation': 'Two separately retrained processing/model/calibration pipelines. Common-path comparison controls acceptance selection only; it does not isolate artifact removal as the cause of a classification difference.'})
    return comparisons


def compact_task(report):
    return {key: report.get(key) for key in ('task_id', 'model', 'source_seconds', 'wall_seconds', 'timing_kind',
             'platform', 'paths', 'baseline', 'eog_calibration', 'eog_counts', 'contract_hash',
             'processing_seconds', 'inference_p95_seconds', 'window_completion_lag_p95_seconds',
             'peak_process_rss_bytes', 'process_cpu_seconds', 'tool_counts', 'tool_seconds', 'cloud', 'stopped')}


def number(value, places=4):
    return '不可用' if value is None else ('%.*f' % (places, value))


def percent(value):
    return '不可用' if value is None else ('%.2f%%' % (100*value))


def render(report):
    controlled, natural = report['controlled'], report['natural_data']
    overall = natural['paired_all_applied_windows']
    corr = overall['frontal_max_abs_eeg_eog_correlation']
    totals = natural['totals']
    lines = ['# 眼电参考去伪迹实施与效果报告', '',
             '本版在因果P0处理后增加VEOG／HEOG参考RLS眼电回归，并分别重训EEGNet、EEGConformer。以下将受控注入效果、真实数据测量、分类评价和运行验收分别报告。', '',
             '## 处理方法与适用边界', '',
             '先用首条任务前已到达、预热后合格的10—60秒无标签基线初始化；随后按256 Hz逐样本先预测眼电贡献并减去，再用该样本更新系数。遗忘时间固定10秒，休息段持续处理；断流后重置为先行初始化并重新预热。算法参数在任务中冻结，自适应系数持续更新。',
             '',
             '这属于目标被试的因果无标签预处理适配，不属于仅使用训练被试信息的源域模型推理。训练、验证和测试各自按被试完整到达流准备数据，分类器归一化仍只从训练集拟合。',
             '',
             '基线频谱与协方差采用初始冻结回归映射清理；当前任务窗口采用更新中的RLS系数。相对基线变化可能同时反映信号变化和处理映射变化，不能直接视为症状变化。',
             '',
             '初始化失败时保留P0测量并拒绝P1分类；清理后残余相关或质量门控未通过时，拒绝当前窗口，不将其回退成有效P0分类输入。连续采流继续运行。云端不执行滤波或眼电数值计算。', '',
             '## 有干净参考的受控注入', '',
             '验证使用独立校准／留出随机种子，并在留出阶段将眼电混合系数增加到初始值的%s倍，检查对混合变化的适配。'%number(controlled['mixing_step_factor'], 1),
             '',
             '- 全部留出记录的眼电误差RMSE：%s → %s μV，下降%s；SNR改善%s dB。'%(
                 number(controlled['entire_recording']['artifact_rmse_uv_before']), number(controlled['entire_recording']['artifact_rmse_uv_after']),
                 percent(controlled['entire_recording']['artifact_rmse_reduction_fraction']), number(controlled['entire_recording']['snr_improvement_db'], 2)),
             '- 首20秒之后的40秒区间：RMSE下降%s；SNR改善%s dB。'%(
                 percent(controlled['after_first_20_seconds']['artifact_rmse_reduction_fraction']), number(controlled['after_first_20_seconds']['snr_improvement_db'], 2)),
             '- α／β平均相对功率误差：%s／%s；无眼电混合负对照的相对RMS变化为%s。'%(
                 percent(controlled['band_power_fidelity']['alpha']['mean_absolute_relative_error']),
                 percent(controlled['band_power_fidelity']['beta']['mean_absolute_relative_error']),
                 percent(controlled['no_mix_negative_control_relative_rms_change'])),
             '- 边界对照：当真实低频神经活动与EOG参考共享成分时，仅保留%s的该低频功率。回归会同时移除相关的真实脑电；物理眼电电极也不能保证完美分离。'%percent(controlled['correlated_neural_limit']['low_band_power_retained_fraction']),
             '',
             '以上证明特定受控条件下的数值去除与保真，并未证明自然数据中的全部神经活动都被保留。', '',
             '## 原始CDT的成对测量', '',
             '总计%d名被试、%d个完整任务窗口；先行校准可用%d人，失败%d人。P0质量合格%d窗，P1分类接受%d窗、拒绝%d窗。'%(
                 natural['subjects'], totals['windows'], natural['calibration_available_subjects'], len(natural['calibration_failed_subjects']),
                 totals['P0_quality_valid_windows'], totals['P1_accepted_windows'], totals['P1_rejected_windows']),
             '',
             '对实际应用RLS的%d个同一窗口对，额区最大绝对EEG–EOG相关中位数为%s → %s，逐窗差值中位数%s；最大通道峰峰值中位数为%s → %s μV。'%(
                 overall['windows'], number(corr['before']['median']), number(corr['after']['median']),
                 number(corr['paired_after_minus_before']['median']),
                 number(overall['maximum_channel_peak_to_peak_uv']['before']['median']),
                 number(overall['maximum_channel_peak_to_peak_uv']['after']['median'])),
             '',
             '去除量的最大通道去均值RMS中位数为%s μV；完整分布、所有清理窗口与接受窗口的分层统计在JSON中保存。相关下降可能由回归机制本身产生，只说明测量变化。'%number(overall['maximum_channel_removed_demeaned_rms_uv']['median']), '',
             '| 本地被试编号 | 校准 | P0合格窗 | P1接受／总窗 | 相关中位数：前→后 | 主要拒绝原因 |',
             '|---|---|---:|---:|---|---|']
    for row in natural['per_subject']:
        pair = row['paired_all_applied_windows']['frontal_max_abs_eeg_eog_correlation']
        reasons = '；'.join('%s:%d' % item for item in sorted(row['rejection_reason_counts'].items())) or '无'
        lines.append('| %02d | %s | %d | %d／%d | %s → %s | %s |'%(
            row['subject'], '可用' if row['calibration']['available'] else row['calibration']['reason'],
            row['P0_quality_valid_windows'], row['P1_accepted_windows'], row['windows'],
            number(pair['before']['median']), number(pair['after']['median']), reasons.replace('|', '/')))
    lines += ['', '一个拒绝窗口可能有多个原因，原因计数不能相加当作拒绝窗数。', '',
              '## 固定跨被试划分的分类评价', '',
              '每种模型均以全部23条测试路径作为覆盖率分母；BACC、AUROC等分类指标仅在接受路径上计算。另比较两套流程都接受的路径，避免不同拒绝集合造成表面性能差异。当前数据曾用于原型开发，结果属于开发数据评价，不能称为未接触的独立测试。', '',
              '| 模型／输入 | 接受路径／23 | BACC | AUROC | Macro-F1 | Brier | ECE | 温度／阈值 |',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for comparison in report['classification_comparison']:
        for profile in ('P0', 'P1'):
            data = comparison[profile]
            measures, model = data['all_paths'], data['model']
            lines.append('| %s／%s | %d／23 | %s | %s | %s | %s | %s | %s／%s |'%(
                comparison['model_id'], profile, measures['accepted_paths'], number(measures['BACC']), number(measures['AUROC']),
                number(measures['Macro_F1']), number(measures['Brier']), number(measures['ECE_10_equal_width']),
                number(model['temperature']), number(model['threshold'])))
    lines += ['', '| 模型 | 共同接受路径 | P0 BACC | P1 BACC | 差值 | P0／P1 AUROC |',
              '|---|---:|---:|---:|---:|---|']
    for comparison in report['classification_comparison']:
        before, after = comparison['P0']['common_accepted_paths'], comparison['P1']['common_accepted_paths']
        lines.append('| %s | %d | %s | %s | %s | %s／%s |'%(
            comparison['model_id'], comparison['common_accepted_count'], number(before['BACC']), number(after['BACC']),
            number(comparison['common_accepted_metric_deltas_P1_minus_P0']['BACC']), number(before['AUROC']), number(after['AUROC'])))
    lines += ['', '| 模型／输入 | 低类接受／总数 | 高类接受／总数 | Sensitivity | Specificity |',
              '|---|---:|---:|---:|---:|']
    for comparison in report['classification_comparison']:
        for profile in ('P0', 'P1'):
            measures = comparison[profile]['all_paths']
            low, high = measures['class_acceptance']['0'], measures['class_acceptance']['1']
            lines.append('| %s／%s | %d／%d | %d／%d | %s | %s |'%(
                comparison['model_id'], profile, low['accepted'], low['total'], high['accepted'], high['total'],
                number(measures['sensitivity']), number(measures['specificity'])))
    changes = [comparison['common_accepted_metric_deltas_P1_minus_P0']['BACC']
               for comparison in report['classification_comparison']]
    if all(change is not None and change < 0 for change in changes):
        lines += ['', '本轮两种重训模型在共同接受路径上的BACC均降低，P1接受路径也减少；因此本轮未证明分类改善。眼电去除的受控效果与模型分类效度需要分别判断。']
    lines += ['', 'JSON保存分类别接受率、拒绝原因、共同路径清单和逐路径预测。两套流程分别重训并拟合归一化、温度和阈值，因此任何指标差值都不能单独归因于眼电去除；不预设清理必然提高分类性能。', '',
              '## 运行与验收证据', '']
    if report['replay_tasks']:
        for task in report['replay_tasks']:
            lines.append('- 任务%s，%s，%s；源数据%s秒，墙钟%s秒，窗口发布延迟P95 %s秒；语言接口%s。'%(
                task['task_id'], task['model']['model_id'], task['timing_kind'], number(task['source_seconds'], 2),
                number(task['wall_seconds'], 2), number(task.get('window_completion_lag_p95_seconds')),
                'mock模拟' if task.get('cloud', {}).get('simulated') else '按任务文件所记接口'))
    else:
        lines.append('本报告生成时未添加回放任务文件；不能据此声称已完成本版1倍速验收。')
    for item in report['verification_evidence']:
        if item['kind'] == 'test_log':
            lines.append('- 测试日志%s：%s。'%(item['path'], item.get('test_summary') or '未解析到最终测试数量'))
        elif item['kind'] == 'online_offline':
            lines.append('- 数值一致性%s：%s，处理版本%s。'%(item['path'], item.get('status'), item.get('processing_version')))
        else:
            lines.append('- 浏览器验收%s：%s；范围%s。'%(item['path'], item.get('status'), item.get('scope')))
    lines += ['', 'Windows运行、mock Agent调用、真实云端收益及Pi硬件性能分别验收；本脚本不调用云端API。没有逐时症状标签，不能给出真实症状onset、提前量或严重度效度。', '',
              '## 可追溯产物', '',
              'JSON保存本次处理合同、模型及缓存哈希、受控验证、数值一致性、浏览器／测试日志和所指定回放任务的文件哈希。原始CDT直接读取，未复制；API密钥和.env不纳入报告。', '',
              '报告生成时间（UTC）：'+report['generated_utc']+'。']
    return '\n'.join(lines)+'\n'


def summarize(cfg, task_ids=()):
    cfg = profile_config(cfg, 'P1')
    output = Path(cfg['output_root'])
    sources = {}

    def evidence(path, role):
        path = Path(path).resolve()
        item = sources.setdefault(str(path), {'path': str(path), 'sha256': file_hash(path), 'roles': []})
        if role not in item['roles']: item['roles'].append(role)

    natural, manifest = collect_natural(cfg, evidence)
    controlled_file = output / 'reports/eog_controlled_validation.json'
    controlled = read_json(controlled_file)
    if (not controlled.get('passed') or controlled.get('processing_version') != contract(cfg)['eog_cleaning']['version']
            or controlled.get('policy') != contract(cfg)['eog_cleaning']):
        raise ValueError('Controlled validation is absent, failed or from a different RLS policy')
    if controlled.get('processing_code_sha256') != file_hash(PROJECT / 'eeg_agent/eog.py'):
        raise ValueError('Controlled validation does not match current processing code')
    evidence(controlled_file, 'clean_reference_controlled_validation')
    comparisons = compare_models(cfg, manifest, evidence)
    tasks = []
    for task_id in dict.fromkeys(task_ids):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', task_id): raise ValueError('Invalid task identifier')
        task_file = output / 'tasks' / task_id / 'report.json'
        task = read_json(task_file)
        if task.get('contract_hash') != manifest['contract_hash']:
            raise ValueError('Replay task uses a different processing contract')
        tasks.append(compact_task(task))
        evidence(task_file, 'P1_replay_report')
        for name in ('events.jsonl', 'context.json'):
            path = task_file.parent / name
            if path.exists(): evidence(path, 'P1_replay_'+name)
    verification = []
    for path in sorted((output / 'reports').glob('eog_online_offline_*.json')):
        data = read_json(path)
        current = data.get('contract_hash') == manifest['contract_hash']
        verification.append({'kind': 'online_offline', 'path': str(path.resolve()), 'status': data.get('status'),
                             'processing_version': data.get('processing_version'), 'current_contract': current,
                             'models_checked': data.get('models_checked'), 'subject': data.get('subject')})
        evidence(path, 'online_offline_current' if current else 'online_offline_historical')
    browser_files = sorted((output / 'reports').glob('*eog*acceptance*.json'))
    for path in browser_files:
        data = read_json(path)
        verification.append({'kind': 'browser', 'path': str(path.resolve()), 'status': data.get('status'),
                             'scope': data.get('boundary', data.get('scope', '按该验收文件记录的操作范围'))})
        evidence(path, 'EOG_browser_acceptance')
    if not browser_files:
        legacy = output / 'reports/browser_acceptance.json'
        if legacy.exists():
            verification.append({'kind': 'browser', 'path': str(legacy.resolve()), 'status': 'historical_only',
                                 'scope': '既有P0浏览器验收，不作为本版眼电链验收'})
            evidence(legacy, 'historical_P0_browser_acceptance')
    logs = [output / 'reports' / name for name in ('eog_pytest.log', 'eog_final_pytest.log')]
    for path in logs:
        if not path.exists(): continue
        content = path.read_text(encoding='utf-8', errors='replace')
        summaries = re.findall(r'\d+ passed[^\r\n]*', content)
        verification.append({'kind': 'test_log', 'path': str(path.resolve()),
                             'test_summary': summaries[-1] if summaries else None})
        evidence(path, 'EOG_test_log')
    for name in ('eog_training.log', 'eog_evaluation.log', 'eog_prepare.log'):
        path = output / 'reports' / name
        if path.exists(): evidence(path, 'EOG_training_evaluation_or_preparation_log')
    for filename in ('eog.py', 'signal.py', 'config.py', 'learning.py', 'models.py', 'monitor.py'):
        evidence(PROJECT / 'eeg_agent' / filename, 'processing_or_runtime_code')
    audit_file = output / 'reports/data_audit.json'
    if audit_file.exists(): evidence(audit_file, 'raw_data_and_event_audit')
    evidence(Path(__file__), 'report_generator_code')
    report = {'schema': 1, 'generated_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'completed_artifact_summary_with_separate_validation_boundaries',
              'processing_version': contract(cfg)['eog_cleaning']['version'],
              'processing_contract': contract(cfg), 'contract_hash': manifest['contract_hash'],
              'cache_manifest_hash': digest(manifest), 'subject_splits': SPLITS,
              'data_provenance': {'label_sha256': manifest['label_sha256'],
                                  'raw_subject_hashes': {str(entry['subject']): entry['raw_sha256'] for entry in manifest['subjects']},
                                  'raw_copy': manifest.get('raw_copy', False)},
              'controlled': controlled, 'natural_data': natural, 'classification_comparison': comparisons,
              'replay_tasks': tasks, 'verification_evidence': verification,
              'interpretation_boundaries': [
                  'Target-subject causal unlabeled preprocessing adaptation; not source-only inductive inference.',
                  'Natural correlation decrease is not a clean reference or proof of neural fidelity.',
                  'Correlated real neural components can be removed with the EOG reference.',
                  'Initial frozen-map baseline versus adaptive current map can contribute to measured changes.',
                  'All 23 test paths remain the acceptance denominator; accuracy metrics are conditional on acceptance.',
                  'Common-accepted comparison controls acceptance selection only; two pipelines are separately retrained and calibrated.',
                  'Development data previously inspected; not untouched independent testing.',
                  'Path labels do not validate instantaneous symptoms, onset, lead time or severity.',
                  'Windows, mock, real cloud and Raspberry Pi hardware are separate validation scopes.'],
              'source_files': list(sources.values()), 'cloud_requests_by_this_script': 0}
    destination = output / 'reports/eog_effect_report.json'
    write_json(destination, report)
    (destination.with_suffix('.md')).write_text(render(report), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--task', action='append', default=[], help='Completed P1 replay task identifier; repeatable')
    args = parser.parse_args()
    result = summarize(load_config(args.config), args.task)
    print(json.dumps({'report': str(Path(load_config(args.config)['output_root']) / 'reports/eog_effect_report.md'),
                      'subjects': result['natural_data']['subjects'],
                      'P1_accepted_windows': result['natural_data']['totals']['P1_accepted_windows'],
                      'models': [row['model_id'] for row in result['classification_comparison']]}, ensure_ascii=True))
