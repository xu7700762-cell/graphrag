"""Bounded local CUDA worker for raw-only conversation tools."""
import copy
import json
import os
from pathlib import Path
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import raw_tool_agent
from . import raw_independent_judge
from .default_rag import attach as attach_default_rag

_ORIGINAL_LEGACY_CLASSIFY = raw_tool_agent.classify

LOCK = threading.Lock()
ROOT = Path(r'D:\桌面\文献总\EEGAGENT')
PYTHON = '/opt/miniconda3/envs/pytorch/bin/python'
DISTRO = 'Ubuntu-22.04-Bio'


def _wsl(path):
    path = Path(path).resolve()
    return '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]


def public_result(result):
    """Expose cloud conclusions; numeric comparisons and API logs stay local."""
    if not result.get('cloud_requested', result.get('gpt_requested')):
        return result
    result = copy.deepcopy(result)
    for row in result['measurements']['segments']:
        for key in ('numeric_fusion', 'mil_class', 'mil_raw_high_probability', 'threshold',
                    'current_raw_contrastive_context'):
            row.pop(key, None)
        for key in ('rounds', 'tool_outputs', 'tool_timings_ms', 'raw_llm_final'):
            row.get('cloud_judgment', {}).pop(key, None)
    return result


def assess(repository, recording_id, evidence, provider=None, cloud=False, rag_policy=None):
    record = evidence['recording']
    result = {'tool': 'vrms.raw_model', 'domain': 'vrms', 'available': False,
              'input_policy': 'raw_eeg_only', 'classifier_required': True, 'classifier_available': False,
              'signal_sha256': record['signal_sha256'], 'measurements': {'segments': [], 'first_high': None}}
    if rag_policy is not None:
        result['rag_policy'] = copy.deepcopy(rag_policy)
        result['rag_mode'] = rag_policy.get('rag_mode')
    if record['task_count'] is None:
        result['reason'] = '原始文件没有任务开始与结束事件，未形成整路径模型输入；指标工具继续分析'
        return result
    if not record['model_compatible']:
        result['reason'] = '该文件不满足 VRMS 输入合同：需要1024Hz、30个指定EEG通道及M1/M2参考；其他工具继续分析'
        return result
    fold = int(recording_id[-2:]) if recording_id.startswith('subject-') else 1
    if not (ROOT / f'outputs/vrms_model_ablation/v3_20261008/model_results/checkpoints/outer_subject_{fold:02d}_femba_frozen_mil.pt').is_file():
        result['reason'] = '该记录没有对应的留一被试 VRMS 权重；本轮提供原始指标与路径统计'
        return result
    inputs = repository.model_inputs(recording_id)
    directory = repository.root / 'inference' / uuid.uuid4().hex
    directory.mkdir(parents=True)
    request = {'fold': fold,
               'signal_sha256': record['signal_sha256'], 'segments': [], 'baseline_file': False}
    # Recording starts during VR exposure. Its opening waveform is not an
    # established resting reference; use the trained missing-reference branch.
    for segment, (windows, stamps) in zip(record['segments'], inputs):
        row = {k: segment[k] for k in ('ordinal', 'start_seconds', 'end_seconds', 'start_clock', 'end_clock', 'complete')}
        stats = segment['stats']
        row.update(predicted_class=None, reason='')
        if not segment['complete']:
            row['reason'] = '缺少结束标记，未发布整路径分类'
        elif len(windows) < repository.cfg['minimum_path_windows'] or stats.get('window_coverage', 0) < repository.cfg['minimum_path_coverage']:
            row['reason'] = '合格窗口数量或覆盖率未达到模型发布规则'
        else:
            filename = f"segment-{segment['ordinal']:03d}.npy"
            np.save(directory / filename, windows, allow_pickle=False)
            request['segments'].append({'ordinal': segment['ordinal'], 'file': filename})
        result['measurements']['segments'].append(row)
    if not request['segments']:
        result['reason'] = '没有满足完整性、质量和覆盖率要求的路径'
        return result
    request_path = directory / 'request.json'
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
    command = ['wsl.exe', '-d', DISTRO, '--cd', _wsl(ROOT), '--exec', PYTHON,
               '-B', '-m', 'vrms_cloud.raw_inference', _wsl(request_path)]
    try:
        with LOCK:
            completed = subprocess.run(command, capture_output=True,
                timeout=240, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        value = json.loads(completed.stdout.decode('utf-8'))
        (directory / 'worker_result.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        if not value.get('available'):
            result['reason'] = value.get('reason', '原始波形编码未完成')
            return result
        actual = {row['ordinal']: row for row in value['segments']}
        expected = {row['ordinal'] for row in request['segments']}
        if set(actual) != expected or value['signal_sha256'] != record['signal_sha256'] or value.get('fresh_raw_encoding') is not True:
            raise ValueError('Worker/source mismatch')
        for row in result['measurements']['segments']:
            if row['ordinal'] in actual:
                prediction = (attach_default_rag(fold, actual[row['ordinal']], rag_policy=rag_policy)
                              if rag_policy is not None else attach_default_rag(fold, actual[row['ordinal']])) if cloud else actual[row['ordinal']]
                p = prediction['numeric_fusion']['raw_combined_score']
                if not isinstance(p, (int, float)) or not 0 <= p <= 1 or prediction['predicted_class'] != ('High' if p >= .5 else 'Low'):
                    raise ValueError('Invalid raw model prediction')
                row.update(prediction)
        if cloud:
            from .raw_evidence_teaching import build_teaching_context
            teaching = build_teaching_context(repository.cfg)
            # Production always uses the independent six-field judge. The
            # legacy hook remains injectable for existing offline test doubles
            # and explicit rollback tests, without changing the default path.
            classify = (raw_tool_agent.classify
                        if raw_tool_agent.classify is not _ORIGINAL_LEGACY_CLASSIFY
                        else raw_independent_judge.classify)
            provider_model = str(getattr(provider, 'model', '') or '')
            vendor = 'gpt' if provider_model.startswith('gpt-') else 'deepseek' if provider_model.startswith('deepseek-') else 'relay'
            label = 'GPT' if vendor == 'gpt' else 'DeepSeek' if vendor == 'deepseek' else '云端模型'
            predictions = [row for row in result['measurements']['segments'] if row['ordinal'] in actual]
            with ThreadPoolExecutor(max_workers=min(3, len(predictions)), thread_name_prefix='raw-gpt') as pool:
                judgments = list(pool.map(lambda row: classify(provider, row, teaching_context=teaching), predictions))
            for row, judgment in zip(predictions, judgments):
                # Scores and intermediate classes remain intact when the cloud
                # disagrees. A failed cloud request never becomes a final class.
                row['cloud_judgment'] = judgment
                row['predicted_class'] = judgment['final_class'] if judgment['available'] else None
                row['classification_layer'] = vendor + '_tool_agent' if judgment['available'] else 'unavailable_cloud_tool_agent'
                row['reason'] = '' if judgment['available'] else judgment['reason']
                (directory / f"{vendor}_path_{row['ordinal']:03d}.json").write_text(
                    json.dumps(judgment, ensure_ascii=False, indent=2), encoding='utf-8')
            succeeded = sum(judgment['available'] for judgment in judgments)
            result.update(cloud_requested=True, cloud_successful_paths=succeeded,
                          cloud_failed_paths=len(judgments) - succeeded, cloud_provider_model=provider_model)
            result['rag_provenance'] = [{
                'ordinal': row['ordinal'],
                'available': bool(judgment.get('rag_context_available')),
                'schema': judgment.get('rag_schema'),
                'variant': judgment.get('rag_variant'),
                'source_hashes': copy.deepcopy(judgment.get('rag_source_hashes', {})),
                'payload_sha256': judgment.get('payload_sha256'),
                'system_sha256': judgment.get('system_sha256'),
            } for row, judgment in zip(predictions, judgments)]
            if vendor == 'gpt':
                result.update(gpt_requested=True, gpt_successful_paths=succeeded,
                              gpt_failed_paths=len(judgments) - succeeded)
        high = next((row for row in result['measurements']['segments'] if row['predicted_class'] == 'High'), None)
        if high:
            result['measurements']['first_high'] = {k: high[k] for k in ('ordinal', 'start_seconds', 'end_seconds', 'start_clock', 'end_clock')}
            result['measurements']['first_high']['decision_available_seconds'] = high['end_seconds']
        result.update(available=True, classifier_available=True, status='ok', reason='VRMS 模型已重新编码原始 EEG 并完成整路径分类',
                      fresh_raw_encoding=True, model_sha256=value['model_sha256'],
                      encoder_sha256=value['encoder_sha256'], normalization_sha256=value['normalization_sha256'])
        result.update({key: value[key] for key in ('numeric_first_sha256', 'numeric_second_sha256', 'numeric_selector_sha256')})
        if cloud:
            result.update(status='partial_fallback' if result['cloud_failed_paths'] else 'ok',
                reason='已重新编码原始 EEG 并完成 ' + label + ' 工具 Agent 判断' if not result['cloud_failed_paths'] else '部分云端判断未完成，数值结果已保留且未代替云端分类')
        result['measurements'].update(readout='whole_path_' + vendor + '_tool_agent' if cloud else 'whole_path_mil_numeric_fusion', probability_calibrated=False,
            baseline_policy='disabled', comparison_policy='event_segments_only',
            onset_scope='首条高类完整路径的时间区间；判定使用整条路径数据，不是路径内首次症状发生时刻')
    except subprocess.TimeoutExpired:
        result['reason'] = '原始波形编码超过四分钟，本轮未发布分类；工具测量已保留'
    except Exception:
        result['reason'] = '原始波形推理进程或输出校验失败，本轮未发布分类'
    return result
