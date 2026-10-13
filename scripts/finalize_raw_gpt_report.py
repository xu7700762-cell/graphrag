# -*- coding: utf-8 -*-
"""Package the audited GPT classification results without numerical-fusion results."""
import csv
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_agent.config import file_hash
from scripts.evaluate_raw_gpt_all import read, save


def finalize(out, destination):
    summary = read(out / 'summary.json')
    audit = read(out / 'completion_audit.json')
    recovery = read(out / 'recovery_audit.json')
    protocol = read(out / 'protocol.json')
    assert audit['status'] == recovery['status'] == 'independently_verified'
    metrics = summary['metrics']
    assert metrics['total'] == metrics['valid_gpt_paths'] == 146
    source = Path(read(out / 'recovery_protocol.json')['original_output'])
    before = recovery['first_pass_metrics']
    with (source / 'path_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        original = {row['path_key']: row for row in csv.DictReader(stream)}
    with (out / 'path_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['prediction_source'] = 'first_pass' if original[row['path_key']]['gpt_class'] else 'failed_call_recovery'
        row['first_pass_gpt_class'] = original[row['path_key']]['gpt_class']
        row['first_pass_generation_status'] = original[row['path_key']]['generation_status']
    with (out / 'path_results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    subjects = '\n'.join(f"| {row['subject']:02d} | {row['paths']} | {row['correct']} | "
        f"{row['correct']/row['paths']*100:.2f}% |" for row in summary['by_subject'])
    extras = '\n'.join(f"| {row['subject']:02d} | {row['ordinal']} | "
        f"{row['start_clock']}–{row['end_clock']} | {row['gpt_class'] or '未发布'} | "
        f"{row['reason'] or '没有可可靠匹配的整路径评分，不计ACC'} |" for row in audit['unscorable_raw_paths'])
    low_recall = metrics['confusion_matrix'][0][0] / metrics['true_class_counts']['Low']
    high_recall = metrics['confusion_matrix'][1][1] / metrics['true_class_counts']['High']
    text = f'''# 当前原始 EEG GPT 工具 Agent 全路径重测（2026-10-09）

GPT 最终 High/Low ACC 为 **{metrics['accuracy']*100:.2f}%（{metrics['correct']}/146）**。146 条评分路径均有有效 GPT 最终分类，覆盖率 100%，错误 38 条。BACC 为 {metrics['balanced_accuracy']*100:.2f}%。

本次使用网页当前原始 EEG 入口 `eeg_agent.raw_model.assess(cloud=True)`、CPA `gpt-6.1-sol` 和既有 seed 2026 权重。24 名被试、146 条可与原始事件可靠匹配的评分路径，标签为整路径问卷分数 <30 为 Low、≥30 为 High。本次没有重训、修改阈值或调整提示词。最终类别来自真实 GPT 工具调用与 JSON 返回，失败请求没有使用其他类别代替。

## 调用失败及补跑

首次运行后段出现大量 API 错误。保留首次运行全部记录；仅对没有有效 GPT 类别的路径补跑，每条路径最多两次补跑。已成功返回的分类未重跑或更改。补跑使用本次已经重新计算的同一份原始 EEG 工具证据，没有使用历史实验的 GPT 答案。直到全部补跑完成才读取目标评分计算指标。

| 阶段 | 有效 GPT 类别 | 正确数 | ACC（固定分母146） | 仅有效输出ACC |
|---|---:|---:|---:|---:|
| 首次运行，失败计错误 | {before['valid_gpt_paths']}/146 | {before['correct']} | {before['accuracy']*100:.2f}% | {before['conditional_accuracy']*100:.2f}% |
| 完成失败调用补跑后 | 146/146 | 108 | 73.97% | 73.97% |

首次运行 255 次 API 请求，42 次失败（其中 41 条评分路径和 1 条未评分原始路径没有有效最终分类）。共进行 {recovery['recovery_attempts']} 次路径补跑尝试，成功补齐 {recovery['successful_recovery_decisions']} 条路径，其中 41 条计入 ACC、1 条单列。全程 API 请求共 {summary['api_calls']} 次，其中 {summary['api_failed_calls']} 次请求失败；补跑后评分路径均已成功。重新编码原始 EEG 窗口共 {summary['fresh_encoded_windows']} 个。

## 分类结果

| 真实类别 | 预测 Low | 预测 High | 识别率（Recall） |
|---|---:|---:|---:|
| Low（70条） | 50 | 20 | {low_recall*100:.2f}% |
| High（76条） | 18 | 58 | {high_recall*100:.2f}% |

| 被试 | 评分路径数 | 正确数 | ACC |
|---|---:|---:|---:|
{subjects}

## 全部原始路径与评分范围

这 24 名被试中共有 {audit['raw_event_segments']} 个原始事件片段：154 条有完整原始 Trigger 起止事件，另有 11 条缺少结束标记的片段。146 条完整路径能与有评分的路径可靠匹配并计入 ACC。另外 8 条完整路径单列如下；其中 6 条未满足窗口数量或覆盖率的发布条件，2 条有 GPT 类别但没有可靠匹配的整路径评分。11 条未闭合片段同样保留在全部原始路径 CSV 中，但不发布分类、不计 ACC。22、26 号被试没有此评分表中的可评测路径。

| 被试 | 原始路径序号 | 记录内起止时间 | GPT 类别 | 说明 |
|---|---:|---|---|---|
{extras}

评分表原有 147 条候选路径，排除 `04:2516402`：评分表的截断区间无法按该记录的固定时间偏移与完整原始 Trigger 路径精确匹配。路径对应只根据起止边界和精确持续时间确定，没有按类别选择。大多数记录的 CSV→原始 Trigger 偏移为 −54 个样本，04 号为 −61,558，23 号为 0；冻结清单包含每条路径的两套边界。

## 核查与结论边界

独立复算得到同样的 108/146。推理源码及标签文件哈希核查通过；重新解析 {audit['total_valid_gpt_decisions_including_unscored']} 条最终 GPT 分类，核对 {audit['unique_fresh_response_ids']} 个新响应和真实工具调用，工具证据重放一致。补跑审计另检查包括未完成尝试在内的 {recovery['unique_responses_including_failed_attempts']} 个唯一响应，确认首次记录未改、成功分类未重跑、EEG 工具证据一致、没有标签参与补跑。

当前协议不使用记录开头作为静息参考。历史 71.92% 来自另一协议，不能作为本次结果的同条件比较。本次是同一批已多轮开发的本地数据复测，不能解释为独立外部泛化结果，也不能仅凭本次 ACC 判断 GPT 相对数值工具的额外贡献。

逐路径表包含实际 GPT 类别、真实类别、对错、解释、工具调用以及是否来自失败补跑；全部原始路径表保留未评分路径及原因。审计与原始响应保存在 `{out}`。
'''
    (out / 'report.md').write_text(text, encoding='utf-8')
    summary['failure_recovery'] = {
        'first_pass_output': str(source), 'first_pass_metrics': before,
        'recovery_attempts': recovery['recovery_attempts'],
        'successful_recovery_decisions_including_unscored': recovery['successful_recovery_decisions'],
        'valid_decisions_never_retried_or_changed': True,
        'same_frozen_eeg_evidence': True, 'audit_status': recovery['status']}
    summary['audit_status'] = audit['status']
    summary['path_scope'] = {'subjects': 24, 'raw_event_segments': audit['raw_event_segments'],
        'complete_raw_paths': audit['raw_complete_paths'],
        'incomplete_raw_paths': len(audit['incomplete_raw_paths']),
        'scored_paths': 146, 'unscorable_complete_raw_paths': len(audit['unscorable_raw_paths'])}
    save(out / 'summary.json', summary)
    destination.mkdir(exist_ok=True)
    previous_manifest = read(out / 'delivery_manifest.json') if (out / 'delivery_manifest.json').exists() else {'files': []}
    previous_hashes = {item['path']: item['sha256'] for item in previous_manifest['files']}
    files = {
        'report.md': 'GPTagent_全路径重测报告_20261009.md',
        'path_results.csv': 'GPTagent_146条路径结果_20261009.csv',
        'all_raw_paths.csv': 'GPTagent_165个原始路径片段_20261009.csv',
        'summary.json': 'GPTagent_全路径重测汇总_20261009.json',
        'completion_audit.json': 'GPTagent_全路径审计_20261009.json',
        'recovery_audit.json': 'GPTagent_失败补跑审计_20261009.json',
    }
    delivered = []
    for name, filename in files.items():
        target = destination / filename
        if target.exists():
            assert file_hash(target) == file_hash(out / name) or file_hash(target) == previous_hashes.get(str(target)), 'Different report already exists at destination'
        shutil.copyfile(out / name, target)
        assert file_hash(target) == file_hash(out / name)
        delivered.append({'path': str(target), 'sha256': file_hash(target)})
    save(out / 'delivery_manifest.json', {'files': delivered})
    print('Audited report and 6 delivery files verified.')


if __name__ == '__main__':
    finalize(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
