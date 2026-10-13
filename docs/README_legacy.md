# EEG Agent 本地研究原型

在 Windows 上回放城市巡航原始 CDT，持续运行因果信号处理与冻结 EEGNet，使用 BrainAgent 风格的 Supervisor、SignalQualityAgent、VRAnalysisAgent 组织专业工具。提供命令行、网页控制台和树莓派部署入口。

## 启动

项目目录：`D:\桌面\2`。本次创建的 `.venv` 使用 `--system-site-packages` 复用现有 CUDA PyTorch，新增依赖只安装到本项目虚拟环境，其他项目环境没有改动。

```powershell
Set-Location -LiteralPath 'D:\桌面\2'
.\.venv\Scripts\python.exe -m eeg_agent serve --port 8880
```

浏览器打开 <http://127.0.0.1:8880>。也可在本项目目录运行 `.\start.ps1`。

网页可以选择被试、固定配置/规则/单Agent/分层Agent、回放速度和源时长，输入自然语言任务，启动/停止并查看SSE实时日志。默认130秒回放可覆盖Acquisition 01首条完整路径。正常采集窗口需要等待5秒，首任务开始前会持续处理基线；未加载模型时明确显示预测不可用。

## 数据协议与重要审计修正

原始数据直接从 `D:\桌面\数据集\城市巡航` 读取，不复制到项目。26份原始文件，1024Hz、37通道；模型只使用30头皮EEG通道，EOG用于校准/测量，Trigger用于当前到达的事件。原始数值单位为 **uV**，处理、缓存及模型合同保持该单位，避免与V混用。

CSV有24人147条完整标记且有评分的候选路径，`<30`为低类71条，`≥30`为高类76条（恰好30有4条）。原始事件审计进一步发现Acquisition 04最后一条CSV“完整”路径的结束被截到文件尾：CDT在2626240样本结束，CEO的真实mark22位于2645592，尚未到达。因此**147条候选中只有146条原始事件完整**，该条保留清单并拒绝训练，不能补造结束事件。严格集合低类70、高类76。

固定被试划分：

| 划分 | 被试 | CSV候选路径 | 原始事件完整 |
|---|---|---:|---:|
| 训练 | 2,3,4,5,6,7,10,11,14,15,16,18,21,23,24 | 88 | 87 |
| 验证 | 8,13,20,25 | 36 | 36 |
| 测试 | 1,9,12,17,19 | 23 | 23 |

数据介绍中的154条与当前CSV不符。Acquisition 04、23的DPO样本数也与实际长度不符。读取器按实际完整float32样本帧读取，检查通道、单位、端序、缩放及事件边界；报告记录差异和SHA256，不修改源文件。

## 连续处理与P0/P1

- P0：M1/M2平均参考，50Hz/Q30陷波，4阶0.5–45Hz Butterworth带通，椭圆抗混叠（100Hz通带、128Hz阻带、0.1dB波纹、80dB衰减），全部采用状态化因果SOS；固定抽取相位1024→256Hz。
- 滤波状态跨块、完整窗、路径及休息段连续保存。启动/断流后预热10秒；缺失/非有限值重置状态并标记不足，未完成窗口丢弃，不把断流两侧拼接成完整窗。窗口为5秒、无重叠，部分路径尾窗丢弃并体现在覆盖中。
- P1：同一公共链输出加冻结的中心化EOG线性回归。只取首任务开始前、预热后已经到达的最多60秒数据，至少10秒；要求EOG方差、矩阵条件数与系数稳定性合法。校准不足禁用P1，不重新拟合，不从后续测试记录挑校准。
- 所有组共享校准、QC、模型和门控。P1处理后幅值仍需满足合同。相关下降是回归构造的结果，不能用它证明信号清理有效。
- QC同时报告原始工频、动态幅值范围、平直、突变与处理后幅值及EOG相关。原始电极DC偏移不等于饱和；当前动态范围门限5000uV、处理后峰峰值500uV是**开发原型门限**，不是已校准的硬件饱和阈值。第一版在训练前修正了原始50Hz/直流偏移误判，最终门限对全部模式一致冻结。
- 规则控制使用EOG相关0.4进入、0.25退出、连续2窗及至少15秒保持；过期QC回到P0。Agent默认每30秒审查一次，仅一个规划任务在途，建议TTL为10秒数据时间；非法、未来、过期或模型不支持的建议拒绝执行。

## 模型、评价和正式实验入口

```powershell
.\.venv\Scripts\python.exe -m eeg_agent audit
.\.venv\Scripts\python.exe -m eeg_agent prepare
.\.venv\Scripts\python.exe -m eeg_agent train --epochs 20
.\.venv\Scripts\python.exe -m eeg_agent evaluate --split validation
.\.venv\Scripts\python.exe -m eeg_agent evaluate --split test
.\.venv\Scripts\python.exe -m eeg_agent export
.\.venv\Scripts\python.exe -m eeg_agent replay --subject 1 --mode hierarchical --backend mock --speed 1 --max-seconds 130
.\.venv\Scripts\python.exe -m eeg_agent compare --subject 1 --backend mock --speed 20 --max-seconds 130
.\.venv\Scripts\python.exe -m pytest -q
```

基础EEGNet从零训练，不使用旧模型项目。seed42、Adam/0.001、batch32、最多20轮、连续5轮验证路径BACC无改善停止，以规则轨迹的验证路径BACC选检查点。归一化仅用训练被试合法P0/P1视图；每条路径按视图数、窗口数及完整训练路径类别频率加权。P0、P1、规则及15秒保持的切换输入分别进行开发验证，数值匹配与信号效度分开报告。训练划分为15人87条原始完整路径，其中14人82条路径有合格窗口参与实际拟合；Acquisition 14因质量门控无合格窗口，仍保留审计记录。

路径概率为有效窗口概率均值，阈值0.5；至少3个有效窗、覆盖80%才能发布最终类别。不足路径保留在分母并拒绝，BACC/AUROC/Macro-F1等是**接受路径上的条件指标**。只有路径评分真值，没有5秒即时症状真值。

正式五折与真实API入口已提供，**本轮不自动启动**：

```powershell
# 五折外层被试隔离，内层被试隔离选模型，归一化逐折重拟合
.\.venv\Scripts\python.exe -m eeg_agent fivefold --epochs 20
# 每组均用同一原始流、模型、工具、校准和调用上限，费用自备
.\.venv\Scripts\python.exe -m eeg_agent compare --subject 1 --backend api --speed 1 --max-seconds 130
```

数据已经用于开发，后续五折是内部交叉验证，不称为未经接触的独立测试。固定P1校准不可用时回退P0；固定P0/规则组不调用语言API，费用为零，Agent各组共享50次调用上限、不自动重试，记录所有成功/失败调用、token及时间。未提供供应商价格时不伪造人民币费用。

## 云端接口

首轮语言规划为明确标注的 `mock`，专业工具和模型推理使用真实数据。模拟响应不是真实LLM推理，不能用于声称Agent优于规则。

复制 `.env.example` 为 `.env`，填写自己的OpenAI兼容接口、Model ID及密钥，或设置同名环境变量。密钥不写入代码、日志或网页响应，不从其他应用读取。`mock`不会读取密钥或请求外部API。

真实模式的Supervisor只分派两个子Agent；SignalQualityAgent执行受限function calling（质量工具→ProposeProfile），本地校验后才能在未来窗生效；VRAnalysisAgent调用冻结预测及路径汇总工具，云端汇总不能改写本地数值。数值数据面不等待云端。云端上下文只含匿名质量/校准摘要及匿名聚合输出，不含EEG数组、被试ID、路径评分或测试标签。

## 产物与接口

- `outputs/reports/`：数据审计、冻结后评价、控制器比较与最终验收记录。
- `outputs/cache/`：按被试压缩的float32 P0/P1窗口及逐窗QC，只是派生缓存。
- `outputs/models/baseline/`：最佳检查点、ONNX、训练历史、输入合同、归一化及哈希。
- `outputs/runs/<run-id>/`：JSONL全过程事件、最终报告/失败原因、覆盖和全部调用成本。
- HTTP：`GET /api/config`、`POST /api/runs`、`GET /api/runs/{id}`、`POST /api/runs/{id}/stop`、`GET /api/runs/{id}/events`（SSE）、`GET /api/reports`。服务默认仅监听本机127.0.0.1。

## 树莓派5

提供 `deploy/pi.yaml` 和systemd示例，不需要Pi本地LLM。将项目代码、冻结ONNX/合同及需要回放的原始文件放到Pi，设置真实数据路径。Python3.11或以上的ARM64系统创建普通venv，安装项目运行依赖（不需要CUDA/PyTorch训练库），运行：

```bash
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m eeg_agent --config deploy/pi.yaml serve --port 8880
```

`deploy/eeg-agent.service` 中的用户和路径需与实际Pi一致。Pi的网络连接、ARM64依赖、ONNX一致性、1倍速延迟和内存还须现场验证，Windows指标不能替代它们。

设计依据：用户提供的 `BrainAgent_差异与创新候选评估.md`、`树莓派5_EEG_Agent_研究方向与实施建议.md`。本项目先验证架构与工程条件，不将云端API、Pi部署、工具规划或二选一配置宣称为已证实的研究创新。
