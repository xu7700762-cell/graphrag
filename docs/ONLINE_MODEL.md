# 在线 CPU 模型与冻结资产

新增模型入口为 `eeg_agent.online.model.OnlineModel`。该模块独立于现有离线分析：原始离线 CUDA 工作进程、历史权重与结果均未修改。在线模式使用相同冻结 FEMBA 编码器、MIL、归一化及八路数值估计器，执行后端固定为 CPU FP32。

## Mamba 移植边界

`minimal_mamba.py` 改编自 [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal/tree/03de542a36d873f6e6c4057ad687278cc6ae944d)，固定 commit `03de542a36d873f6e6c4057ad687278cc6ae944d`。完整 Apache-2.0 许可证保存在 `eeg_agent/online/licenses/mamba-minimal-LICENSE`，修改说明保存在源文件开头。

只采用其 `MambaBlock` 和 selective scan。移植去除了语言模型、RMSNorm、einops 与 CUDA 扩展依赖；线性投影、分组 Conv1d、SiLU、Softplus、指数和 80 步递推均使用普通 PyTorch 操作。原 EEG 模型的 PatchEmbed、位置嵌入、四层双向相加、残差与 LayerNorm 保留。反向分支有独立权重，不能复用正向权重。

输入为 float32 `30×1280`、5 秒、256 Hz、固定通道与 μV 单位。每方向 `d_model=525`、`expand=4`、`d_inner=2100`、`d_state=16`、`d_conv=4`、`dt_rank=33`；编码器完整加载 83 个张量、28,540,155 个参数。前缀剥离和旧 `core.` 键映射必须唯一，任何缺键、额外键、形状冲突、非有限张量或哈希冲突都会拒绝加载。

每个独立窗口的网络卷积与 SSM 状态从零开始，保持与原离线逐窗编码一致。信号预处理状态仍由流处理模块跨数据块维护。双向计算只使用已经完整到达的 5 秒窗，不读取下一窗口。

## 模型接口

```python
from eeg_agent.online.model import OnlineModel

model = OnlineModel('outputs/online_assets/seed2026/manifest.json', threads=2)
path = model.start_path(fold=1)
snapshot = path.add(window30x1280)
snapshot = path.result()
# Optional: share the same float32 window's Welch PSD with descriptive tools.
snapshot = path.add_shared(window30x1280, frequencies, psd)
```

公共编码器常驻，当前折的 MIL、归一化和数值工具懒加载。更换被试只加载有对应留一权重的折；缺失权重直接报错，不能转用第一折或旧监测模型。已返回的路径对象保留自己的折引用；服务应在路径结束后释放它。

每窗只编码和提取一次频谱、协方差、滤波器组协方差。后续前缀聚合使用缓存，不重读整条波形，也不重算旧窗口 FFT。结果以副本返回，同一前缀重复 `result()` 不再次运行八工具。

`add_shared` 只共享 `fs=256`、512 点、50% 重叠、Hann、去均值、density Welch 的频率和 PSD，不共享频带功率。原数值工具使用半开频带边界，描述工具使用闭合端点，两者分别从同一 PSD 积分，保持各自定义。共享路径与默认路径的真实资产 3 窗结果已逐字段核对相同。

结果保留原 D2 八工具名称：`femba_features`、`spectral`、`spatial_covariance`、`femba_mean`、`femba_temporal`、`spectral_compact`、`filterbank_csp`、`case_retrieval`。返回 `mil_raw_high_probability`、`mil_class`、`threshold`、`numeric_fusion`、`accepted_windows`、编码器/MIL/归一化/三组数值资产哈希。

不足 3 个有效窗口时 `available=false`，可附未校准 MIL 预览；不运行需要三段划分的八工具。3 窗后 `available=true` 只表示数值证据可用，仍有 `prefix_provisional=true` 与 `classification_scope=whole_path_prefix`。最终路径发布规则由在线会话在结束事件到达后检查；前缀输出不能称即时症状检测。

MIL、融合分数均保持 `probability_calibrated=false`；工具自身独立校准字段按原资产返回。无记录开头静息参考。训练案例解释库未随本次资产包导出，案例投票估计器仍按原定义运行，额外案例解释显式不可用，不修改数值判断。云端判断由会话层仅在路径结束后触发。

## 资产准备

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_online_assets `
  --source-root 'D:\桌面\文献总\EEGAGENT' `
  --out outputs/online_assets/seed2026
```

导出只读取 D2 原始工作进程实际使用的已有参数。源编码器、MIL、归一化与三个数值包分别核对原 `preflight.json`、`fold_audit.json` 和 `training_summary.json`。导出不训练、不读取或复制数据集，不读取 API 密钥。

清单使用相对路径，包含 24 折、文件 SHA256 和原源文件 SHA256。Torch 文件去除无关字段；joblib 中旧 CSP 模块映射到 `eeg_agent.online.features`，估计器参数和工具名不变。运行时 SHA256 指向实际加载的可移植文件，`source_sha256` 保留原参数来源。首次生成资产包约 164.7 MB。

目标目录必须不存在，禁止覆盖历史资产。清单及所有参数应作为整体复制到树莓派。只加载自己持有且可信的 Torch/joblib 资产。建议树莓派固定 scikit-learn 1.6.1，与资产保存版本一致；跨版本必须重新做数值核对。

## 已完成验证与未完成部分

`tests/test_online_model.py` 验证 scan 递推、因果卷积、83 个参数键、LayerNorm、哈希/路径校验、重复结果缓存、输入拒绝及真实资产 CPU 加载；追加共享 PSD 的无重复 FFT、完整分数一致性及 CSV/Trigger 固定偏移检查后共 9 项通过。

`outputs/online_implementation/20261011/model_validation/report.json` 保存真实权重的独立 CUDA FP32 对照，使用 8 个真实合格缓存窗口，仅用于无标签数值迁移检查：

| 核对 | 最大绝对误差 |
|---|---:|
| 8 窗 embedding，CPU FP32 / CUDA FP32 | 4.77×10⁻⁷ |
| 8 窗 tokens，CPU FP32 / CUDA FP32 | 3.34×10⁻⁶ |
| 1/3/8 窗 MIL 概率 | 3.58×10⁻⁷ |
| 缓存八工具 / 原全路径重算，使用同一 embedding | 3.33×10⁻¹⁶ |

1/3/8 窗 MIL 类别、3/8 窗八工具及融合类别一致，24 折均通过加载及被试隔离检查。冻结容差为 FP32 `atol=rtol=1e-4`。WSL 对照环境的 scikit-learn 1.7.2 加载 1.6.1 资产产生版本提示，实际上述分数已核对。

全原始路径核验入口为 `python -m scripts.validate_online_model`，需要现有 CUDA 环境仅作独立参考。它从 CDT 波形和 Trigger 按原离线预处理重建窗口，遍历所有有冻结折的完整任务；模型输入不含问卷或目标标签。CSV 仅由验证 evaluator 读取资格与边界，按已有 `evaluate_raw_gpt_all` 的精确时长与固定记录偏移规则对齐，避免把 CEO/CSV 的采样偏移误判为缺失路径。新核验必须选择新的 `--out`；既有结果不会覆盖。

全路径结果在 `outputs/online_implementation/20261011/model_validation/all_paths_report.json`：148 条完整可推理原始路径、3820 个有效窗口，包括全部 146 条可评分路径、3769 个有效窗口。CPU/CUDA FP32 的 embedding、MIL 概率、八工具分数和融合概率最大绝对差分别为 3.04×10⁻⁶、1.25×10⁻⁶、5.62×10⁻⁶、3.34×10⁻⁷，MIL、八工具和融合类别全部一致。缓存数值工具与原全路径重新提取的最大分数差为 1.24×10⁻¹⁴。缺权重的被试 22/26 明确不可推理，被试 25 有真实对应权重。

原始数值比较全部完成后，单独按既有固定偏移规则修正 evaluator 的 CSV 资格字段。初版报告保留为 `all_paths_report_unaligned.json`，最终报告记录其哈希；没有重算或改写数值比较。独立 BF16 诊断在本次 148 条路径中未发现 MIL、八工具或融合类别变化，但它仍不能替代旧协议或云端准确率复现。

既有离线 CUDA BF16 与 FP32 的 8 窗 embedding 最大差为 0.00318；该差异单独记录，没有强行按 1e-4 判定，也没有改动历史离线输出。新 FP32 在线协议不能直接继承原 BF16 准确率。

`scripts/evaluate_online_numeric.py` 已按 256 样本流实际回放 26 份原始记录，所有预测完成后才读取路径标签。146 条评分路径完整结果中，MIL 的 ACC/BACC/AUROC 为 64.38%/64.61%/0.6861，数值融合为 73.97%/73.87%/0.7145，发布覆盖率均为 100%，失败率为 0%。这不是云端 Agent 准确率。每 5 秒前缀、25%/50%/75%/最后一次更新和完整路径终点分别保存；拒绝的当前窗口不发布前缀类别，不足三窗的 MIL 仅为预览。前缀用路径结束标签事后评价，没有即时症状真值，相关窗口不能当作独立受试者样本。结果见 `numeric_evaluation.json` 和本地两个 CSV。

单窗检查 CPU 编码与缓存约 0.087–0.135 秒，执行机是 WSL x86_64，不能作为树莓派速度。最终 Windows 服务 130 秒真实 1 倍速回放收到 19/19 窗，窗口处理 P95 为 0.221 秒，源时钟到完成的延迟 P95 为 0.331 秒。尚未完成 ARM64/Pi 硬件、真实云端类别评价、长时间实时运行和能耗验证。迁移失败、窗口超期或资源不足时显式停止发布，不能自动更换模型、量化、减少层数、调用远程 GPU 或使用历史预测代替在线计算。
