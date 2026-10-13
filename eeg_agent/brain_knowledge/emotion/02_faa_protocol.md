---
{"title":"EmotionAgent的FAA测量合同","source":"brain_knowledge/emotion/02_faa_protocol.md","page":"不适用：新增工具定义","evidence_type":"local_protocol","version":"1.1"}
---
# EmotionAgent的FAA测量合同

## 额区alpha不对称FAA

EmotionAgent读取实际F3与F4通道的alpha频带功率，计算FAA = ln(alpha_F4) - ln(alpha_F3)。左右功率必须是质量合格窗口的真实有限正值，来自匹配频谱计算及同一参考配置。两通道功率可作为匿名数值摘要；不传原始EEG或完整PSD。

只有F3、Fz、F4区域平均功率时FAA不可用，因为区域平均不能还原左右通道。缺通道、非正功率、无效质量或尚未运行频谱计算时明确返回缺失项。公式定义属于本地可复核工具合同，不声称已经复现EEG Emotion Copilot的模型。

## 情绪标签需独立验证

FAA及其符号不直接对应特定情绪、效价、唤醒度或心理诊断。EmotionAgent 采用 EEG 工具分析模式，不使用也不要求部署深度学习分类模型。工具提供FAA数值与证据完整性检查，Agent 检索共享库和情感领域库，再调用真实大模型 API 解释测量及引用，由 Supervisor 汇总。未取得 EEG 时可以进行知识问答；缺失分类器不影响工具分析。

工具分析报告不宣称情绪分类效度。若以后需要评价情绪类别或连续维度，应另设独立标签与匹配的数据验证协议，不能将FAA或大模型解释直接作为经验证的情绪标签。

FatigueAgent与EmotionAgent可以解释同一窗口的不同测量；它们共享信号质量与数值合同，并分别检索领域知识库。Supervisor汇总各领域的可用证据与不确定性，不为缺失领域编造标签。
