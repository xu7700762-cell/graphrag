---
{"title":"BrainAgent情感工具的频谱与波动指标","source":"01_BrainAgent_arXiv_2606.25400.pdf","source_sha256":"1ce600063c74d4ab3753ab4128201437cd3f4fb5cf7255f10e8cd2e394ad441e","url":"https://arxiv.org/abs/2606.25400","page":"PDF页序7 Figure 5；13 A.3","evidence_type":"paper_excerpt","version":"2026-10-09原文核对"}
---
# BrainAgent情感工具的频谱与波动指标

## 指标与论文工具

BrainAgent附录A.3列出EmotionFAAAnalyzer，计算左右额区alpha不对称；EmotionBiomarkerExtractor将Beta/Alpha Ratio（BAR）关联到压力与唤醒，Theta/Beta Ratio（TBR）关联到注意控制与走神，Theta/Alpha Ratio（TAR）关联到嗜睡和疲劳。EmotionStabilityAnalyzer考察额区和中央区信号的时间变异与复杂度。Figure 5报告具体alpha/beta变异系数，作为执行报告的数值证据。

本项目现有工具计算F3/F4的FAA、BAR/TBR/TAR，以及连续额区alpha/beta功率的变异系数与时间趋势。FAA定义为ln(alpha_F4)-ln(alpha_F3)。FAA小于零表示这里的F4 alpha功率低于F3，大于零表示F4高于F3；这是该公式下的信号方向。变异系数定义为样本标准差除以均值，直接描述所分析窗口间的功率波动。可以明确报告这些方向、比值及变化，随后引用文献说明研究关联。

## 解读范围

不能只凭FAA的正负直接命名具体情绪，也不能把功率波动直接改名为心理状态不稳定。论文描述的MentalHealthRiskAssessor属于另一个规则汇总工具，要求至少两个前置工具，但附录未公开足以复现其风险分级的数值阈值；本项目未实现该风险分级器。本文所述研究关联可以用于解释指标，不能把论文示例数字当成当前EEG测量。知识与方法问题应直接解释指标，不以缺少当前EEG为理由拒绝回答。
