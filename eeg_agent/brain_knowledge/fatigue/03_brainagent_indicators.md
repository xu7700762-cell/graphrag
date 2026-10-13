---
{"title":"BrainAgent疲劳工具的指标与报告组织","source":"01_BrainAgent_arXiv_2606.25400.pdf","source_sha256":"1ce600063c74d4ab3753ab4128201437cd3f4fb5cf7255f10e8cd2e394ad441e","url":"https://arxiv.org/abs/2606.25400","page":"PDF页序7 Figure 5；13 A.3","evidence_type":"paper_excerpt","version":"2026-10-09原文核对"}
---
# BrainAgent疲劳工具的指标与报告组织

## TAR与时间趋势

BrainAgent附录A.3把Theta/Alpha Ratio（TAR，theta/alpha）列为与嗜睡和疲劳相关的指标，把Theta/Beta Ratio（TBR）列为与注意控制及走神相关的指标。MentalWorkloadCalculator使用EEG指标及时间趋势描述任务中的认知负荷和疲劳积累。上述是论文的工具设计与解释方向。

Figure 5的示例先加载具体EEG文件，报告TAR、theta和beta功率变化，再与另一个领域的工具结果汇总。本项目应先报告实际算出的比值、增加/下降方向和时间范围，并结合所检索文献解释研究关联，避免用笼统的“证据不足”覆盖已经成功完成的测量。

## 指标定义与回答范围

本地工具确定性计算TAR、TBR、Beta/Alpha Ratio（BAR）和慢快比，并分析连续窗口的斜率、前后均值及变异系数。正斜率表示当前指标上升，负斜率表示下降；变化来自所分析时间段，不自动等于相对休息基线的变化。根据这些工具结果可以明确回答“频谱比值是多少、近期是上升还是下降”。

论文示例中的TAR=1.97属于该示例记录，不是本次测量或通用疲劳阈值。附录工具说明未给出可直接迁移到本项目的统一疲劳分级阈值。生理状态研判需要引用适用来源；不要虚构分级阈值或校准概率。知识与方法问题不要求用户先提供EEG记录。
