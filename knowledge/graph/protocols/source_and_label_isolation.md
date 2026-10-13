---
{"evidence_level": "high", "review_status": "reviewed", "source_refs": ["local-vrms-protocol"], "source_sha256": {"local-vrms-protocol": "e1d5d011e0f3fa1f66c5d9ab8ad55ee68679dd90b01bc798abdb99e1274a4591"}, "title": "来源、标签隔离与会话证据协议", "version": "state_reasoning_v1"}
---

# 来源、标签隔离与会话证据协议

静态输入为sources.json与claims.json；只物化reviewed来源、节点与边，candidate仅保留清单。source_sha256是source记录去除自身哈希后的canonical JSON摘要（Python sort_keys=True、ensure_ascii=False、allow_nan=False、默认分隔符）。node/edge的source与content摘要由确定性构图器补齐。修改reviewed记录需重建manifest，不能按文件时间选用案例库。

外部医学关系要有DOI、标题、年份与章节定位。local_protocol只能支持代码/流程规则，不能充当医学关系的论文证据。paper_evidence、domain_knowledge、counter_evidence、local_protocol需在检索返回中清晰区分。independence_group按来源研究团队合并；同DOI的多个章节记录是一篇来源。

当前观测只能来自实际EEG描述，不可从类别分数、模型概率或查询问卷反推biomarker方向。模型分数进入model_evidence时仅指冻结模型输出，不构成独立生理证据。

model_evidence仅保存当前会话prediction、实际EEG变化、支持/反证路径及证据强度。不得保存当前真值、问卷、身份或隐藏标签；不得写回静态图或成为后续检索案例。反证限制解释，不作为类别票。
