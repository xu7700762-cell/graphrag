---
{"evidence_level": "low", "review_status": "reviewed", "source_refs": ["lee2026-eeg", "lee2026-mechanism"], "source_sha256": {"lee2026-eeg": "2ac6fcc15f97c0b7638d65e51847be54dec71a2ab6404d4541e749d30ef9264b", "lee2026-mechanism": "9ba89cf648b62468f6002e5ebe39985e5da2c02ea06093c4957f598a5db5d7ca"}, "title": "VR motion sickness：研究状态定义", "version": "state_reasoning_v1"}
---

# VR motion sickness：研究状态定义

本图首版只支持VR motion sickness（VRMS）研究状态，包括文献语境中的cybersickness/VR sickness/VIMS。state不是临床诊断节点，也不是当前查询的High/Low真值。

supporting_paths给出观测与研究关联；contradictory_paths给出可限制解释的证据。路径数、节点度数及案例类别占比均不能转换成类别概率。若支持与反证同时存在，保留冲突并降低证据强度，不把反证作为另一类票。

2026论文指出EEG与SSQ的时序和测量含义不同，因此EEG加工反应与事后主观严重度不必一致。当前模型结论仍由冻结模型及LLM约定协议输出，知识图不替代数值模型。
