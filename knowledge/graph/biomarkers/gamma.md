---
{"evidence_level": "low", "review_status": "candidate", "source_refs": ["lee2026-eeg", "local-vrms-protocol"], "source_sha256": {"lee2026-eeg": "2ac6fcc15f97c0b7638d65e51847be54dec71a2ab6404d4541e749d30ef9264b", "local-vrms-protocol": "e1d5d011e0f3fa1f66c5d9ab8ad55ee68679dd90b01bc798abdb99e1274a4591"}, "title": "Gamma：频带兼容性候选", "version": "state_reasoning_v1"}
---

# Gamma：频带兼容性候选

项目描述工具gamma_low为30–45 Hz；当前在线整路径模型的冻结特征仅到beta。2026文献采用30–50 Hz，并报告额区gamma增加，但尚未核验到与本项目频带及脑区一致的关系。

此节点及机制边为candidate，不进入reviewed graph或LLM生产判读。不能将30–50 Hz结论直接复制到30–45 Hz，也不能推断在线分类头已经使用gamma。该论文曲面显示器中央gamma的叙述与Table 1不一致，本版不采用该区域方向。

定位：10.1038/s41598-025-32858-2，EEG recording and data process、Tables 1–2、Discussion gamma段。
