---
{"evidence_level": "high", "review_status": "reviewed", "source_refs": ["local-vrms-protocol", "lee2026-eeg"], "source_sha256": {"lee2026-eeg": "2ac6fcc15f97c0b7638d65e51847be54dec71a2ab6404d4541e749d30ef9264b", "local-vrms-protocol": "e1d5d011e0f3fa1f66c5d9ab8ad55ee68679dd90b01bc798abdb99e1274a4591"}, "title": "Raw EEG path 方法 sidecar", "version": "state_reasoning_v1"}
---

# Raw EEG path 方法 sidecar

## Task

处理当前原始EEG的完整路径，在冻结本地模型与固定RAG policy下解释研究分类结果。既有v1/v2/v3方法源和历史policy保持可复算。知识图只扩展可追溯解释，不重新拟合模型或校准器。

## EEG Features

描述工具theta 4–8、alpha 8–13、beta 13–30、gamma_low 30–45 Hz来自实际窗口PSD。在线冻结特征仅使用delta/theta/alpha/beta。spectral、spectral_compact、spatial_covariance、filterbank_csp的分类分数不是相应生理量；同源表示不算独立观测。未实际计算的entropy/connectivity保持不可用。

## Physiological Interpretation

alpha、theta、beta文献关系只提供有范围的关联。感觉冲突、视觉–前庭失配、视觉加工负荷保留hypothesis。Gamma频带、EDA/ECG模态不兼容时不迁移。没有合法基线时绝对功率不叫“变化”，路径首末差不叫静息基线差。

## Evidence Quality

文献source包含DOI、发表年、章节定位、原文哈希、独立来源组与审阅状态。reviewed表示本次来源核验，不代表临床验证。2025与2026 Lee团队论文合并独立性；多篇论文和更多路径不会自动提高证据等级。支持与反证冲突返回low；无可用观测/关系返回unavailable及具体原因。
