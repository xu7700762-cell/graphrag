---
{"evidence_level": "low", "review_status": "reviewed", "source_refs": ["long2025-autonomic"], "source_sha256": {"long2025-autonomic": "fa86821d993c6d64d5e3d0a2bf239ebd95a73b872d68f8cb987da7f613342ed6"}, "title": "自主神经反应：模态边界", "version": "state_reasoning_v1"}
---

# 自主神经反应：模态边界

Long等在VR导航任务中测量EDA和ECG，将皮肤电特征与VRSQ预测及交感神经唤醒联系起来。这为VRMS研究提供自主神经背景，并非本项目EEG直接检测交感反应的证据。

静态背景节点保留，通向state的边要求EDA/ECG模态。当前EEG-only输入不得声称观测到皮肤电、HRV或自主神经变化；论文回归性能不能迁移到冻结EEG模型。反证节点明确这一模态边界。

定位：10.3390/s25185828，Abstract、2.1.3、4.2、4.4。
