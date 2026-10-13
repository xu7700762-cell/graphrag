# 默认 RAG 与后续优化

新增显式普通 RAG / GraphRAG / 语义 GraphRAG 入口，详见 [图检索与单轮比较](GRAPH_RAG.md)。
普通固定入口保留历史检索与高推理强度协议；原默认在 Graph 达到112/146、
BACC不下降且工程验收通过前保持原状。会话创建时固定版本，比较页只读已有结果。

2026-10-12 起，首页原始 EEG 云端分类和 `/online` 完整路径结束后的云端分类共用 `eeg_agent.default_rag`。当前默认版本为 `raw_decisive_rag_v4 / percentile_group`。

该检索使用八个工具原始分数在同折参考库中的经验百分位，按 FEMBA 表征、频谱、空间/CSP、案例检索四个来源组等权计算距离。MIL 作为实际读数展示，不参与距离。每类选取至多两条核心案例，优先覆盖不同内部被试，必要时补足缺失被试，最多六条。上下文还包含原始读数、可用校准值、整库类别中位数/IQR、方向错误计数和来源组符号模式。最终 High/Low 由 LLM 独立判定，冻结数值融合类别不进入该判读请求。

24 折参考库从已校验的 `v3_current_raw_policy_heldout_20261011` 导出到 `outputs/rag_assets/current_raw_policy_v3`。程序核验固定清单 SHA256、每折 SHA256、参考被试角色分离和读数一致性。损坏、缺失或无对应折时明确报告 RAG 不可用，禁止静默退回旧案例版本。参考被试未参与该折头与校准器拟合，但曾用于 selector 开发，不是完全未查看的独立测试集。在线 CPU FP32 和原参考生成的数值后端存在精度差异，原实验 ACC 不能直接当作在线版性能。

方法知识库继续使用 BM25。知识文本与默认案例策略保持一致；历史输出和历史实验的显式版本不随默认升级改变。

后续 RAG 优化完成后，默认同步切换，无需用户再次要求。修改默认入口、方法知识卡和部署白名单，核验参考来源与标签隔离，验证首页及在线请求，更新 `/api/brain/catalog` 与 `/api/online/catalog` 显示的版本，重载空闲服务，并记录部署结果。优化只存在于实验文件、不进入默认服务，不算完成。历史对照必须显式固定 variant。

验证命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_default_rag.py tests/test_raw_independent_judge.py tests/test_online_cloud.py
```

服务状态接口的 `vrms_case_rag` 字段给出当前 schema、variant、参考库清单哈希和默认入口。更换默认版本后，先确认没有运行中的任务，再重载服务；现有会话资料保留。

2026-10-13 的语义图升级已接入同一默认入口的显式 `semantic_graph` 分支，
并同步知识卡、网页选项、部署白名单及版本状态接口 `/api/rag/version`。
`semantic_graph_rag` 字段显示 `raw_eeg_state_graph_v1 / state_reasoning_v1` 与
`promotion_status=not_evaluated`。新语义版本独立保存，默认仍保持普通RAG。
其新评估必须146条有效、至少112条正确、BACC≥73.2143%，并全部通过支持/反证
引用及标签隔离核验，才按上述默认发布规则切换。历史v1成绩和146条无云端工程
审计不能代替这次评估；旧会话的policy保持创建时版本，不自动升级。
