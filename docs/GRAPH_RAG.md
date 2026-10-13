# EEG 证据图与普通 RAG 对比

2026-10-13 新增独立状态推理入口 `semantic_graph`，schema 为
`raw_eeg_state_graph_v1`，variant 为 `state_reasoning_v1`。首页与 `/online`
创建会话时可选“语义 GraphRAG（状态推理实验）”；版本及实现、协议、资产哈希
随会话固定。当前默认仍为 `regular / percentile_group`，新版本状态为
`not_evaluated`。下面的107/146属于旧 GraphRAG v1，不能作为新语义图的准确率。

语义链为实际 EEG 观测 → biomarker → mechanism → VR motion sickness →
论文来源及方法协议。新增资产位于 `outputs/rag_assets/eeg_state_graph_v1/`，
来源和关系位于 `knowledge/graph/sources.json`、`claims.json`，短卡片按
biomarkers、mechanisms、states、methods、protocols 分目录保存。
首版包含19个已审阅节点、30条边；旧v1/v2/v3和旧`method_source.md`保持原字节。
每条节点和边保留来源层、证据等级、来源引用、来源/内容SHA256与版本；
构建及加载核验所有sidecar、原始协议文件和固定参考资产。

当前真正测量theta 4–8、alpha 8–13、beta 13–30 Hz（左闭右开），以已接受的
30通道完整窗口计算路径末三分之一与首三分之一的平均log功率差。
10%几何功率变化阈值只用于描述增/减/稳定。首段是暴露路径的一部分，
不是静息基线；八工具分类分数不替代功率变化。在线复用缓存频谱，
不增加FFT；首页仅在选择语义模式时从本轮原始窗口计算同定义观测。
未完成路径、不足三个合格窗或缺少实际测量均明确报告不可用。

支持路径、反证路径及其来源完整打包，按证据等级、来源独立性和层级排序，
在24000字符内整包截断并记录遗漏数。实际观测在截断时保留。
`supporting_paths`、`contradictory_paths`、`confidence_basis`、`evidence_strength`
进入LLM上下文；强度只允许high/medium/low，有反证时当前状态推断强度为low。
无适用审阅关系时使用`evidence_status=unavailable`及具体`fallback_reason`。
这些是研究关联与假说，不是诊断、类别票数或疾病概率。两篇Lee团队论文
归入同一来源独立组；区域、显示器差异及未显著EEG–SSQ相关均保留为约束。
gamma频段不兼容、entropy无审阅来源、connectivity只有摘要，均保留为候选，
不会进入reviewed graph或生产判读上下文。未测EDA/ECG不推断自主反应机制。

语义模式保留v3同折案例、边界反例和原始数值读数，去除整库类别统计摘要。
论文、领域知识、动态模型证据、反证与本地协议分层标注；案例标签仅用于
逐例匿名比较。真值、问卷、身份、旧答案及融合类别/分数不进入实际判读请求。
输出继续使用原六字段协议。每次完整路径解释写入当前会话的
`model_evidence.jsonl`；只有真实LLM成功时才保存其预测，无云端时预测为空。
会话记录不写入静态案例图，也不参与后续检索；没有标签反馈。

验收命令：

```powershell
.\.venv\Scripts\python.exe -m scripts.build_state_graph_assets
.\.venv\Scripts\python.exe -m pytest -q tests/test_state_graph_rag.py tests/test_semantic_observations.py tests/test_semantic_modes.py
.\.venv\Scripts\python.exe -m scripts.audit_state_graph_rag --help
```

审计旧146条冻结数值输入只检查资产、同折隔离与请求结构，不能补造其缺失的
功率观测，更不能复用旧答案为新版本计分。只有新语义版本146条真实判读均有效、
至少112/146正确、BACC不低于73.2143%，并通过工程、引用与标签隔离核验，
才自动同步为默认版本。尚未开展该轮真实模型评估。版本状态见
`/api/rag/version`、`/api/brain/catalog`与`/api/online/catalog`。

2026-10-12 的单轮开发对照已完成，九条技术失败经用户单独授权各恢复一次。
最终 GraphRAG v1 与普通 RAG 均为 **107/146，ACC 73.2877%**，
改正4条、误改4条、净增0条。Graph 未达到至少112条正确的发布门槛，
当前默认继续使用普通 RAG；Graph v1 保留实验入口。

首页与 `/online` 新建会话可选择默认版本、普通 RAG 固定基线或 GraphRAG。
版本在会话创建时固定，运行中不切换；历史缺少记录时显示“历史未记录版本”。
`/rag-comparison` 只读取已保存结果，不发起模型判读。

普通分支固定 `raw_decisive_rag_v4 / percentile_group`、同折参考资产和判读协议。
`outputs/rag_assets/regular_baseline_v1` 保存经哈希核验的协议与旧方法教学快照；
`eeg_agent/regular_rag.py` 是固定检索实现。显式普通/Graph入口均使用
`gpt-6.1-sol`、Responses、high、8192 和历史六字段格式。
原默认在 Graph 验收通过前保留原生产判读协议。

GraphRAG 使用本地 JSON 方法图与按 fold 隔离的案例图，资产为
`outputs/rag_assets/eeg_evidence_graph_v1`。从数值近邻和本次真实冲突出发，
最多两跳连接共享特征、字段定义、实际验证资料和同折符号模式案例，最多六条。
MIL 不参与选例，重复来源不增加独立票数；图不生成查询概率或类别。
资产不符时明确不可用；没有冲突或模式匹配时保留原选例并记录原因。

方法知识图不含带标签案例；各折案例不能跨折合并或生成全局标签摘要。
图不读取查询问卷、真值、身份、旧分类答案或隐藏融合结果。
预处理、冻结模型、历史缓存与 numerical worker 未因建图改写。

单轮比较复用普通 RAG 已有结果，**普通新增判读0次**。
Graph146个单元先24再122，前24条计入同一轮，未依据其结果改规则。
原始实验执行146次初始尝试和最多8次技术重试，共154次，
得到137条有效答案、9条失败。原始实验和失败账本保持只读。
用户随后单独授权九条失败各恢复一次，字面复用冻结请求和原判读参数，
增加9次尝试，均有效；总计 **154 + 9 = 163次尝试、146条有效答案、0条未解决失败**。
九次恢复属于新增故障恢复授权，不计入原八次技术重试额度。
合并保留原始首次有效答案，仅对原始失败采用恢复首次有效答案；
未重新采样有效答案，也未重新判读普通 RAG。

| 指标 | 普通 RAG 历史基线 | GraphRAG v1 恢复后合并 |
|---|---:|---:|
| 正确 / 固定分母 | 107/146 | 107/146 |
| ACC | 73.2877% | 73.2877% |
| BACC | 73.2143% | 73.2707% |
| High召回 | 75.0000% | 73.6842% |
| Low召回 | 71.4286% | 72.8571% |
| 有效答案 / 未解决失败 | 146 / 0 | 146 / 0 |

改正为q033、q111、q112、q118；误改为q001、q045、q107、q146。
路径ACC差值为0个百分点；配对被试聚类95%置信区间为
**[-4.7945，+5.2326]个百分点**，以24名被试为重采样单位，
10,000次percentile bootstrap，seed20261012。
全部146条通过生产六字段及引用校验，引用警告和实际模型违规均为0。
工程门槛通过，分类性能门槛未通过；BACC的微小上升不能替代ACC目标。

默认发布门槛：Graph至少112/146（ACC76.71%），BACC不低于普通基线，
146条均有效并且工程核验通过；不要求单轮置信区间下界为正。
未达标保留普通默认和 Graph 实验入口。通过后自动同步默认入口、知识卡、
状态与部署资产，空闲时重载；普通固定入口继续可选。

当前默认实际模式为 `regular`，判读协议为 `legacy_production`；
显式 `regular` 和 `graph` 使用上述历史高推理强度对比协议。
已有会话固定的版本不随本次结果发布改变。v2当前是独立离线候选：
只由原始与校准方向冲突触发案例扩展；共享特征方向分歧独立出现时保留普通案例。
冲突匹配要求参考在查询冲突的同一工具上具有同一raw→cal转换。
最近两类锚点和被试覆盖规则保留；联合排序仍可能补入没有匹配组的类别案例，
这些案例明确标记为未匹配，不能称为冲突匹配证据。
候选使用独立module、asset及request，未接入网页或默认入口，
不纳入本次v1部署包。尚未进行新的真实API分类评估，准确率未知。

Windows实验资料保存在
`D:\桌面\文献总\EEGAGENT\outputs\graphrag_upgrade\20261012`：
`experiment`保留原始实验，`recovery_v1`保存授权、九条恢复原生响应、
`combined_summary.json`、`all146_combined_results.csv`和`COMBINED_REPORT.md`。
网页 `/rag-comparison` 展示合并结果，其只读资产是
`outputs/rag_assets/comparison_v1/comparison.json`，打开页面不触发云端调用。

旧基线由不同供应商接续，数据也曾用于开发。本比较属于复用历史结果的
单轮开发对照，不证明严格供应商控制后的因果提升、重复稳定性或外部泛化。
Graph同时改变参考选例、被试覆盖和上下文，不能把结果单独归因于图结构。
本轮和恢复期间维持相同逻辑CPA入口及模型参数，但服务重启前后上游路由与
后端权重连续性不能仅由入口地址和原生model字段证明。
在线CPU FP32与参考生成后端存在精度差异，固定离线输入比较不等于在线准确率。
