---
{"title":"普通 RAG、GraphRAG与语义状态推理：入口、来源与版本边界","source":"eeg_agent/default_rag.py; eeg_agent/state_graph_rag.py; eeg_agent/semantic_observations.py; knowledge/graph/sources.json; outputs/rag_assets/eeg_state_graph_v1/manifest.json; docs/GRAPH_RAG.md","evidence_type":"local_protocol","version":"graph-rag-entry-2.1","scope":"整路径 High/Low 的检索解释与来源边界；15条开发抽查不代表146条发布评估，不包含当前查询答案"}
---
# 普通 RAG 与 GraphRAG

2026-10-13 增加 semantic_graph 显式实验模式，固定 raw_eeg_state_graph_v1 / state_reasoning_v1，默认仍为 regular / percentile_group。以下旧v1比较成绩不代表语义版性能。语义版只有146条真实新判读全部有效、至少112条正确、BACC不低于73.2143%，并全部通过工程、来源引用与标签隔离核验，才自动发布为默认。

实际观测只测量合格整路径窗口的theta 4–8、alpha 8–13、beta 13–30 Hz功率，使用30通道mean log power的末三分之一减首三分之一。首段不是静息基线，分类工具分数不代表功率；增减阈值仅用于描述。沿biomarker→机制假说→VR motion sickness追溯论文的DOI、年份及章节，不把方向、图度数、路径数或案例类别比例当作概率或类别票。论文来源、领域知识、模型观测、反证和协议分别标注，每条节点及边带来源与内容哈希。两篇同作者团队文献属于同一来源独立组。

语义上下文包含supporting_paths、contradictory_paths、evidence_strength、confidence_basis、evidence_status与fallback_reason。通用方向、基线、区域、模态和EEG–问卷边界会保留在反证路径中，但不自动把所有路径压成low；实测biomarker有直接审阅论文支持且没有直接适用的实质反证时为medium。缺少直接论文支持或存在material_counterevidence时为low；未分类的新反证按实质反证处理，实质反证优先保留。medium评价实测biomarker的论文关联，机制假说路径自身仍为low；它与LLM的分类置信等级分别报告。未测或无适用审阅证据时明确unavailable，不补造观测。alpha方向差异、区域和显示器依赖及未显著EEG–SSQ相关均作为解释边界。gamma、entropy、connectivity当前只作候选，不进入已审阅上下文；没有EDA/ECG不能推断自主反应机制。

语义模式保留v3同折个别案例及边界反例、实际数值，排除整库类别统计摘要。LLM仍输出原六字段，支持与反证引用须存在于提供的上下文。动态model_evidence仅写入当前会话model_evidence.jsonl，无真实云端成功时预测为空；不读取真值、身份、问卷或融合类别，不进入静态图或下次检索。新建会话固定policy，旧会话不自动升级。旧图资产及method_source.md快照保持不变。

首页和在线页面均提供 default、regular、graph、semantic_graph 四种选择。default 在创建会话时解析为当时发布的版本；regular 固定 raw_decisive_rag_v4 / percentile_group；graph 固定旧本地 JSON 证据图版本；semantic_graph 固定 state_reasoning_v1 与 semantic_state_high_v1 协议。检索模式、判读协议及资产哈希随会话保存，运行中不切换。缺少版本记录的旧会话显示“历史未记录版本”，不能把新默认补写成历史实际版本。

普通 RAG 用八工具同折经验百分位及四来源组等权距离选取两类参考案例。GraphRAG 沿用相同距离，首先取最近 High、Low 各一条，再根据原始与校准方向冲突、共享特征工具分歧，遍历同折符号模式与方法关系，最多两跳。补充案例联合排序依次考虑模式匹配组数、被试覆盖、距离总和和原始索引，必要时补齐被试覆盖，总数最多六条。MIL 仍展示为实际读数，不参与距离或模式匹配。

方法图连接工具、共享特征、协议、知识片段及实际验证资料；各折案例图连接内部参考案例、匿名参考被试及八工具原始符号模式。关系记录来源及哈希。带标签案例不能跨 fold 检索，当前查询真值、问卷、旧答案和数值融合类别或分数不进入判读请求。没有冲突或没有匹配案例时保留普通选例并记录原因；资产缺失或校验失败时明确不可用。

图提供的关系路径最多4000字符，并保留实际读数、校准值、检索距离、参考统计及适用限制。关系和案例不构成独立票数，图不生成查询类别、概率或虚构验证指标。最终 High/Low 仍由 LLM 一次独立输出。显式 regular 与 graph 均采用固定的 gpt-6.1-sol / Responses / high / 8192 协议及六字段格式；它们不额外注入历史协议未包含的教学字段。

比较页只读取普通 RAG 的已有历史结果及保存的 GraphRAG 结果，不因打开页面而调用模型。GraphRAG 必须达到预先规定的分类与工程验收门槛才成为新默认；实现图检索本身不能证明准确率提高。旧基线混合供应商、内部案例开发使用及图分支选例数量变化均限制性能归因，固定离线输入上的结果不能直接当作在线 CPU 版的准确率。

2026-10-12 的146条单轮开发比较已完成，普通 RAG 复用旧答案，新增调用0次。原始 GraphRAG 执行154次尝试，得到137条有效答案和9条技术失败；用户另授权九条失败各恢复一次，均有效，总计163次尝试、146条有效答案、0条未解决失败。九次恢复独立记录，不属于原八次技术重试额度。原始首次有效答案优先，只有原始失败才采用恢复首次有效答案，有效答案未为提高分数而重新采样。

普通 RAG 为107/146，ACC73.2877%、BACC73.2143%，High召回75.0000%、Low召回71.4286%；GraphRAG v1恢复后同为107/146，ACC73.2877%、BACC73.2707%，High召回73.6842%、Low召回72.8571%。改正4条、误改4条、净增0条。配对路径ACC差值的被试聚类95%置信区间为[-4.7945，+5.2326]个百分点，24名被试、10,000次percentile bootstrap、seed20261012。

146条均通过生产六字段及引用校验，引用警告和实际模型违规为0。工程验收通过，但尚未达到至少112/146正确且BACC不低于73.2143%的分类性能门槛，当前默认仍为普通RAG的legacy_production协议，Graph v1保留实验入口。

v2目前是独立离线候选，只由原始与校准方向冲突触发案例扩展；共享特征方向分歧独立出现时保留普通案例。冲突匹配要求参考在查询冲突的同一工具上具有同一raw→cal转换。最近两类锚点和被试覆盖规则保留；联合排序仍可能补入没有匹配组的类别案例，明确标记为未匹配，不能称为冲突匹配证据。候选使用独立module、asset及request，未接入网页或默认入口，不纳入本次v1部署包；尚未进行新的真实API分类评估，准确率未知。
