# EEG 状态解释知识 sidecar

此目录为`raw_eeg_state_graph_v1` / `state_reasoning_v1`的确定性输入，首版仅覆盖VR motion sickness。旧方法源和图资产不在这里重写。

`claims.json`为构图输入，`sources.json`为来源登记；Markdown是便于审阅的知识卡。每个卡片的JSON frontmatter携带source记录摘要。图构建应读取JSON，不按Markdown文本自动推断关系。

## 审阅与来源

本次按nature-academic-search的多源检索流程执行。没有可用学术MCP工具，使用公开Crossref DOI元数据和Europe PMC/PubMed索引及JATS全文；一次Europe PMC查询503、一次Cavanagh全文500后继续其余来源，不把失败文献用于事实边。reviewed表示已核对元数据与指定原文段落，既不是人工临床审阅，也不是当前模型的效果验收。

Lee 2025与Lee 2026为同一作者团队，`independence_group`相同；两个记录不能当独立验证。2026 Scientific Reports DOI含025但实际发表年为2026。原文曲面显示器中央gamma叙述与Table 1不一致，本版不采用其方向。

## 已审阅关系

alpha、theta、beta仅采用有范围的研究关联；机制边属于解释假说。theta/beta方向依赖脑区和显示器，alpha跨研究增减不一，EEG与SSQ在相关研究中未显著关联。未建立通用方向、严重度阈值或真实症状概率。source的excerpt为本次核验的短释义，locator对应具体原文章节；`fetched_content_sha256`绑定核验时JATS内容。

自主神经节点来自EDA/ECG研究背景，不作为EEG-only观测推理路径。静态医学关系在论文层提供源证据，图知识及反证层由构图器标明角色，local_protocol只说明本地工具与方法。

## 候选清单

- Gamma：论文30–50 Hz与本地gamma_low 30–45 Hz不同，当前在线冻结路径特征仅到beta。候选不进检索。
- Entropy：缺少兼容EEG熵定义与已核验VRMS关系。不得拿光流/HRV熵替代，不输出虚构观测。
- Connectivity：已核验2025 TVCG索引摘要，尚未核验全文、具体指标和机制。协方差/CSP分类分数不能替代测得连接性。

候选只用于说明覆盖缺口，不能进入reviewed graph或LLM判读。有限检索未找到关系不代表不存在关系。

## 版本与完整性

所有医学节点及边必须指向来源ID，source DOI/年/章节定位不可缺。source_sha256为去除自身字段后的canonical JSON SHA256，canonical规则为Python `json.dumps(..., sort_keys=True, ensure_ascii=False, allow_nan=False)`默认分隔符。图构图器补source_layer、content_sha256及version。当前分类头分数不产生观测节点；标签、身份与问卷不进入动态解释证据。
