# EEG Agent：多领域对话与 VRMS 脑电监测

首页和在线页面新增普通 RAG / GraphRAG 模式选择，创建会话时固定版本。
普通 RAG 固定基线保留；Graph 为按 fold 隔离的局部证据图。
新增 `semantic_graph` 状态推理实验入口，以真实theta/alpha/beta路径功率变化查询
已审阅的VR motion sickness关联、机制假说与反证。来源sidecar在`knowledge/graph/`，
独立图资产在`outputs/rag_assets/eeg_state_graph_v1/`，动态证据仅记入当前会话。
新版本尚未完成146条真实判读评估，默认仍为普通RAG；版本见`/api/rag/version`。
访问 `/rag-comparison` 查看已保存的前后结果，不触发普通基线重判。
实现、判读协议与默认发布门槛见 [GraphRAG说明](docs/GRAPH_RAG.md)。

新增“[实时在线分析](docs/ONLINE_ANALYSIS.md)”页面 `/online`：同一城市巡航 CDT 数据按原始顺序回放，每 5 秒更新三个领域的本地结果；冻结 VRMSModel 使用 mamba-minimal CPU 后端，云端仅在路径结束后调用。现有离线模型、预处理和历史结果保持原样。树莓派安装与实机验收见 [在线部署说明](docs/ONLINE_DEPLOYMENT.md)。

首页现支持直接提问“分析被试10的路径、疲劳相关指标、心理负荷及高类路径时间”。无需先选研究路径：对话自动定位本地原始CDT，三个Agent共享从同一原始波形计算的匿名工具结果。分析参数内可选原始文件或上传EDF。路径由内嵌Trigger统计；FatigueAgent计算TAR及变化，EmotionAgent计算额区theta/顶区alpha与FAA；VRMSAgent重新执行原始波形编码、MIL、八路数值工具和冻结融合，再由所选GPT或DeepSeek读取本轮实际工具证据和默认优化RAG，独立给出最终High/Low。GPT使用Responses，DeepSeek使用Chat Completions；报告明确标明实际提供方。MIL、融合、云端类别分别保存，首条高类按所选云端最终类别定位；云端失败的路径不发布最终类别，数值结果单独保留。同一文件的预处理窗口与频谱指标可复用内存缓存；VRMS每轮重新编码并执行MIL与八路数值工具，云端函数调用读取本轮已计算证据。Supervisor使用当前接口解释并汇总已执行结果。问卷、自评、目标标签、预处理MAT、历史预测及被试答案不进入推理；只有训练期冻结参数在本地模型工具内使用。原始推理在分类前提供已审阅的九路数值方法教学和已校验的匿名训练案例；用户资料仍可用于独立知识问答。高类时间表示首条高类完整路径的区间，整路径判定在路径结束后形成，不是首次症状秒数。一般EDF可运行指标工具；VRMS分类还需要匹配1024Hz、30个指定通道及M1/M2参考。旧研究入口及网页实验评价已移除。本无参考原始协议尚未重新评价ACC，历史71.92%不能直接作为本入口的准确率。

原始记录开始时已戴上VR眼镜，本分析取消记录开头五至三十五秒的“任务前静息基线”。完整路径按20→22划分，完整休息段按22→下一20划分，分别给出首末变化与最高片段及区间；开头和未闭合片段不归为完整休息。原始VRMS数值工具采用已有无参考分支，不读取该开头波形，训练归一化与冻结权重保持不变。旧原始回答保留并标示参考已停用；历史研究评价不作为本无参考协议效果的证据。

多领域对话工作台仅保留原始 EEG 分析与方法问答，详见 [工作台说明](docs/BRAIN_WORKSPACE.md)。旧研究路径选择、缓存特征推理入口和网页全路径实验评价已移除；旧研究对话仅供查看和下载，不再接受追问。历史实验文件和当前原始推理依赖的冻结权重、训练归一化及数值工具参数保留。首页可切换 DeepSeek / CPA GPT，分别保存密钥。各 Agent 完成工具后检索共享库和领域库，由真实 API 生成报告，再由 Supervisor 汇总。原监测控制台 `/monitor` 保留，其旧监测模型不作为新版 VRMSAgent 分类来源。

在 Windows 开发及训练，树莓派5／8GB执行因果预处理、ONNX检测、时间统计、频谱／协方差测量、本地BM25和网页服务。云端 Supervisor 理解与汇总，VRMSAgent调用登记工具并解释。辅助工具不改写所选深度模型的路径类别。

## 启动与网页操作

### 当前默认：优化案例 RAG 与九路方法教学

首页原始 EEG 与在线完整路径结束判读统一使用 `eeg_agent.default_rag`，版本 `raw_decisive_rag_v4 / percentile_group`。八工具按同折参考库经验百分位、四来源组等权距离检索 High/Low 案例，最多六条；MIL不参与检索距离。24折参考资产经过来源、哈希和角色分离核验，存于 `outputs/rag_assets/current_raw_policy_v3`。LLM读取实际九路证据与案例后独立给出最终类别，冻结融合类别和分数不进入这一默认判读请求。方法教学来自已审阅的九路知识卡，用户上传资料不进入分类方法入口。

后续已验证RAG优化必须同步成为默认，更新方法知识、部署资产与状态接口，核验请求并重载空闲服务；无需用户再次要求切换。历史实验固定其显式variant，原模型、预处理和缓存不随默认更新改写。缺失或损坏参考库时明确不可用，不静默退回旧检索。流程及验证入口见 [默认RAG说明](docs/RAG_DEFAULT.md)。`/api/brain/catalog` 与 `/api/online/catalog` 的 `vrms_case_rag` 字段显示当前版本。

参考被试未参与该折头与校准器拟合，但曾用于selector开发。在线CPU FP32与参考生成后端存在精度差异，原RAG实验ACC不直接代表网页或在线版的新性能。

```powershell
Set-Location -LiteralPath 'D:\桌面\2'
.\.venv\Scripts\python.exe -m eeg_agent serve --port 8880
```

打开 http://127.0.0.1:8880/ 进入多领域对话；点击“EEG 监测”或访问 http://127.0.0.1:8880/monitor 进入监测控制台。选择EEGNet或EEGConformer及检查点，再选择Adaptive、Deep only或Full evidence并启动。优先选择 `eog_rls_seed42`（P1眼电清理版），`fixed_seed42`保留作P0对照。默认Acquisition 01、130秒、1倍速、mock。首次输出须等待先行基线和完整任务窗口；每5秒更新。

模型、温度、阈值和归一化在任务开始时固定，运行中不能热切换。更换模型应停止后新建任务。刷新页面会恢复当前任务；服务重启后仅恢复资料，原运行任务标为中断，继续回放需创建新会话并重新预热。

网页展示计划、质量与覆盖、模型曲线及阈值、时间趋势、按需EEG测量、调用原因、知识引用和报告。可追问并下载Markdown／JSON。未训练或不兼容模型明确显示“预测不可用”，不偷偷替换模型。

## 数据与模型合同

直接读取 `D:\桌面\数据集\城市巡航`，不复制原始文件。24人147条CSV候选，146条原始事件完整；Acquisition 04末条评分路径在EOF截断，拒绝分类训练。154条描述、Acquisition 04及23的元数据异常保留审计。评分＜30为低类，≥30为高类。

训练15人87条完整路径，验证4人36条，测试5人23条。实际参与训练的合法窗口及路径数另见模型合同，质量拒绝不会偷偷删除审计分母。归一化只来自训练被试。

P0固定乳突平均参考、50Hz因果陷波、0.5–45Hz因果带通、抗混叠与1024→256Hz降采样。模型输入30×1280、μV、5秒完整无重叠窗口。状态跨块、路径和休息段连续，断流／非有限样本后预热10秒，不能拼接两侧窗口。P0与P1有独立处理、归一化、模型和阈值合同，不能混用。

### 使用VEOG／HEOG电极去眼电

P1由10–60秒合格先行基线初始化两参考岭回归，随后用逐样本因果RLS跟踪眼电向30个EEG通道的耦合。每个输出先用过去系数减去参考预测分量，再用当前已到达的无标签EEG／EOG更新；遗忘时间固定10秒。任务冻结算法、安全参数和模型，回归系数允许因果更新。这属于被试内无标签自适应预处理，不能称为完全无目标被试适应的跨被试模型。

状态跨路径及休息连续；断流／非有限值时回到先行初始化，预热期间不更新，跳变和异常幅值跳过更新。清理后重新测量幅值、平直、突变和EEG–EOG相关。额区残余相关>0.6或最大通道移除RMS／原始RMS>1.5时拒绝分类；这是未用测试标签调节的工程门控。缺基线、EOG退化或参数不稳定时继续采流及明确标记的P0测量，P1模型预测不可用，不自动切换模型。

频谱／协方差基线在初始化到达时用同一初始去眼电映射重建并冻结，任务段由自适应映射处理。测量变化可能包含回归系数变化，不能全部归为生理变化。与EOG参考相关的真实脑活动也可能被回归移除；低频及眼动相关脑活动尤其需要谨慎解释。受控注入检查已知干净参考的误差及α／β保真；自然相关下降不能证明全部脑电保留。

网页“眼电去伪迹”展示方法、初始化状态、前后相关／幅值、移除比例、更新／跳过数和拒绝原因。本地报告保留摘要；原始EOG、回归矩阵及身份不上传云端。

EEGNet为2866参数小模型；EEGConformer采用40维、6层、10头和卷积token化，986626参数。两者从随机初始化训练，没有公开预训练或旧项目权重。seed42、Adam0.001、batch32、最多20轮、patience5；以验证路径BACC（raw概率0.5）选检查点，再按验证路径NLL拟合温度、验证路径BACC选阈值。

路径概率为有效窗口校准概率均值，至少3个有效窗、有效时长覆盖80%才发布类别。5秒窗的训练标签是广播的路径评分，不是即时症状真值。温度校准也仅按路径评价。

## 工具和分析策略

先行基线只用首任务前、预热后已到达且质量合格的完整5秒窗口，至少10秒、最多60秒，在首任务开始冻结，不训练分类器。缺基线仍可计算绝对功率及比值，变化和协方差距离返回不足。

频谱使用2秒Hann Welch、50%重叠、去均值；提供δ／θ／α／β／30–45Hz功率、相对功率、额区及枕区汇总、θ/α、θ/β、慢快比和相对基线变化。协方差为固定10%向trace-scaled单位阵收缩，参考为基线矩阵均值，距离采用AIRM。距离无高低类方向，不作为已验证生物标志物。

时间分析保留最近30秒最多6窗，至少3个有效窗才发布趋势，斜率单位为概率／分钟，波动和跳变描述模型输出。连续3个有效完整窗≥阈值产生“模型持续高类候选段”，分别记录首窗起点与第三窗完成确认时间；无效窗、断流、路径切换打断。

Adaptive的频谱触发为历史＜3窗、margin＜0.10、当前与近期均值异侧、std＞0.15或相邻跳变＞0.20；协方差触发为异侧、跳变＞0.20或完整分析请求。Early Exit只省去当前窗的辅助工具，检测与采流继续。规则不是已验证临床置信度。

RAG为本地BM25，600字符分块、100重叠、top5，保存引用ID、来源、章节、版本与哈希。知识资料不得覆盖模型合同。数值由本地工具计算。云端报告选择本轮事实编号，程序还原精确来源后校验数值、时间与类别；报告校验失败允许一次带反馈的修正，仍失败时回退模板。API连接或超时失败不自动重试。

## 命令行

```powershell
.\.venv\Scripts\python.exe -m eeg_agent audit
.\.venv\Scripts\python.exe -m eeg_agent prepare
.\.venv\Scripts\python.exe -m eeg_agent train --model both --epochs 20
.\.venv\Scripts\python.exe -m eeg_agent evaluate --model both --split test
.\.venv\Scripts\python.exe -m eeg_agent export --model eegconformer
.\.venv\Scripts\python.exe -m eeg_agent models
.\.venv\Scripts\python.exe -m eeg_agent knowledge-index
.\.venv\Scripts\python.exe -m eeg_agent monitor --model eegnet --subject 1 --speed 1 --max-seconds 130
.\.venv\Scripts\python.exe -m eeg_agent compare-strategies --model eegnet --speed 100 --max-seconds 130
.\.venv\Scripts\python.exe scripts/verify_online_offline.py
# 眼电清理版：单独准备缓存并从零重训两个模型
.\.venv\Scripts\python.exe -m eeg_agent prepare --profile P1
.\.venv\Scripts\python.exe -m eeg_agent train --profile P1 --model both --epochs 20
.\.venv\Scripts\python.exe -m eeg_agent evaluate --profile P1 --model both --split test
.\.venv\Scripts\python.exe scripts/verify_eog_online_offline.py
.\.venv\Scripts\python.exe scripts/validate_eog.py --method adaptive --output outputs/vrms_v2/reports/my_eog_validation.json
.\.venv\Scripts\python.exe -m pytest -q
# 正式内部五折入口，本次不自动运行
.\.venv\Scripts\python.exe -m eeg_agent fivefold --model eegnet --epochs 20
```

新增产物位于 `outputs/vrms_v2/`：P0缓存为`cache`，眼电缓存为`cache_eog_rls_v1`；眼电检查点为`models/<model>/eog_rls_seed42`。原P0检查点、旧 `outputs/cache`、`outputs/models/baseline`和旧报告保留。重训已有检查点自动使用新时间戳目录。旧回放入口仍在，但旧模型处理合同不能用于新版分类。

## API与云端设置

在“API 与模型配置”中选择“中转站”，填写服务商提供的 Base URL（通常以 `/v1` 结尾）、Model ID 和 API Key，选择 Chat Completions 或 Responses 协议，然后点击“保存并启用配置”和“测试当前配置”。对话页的云端模型下拉框也可切换该渠道。中转站、CPA GPT 和 DeepSeek 分别保存密钥；中转站首次配置需要填写自己的密钥。服务需兼容所选协议和 JSON 输出；VRMS 工具调用流程还需模型支持 function calling。

请求协议保存在本项目 `cloud_profiles.json` 和当前配置的 `EEG_API_PROTOCOL` 中，可取 `chat_completions`、`responses` 或 `auto`。自动模式沿用原有规则；显式选择 Chat Completions 时，GPT 模型也使用 `/chat/completions`。Base URL 填根地址，不附加 `/responses` 或 `/chat/completions`。

任务API为POST／GET `/api/tasks`、详情、`events`、`stop`、`report`、`messages`；目录为 `/api/models`、`/api/tools`；知识查询为 `/api/knowledge/search?q=...`。SSE事件有ID，支持Last-Event-ID和after续传。

在网页左侧打开“云端API设置”，填写OpenAI兼容接口的Base URL、云端模型ID和API密钥，点击“保存配置”。配置保存在本项目 `.env`；保存后无须重启服务，新建任务即可使用。随后可点击“测试已保存配置”，它会使用已保存配置发送一次固定测试请求，可能产生API费用。返回监测页面，手动选择“真实云端API”后启动任务。

设置页以明文回填所选渠道的已保存密钥，可直接查看和编辑；密钥不进入浏览器持久化、日志或报告。留空并保存会保留此渠道已有密钥。“连接测试 · 当前输入”使用输入框中的地址、模型、密钥与协议，无需先保存或启用，也不会修改配置；“测试已启用配置”验证当前生效的渠道。真实API任务使用配置期间禁止修改和测试；需要更换接口时，先停止任务，再保存配置并创建新任务。mock任务不占用云端配置。

连接测试限64个输出token且不重试；推理内容已返回但输出到达上限时，显示“接口已响应”及限制说明。连接成功只验证接口连通，完整Agent规划、工具调用和报告仍需任务验证。

真实API仅使用项目 `.env` 或同名环境变量：`EEG_API_BASE_URL`、`EEG_API_MODEL`、`EEG_API_KEY`，示例见 `.env.example`。项目文件中的显式值优先，未设置的项才读取同名环境变量；网页保存不会修改系统环境或其它应用。不从其它应用读取密钥。设置接口为GET／PUT `/api/cloud-settings`及POST `/api/cloud-settings/test`，读取只返回配置状态。每任务50次共享上限、单次10秒、不自动重试；Supervisor开始规划和结束汇总，VRMSAgent每30秒源时间审查。停止不等待在途云端回复，过期或非法请求拒绝执行。

原生 `api.deepseek.com` 的 `deepseek-flash`／`deepseek-pro` 在Agent工具和报告请求中关闭默认思考模式，使有限输出预算用于工具请求和结构化最终答复；依据[DeepSeek官方说明](https://api-docs.deepseek.com/guides/thinking_mode)。其他接口不附加该参数。调用日志记录结束原因和是否截断，不保存内部思考内容。结束汇总最多等待三轮调用的超时预算，停止仍可立即中断等待。

云端仅接收匿名质量、地区频谱摘要、协方差摘要、模型／时间输出和知识证据。不传原始EEG、完整PSD、被试身份、真实评分或标签。mock是模拟响应，不能用于声称实际LLM收益。

## 树莓派部署

解压新版 `deploy/VRMSAgent_Pi_eog_bundle.zip` 到 `/srv/eeg-agent`，准备原始CDT及对应DPO／CEO文件，修改 `deploy/pi.yaml` 的data_root。原 `VRMSAgent_Pi_bundle.zip` 保留历史版本。系统使用ARM64 Python3.11或以上：

```bash
cd /srv/eeg-agent
python3 -m venv .venv
.venv/bin/pip install -r requirements-runtime.txt
.venv/bin/python -m eeg_agent --config deploy/pi.yaml serve --host 0.0.0.0 --port 8880
```

普通venv不复用Windows环境，不安装PyTorch训练栈。数据根下保持 `data/raw/Acquisition xx.cdt` 等布局。ONNX和合同已包含；本地回放监测不要求把标签CSV送到Pi。systemd示例需按实际用户及路径修改。

使用systemd时，示例服务以`eegagent`用户运行，并只监听树莓派本机的`127.0.0.1:8880`。完成解压、数据路径设置及依赖安装后，在树莓派执行：

```bash
if ! id -u eegagent >/dev/null 2>&1; then
  sudo useradd --system --home-dir /srv/eeg-agent --shell /usr/sbin/nologin eegagent
fi
sudo chown -R eegagent:eegagent /srv/eeg-agent
sudo install -m 644 deploy/eeg-agent.service /etc/systemd/system/eeg-agent.service
sudo systemctl daemon-reload
sudo systemctl enable --now eeg-agent.service
sudo systemctl status eeg-agent.service
```

项目目录的所有权用于允许服务保存任务、知识索引和网页填写的本地API配置。`eegagent`是服务账户，SSH连接使用树莓派原有的可登录账户。在Windows打开终端，保持下面的转发连接运行：

```powershell
ssh -N -L 8881:127.0.0.1:8880 你的SSH用户名@树莓派IP
```

随后在电脑打开 http://127.0.0.1:8881/ 访问树莓派网页。电脑端使用8881，避免与Windows原型的8880冲突；上面的手动启动命令使用`0.0.0.0`，可通过`http://树莓派IP:8880/`访问。

`scripts/verify_eog_online_offline.py`用于Windows的完整缓存验收，需要原始数据和`cache_eog_rls_v1`，不能仅凭Pi部署包直接运行。Pi包不包含原始流、缓存、训练权重或真实API密钥；Pi运行命令须显式提供`--config deploy/pi.yaml`，再通过实际回放验收ARM64推理和运行性能。

Pi依赖安装、ARM64一致性、1倍速延迟／内存和能耗须硬件实测；Windows耗时不能替代它们。实时采集设备驱动尚未接入，本版使用真实CDT回放。

## 评价边界

固定划分属于开发协议，当前数据曾用于开发，不称未经接触的独立测试。分类指标是接受路径上的条件指标，同时报告拒绝／覆盖和分类别接受情况。对比三种工具策略应得到相同路径预测，研究调用、资源和报告证据质量，不预设辅助测量提高准确率。不存在逐时症状、真实onset、提前量或严重度效度验证。
