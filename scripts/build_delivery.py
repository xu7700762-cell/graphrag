"""Assemble traceable Windows acceptance and the small ARM64 deployment bundle."""
import datetime
import importlib.metadata
import json
import platform
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eeg_agent.config import load_config,file_hash,write_json,contract,digest
from eeg_agent.tools import TOOL_CATALOG
from eeg_agent.models import model_catalog

cfg=load_config();OUT=Path(cfg['output_root']);REPORT=OUT/'reports'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

tests=(REPORT/'pytest-result.txt').read_text(encoding='utf-8')
assert '50 passed' in tests and '1 skipped' in tests,tests
equivalence=read(REPORT/'online_offline.json');assert equivalence['status']=='passed'
browser=read(REPORT/'browser_acceptance.json');assert browser['status']=='passed'
browser['actions']['knowledge_no_match']=True
for entry in browser['real_time_tasks']:entry['model_id']=entry['model']['model_id']
write_json(REPORT/'browser_acceptance.json',browser)

catalog=model_catalog(cfg);models={};evaluations={};strategies={}
for item in catalog:
    model=item['model_id'];assert item['available']
    directory=OUT/'models'/model/item['default_checkpoint_id'];models[model]=read(directory/'contract.json')
    evaluations[model]=read(REPORT/f'evaluation_{model}_fixed_seed42_test.json')
    comparison=read(REPORT/f'strategy_comparison_{model}.json');assert comparison['identical_path_results']
    strategies[model]=[{k:r[k] for k in ('tool_counts','early_exits','processing_seconds','process_cpu_seconds',
                    'peak_process_rss_bytes','wall_seconds','source_seconds')}|{'strategy':r['request']['strategy']} for r in comparison['reports']]

versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','onnx','onnxruntime','fastapi','pydantic','openai','pytest','matplotlib')}
sources=[p for p in (ROOT/'eeg_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
sources+=[ROOT/'configs/local.yaml',ROOT/'pyproject.toml']
sources+=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'knowledge').glob('*.md'))
experiment={'version':'VRMSAgent-0.2.0','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'platform':platform.platform(),'python':sys.version,'software':versions,'preprocessing_contract':contract(cfg),
    'contract_hash':digest(contract(cfg)),'cache_manifest_sha256':file_hash(OUT/'cache/manifest.json'),
    'data_audit_sha256':file_hash(REPORT/'data_audit.json'),
    'source_sha256':{p.relative_to(ROOT).as_posix():file_hash(p) for p in sources},
    'models':{k:{f:m[f] for f in ('model_id','checkpoint_id','checkpoint_sha256','onnx_sha256','inference_contract_hash',
                'parameter_count','selected_epoch','temperature','threshold','assigned_training_subjects',
                'normalization_fit_subjects','validation_subjects','actual_subject_splits','training','calibration','threshold_selection')} for k,m in models.items()},
    'temporal_policy':{'history_seconds':30,'maximum_slots':6,'minimum_valid_windows':3,'persistent_high_windows':3},
    'auxiliary_policy':{'margin_lt':.10,'std_gt':.15,'jump_gt':.20,'changes_class':False},
    'boundaries':['development subject-disjoint split','conditional accepted-path classification','no instantaneous symptom labels',
                  'mock acceptance, not paid cloud efficacy','Windows timing, Pi hardware not tested','formal fivefold not executed']}
write_json(REPORT/'experiment_manifest.json',experiment)
write_json(REPORT/'acceptance.json',{'status':'passed_with_explicit_external_validation_boundaries',
    'tests':{'passed':50,'skipped':1,'skip_reason':'legacy baseline artifact under new output root; both new model artifacts checked separately'},
    'numerical_online_offline':equivalence,'browser':browser,'tools_quality':read(REPORT/'quality_measurements.json'),
    'onnx':{k:m['onnx_validation'] for k,m in models.items()},'strategies':strategies,
    'separate_acceptance':{'paid_cloud_api':'not_run','Pi_hardware':'not_run','formal_fivefold':'not_run','live_acquisition_driver':'not_integrated'}})

lines=['# VRMSAgent 双模型实施与验收报告','',
    '本版已完成P0数值修复、EEGNet／EEGConformer从零训练、路径校准、UI模型选择、辅助EEG工具、自适应调度、本地RAG和共享任务服务。',
    '', '## 模型评价','', '以下为固定被试划分开发结果，指标仅在接受路径计算，同时保留拒绝分母。验证集用于检查点、温度和阈值选择，测试集曾用于原型开发。',
    '', '| 模型 | 测试接受／全部 | BACC | AUROC | Macro-F1 | Brier | ECE | 温度 | 阈值 |', '|---|---|---|---|---|---|---|---|---|']
for model,r in evaluations.items():
    m=r['metrics'];lines.append(f"| {model} | {m['accepted_paths']}/{m['total_paths']} | {m['BACC']:.4f} | {m['AUROC']:.4f} | {m['Macro_F1']:.4f} | {m['Brier']:.4f} | {m['ECE_10_equal_width']:.4f} | {r['temperature']:.4f} | {r['threshold']:.2f} |")
lines+=['','两模型差异较小，不能据此认定稳定优劣或泛化优势。路径概率ECE仍较大，温度拟合不保证测试校准改善；不得将高概率称为临床高置信度。',
    '', '![路径可靠性图](figures/path_reliability.png)','',
    '## 数值与模型验收','',
    '- 50项测试通过；1项旧baseline产物测试跳过，新两模型的真实ONNX另外完成验收。',
    '- 正常／NaN／Inf分块一致、未来隔离、事件到达、完整原始窗口、断流、先行基线冻结和停止均有测试。',
    '- 真实CDT前130秒以256／777／1024块大小核对：每组19窗，P0误差0，QC一致，边界一致；两模型logits／概率误差0。',
    '- 原始及处理后的幅值、平直、突变、工频和EEG–EOG相关均可测量；新增测量没有改变质量门控或P0缓存。']
for model,m in models.items():lines.append(f"- {model}：{m['parameter_count']}参数；实际训练{len(read(OUT/'models'/model/'fixed_seed42/training_history.json'))}轮，选第{m['selected_epoch']}轮；真实及合成输入ONNX最大误差{m['onnx_max_abs_error']:.3g}。")
lines+=['','## 网页与真实1倍速回放','',
    '实际通过网页选择两种模型并分别完成130秒1倍速回放，检查模型锁定、刷新恢复、报告、追问、停止、RAG匹配与无匹配、指标和协方差展示。',
    '', '| 模型 | 源时长 | 墙钟时长 | 窗口发布延迟P95 | 服务进程峰值RSS | 任务ID |','|---|---|---|---|---|---|']
for t in browser['real_time_tasks']:
    lines.append(f"| {t['model_id']} | {t['source_seconds']:.0f}s | {t['wall_seconds']:.2f}s | {t['lag_p95_seconds']*1000:.2f}ms | {t['rss_bytes']/1024**2:.1f}MiB | {t['id']} |")
lines+=['','窗口发布延迟从完整窗口的源时间终点计算，不是从第一样本开始计算，因而不包括5秒采窗时长。进程RSS是Windows服务进程观测值。',
    '', '## 辅助工具成本对照','',
    '同一原始流、同一模型、130秒源时长、100倍速、mock；三策略路径类别、概率和覆盖完全一致。以下是该路径的测量，不代表其它路径或Pi性能。',
    '', '| 模型／策略 | 频谱调用 | 协方差调用 | Early Exit | 数值处理总耗时 |','|---|---|---|---|---|']
for model,runs in strategies.items():
    for r in runs:lines.append(f"| {model}／{r['strategy']} | {r['tool_counts'].get('BiomarkerCalculator',0)} | {r['tool_counts'].get('CovarianceAnalyzer',0)} | {r['early_exits']} | {r['processing_seconds']:.4f}s |")
lines+=['','![实际辅助工具调用](figures/tool_usage.png)','',
    'Early Exit只退出额外工具链；基础检测仍持续。图示不主张分类准确率提高，成本受模型输出和回放条件影响。能耗未测量，费用未提供价格，不捏造金额。',
    '', '## 产物与部署','',
    '- 使用说明：项目README；模型冻结合同：outputs/vrms_v2/models下各contract.json。',
    '- 实验清单：experiment_manifest.json；测试：pytest-result.txt；数值：online_offline.json；浏览器：browser_acceptance.json。',
    '- 知识来源：knowledge目录，当前5个来源21个分块，模型来源为原始论文和作者代码链接；不在知识上下文填入本项目测试成绩。',
    '- 本轮仅mock语言接口；真实API、正式五折和Pi硬件实测保留独立入口。',
    '- Pi包只含代码、知识资料、两种ONNX／合同及部署说明，不包含原始数据、训练缓存、PT权重或API密钥。',
    '', '当前不存在逐时症状标签，真实onset、提前量、严重度和临床效度均未验证。实时采集驱动仍需后续硬件接入。']
(REPORT/'VRMSAgent_实施与验收报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

docs=ROOT/'docs';docs.mkdir(exist_ok=True)
write_json(docs/'tool_contracts.json',TOOL_CATALOG)
write_json(docs/'model_contract_summary.json',{model:{k:m[k] for k in ('model_id','architecture','contract','allowed_profiles','threshold','temperature','publication_policy','inference_contract_hash')} for model,m in models.items()})

bundle=ROOT/'deploy/VRMSAgent_Pi_bundle.zip'
files=[p for p in (ROOT/'eeg_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += list((ROOT/'knowledge').glob('*'))
files += [ROOT/name for name in ('README.md','pyproject.toml','requirements-runtime.txt','requirements-train.txt','.env.example','deploy/pi.yaml','deploy/eeg-agent.service')]
files += list(docs.glob('*contract*.json'))
for model,item in models.items():
    directory=OUT/'models'/model/item['checkpoint_id'];files += [directory/'model.onnx',directory/'contract.json']
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        if path.is_file():archive.write(path,path.relative_to(ROOT).as_posix())
write_json(ROOT/'deploy/bundle_manifest.json',{'zip_sha256':file_hash(bundle),'zip_bytes':bundle.stat().st_size,
    'files':{p.relative_to(ROOT).as_posix():file_hash(p) for p in files if p.is_file()},
    'pi_hardware_tested':False,'no_raw_data':True,'no_credentials':True,'no_training_checkpoints':True})
print(str(REPORT/'VRMSAgent_实施与验收报告.md'));print(str(bundle))
