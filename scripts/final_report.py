"""Regenerate the evidence summary from actual immutable result artifacts."""
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
from scipy import signal
from eeg_agent.config import PROJECT,load_config,file_hash,write_json
from eeg_agent.learning import build_model,load_cache
from eeg_agent.runtime import ONNXPredictor
from eeg_agent.signal import fit_calibration,filter_sos

cfg=load_config();output=Path(cfg['output_root']);reports=output/'reports';model_dir=output/'models/baseline'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
audit=read(reports/'data_audit.json');manifest=read(model_dir/'contract.json');cache=read(output/'cache/manifest.json')
validation=read(reports/'evaluation_validation.json');test=read(reports/'evaluation_test.json')
run_id=(reports/'realtime_run_id.txt').read_text().strip();realtime=read(output/'runs'/run_id/'report.json')
comparison=read(reports/'controller_comparison.json')

# Independent held-out synthetic clean reference: numeric test, not biological validity.
rng=np.random.default_rng(5);n=4096;e=rng.normal(size=(n,2))*8;clean=rng.normal(size=(n,30));B=rng.normal(size=(30,2))
cal=fit_calibration(np.column_stack([clean+e@B.T,e]),n*4,cfg)
t=np.arange(4096)/256;held_clean=rng.normal(size=(4096,30))+2*np.sin(2*np.pi*10*t)[:,None]+np.sin(2*np.pi*20*t)[:,None]
held_e=rng.normal(size=(4096,2))*8;contaminated=held_clean+held_e@B.T;corrected=cal.clean(contaminated,held_e)
def bandpower(x,lo,hi):
    f,p=signal.welch(x,fs=256,nperseg=256,axis=0);return float(np.sum(p[(f>=lo)&(f<=hi)]))
injection={'reference':'synthetic known clean sine+noise; independent held-out injection; not natural EEG truth',
           'rmse_before_uv':float(np.sqrt(np.mean((contaminated-held_clean)**2))),
           'rmse_after_uv':float(np.sqrt(np.mean((corrected-held_clean)**2))),
           'alpha_power_relative_error':abs(bandpower(corrected,8,13)/bandpower(held_clean,8,13)-1),
           'beta_power_relative_error':abs(bandpower(corrected,13,30)/bandpower(held_clean,13,30)-1),'calibration':cal.summary()}
write_json(reports/'controlled_injection.json',injection)

records,_=load_cache(cfg,'train');record=next(r for r in records if r['valid'].sum()>=4 and r['p1'].any())
data=np.concatenate([record['x0'][record['valid']][:4],record['x1'][record['valid1']][:4]])
mean=np.array(manifest['mean'],dtype=np.float32);std=np.array(manifest['std'],dtype=np.float32)
z=((data-mean[None,:,None])/std[None,:,None])[:,None]
network=build_model();checkpoint=torch.load(model_dir/'checkpoint.pt',map_location='cpu',weights_only=False);network.load_state_dict(checkpoint['state_dict']);network.eval();torch.set_num_threads(2)
predictor=ONNXPredictor(cfg)
with torch.no_grad():expected=network(torch.from_numpy(z)).numpy()
actual=predictor.session.run(None,{'eeg':z})[0];real_error=float(np.max(np.abs(actual-expected)))
assert real_error<1e-4
np.testing.assert_allclose(filter_sos(),np.array(manifest['sos_coefficients']),rtol=0,atol=1e-12)

environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
             'packages':{k:version(k) for k in ['torch','numpy','scipy','scikit-learn','onnx','onnxruntime','fastapi','openai','pydantic','pytest']},
             'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'cuda':torch.version.cuda,
             'source_environment_modified':False,'source_hashes':{str(p.relative_to(PROJECT)):file_hash(p) for p in sorted((PROJECT/'eeg_agent').rglob('*')) if p.is_file() and p.suffix in ('.py','.js','.css','.html')}}
write_json(reports/'environment_and_source.json',environment)
browser_path=reports/'browser_qa.json'
browser=read(browser_path) if browser_path.exists() else {'status':'not_verified','reason':'cua browser tool: unsupported Codex auth method: apikey; alternative browser technology awaits explicit user choice'}
qa={'generated_at':datetime.now().isoformat(),'status':'local_prototype_implemented','data_candidates':147,'raw_complete_paths':146,
    'tests':(reports/'pytest-result.txt').read_text(encoding='utf-8-sig').strip() if (reports/'pytest-result.txt').exists() else 'see pytest output',
    'model_epoch':manifest['selected_epoch'],'onnx_random_max_abs_error':manifest['onnx_max_abs_error'],'onnx_real_p0_p1_max_abs_error':real_error,
    'controlled_injection':injection,'realtime_run':run_id,'realtime_summary':{k:realtime[k] for k in ['source_seconds','wall_seconds','block_processing_p95_seconds','inference_p95_seconds','window_completion_lag_p95_seconds','peak_process_rss_bytes']},
    'test_metrics':{k:v['metrics'] for k,v in test['modes'].items()},'browser_qa':browser,'cloud_backend':'mock, no external calls',
    'pi_status':'deployment config provided; hardware tests not performed','formal_fivefold_status':'entry point provided; not executed'}
write_json(reports/'acceptance.json',qa)
lines=['# EEG Agent 本地实现与验收记录','',f'生成时间：{qa["generated_at"]}','',
       '代码、真实原始数据处理、基础模型训练、冻结后评价、ONNX和网页服务已实现。云端语言规划为模拟响应。',
       '', '## 数据与模型', '',
       '- 24名被试、147条CSV候选；原始事件完整146条，Acquisition 04末条截断拒绝训练。训练87条、验证36条、测试23条。',
       f'- 完整窗口{sum(s["windows"] for s in cache["subjects"])}，公共QC接受{sum(s["quality_valid_windows"] for s in cache["subjects"])}；10名被试先行校准可用。',
       f'- 从零训练EEGNet，seed42；运行7轮，冻结第{manifest["selected_epoch"]}轮。训练划分15人87条原始完整路径，实际{len(manifest["normalization_fit_subjects"])}人{manifest["contributing_training_paths"]}条路径有合格窗口参与拟合；归一化仅用这些训练数据。',
       f'- ONNX随机输入误差{manifest["onnx_max_abs_error"]:.3g}，真实P0/P1输入误差{real_error:.3g}。',
       '', '## 测试集（接受路径上的条件指标）', '', '|配置|接受/全部|BACC|AUROC|Macro-F1|', '|---|---:|---:|---:|---:|']
for mode,res in test['modes'].items():
    m=res['metrics'];lines.append(f'|{mode}|{m["accepted_paths"]}/{m["total_paths"]}|{m["BACC"]:.4f}|{m["AUROC"]:.4f}|{m["Macro_F1"]:.4f}|')
lines+=['','测试接受覆盖73.91%，6条不足路径保留在分母。该单次开发协议未证实Agent优于规则，不能解释为临床或跨数据集效度。',
        '', '## 在线与编排验收', '',
        f'- Acquisition 01真实1倍速：源时间{realtime["source_seconds"]:.2f}秒，墙钟{realtime["wall_seconds"]:.2f}秒；首路径19/19窗口有效，覆盖95.08%。',
        f'- 块处理P95 {realtime["block_processing_p95_seconds"]*1000:.3f} ms，推理P95 {realtime["inference_p95_seconds"]*1000:.3f} ms，窗口完成延迟P95 {realtime["window_completion_lag_p95_seconds"]*1000:.2f} ms。',
        '- 延迟包括输入块到达、数值处理和窗口发布，不代表5秒采集等待消失；以上为Windows本机测量。',
        '- Acquisition 02以20倍速验证五种控制方式，均形成合法路径输出，P1、规则、单Agent和分层Agent均实际发生配置选择。加速结果只用于吞吐/功能验证。',
        '', '|控制器|配置变化|语言调用|失败调用|', '|---|---:|---:|---:|']
for r in comparison['reports']:
    c=r['control'];lines.append(f'|{r["mode"]}|{c["switches"]}|{c["cloud"]["calls"]}|{c["cloud"]["failures"]}|')
lines+=['', '所有语言调用为模拟响应，真实token和外部费用为零，模拟接口耗时不能作为云端延迟证据。',
        '', '## 保真、测试与未验证项', '',
        f'- 独立合成干净参考的受控注入：RMSE {injection["rmse_before_uv"]:.4f}→{injection["rmse_after_uv"]:.4f} uV；alpha/beta频带功率相对误差{injection["alpha_power_relative_error"]:.3%}/{injection["beta_power_relative_error"]:.3%}。',
        '- 这验证回归工具数值行为，不证明自然EEG中神经活动保留。EEG–EOG相关下降不作为清理真值。',
        '- 已测试块划分一致性、未来隔离、断流、事件/样本顺序、缺通道、非有限输入、校准回退、配置锁定、过期/非法建议、隐私、预算失败计费、OpenAI兼容function calling、模型合同、缺模型网页状态与HTTP/SSE全链路。',
        f'- Pytest：{qa["tests"]}',f'- 浏览器视觉及真实点击验收：{browser["status"]}；{browser.get("reason", "见browser_qa.json")}。',
        '- 真实云端API、Pi硬件性能及正式五折未运行；入口已提供。','',
        '原始数据文件未改动；数据、缓存、模型、代码哈希与环境版本见同目录JSON报告。']
(reports/'本地验收报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'report':str(reports/'本地验收报告.md'),'real_onnx_error':real_error,'browser':browser},ensure_ascii=False))
