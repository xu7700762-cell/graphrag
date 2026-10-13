"""Shared arrived-only task service for the CLI and HTTP console."""
import json
import platform
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil

from .agents import CloudProvider
from .config import contract, digest, write_json, profile_config
from .data import CDTSource
from .models import ONNXPredictor
from .runtime import aggregate
from .signal import StreamPipeline
from .tools import BaselineCollector, TemporalAnalyzer, biomarkers, covariance, analysis_policy, TOOL_CATALOG
from .knowledge import KnowledgeIndex
from .eog import process_window

TERMINAL=('completed','failed','stopped','interrupted')

def _scalar_fields(value, keys):
    if not isinstance(value, dict): return {}
    return {k:value[k] for k in keys if k in value and
            (value[k] is None or isinstance(value[k], (str, bool, int, float)))}

def _band_summary(value, keys=('delta','theta','alpha','beta','gamma_low')):
    if not isinstance(value, dict): return {}
    return {k:value[k] for k in keys if k in value and
            (value[k] is None or (isinstance(value[k], (int,float)) and
                                 not isinstance(value[k], bool) and np.isfinite(value[k])))}

def _cloud_auxiliary_summary(result):
    """The same scalar-only contract is used for evidence and cloud tool replies."""
    result=result if isinstance(result,dict) else {}
    summary=_scalar_fields(result, ('status','reason'))
    bio=result.get('biomarkers')
    if isinstance(bio,dict):
        safe=_scalar_fields(bio, ('tool','available','version','unit','baseline_available','reason'))
        regions=bio.get('regions') or {}
        safe['regions']={name:{field:_band_summary(region.get(field),
                           ('theta_alpha','theta_beta','slow_fast') if field=='ratios' else
                           ('delta','theta','alpha','beta','gamma_low'))
                           for field in ('absolute_power_uv2','relative_power','ratios') if field in region}
                         for name,region in regions.items() if name in ('all','frontal','occipital') and isinstance(region,dict)}
        change=bio.get('baseline_change')
        changes=change.get('regions',change) if isinstance(change,dict) else None
        safe['baseline_change']=({name:_band_summary(values) for name,values in changes.items()
                                 if name in ('all','frontal','occipital')} if isinstance(changes,dict) else None)
        summary['biomarkers']=safe
    cov=result.get('covariance')
    if isinstance(cov,dict):
        summary['covariance']=_scalar_fields(cov, ('tool','version','available','unit','distance_unit',
            'shrinkage','trace_uv2','minimum_eigenvalue','maximum_eigenvalue','baseline_available',
            'baseline_windows','airm_distance','z_score','reason','interpretation',
            'source_start_sample','source_end_sample','profile'))
    return summary

def _cloud_eog_summary(value):
    """Only scalar cleaning evidence leaves the local numerical service."""
    summary=_scalar_fields(value, ('applied','input_available','measurement_fallback','requested_profile','applied_profile','status','reason',
        'pre_frontal_eog_correlation','post_frontal_eog_correlation','pre_peak_to_peak_uv',
        'post_peak_to_peak_uv','removed_rms_ratio','removed_rms_uv','processing_version','method',
        'parameters_frozen','coefficients_frozen'))
    if isinstance(value,dict):
        summary['calibration']=_scalar_fields(value.get('calibration'), ('available','reason','samples',
            'cutoff_sample','condition','version','method','seconds','heldout_seconds','usable_channels',
            'heldout_error_ratio','coefficient_max_abs','processing_version'))
        summary['adaptive']=_scalar_fields(value.get('adaptive'), ('available','reason','samples',
            'updates','coefficient_version','version','method','forgetting_seconds','ridge_fraction',
            'condition','coefficient_max_abs','parameters_frozen','coefficients_frozen','skipped_updates',
            'prediction_before_update','labels_used','initialization_cutoff_sample'))
    return summary

def _cloud_failure_reason(exc):
    # Provider error bodies may contain sensitive request details; never persist them.
    known=('Invalid explanation','Invalid citation','Missing citation','Numeric explanation must use local rendering',
           'Classification explanation must use local rendering','Unsupported symptom claim','Unregistered tool',
           'Invalid routing','Invalid task type','Illegal tool call','Tool step budget exceeded')
    return ('ValueError: '+str(exc) if type(exc) is ValueError and str(exc) in known else type(exc).__name__)

def atomic_json(path,value):
    path=Path(path); temporary=path.with_suffix('.tmp')
    write_json(temporary,value); temporary.replace(path)

class MonitorTask:
    def __init__(self,cfg,request,knowledge=None):
        self.cfg=cfg; self.request=dict(request); self.id=uuid.uuid4().hex[:12]
        self.state='created'; self.events=[]; self.lock=threading.RLock(); self.stop_event=threading.Event()
        self.directory=Path(cfg['output_root'])/'tasks'/self.id; self.directory.mkdir(parents=True,exist_ok=True)
        self.latest=None; self.result=None; self.error=None; self.plan=None; self.messages=[]
        self.paths=[]; self.baseline=BaselineCollector(cfg); self.last_x=None; self.last_window=None
        self.knowledge=knowledge or KnowledgeIndex(cfg)
        self.source_sample=int(request.get('start_sample',0)); self.tool_counts={}; self.tool_seconds={}; self.early_exits=0
        self.source=CDTSource(cfg['data_root'],request.get('subject',1))
        if not 0<=self.source_sample<len(self.source.raw): raise ValueError('Invalid start_sample')
        predictor_cfg=cfg
        if not request.get('checkpoint_id') and cfg.get('prefer_eog_models'):
            candidate=Path(cfg['output_root'])/'models'/request.get('model_id','eegnet')/'eog_rls_seed42'/'model.onnx'
            if candidate.exists(): predictor_cfg=profile_config(cfg,'P1')
        try:
            self.predictor=ONNXPredictor(predictor_cfg,model_id=request.get('model_id','eegnet'),checkpoint_id=request.get('checkpoint_id'))
        except FileNotFoundError: self.predictor=None
        self.preprocessing_profile=(self.predictor.profiles[0] if self.predictor else cfg.get('classification_profile','P0'))
        self.cfg=profile_config(cfg,self.preprocessing_profile)
        self.contract_hash=digest(contract(self.cfg)); self.baseline_profile='P0'
        self._prior_baseline_windows=[]; self.eog_calibration=None
        self.eog_counts={'total_windows':0,'cleaned_windows':0,'quality_rejected_windows':0,'input_unavailable_windows':0,
                         'calibration_unavailable_windows':0,'measurement_fallback_windows':0}
        self.provider=CloudProvider(request.get('backend','mock'),cfg,self.emit)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='vrms-cloud')
        self.cloud_future=None; self.last_review=-1e30; self.language=[]; self.references={}; self.pending_full=False
        self.thread=None; self.context_version=1

    def snapshot(self):
        with self.lock:
            return {'id':self.id,'state':self.state,'request':self.request,'event_count':len(self.events),
                    'latest':self.latest,'plan':self.plan,'model':self.model_info(),'result':self.result,'error':self.error,
                    'source_sample':self.source_sample,'simulated_agents':self.request.get('backend','mock')=='mock'}

    def model_info(self):
        p=self.predictor
        return {'model_id':self.request.get('model_id','eegnet'),'available':p is not None,
                'model_hash':p.model_hash if p else None,'threshold':p.threshold if p else None,
                'inference_contract_hash':p.manifest.get('inference_contract_hash') if p else None,
                'checkpoint_id':p.directory.name if p else None,
                'temperature':p.temperature if p else None,
                'preprocessing_profile':self.preprocessing_profile,
                'reason':None if p else '未加载有效模型，预测不可用'}

    def baseline_summary(self):
        return {**self.baseline.summary(),'profile':self.baseline_profile}

    def _observe_baseline(self,w):
        self.baseline.observe(w['x'],w['quality'],w['start_sample']/1024,w['end_sample']/1024)
        # These at most twelve already-arrived windows are held only until calibration freezes.
        if self.preprocessing_profile=='P1' and len(self._prior_baseline_windows)<12 and w['quality']['valid']:
            self._prior_baseline_windows.append({**w,'x':w['x'].copy(),'e':w['e'].copy()})

    def _freeze_calibration(self,c):
        self.eog_calibration=_cloud_eog_summary({'calibration':c.summary()})['calibration']
        cutoff=c.cutoff_sample/1024
        if self.preprocessing_profile=='P1' and c.available:
            self.baseline=BaselineCollector(self.cfg); self.baseline_profile='P1'
            for w in self._prior_baseline_windows:
                if w['end_sample']>c.cutoff_sample: continue
                prepared=process_window({**w,'calibration':c,'calibration_reference':True,
                                         'observed_end_sample':c.cutoff_sample},self.cfg,'P1')
                self.baseline.observe(prepared['x'],prepared['quality'],w['start_sample']/1024,w['end_sample']/1024)
        self._prior_baseline_windows.clear()
        self.baseline.freeze(cutoff)
        self.emit('eog_calibration',{'requested_profile':self.preprocessing_profile,
                  'calibration':self.eog_calibration,'baseline':self.baseline_summary()})

    def _prepare_window(self,w):
        prepared=process_window(w,self.cfg,self.preprocessing_profile)
        cleaning=prepared['eog_cleaning']; self.eog_counts['total_windows']+=1
        if self.preprocessing_profile=='P1' and not cleaning.get('applied',False):
            # Keep measuring the arrived stream, while refusing the P1-only classifier input.
            fallback=process_window(w,self.cfg,'P0')
            prepared={**fallback,'eog_cleaning':{**cleaning,'applied_profile':'P0','measurement_fallback':True}}
            cleaning=prepared['eog_cleaning']
        if cleaning.get('applied'): self.eog_counts['cleaned_windows']+=1
        if not prepared['quality']['valid']: self.eog_counts['quality_rejected_windows']+=1
        if self.preprocessing_profile=='P1' and not cleaning.get('input_available',cleaning.get('applied',False)):
            self.eog_counts['input_unavailable_windows']+=1
            if not cleaning.get('calibration',{}).get('available'):
                self.eog_counts['calibration_unavailable_windows']+=1
        if cleaning.get('measurement_fallback'): self.eog_counts['measurement_fallback_windows']+=1
        prepared['profile']=cleaning.get('applied_profile',self.preprocessing_profile)
        return prepared

    def emit(self,kind,payload):
        with self.lock:
            item={'sequence':len(self.events),'kind':kind,'wall_time':time.time(),'payload':payload}
            self.events.append(item)
            with (self.directory/'events.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
            if kind in ('window','path','plan','status','language_report','message','report'):
                atomic_json(self.directory/'context.json',self.snapshot())
        return item

    def start(self):
        self.thread=threading.Thread(target=self.execute,daemon=True,name='vrms-'+self.id)
        self.thread.start(); return self

    def request_stop(self): self.stop_event.set()
    def report(self): return self.result

    def evidence(self):
        # Construct an allowlist: identities, labels, file references and arrays never enter cloud input.
        with self.lock:
            latest=self.latest or {}
            keys=('quality','deep','temporal','biomarkers','covariance','coverage','reasons','prediction_available')
            arrived={k:latest[k] for k in keys if k in latest}
            arrived.update(_cloud_auxiliary_summary(arrived))
            arrived['eog_cleaning']=_cloud_eog_summary(latest.get('eog_cleaning'))
            deep=arrived.get('deep')
            if deep: arrived['deep']={k:v for k,v in deep.items() if k!='logits'}
            trend=arrived.get('temporal')
            if trend: arrived['temporal']={k:v for k,v in trend.items() if k!='history'}
            paths=[{k:p[k] for k in ('status','probability_high','predicted_class','valid_windows','total_windows','coverage','threshold') if k in p} for p in self.paths]
            return {'arrived':arrived,'paths':paths,'baseline':self.baseline_summary(),'model':self.model_info(),
                    'source_time_seconds':self.source_sample/1024,'contract_hash':self.contract_hash}

    def request_auxiliary(self,tools,stamp):
        with self.lock:
            w=self.last_window
            if self.stop_event.is_set() or self.state in ('failed','stopped','interrupted'): return {'status':'rejected','reason':'task_ended'}
            if not w or not w['quality']['valid'] or self.last_x is None: return {'status':'rejected','reason':'quality_or_history'}
            if stamp!=w['end_sample'] or not 0<=self.source_sample-stamp<=self.cfg['proposal_ttl_seconds']*1024:
                return {'status':'rejected','reason':'expired_or_future'}
            if any(t not in ('BiomarkerCalculator','CovarianceAnalyzer') for t in tools): return {'status':'rejected','reason':'unregistered_tool'}
            result={}
            for name in tools:
                field='biomarkers' if name=='BiomarkerCalculator' else 'covariance'
                if self.latest.get(field) is None:
                    self.latest[field]=self.run_tool(name,self.last_x,w,['user_or_agent_request'])
                result[field]=self.latest[field]
            self.emit('auxiliary',{'source_end_sample':stamp,**result})
            return {'status':'success',**result}

    def run_tool(self,name,x,w,reasons):
        begin=time.perf_counter()
        try:
            output=(biomarkers if name=='BiomarkerCalculator' else covariance)(x,self.baseline)
        except (ValueError,np.linalg.LinAlgError) as exc:
            output={'available':False,'reason':type(exc).__name__}
        elapsed=time.perf_counter()-begin
        output.update(source_start_sample=w['start_sample'],source_end_sample=w['end_sample'],profile=w.get('profile','P0'))
        with self.lock:
            self.tool_counts[name]=self.tool_counts.get(name,0)+1
            self.tool_seconds[name]=self.tool_seconds.get(name,0)+elapsed
        self.emit('tool_call',{'tool':name,'version':'2.0','source_start_sample':w['start_sample'],
                             'source_end_sample':w['end_sample'],'status':'success' if output.get('available',True) else 'insufficient',
                             'reasons':reasons,'seconds':elapsed})
        return output

    def message(self,text):
        if not isinstance(text,str) or not text.strip() or len(text)>2000: raise ValueError('Message must contain 1-2000 characters')
        self.messages.append({'text':text,'wall_time':time.time()}); self.emit('message',{'text':text,'status':'accepted'})
        requested=[]
        if any(word in text.lower() for word in ('频谱','功率','完整分析','全面评估','spectrum')): requested.append('BiomarkerCalculator')
        if any(word in text.lower() for word in ('协方差','完整分析','全面评估','covariance')): requested.append('CovarianceAnalyzer')
        if requested: self.request_auxiliary(requested,self.last_window['end_sample'] if self.last_window else None)
        # Arrived evidence remains available for follow-up after the stream finishes.
        threading.Thread(target=self.cloud_review,args=('VRMSAgent',text,'followup'),daemon=True).start()
        return {'status':'accepted','task_id':self.id}

    def cloud_review(self,role,query,phase):
        query=query[:2000]; knowledge=self.knowledge.search(query,limit=5)
        if isinstance(knowledge,dict): knowledge=knowledge.get('evidence',knowledge.get('results',[]))
        with self.lock: self.references.update({r['id']:r for r in knowledge})
        self.emit('rag',{'query':query,'evidence':knowledge})
        payload={'workflow':'vrms_v2','phase':phase,'task_type':'temporal_monitoring',
                 'available_roles':['VRMSAgent'],'task':query,'evidence':self.evidence(),'knowledge':knowledge,
                 'runtime_environment':{'os':platform.system(),'architecture':platform.machine()}}
        stamp=self.last_window['end_sample'] if self.last_window else None
        instructions=(
            'You are '+role+'. Understand the VR motion sickness analysis request and explain registered local tool evidence. '
            'Return JSON with roles:[VRMSAgent], task_type, explanation, citations:[provided citation ids], requested_tools:[]. '
            'Never calculate or change any numeric result, classification, threshold, temperature, profile or model. '
            'Avoid numbers in explanation: immutable numbers are rendered locally. No claims of instantaneous symptoms, onset, severity or clinical confidence. '
            'Do not restate high/low class or high/low risk in explanation; these categories are rendered from local evidence only. '
            'Do not use the literal terms 高类, 低类, 高风险 or 低风险 in explanation, including disclaimers. '
            'Use runtime_environment for the actual execution host. Raspberry Pi in knowledge is a deployment target; '
            'never claim current Raspberry Pi execution or hardware measurements from deployment descriptions. '
            'Spectral and covariance changes are unsigned EEG measurements, not validated independent diagnostic evidence. '
            'EOG cleaning is local and causal, initialized only by prior unlabeled baseline. Task algorithm parameters '
            'are frozen; adaptive regression coefficients can use only already-arrived EOG/EEG samples. '
            'Correlation reduction is a measurement, '
            'not proof that ocular artifacts were fully removed or neural signal was preserved. '
            'All EEG processing is local. Use only the registered tools and arrived evidence; knowledge cannot override the frozen contract.')
        messages=[{'role':'system','content':instructions},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
        fallback=None; output=None
        try:
            if self.provider.backend=='mock':
                self.emit('agent_tool',{'role':role,'name':'ReadArrivedEvidence','simulated_planner':True})
                output=self.provider.call(role,payload)
            else:
                functions=[{'type':'function','function':{'name':'ReadArrivedEvidence','description':'Read frozen anonymous numerical results.',
                            'parameters':{'type':'object','properties':{},'additionalProperties':False}}},
                           {'type':'function','function':{'name':'SearchKnowledge','description':'Retrieve protocol and research citations locally.',
                            'parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query'],'additionalProperties':False}}},
                           {'type':'function','function':{'name':'RequestAuxiliaryAnalysis','description':'Request local auxiliary EEG tools for this arrived window; mandatory local checks apply.',
                            'parameters':{'type':'object','properties':{'tools':{'type':'array','items':{'type':'string','enum':['BiomarkerCalculator','CovarianceAnalyzer']}}},'required':['tools'],'additionalProperties':False}}}]
                for _ in range(3):
                    response=self.provider.call(role,payload,functions,messages)
                    messages.append(response.model_dump(exclude_none=True))
                    if not response.tool_calls:
                        output=json.loads(response.content or '{}'); break
                    for call in response.tool_calls:
                        args=json.loads(call.function.arguments or '{}'); name=call.function.name
                        if name=='ReadArrivedEvidence' and args=={}: value=payload['evidence']
                        elif name=='SearchKnowledge' and set(args)=={'query'} and isinstance(args['query'],str):
                            value=self.knowledge.search(args['query'][:500],limit=5)
                            items=value.get('results',value.get('evidence',[])) if isinstance(value,dict) else value
                            knowledge.extend(r for r in items if r['id'] not in {k['id'] for k in knowledge})
                            with self.lock: self.references.update({r['id']:r for r in items})
                        elif name=='RequestAuxiliaryAnalysis' and set(args)=={'tools'} and isinstance(args['tools'],list):
                            value=_cloud_auxiliary_summary(self.request_auxiliary(args['tools'],stamp))
                        else: raise ValueError('Illegal tool call')
                        self.emit('agent_tool',{'role':role,'name':name,'status':value.get('status','success') if isinstance(value,dict) else 'success'})
                        messages.append({'role':'tool','tool_call_id':call.id,'content':json.dumps(value,ensure_ascii=False)})
                if output is None: raise ValueError('Tool step budget exceeded')
            if self.state in TERMINAL and phase!='followup':
                self.emit('agent_reply_rejected',{'role':role,'reason':'task_closed','phase':phase})
                return {'status':'rejected','reason':'task_closed'}
            if not isinstance(output,dict) or not isinstance(output.get('explanation'),str): raise ValueError('Invalid explanation')
            ids={r['id'] for r in knowledge}; citations=output.get('citations',[])
            if not isinstance(citations,list) or any(i not in ids for i in citations): raise ValueError('Invalid citation')
            if knowledge and not citations: raise ValueError('Missing citation')
            text=output['explanation'][:2000]
            if re.search(r'\d',text): raise ValueError('Numeric explanation must use local rendering')
            if any(term in text for term in ('高类','低类','高风险','低风险','置信度高','严重眩晕')): raise ValueError('Classification explanation must use local rendering')
            if any(term in text for term in ('严重度为','症状已确诊','真实onset','确诊眩晕')): raise ValueError('Unsupported symptom claim')
            requested=output.get('requested_tools',[])
            if not isinstance(requested,list) or any(t not in ('BiomarkerCalculator','CovarianceAnalyzer') for t in requested): raise ValueError('Unregistered tool')
            if requested: self.request_auxiliary(requested,stamp)
            if phase=='planning':
                if output.get('roles')!=['VRMSAgent']: raise ValueError('Invalid routing')
                if output.get('task_type') not in ('quick_detection','temporal_monitoring','explanation','full_assessment'): raise ValueError('Invalid task type')
                self.pending_full=output.get('task_type')=='full_assessment'
                self.plan={'roles':['VRMSAgent'],'strategy':self.request.get('strategy','adaptive'),
                           'task_type':output.get('task_type','temporal_monitoring'),
                           'steps':['质量检查','所选模型检测','时间监测','按本地规则补充EEG测量','RAG证据解释'],
                           'simulated':self.provider.backend=='mock'}
                self.emit('plan',self.plan)
        except Exception as exc:
            fallback=_cloud_failure_reason(exc)
            text='语言接口或证据校验未完成，采用本地确定性报告。模型输出为路径高低类估计，辅助指标只描述脑电变化。'
            citations=[]
        entry={'role':role,'explanation':text,'citations':citations,'fallback_reason':fallback,
               'phase':phase,'source_end_sample':stamp,'simulated':self.provider.backend=='mock'}
        if self.state in TERMINAL and phase!='followup':
            self.emit('agent_reply_rejected',{'role':role,'reason':'task_closed','phase':phase})
            return entry
        with self.lock: self.language.append(entry)
        self.emit('language_report',entry)
        return entry

    def execute(self):
        begin=time.perf_counter(); cpu_begin=psutil.Process().cpu_times(); processing=[]; inference=[]; latency=[]; peak=0
        probs=[]; total=0; path_id=None; temporal=TemporalAnalyzer(self.predictor.threshold if self.predictor else .5)
        self.state='running'; self.emit('model',self.model_info()); self.emit('status',{'state':self.state})
        self.plan={'roles':['VRMSAgent'],'strategy':self.request.get('strategy','adaptive'),'task_type':'temporal_monitoring',
                   'steps':['质量检查','所选模型检测','时间监测','按需频谱及协方差','证据报告'],'simulated':self.provider.backend=='mock'}
        self.emit('plan',self.plan)
        self.cloud_future=self.executor.submit(self.cloud_review,'Supervisor',self.request.get('task','分析VR晕动症路径高低类'),'planning')
        stream_begin=time.perf_counter(); source_origin=self.source_sample; speed=self.request.get('speed',1)
        try:
            def on_baseline(w): self._observe_baseline(w)
            def on_gap(sample,reason):
                temporal.reset();self.emit('gap',{'source_sample':sample,'reason':reason})
            def on_window(w):
                nonlocal total,path_id
                prepare_begin=time.perf_counter(); w=self._prepare_window(w)
                cleaning=w['eog_cleaning']; cleaning_seconds=time.perf_counter()-prepare_begin
                if w['path_start_sample']!=path_id:
                    temporal.reset();path_id=w['path_start_sample']
                self.baseline.freeze(w['path_start_sample']/1024)
                total+=1; deep=None; probability=None; reasons=list(w['quality']['reasons']); t=time.perf_counter()
                cleaning_input=(self.preprocessing_profile!='P1' or cleaning.get('input_available',cleaning.get('applied',False)))
                if not cleaning_input: reasons.append('eog_calibration_unavailable' if not cleaning.get('calibration',{}).get('available') else 'eog_cleaning_input_rejected')
                if self.predictor and not reasons:
                    deep=self.predictor.predict_details(w['x']);probability=deep['probability_high'];probs.append(probability)
                    deep.update(tool='DeepVRMSDetector',version=self.predictor.manifest['architecture']['version'],unit='probability',
                        available=True,source_start_sample=w['start_sample'],source_end_sample=w['end_sample'])
                    inference.append(time.perf_counter()-t)
                elif not self.predictor: reasons.append('model_unavailable')
                self.tool_counts['EOGArtifactCleaner']=self.tool_counts.get('EOGArtifactCleaner',0)+1
                self.tool_seconds['EOGArtifactCleaner']=self.tool_seconds.get('EOGArtifactCleaner',0)+cleaning_seconds
                self.emit('tool_call',{'tool':'EOGArtifactCleaner','version':cleaning.get('processing_version','1.0'),
                    'method':cleaning.get('method'),
                    'source_start_sample':w['start_sample'],'source_end_sample':w['end_sample'],
                    'status':cleaning.get('status'),'reasons':cleaning.get('reasons',[]),'seconds':cleaning_seconds,
                    'requested_profile':self.preprocessing_profile,'applied_profile':w['profile']})
                trend=temporal.observe(w['start_sample']/1024,w['end_sample']/1024,probability,path_id)
                for tool in ('SignalQualityChecker','DeepVRMSDetector','TemporalAnalyzer'):
                    self.tool_counts[tool]=self.tool_counts.get(tool,0)+1
                    self.emit('tool_call',{'tool':tool,'version':'2.0','source_start_sample':w['start_sample'],
                        'source_end_sample':w['end_sample'],'status':'success' if (not reasons or tool=='SignalQualityChecker') else 'insufficient','reasons':reasons})
                full_requested=self.pending_full or any(s in self.request.get('task','') for s in ('完整分析','全面评估'))
                policy=analysis_policy(self.request.get('strategy','adaptive'),deep,trend,full_requested=full_requested)
                self.pending_full=False
                auxiliary={'biomarkers':None,'covariance':None}
                if w['quality']['valid'] and (probability is not None or full_requested or self.request.get('strategy')=='full_evidence'):
                    if probability is None:
                        policy={'tools':['BiomarkerCalculator','CovarianceAnalyzer'],
                                'reasons':['measurement_without_classifier'],'early_exit':False}
                    for tool in policy.get('tools',[]):
                        auxiliary['biomarkers' if tool=='BiomarkerCalculator' else 'covariance']=self.run_tool(tool,w['x'],w,policy.get('reasons',[]))
                    if not policy.get('tools'): self.early_exits+=1
                lag=max(time.perf_counter()-stream_begin-(w['end_sample']-source_origin)/1024/speed,0);latency.append(lag)
                duration=(w['end_sample']-w['path_start_sample'])/1024
                current={'start_sample':w['start_sample'],'end_sample':w['end_sample'],'profile':w['profile'],'quality':w['quality'],
                             'requested_profile':self.preprocessing_profile,'eog_cleaning':cleaning,
                             'eog_counts':dict(self.eog_counts),'eog_calibration':self.eog_calibration,
                             'calibration':self.baseline_summary(),'probability_high':probability,'raw_probability':deep.get('raw_probability') if deep else None,
                             'threshold':self.predictor.threshold if self.predictor else None,'prediction_available':probability is not None,
                             'reasons':reasons,'completion_lag_seconds':lag,'deep':deep,'temporal':trend,'policy':policy,
                             'coverage':min(len(probs)*5/max(duration,1e-9),1),**auxiliary}
                with self.lock:
                    self.source_sample=max(self.source_sample,w['end_sample'])
                    self.latest=current;self.last_x=w['x'].copy();self.last_window=w
                    self.emit('window',current)
                for event in trend.get('events',[]): self.emit('temporal_event',event)
                now=w['end_sample']/1024
                if now-self.last_review>=self.cfg['agent_review_seconds'] and (self.cloud_future is None or self.cloud_future.done()):
                    self.last_review=now;self.cloud_future=self.executor.submit(self.cloud_review,'VRMSAgent',self.request.get('task','分析当前EEG趋势'),'review')
            def on_path(p):
                nonlocal probs,total,path_id
                if not self.baseline.frozen: self.baseline.freeze(p['start_sample']/1024)
                result=aggregate(probs,total,p['duration_seconds'],self.cfg,self.predictor.threshold if self.predictor else .5)
                if not p['complete']: result.update(status='incomplete',probability_high=None,predicted_class=None)
                result.update(p);self.paths.append(result);self.emit('path',result)
                probs=[];total=0;path_id=None;temporal.reset()
            pipeline=StreamPipeline(self.source.meta['channels'],self.cfg,on_window,on_path,
                on_baseline=on_baseline,on_gap=on_gap,on_calibration=self._freeze_calibration)
            for start,raw,events in self.source.chunks(self.cfg['block_samples'],self.request.get('max_seconds'),source_origin):
                target=stream_begin+(start+len(raw)-source_origin)/1024/speed
                if self.stop_event.wait(max(target-time.perf_counter(),0)): break
                t=time.perf_counter();pipeline.feed(start,raw,events);processing.append(time.perf_counter()-t)
                self.source_sample=start+len(raw);peak=max(peak,psutil.Process().memory_info().rss)
                # Freeze even if a task has not produced its first complete window.
                if pipeline.first_task and not self.baseline.summary().get('frozen',False):
                    self.baseline.freeze((pipeline.active or self.source_sample)/1024)
            pipeline.eof()
            self.state='finalizing';self.emit('status',{'state':self.state})
            if not self.stop_event.is_set():
                final=threading.Thread(target=self.cloud_review,args=('Supervisor','汇总已到达工具证据及任务报告','final'),daemon=True)
                final.start();deadline=time.perf_counter()+3*self.cfg['api_timeout_seconds']+1
                while (final.is_alive() or (self.cloud_future is not None and not self.cloud_future.done())) and time.perf_counter()<deadline:
                    if self.stop_event.wait(.05): break
            cpu_end=psutil.Process().cpu_times()
            self.result={'task_id':self.id,'model':self.model_info(),'request':self.request,'paths':self.paths,'baseline':self.baseline_summary(),
                         'eog_calibration':self.eog_calibration,'eog_counts':dict(self.eog_counts),
                         'contract_hash':self.contract_hash,'source_start_sample':source_origin,'source_end_sample':self.source_sample,
                         'source_seconds':(self.source_sample-source_origin)/1024,'wall_seconds':time.perf_counter()-begin,
                         'timing_kind':'real_time_1x' if speed==1 else 'accelerated_throughput','platform':platform.platform(),
                         'processing_seconds':sum(processing),'inference_p95_seconds':float(np.percentile(inference,95)) if inference else None,
                         'window_completion_lag_p95_seconds':float(np.percentile(latency,95)) if latency else None,
                         'peak_process_rss_bytes':peak,'process_cpu_seconds':cpu_end.user+cpu_end.system-cpu_begin.user-cpu_begin.system,
                         'tool_counts':dict(self.tool_counts),'tool_seconds':dict(self.tool_seconds),'early_exits':self.early_exits,
                         'cloud':self.provider.summary(),'language_reports':list(self.language),'references':list(self.references.values()),'source_warnings':self.source.meta['warnings'],
                         'stopped':self.stop_event.is_set(),'boundary':'路径评分分类；辅助EEG测量不改变类别；mock不是实际云端收益；Windows不是Pi实测'}
            self.result['markdown']=self.render_report()
            write_json(self.directory/'report.json',self.result);(self.directory/'report.md').write_text(self.result['markdown'],encoding='utf-8')
            self.state='stopped' if self.stop_event.is_set() else 'completed'
            self.emit('report',{'markdown':self.result['markdown']});self.emit('status',{'state':self.state})
        except Exception as exc:
            self.error=type(exc).__name__+': '+str(exc);self.state='failed';self.emit('error',{'reason':self.error})
        finally:
            self.executor.shutdown(wait=False,cancel_futures=True)

    def render_report(self):
        with self.lock: return self._render_report()

    def _render_report(self):
        lines=['# VRMSAgent EEG分析报告','',f"模型：{self.request.get('model_id','eegnet')}；策略：{self.request.get('strategy','adaptive')}；语言接口：{self.request.get('backend','mock')}。",
               '', '以下为路径评分高低类估计，5秒输出不是即时症状真值。频谱与协方差描述EEG变化，不修改分类结果。','',
               '| 路径 | 状态 | 高类概率 | 类别 | 有效窗 | 覆盖 |','|---|---|---|---|---|---|']
        for i,p in enumerate(self.paths,1):
            prob='不可用' if p['probability_high'] is None else f"{p['probability_high']:.4f}"
            label='不足' if p['predicted_class'] is None else ('高类' if p['predicted_class'] else '低类')
            lines.append(f"| {i} | {p['status']} | {prob} | {label} | {p['valid_windows']} | {p['coverage']:.1%} |")
        lines+=['',f"输入处理配置：{self.preprocessing_profile}；眼电校准：{json.dumps(self.eog_calibration,ensure_ascii=False)}。",
                f"眼电处理计数：{json.dumps(self.eog_counts,ensure_ascii=False)}。",
                '眼电回归使用首任务前已到达的无标签基线初始化；任务冻结算法及参数，自适应系数仅随已到达样本更新。校准失败时保留P0质量测量，拒绝P1模型分类输入。相关下降不能单独证明去伪迹有效或脑电保真。',
                '',f"先行基线：{json.dumps(self.baseline_summary(),ensure_ascii=False)}",'',
                f"辅助工具调用：{json.dumps(self.tool_counts,ensure_ascii=False)}；Early Exit：{self.early_exits}。",'',
                f"云端调用：{self.provider.summary()['calls']}；模拟状态：{self.provider.backend=='mock'}。"]
        for entry in self.language:
            lines+=['',entry['role']+'：'+entry['explanation']]
            if entry['citations']: lines.append('引用：'+', '.join(entry['citations']))
            if entry['fallback_reason']: lines.append('模板回退原因：'+entry['fallback_reason'])
        if self.references:
            lines+=['','## 本地知识引用','']
            for identity,reference in self.references.items():
                lines.append(f"- {identity}：{reference['source']} / {reference['section']}；版本 {reference['version']}；SHA256 {reference.get('source_sha256','')}。")
        if self.latest:
            bio=self.latest.get('biomarkers')
            concise_bio=None
            if bio:
                concise_bio={k:bio[k] for k in ('available','unit','regions','baseline_available','reason') if k in bio}
                change=bio.get('baseline_change');concise_bio['baseline_change']=change.get('regions') if change else None
            lines+=['','最后到达窗口的辅助测量：', '```json',json.dumps({k:self.latest.get(k) for k in ('temporal','biomarkers','covariance')},ensure_ascii=False,indent=2),'```']
            lines[-2]=json.dumps({'eog_cleaning':_cloud_eog_summary(self.latest.get('eog_cleaning')),
                'temporal':self.latest.get('temporal'),'biomarkers':concise_bio,'covariance':self.latest.get('covariance')},ensure_ascii=False,indent=2)
        lines+=['','当前数据没有逐时症状标签；真实onset、提前量、严重度与临床置信度未验证。Pi硬件性能需单独实测。']
        return '\n'.join(lines)+'\n'

class RestoredTask:
    def __init__(self,directory,context):
        self.directory=directory;self.context=context;self.id=context['id'];self.lock=threading.RLock()
        self.state=context['state'] if context['state'] in TERMINAL else 'interrupted'
        self.context['state']=self.state;self.events=[]
        file=directory/'events.jsonl'
        if file.exists():
            for line in file.read_text(encoding='utf-8').splitlines():
                try:self.events.append(json.loads(line))
                except json.JSONDecodeError:break
        atomic_json(directory/'context.json',self.context)
    def snapshot(self): return {**self.context,'event_count':len(self.events)}
    def request_stop(self): return None
    def report(self):
        p=self.directory/'report.json'
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    def message(self,text):
        report=self.report()
        return {'status':'restored_evidence','answer':report['markdown'] if report else '任务已中断；历史证据已恢复，请创建新会话继续回放。'}

class TaskManager:
    def __init__(self,cfg):
        self.cfg=cfg;self.tasks={};self.lock=threading.Lock();self.knowledge=KnowledgeIndex(cfg)
        root=Path(cfg['output_root'])/'tasks'
        for path in root.glob('*/context.json'):
            try:
                context=json.loads(path.read_text(encoding='utf-8'));self.tasks[context['id']]=RestoredTask(path.parent,context)
            except (ValueError,OSError): continue
    def create(self,request):
        request={'subject':1,'model_id':'eegnet','strategy':'adaptive','backend':'mock','speed':1,'task':'分析VR晕动症路径高低类',**request}
        if request['model_id'] not in ('eegnet','eegconformer'): raise ValueError('Unknown model_id')
        if request['strategy'] not in ('deep_only','adaptive','full_evidence'): raise ValueError('Unknown strategy')
        if not 0<request['speed']<=1000: raise ValueError('Invalid replay speed')
        with self.lock:
            if any(t.state not in TERMINAL for t in self.tasks.values()): raise RuntimeError('已有任务正在运行')
            task=MonitorTask(self.cfg,request,self.knowledge);self.tasks[task.id]=task;task.start()
            return task.snapshot()
    def list(self): return [t.snapshot() for t in reversed(list(self.tasks.values()))]
    def get(self,task_id):
        if task_id not in self.tasks: raise KeyError(task_id)
        return self.tasks[task_id]
