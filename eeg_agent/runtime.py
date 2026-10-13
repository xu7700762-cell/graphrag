import json
import threading
import time
import uuid
from pathlib import Path
import numpy as np
import psutil
from .agents import Controller
from .config import digest, contract, file_hash, write_json
from .data import CDTSource
from .signal import StreamPipeline

def aggregate(probabilities,total_windows,duration,cfg,threshold=.5):
    valid=len(probabilities); coverage=min(valid*5/max(duration,1e-9),1)
    accepted=valid>=cfg['minimum_path_windows'] and coverage>=cfg['minimum_path_coverage']
    probability=float(np.mean(probabilities)) if accepted else None
    return {'status':'accepted' if accepted else 'insufficient','probability_high':probability,
            'predicted_class':int(probability>=threshold) if accepted else None,'valid_windows':valid,'total_windows':total_windows,'coverage':coverage,'threshold':float(threshold)}

class ONNXPredictor:
    def __init__(self,cfg,model_dir=None):
        import onnxruntime as ort
        self.directory=Path(model_dir) if model_dir else Path(cfg['output_root'])/'models/baseline'
        manifest=json.loads((self.directory/'contract.json').read_text(encoding='utf-8'))
        if manifest['contract_hash']!=digest(contract(cfg)): raise ValueError('Model preprocessing contract mismatch')
        self.model_hash=file_hash(self.directory/'model.onnx')
        if self.model_hash!=manifest['onnx_sha256']: raise ValueError('ONNX hash mismatch')
        self.profiles=manifest['allowed_profiles']; self.mean=np.array(manifest['mean'],dtype=np.float32)[:,None]
        self.std=np.array(manifest['std'],dtype=np.float32)[:,None]
        options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(self.directory/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])

    def predict(self,x):
        if x.shape!=(1280,30) or not np.isfinite(x).all(): raise ValueError('Invalid model input')
        z=((x.T.astype(np.float32)-self.mean)/self.std)[None,None]
        logits=self.session.run(None,{'eeg':z})[0][0]; logits=logits-logits.max()
        probs=np.exp(logits); probability=float(probs[1]/probs.sum())
        if not np.isfinite(probability): raise ValueError('Nonfinite prediction')
        return probability

class ReplayRun:
    def __init__(self,cfg,subject=1,mode='hierarchical',backend='mock',speed=1,max_seconds=None,task='分析当前到达的EEG信号',model_dir=None):
        if speed<=0 or speed>1000: raise ValueError('speed must be in (0,1000]')
        self.cfg=cfg; self.subject=subject; self.mode=mode; self.backend=backend; self.speed=speed; self.max_seconds=max_seconds; self.task=task; self.model_dir=model_dir
        self.id=uuid.uuid4().hex[:12]; self.state='created'; self.events=[]; self.lock=threading.Lock(); self.stop_event=threading.Event(); self.thread=None
        self.directory=Path(cfg['output_root'])/'runs'/self.id; self.result=None

    def emit(self,kind,payload):
        with self.lock:
            item={'sequence':len(self.events),'kind':kind,'wall_time':time.time(),'payload':payload}
            self.events.append(item)
        if hasattr(self,'log'):
            with self.log_lock:
                self.log.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n'); self.log.flush()

    def start(self):
        self.thread=threading.Thread(target=self.execute,name='replay-'+self.id,daemon=True); self.thread.start(); return self

    def snapshot(self):
        return {'id':self.id,'state':self.state,'mode':self.mode,'backend':self.backend,'simulated_agents':self.backend=='mock','speed':self.speed,
                'event_count':len(self.events),'result':self.result,'error':getattr(self,'error',None)}

    def execute(self):
        self.directory.mkdir(parents=True,exist_ok=True); self.log_lock=threading.Lock(); self.log=(self.directory/'events.jsonl').open('w',encoding='utf-8')
        begin=time.perf_counter(); processing=[]; inference=[]; completion=[]; paths_out=[]; probs=[]; total=0; window_profiles=[]; peak=0
        controller=None
        try:
            source=CDTSource(self.cfg['data_root'],self.subject)
            try: predictor=ONNXPredictor(self.cfg,self.model_dir)
            except FileNotFoundError: predictor=None
            self.emit('model',{'available':predictor is not None,'reason':None if predictor else '未加载有效模型，预测不可用',
                               'hash':predictor.model_hash if predictor else None})
            controller=Controller(self.cfg,self.mode,self.backend,self.emit,predictor.profiles if predictor else ('P0',),self.task)
            self.state='running'; self.emit('status',{'state':self.state,'simulated_agents':self.backend=='mock','timing_kind':'real_time_1x' if self.speed==1 else 'accelerated_throughput'})
            def on_window(w):
                nonlocal total
                total+=1; q=w['quality']; cal=w['calibration']; profile=w['profile']; x=w['x']
                controller.cal=cal
                if profile=='P1': x=cal.clean(x,w['e'])
                probability=None; reasons=list(q['reasons'])
                if not np.isfinite(x).all() or np.ptp(x,axis=0).max()>self.cfg['qc']['peak_to_peak_uv']: reasons.append('processed_amplitude')
                if profile=='P1' and (not cal.available or not predictor or 'P1' not in predictor.profiles): reasons.append('unsupported_P1')
                if predictor and not reasons:
                    t=time.perf_counter(); probability=controller.vr.predict(predictor,x); inference.append(time.perf_counter()-t); probs.append(probability)
                elif not predictor: reasons.append('model_unavailable')
                feedback={'profile':profile,'post_eog_correlation':self.post_correlation(x,w['e']),
                          'post_peak_to_peak_uv':float(np.ptp(x,axis=0).max()),'post_amplitude_valid':not ('processed_amplitude' in reasons),
                          'note':'correlation reduction is not evidence of cleaning validity'}
                controller.observe(q,cal,feedback)
                window_profiles.append(profile)
                latency=time.perf_counter()-stream_begin-(w['end_sample']/1024/self.speed); completion.append(max(latency,0))
                self.emit('window',{'start_sample':w['start_sample'],'end_sample':w['end_sample'],'profile':profile,'quality':q,
                                    'calibration':cal.summary(),'probability_high':probability,'prediction_available':probability is not None,
                                    'reasons':reasons,'completion_lag_seconds':max(latency,0),'feedback':feedback})
            def on_path(p):
                nonlocal probs,total,window_profiles
                result=aggregate(probs,total,p['duration_seconds'],self.cfg)
                if not p['complete']: result.update(status='incomplete',probability_high=None,predicted_class=None)
                result.update(p); result['profiles']=window_profiles; paths_out.append(result)
                self.emit('path',result)
                if self.mode=='hierarchical': controller.executor.submit(controller.vr.report,result)
                probs=[]; total=0; window_profiles=[]
            pipeline=StreamPipeline(source.meta['channels'],self.cfg,on_window,on_path,lambda s:self.select(controller,pipeline,s))
            stream_begin=time.perf_counter()
            for start,raw,events in source.chunks(self.cfg['block_samples'],self.max_seconds):
                target=stream_begin+(start+len(raw))/1024/self.speed
                if self.stop_event.wait(max(target-time.perf_counter(),0)): break
                t=time.perf_counter(); pipeline.feed(start,raw,events); processing.append(time.perf_counter()-t)
                peak=max(peak,psutil.Process().memory_info().rss)
            pipeline.eof(); self.state='finalizing'; control_stats=controller.close()
            self.result={'run_id':self.id,'mode':self.mode,'agent_backend':self.backend,'simulated_agents':self.backend=='mock',
                         'speed':self.speed,'timing_kind':'real_time_1x' if self.speed==1 else 'accelerated_throughput',
                         'source_seconds':pipeline.last_observed/1024,'wall_seconds':time.perf_counter()-begin,'stopped':self.stop_event.is_set(),
                         'processing_seconds':sum(processing),'block_processing_p95_seconds':float(np.percentile(processing,95)) if processing else None,
                         'inference_p95_seconds':float(np.percentile(inference,95)) if inference else None,
                         'window_completion_lag_p95_seconds':float(np.percentile(completion,95)) if completion else None,
                         'peak_process_rss_bytes':peak,'source_warnings':source.meta['warnings'],'calibration':pipeline.calibration.summary(),
                         'paths':paths_out,'control':control_stats,'contract_hash':digest(contract(self.cfg)),
                         'model_hash':predictor.model_hash if predictor else None,'platform':'Windows local; not Raspberry Pi measurements'}
            write_json(self.directory/'report.json',self.result); self.state='completed'; self.emit('status',{'state':self.state})
        except Exception as exc:
            self.error=type(exc).__name__+': '+str(exc); self.state='failed'; self.emit('error',{'reason':self.error})
            write_json(self.directory/'failure.json',{'error':self.error})
            if controller:
                try: controller.close()
                except Exception: pass
        finally: self.log.close()

    @staticmethod
    def select(controller,pipeline,start):
        controller.cal=pipeline.calibration
        return controller.choose(start)

    @staticmethod
    def post_correlation(x,e):
        a=x-x.mean(axis=0); b=e-e.mean(axis=0)
        corr=a.T@b/np.maximum(np.sqrt(np.sum(a*a,axis=0))[:,None]*np.sqrt(np.sum(b*b,axis=0))[None,:],1e-12)
        return float(np.max(np.abs(corr[:9])))
