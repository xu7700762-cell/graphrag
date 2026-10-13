import csv
import json
import random
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix
from .config import SPLITS, contract, digest, file_hash, write_json, cache_path, profile_config
from .data import CDTSource, paths
from .signal import StreamPipeline, filter_sos
from .runtime import aggregate
from .models import ARCHITECTURES, MODEL_IDS, ONNXPredictor, build_model as model_factory, model_directory, probabilities_from_logits, inference_contract, publication_policy

def prepare(cfg):
    audit_path=Path(cfg['output_root'])/'reports/data_audit.json'
    if not audit_path.exists(): raise ValueError('Run audit first')
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['status']=='failed': raise ValueError('Audit failed')
    label_path=Path(cfg['data_root'])/'data/labels/task_segments_with_path_scores.csv'
    if file_hash(label_path)!=audit['label_sha256']: raise ValueError('Labels changed since audit')
    cache=cache_path(cfg); cache.mkdir(parents=True,exist_ok=True)
    profile=cfg.get('classification_profile','P0')
    selected=paths(cfg); summary=[]; begin=time.perf_counter()
    for sid in sorted(set(r['subject'] for r in selected)):
        src=CDTSource(cfg['data_root'],sid)
        original=next(f for f in audit['files'] if f['subject']==sid)
        if file_hash(src.path)!=original['sha256']: raise ValueError('Raw data changed since audit')
        lookup={r['start']:r for r in selected if r['subject']==sid}
        rows=[]; matrices=[]; cleaned=[]; validity=[];validity1=[];p1_available=[];correlations=[];starts=[];ends=[];indices=[];qc_records=[];eog_records=[]
        def on_window(w):
            if w['path_start_sample'] not in lookup: return
            r=lookup[w['path_start_sample']]
            if w['end_sample']>r['end']: return
            processed=w
            if profile=='P1':
                from .eog import process_window
                processed=process_window(w,cfg,'P1');clean=processed['x']
                eog_records.append({'start_sample':w['start_sample'],'end_sample':w['end_sample'],**processed['eog_cleaning']})
            else: clean=w['calibration'].clean(w['x'],w['e'])
            matrices.append(w['x'].T.astype(np.float32));cleaned.append(clean.T.astype(np.float32))
            validity.append(w['quality']['valid']); p1_available.append(w['calibration'].available)
            validity1.append(processed['quality']['valid'] if profile=='P1' else
                             w['quality']['valid'] and np.isfinite(clean).all() and np.ptp(clean,axis=0).max()<=cfg['qc']['peak_to_peak_uv'])
            correlations.append(w['quality']['frontal_eog_correlation']); starts.append(w['start_sample']); ends.append(w['end_sample'])
            qc_records.append({'start_sample':w['start_sample'],**processed['quality']})
            indices.append(r['start'])
        def on_path(p):
            if p['start_sample'] in lookup and p['end_sample']==lookup[p['start_sample']]['end']:
                r=lookup[p['start_sample']]
                if p['complete']!=r['raw_complete']: raise ValueError('Raw completeness changed since audit')
                rows.append(r)
        pipe=StreamPipeline(src.meta['channels'],cfg,on_window,on_path)
        for start,raw,events in src.chunks(cfg['block_samples']): pipe.feed(start,raw,events)
        pipe.eof()
        if len(rows)!=len(lookup): raise ValueError('Event/label path alignment differs')
        np.savez_compressed(cache/('subject_%02d.npz'%sid),x0=np.asarray(matrices),x1=np.asarray(cleaned),valid=np.asarray(validity),valid1=np.asarray(validity1),
                            p1=np.asarray(p1_available),correlation=np.asarray(correlations),window_start=np.asarray(starts,dtype=np.int64),
                            window_end=np.asarray(ends,dtype=np.int64),path_start=np.asarray(indices,dtype=np.int64))
        entry={'subject':sid,'windows':len(matrices),'quality_valid_windows':int(np.sum(validity)),
               'classification_valid_windows':int(np.sum(validity1 if profile=='P1' else validity)),
               'paths':rows,'calibration':pipe.calibration.summary(),'raw_sha256':original['sha256'],
               'cache_sha256':file_hash(cache/('subject_%02d.npz'%sid))}
        summary.append(entry); print('prepared subject %02d: %d windows, %d valid, P1 %s'%(sid,len(matrices),sum(validity),pipe.calibration.available),flush=True)
        write_json(cache/('quality_%02d.json'%sid),qc_records)
        if profile=='P1': write_json(cache/('eog_%02d.json'%sid),eog_records)
    manifest={'schema':2,'contract_hash':digest(contract(cfg)),'label_sha256':audit['label_sha256'],'subjects':summary,
              'elapsed_seconds':time.perf_counter()-begin,'splits':SPLITS,'unit':'uV','precision':'float32','raw_copy':False,
              'classification_profile':profile,'processing_contract':contract(cfg)}
    if profile=='P1':
        manifest['processing_source_hashes']={name:file_hash(Path(__file__).parent/name) for name in ('config.py','signal.py','eog.py')}
    write_json(cache/'manifest.json',manifest); return manifest

def load_cache(cfg,split,verify=True):
    cache=cache_path(cfg); manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    if manifest['contract_hash']!=digest(contract(cfg)): raise ValueError('Cache preprocessing mismatch')
    if manifest['label_sha256']!=file_hash(Path(cfg['data_root'])/'data/labels/task_segments_with_path_scores.csv'): raise ValueError('Cache label mismatch')
    records=[]; selected={r['key']:r for r in paths(cfg)}
    for sid in SPLITS[split]:
        p=cache/('subject_%02d.npz'%sid); entry=next(x for x in manifest['subjects'] if x['subject']==sid)
        if verify and file_hash(p)!=entry['cache_sha256']: raise ValueError('Cache hash mismatch')
        with np.load(p) as z: arrays={k:z[k] for k in z.files}
        for row in entry['paths']:
            if row!=selected[row['key']]: raise ValueError('Cache path labels differ')
            take=arrays.get('path_start',arrays.get('path'))==row['start']
            record={**row,'path_start':row['start'],'path_end':row['end'],
                    **{k:v[take] for k,v in arrays.items() if k not in ('start','end','path','path_start')}}
            record['window_start']=arrays.get('window_start',arrays.get('start'))[take]
            record['window_end']=arrays.get('window_end',arrays.get('end'))[take]
            if np.any(record['window_start']<row['start']) or np.any(record['window_end']>row['end']):
                raise ValueError('Window outside path boundaries')
            records.append(record)
    return records,manifest

def trajectory(row,mode,cfg):
    n=len(row['valid']); profiles=np.zeros(n,dtype=np.int8)
    current=0; changed=-100000000; enter=0; leave=0
    starts=row['window_start'] if 'window_start' in row else row['start']
    for i in range(n):
        if mode=='fixed_p1': current=int(row['p1'][i])
        elif mode in ('rules','switch'):
            # Choose before looking at this window; decisions use previous arrived QC only.
            target=current
            if mode=='switch': target=int((i//3)%2==1 and row['p1'][i])
            elif enter>=2 and row['p1'][i]: target=1
            elif leave>=2 or not row['p1'][i]: target=0
            if target!=current and starts[i]-changed>=cfg['minimum_hold_seconds']*1024:
                current=target; changed=starts[i]
        profiles[i]=current
        corr=row['correlation'][i]
        enter=enter+1 if row['valid'][i] and corr>=cfg['qc']['eog_enter_correlation'] else 0
        leave=leave+1 if corr<=cfg['qc']['eog_exit_correlation'] else 0
    return profiles

def metrics(results,threshold=.5):
    from collections import Counter
    accepted=[r for r in results if r['status']=='accepted']
    y=np.asarray([r['label'] for r in accepted],dtype=int); p=np.asarray([r['probability_high'] for r in accepted],dtype=float)
    pred=(p>=threshold).astype(int)
    matrix=confusion_matrix(y,pred,labels=[0,1]) if len(y) else np.zeros((2,2),dtype=int)
    tn,fp,fn,tp=[int(v) for v in matrix.ravel()]
    bins=[];ece=0.
    for i in range(10):
        take=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1)); n=int(take.sum())
        mean_p=float(p[take].mean()) if n else None; fraction=float(y[take].mean()) if n else None
        if n: ece+=n/len(y)*abs(mean_p-fraction)
        bins.append({'lower':i/10,'upper':(i+1)/10,'count':n,'mean_probability':mean_p,'observed_high_fraction':fraction})
    by_class={}
    for label in (0,1):
        rows=[r for r in results if r.get('label')==label]; n=sum(r['status']=='accepted' for r in rows)
        by_class[str(label)]={'total':len(rows),'accepted':n,'rejected':len(rows)-n,'acceptance_rate':n/len(rows) if rows else None}
    return {'total_paths':len(results),'accepted_paths':len(accepted),'rejected_paths':len(results)-len(accepted),
            'path_coverage':len(accepted)/max(len(results),1),'conditional_on_accepted':True,'threshold':float(threshold),
            'rejection_reasons':dict(Counter(r['status'] for r in results if r['status']!='accepted')),'class_acceptance':by_class,
            'mean_valid_duration_coverage':float(np.mean([r.get('coverage',0.) for r in results])) if results else None,
            'BACC':float(balanced_accuracy_score(y,pred)) if len(set(y.tolist()))==2 else None,
            'AUROC':float(roc_auc_score(y,p)) if len(set(y.tolist()))==2 else None,
            'Macro_F1':float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)) if len(y) else None,
            'sensitivity':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,
            'confusion_matrix':matrix.tolist(),'Brier':float(np.mean((p-y)**2)) if len(y) else None,
            'ECE_10_equal_width':float(ece) if len(y) else None,'reliability_bins':bins}

def input_arrays(row,cfg=None):
    profile=(cfg or {}).get('classification_profile','P0')
    x=np.asarray(row['x1' if profile=='P1' else 'x0'])
    valid=np.asarray(row['valid1' if profile=='P1' else 'valid'],dtype=bool)
    if valid.shape!=(len(x),) or x.shape[1:]!=(30,1280): raise ValueError('Invalid cached model input')
    if profile=='P1':
        calibrated=np.asarray(row['p1'],dtype=bool)
        if calibrated.shape!=valid.shape or np.any(valid & ~calibrated):
            raise ValueError('Uncalibrated P1 input cannot enter classifier')
    if valid.any() and not np.isfinite(x[valid]).all(): raise ValueError('Nonfinite cached window passed quality gate')
    return x,valid


def evaluate_records(records,predict_batch,cfg,mode='fixed_p0',threshold=.5):
    profile=cfg.get('classification_profile','P0')
    if mode not in ('fixed_p0','fixed_p1') or (mode=='fixed_p1' and profile!='P1'):
        raise ValueError('P0 only for a P0 contract; classifier preprocessing must match frozen model; switching is not supported')
    results=[]
    for row in records:
        x,valid=input_arrays(row,cfg)
        p=np.asarray(predict_batch(x[valid]),dtype=float) if valid.any() and row.get('raw_complete',True) else np.array([])
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError('Invalid classification probability')
        result=aggregate(p,len(x),row['duration'],cfg)
        if result['status']=='accepted': result['predicted_class']=int(result['probability_high']>=threshold)
        if not row.get('raw_complete',True): result.update(status='raw_boundary_incomplete',probability_high=None,predicted_class=None)
        result.update(key=row['key'],subject=row['subject'],label=row['label'],mode='fixed_p1' if profile=='P1' else 'fixed_p0',threshold=float(threshold),
                      path_start=row.get('path_start',row.get('start')),path_end=row.get('path_end',row.get('end')),profile_switches=0)
        results.append(result)
    return {'metrics':metrics(results,threshold),'paths':results}


def accepted_logit_paths(records,logits_batch,cfg):
    output=[]
    for row in records:
        if not row.get('raw_complete',True): continue
        x,valid=input_arrays(row,cfg);coverage=min(int(valid.sum())*5/max(row['duration'],1e-9),1)
        if int(valid.sum())>=cfg['minimum_path_windows'] and coverage>=cfg['minimum_path_coverage']:
            output.append({'label':row['label'],'key':row['key'],'logits':np.asarray(logits_batch(x[valid]),dtype=float)})
    return output


def calibrate_temperature(logit_paths):
    from collections import Counter
    from scipy.optimize import minimize_scalar
    labels=[r['label'] for r in logit_paths]
    if len(logit_paths)<10 or set(labels)!={0,1}:
        return {'temperature':1.,'status':'unavailable','reason':'requires_at_least_10_accepted_validation_paths_and_two_classes',
                'accepted_paths':len(logit_paths),'class_counts':dict(Counter(labels))}
    def objective(log_temperature):
        t=float(np.clip(np.exp(log_temperature),.05,20))
        p=np.asarray([probabilities_from_logits(r['logits'],t).mean() for r in logit_paths]);p=np.clip(p,1e-12,1-1e-12);y=np.asarray(labels)
        return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))
    fitted=minimize_scalar(objective,bounds=(np.log(.05),np.log(20)),method='bounded',options={'xatol':1e-8})
    candidates=[0.,np.log(.05),np.log(20)]
    if fitted.success and np.isfinite(fitted.fun): candidates.append(float(fitted.x))
    best=min(candidates,key=lambda v:(objective(v),abs(v))); t=float(np.clip(np.exp(best),.05,20))
    return {'temperature':t,'status':'fitted','accepted_paths':len(logit_paths),'class_counts':dict(Counter(labels)),
            'objective':'validation accepted-path mean-probability NLL','uncalibrated_NLL':objective(0.),'calibrated_NLL':objective(best),'bounds':[.05,20]}


def select_threshold(logit_paths,temperature=1.):
    labels=[r['label'] for r in logit_paths]
    if set(labels)!={0,1}: return {'threshold':.5,'status':'unavailable','reason':'accepted_validation_paths_must_include_both_classes','search':[]}
    p=np.asarray([probabilities_from_logits(r['logits'],temperature).mean() for r in logit_paths])
    scores=[(i/100,float(balanced_accuracy_score(labels,p>=i/100))) for i in range(10,91)]
    maximum=max(v for _,v in scores);tied=[t for t,v in scores if np.isclose(v,maximum,rtol=0,atol=1e-12)]
    threshold=min(tied,key=lambda v:(round(abs(v-.5),12),v))
    return {'threshold':threshold,'status':'frozen','BACC':maximum,'tie_rule':'closest to 0.5, then smaller',
            'search':[{'threshold':t,'BACC':v} for t,v in scores]}


def training_arrays(records,cfg=None):
    contributing=[r for r in records if r.get('raw_complete',True) and input_arrays(r,cfg)[1].any()]
    counts={k:sum(r['label']==k for r in contributing) for k in (0,1)}
    if not contributing or not all(counts.values()): raise ValueError('Valid training paths must include both classes')
    xs=[];ys=[];weights=[]
    for row in contributing:
        x,valid=input_arrays(row,cfg);values=x[valid]
        if values.shape[1:]!=(30,1280) or not np.isfinite(values).all(): raise ValueError('Invalid training input')
        n=len(values);xs.append(values);ys.extend([row['label']]*n);weights.extend([1/(n*counts[row['label']])]*n)
    x=np.concatenate(xs).astype(np.float32,copy=False)
    mean=x.mean(axis=(0,2),dtype=np.float64).astype(np.float32);std=x.std(axis=(0,2),dtype=np.float64).clip(.1).astype(np.float32)
    normalized=(x-mean[None,:,None])/std[None,:,None];weight=np.asarray(weights,dtype=np.float32);weight/=weight.mean()
    return normalized,np.asarray(ys,dtype=np.int64),weight,mean,std,contributing


def build_model(model_id='eegnet'):
    return model_factory(model_id)


def train(cfg,epochs=20,model_name=None,train_records=None,val_records=None,model_id='eegnet',checkpoint_id=None):
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    if model_id not in MODEL_IDS or not 1<=epochs<=20: raise ValueError('Use EEGNet/EEGConformer and 1-20 training epochs')
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    manifest_path=cache_path(cfg)/'manifest.json'
    cache_manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    if train_records is None: train_records,cache_manifest=load_cache(cfg,'train')
    if val_records is None: val_records,_=load_cache(cfg,'validation')
    train_records=[r for r in train_records if r.get('raw_complete',True)];val_records=[r for r in val_records if r.get('raw_complete',True)]
    if set(r['subject'] for r in train_records)&set(r['subject'] for r in val_records): raise ValueError('Subject leakage')
    profile=cfg.get('classification_profile','P0')
    x,ys,w,mean,std,contributing=training_arrays(train_records,cfg)
    dataset=TensorDataset(torch.from_numpy(x[:,None]),torch.from_numpy(ys),torch.from_numpy(w))
    loader=DataLoader(dataset,batch_size=32,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42))
    chosen_id=checkpoint_id or model_name or ('eog_rls_seed42' if profile=='P1' else 'fixed_seed42');directory=model_directory(cfg,model_id,chosen_id)
    if (directory/'checkpoint.pt').exists():
        chosen_id+='_'+time.strftime('%Y%m%d_%H%M%S');directory=model_directory(cfg,model_id,chosen_id)
    directory.mkdir(parents=True,exist_ok=True)
    model=build_model(model_id);device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);criterion=torch.nn.CrossEntropyLoss(reduction='none')
    history=[];best=-1.;stale=0;begin=time.perf_counter()
    def logits_batch(data):
        outputs=[];model.eval()
        with torch.no_grad():
            for offset in range(0,len(data),64):
                z=(data[offset:offset+64].astype(np.float32)-mean[None,:,None])/std[None,:,None]
                outputs.append(model(torch.from_numpy(z[:,None]).to(device)).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0,2),dtype=np.float32)
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for xb,yb,wb in loader:
            optimizer.zero_grad();loss=(criterion(model(xb.to(device)),yb.to(device))*wb.to(device)).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
        validation=evaluate_records(val_records,lambda a:probabilities_from_logits(logits_batch(a)),cfg)
        score=validation['metrics']['BACC']
        if score is None: raise ValueError('Validation coverage does not permit BACC selection')
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation':validation['metrics'],'selection_threshold':.5,'selection_temperature':1.})
        print('%s epoch %02d loss %.4f validation %s-path BACC %.4f'%(model_id,epoch,history[-1]['train_loss'],profile,score),flush=True)
        if score>best:
            best=score;stale=0
            torch.save({'model_id':model_id,'state_dict':model.state_dict(),'mean':mean,'std':std,'epoch':epoch,'contract_hash':digest(contract(cfg))},directory/'checkpoint.pt')
        else: stale+=1
        write_json(directory/'training_history.json',history)
        if stale>=5: break
    checkpoint=torch.load(directory/'checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['state_dict']);model.eval()
    logit_paths=accepted_logit_paths(val_records,logits_batch,cfg);calibration=calibrate_temperature(logit_paths)
    threshold_selection=select_threshold(logit_paths,calibration['temperature']);temperature=calibration['temperature'];threshold=threshold_selection['threshold']
    raw_validation=evaluate_records(val_records,lambda a:probabilities_from_logits(logits_batch(a)),cfg)
    validation=evaluate_records(val_records,lambda a:probabilities_from_logits(logits_batch(a),temperature),cfg,threshold=threshold)
    manifest={'schema':2,'model_id':model_id,'checkpoint_id':chosen_id,'architecture':ARCHITECTURES[model_id],
              'contract':contract(cfg),'contract_hash':digest(contract(cfg)),'allowed_profiles':[profile],'preprocessing_profile':profile,'initialization':'random; no public or old-project weights',
              'mean':mean.tolist(),'std':std.tolist(),'threshold':threshold,'temperature':temperature,'calibration':calibration,'threshold_selection':threshold_selection,
              'assigned_training_subjects':sorted(set(r['subject'] for r in train_records)),'normalization_fit_subjects':sorted(set(r['subject'] for r in contributing)),
              'contributing_training_paths':len(contributing),'raw_complete_training_paths':len(train_records),
              'training_path_keys':[r['key'] for r in train_records],'validation_path_keys':[r['key'] for r in val_records],
              'validation_subjects':sorted(set(r['subject'] for r in val_records)),
              'actual_subject_splits':{'train':sorted(set(r['subject'] for r in train_records)),
                                       'validation':sorted(set(r['subject'] for r in val_records))},
              'checkpoint_sha256':file_hash(directory/'checkpoint.pt'),
              'selected_epoch':checkpoint['epoch'],'seed':42,'device':device,'parameter_count':sum(p.numel() for p in model.parameters()),
              'training':{'optimizer':'Adam','learning_rate':.001,'batch_size':32,'maximum_epochs':epochs,'patience':5,
                          'selection':'validation '+profile+' path BACC at raw probability threshold 0.5','weighting':'equal valid path contribution within class; equal class total'},
              'validation':validation['metrics'],'validation_raw_0_5':raw_validation['metrics'],'training_seconds':time.perf_counter()-begin,
              'software':{'torch':torch.__version__,'numpy':np.__version__},'cache_manifest_hash':digest(cache_manifest) if cache_manifest else None,
              'window_label_note':'weak path labels; windows are not instantaneous symptom ground truth; temperature is fitted at path level',
              'evaluation_boundary':'development dataset; not untouched independent testing','pi_hardware_status':'not_tested'}
    write_json(directory/'contract.json',manifest)
    real=np.concatenate([input_arrays(r,cfg)[0][input_arrays(r,cfg)[1]][:2] for r in val_records if input_arrays(r,cfg)[1].any()])[:4]
    export_onnx(cfg,directory,model_id=model_id,real_windows=real)
    return json.loads((directory/'contract.json').read_text(encoding='utf-8'))


def export_onnx(cfg,directory=None,model_id='eegnet',checkpoint_id=None,real_windows=None):
    import torch
    import onnx
    import onnxruntime as ort
    directory=Path(directory) if directory else model_directory(cfg,model_id,checkpoint_id)
    checkpoint=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    if checkpoint.get('model_id',model_id)!=model_id or checkpoint['contract_hash']!=digest(contract(cfg)): raise ValueError('Checkpoint identity/preprocessing mismatch')
    manifest=json.loads((directory/'contract.json').read_text(encoding='utf-8'));model=build_model(model_id);model.load_state_dict(checkpoint['state_dict']);model.eval()
    if real_windows is None:
        records,_=load_cache(cfg,'validation');available=[input_arrays(r,cfg)[0][input_arrays(r,cfg)[1]][:2] for r in records if r.get('raw_complete',True) and input_arrays(r,cfg)[1].any()]
        if not available: raise ValueError('ONNX export requires real quality-valid P0 windows')
        real_windows=np.concatenate(available)[:4]
    real_windows=np.asarray(real_windows,dtype=np.float32)
    if real_windows.ndim!=3 or real_windows.shape[1:]!=(30,1280) or not len(real_windows) or not np.isfinite(real_windows).all(): raise ValueError('Invalid real P0 export validation input')
    sample=torch.randn(3,1,30,1280,generator=torch.Generator().manual_seed(42))
    torch.onnx.export(model,sample,directory/'model.onnx',input_names=['eeg'],output_names=['logits'],dynamic_axes={'eeg':{0:'batch'},'logits':{0:'batch'}},opset_version=17,dynamo=False)
    onnx.checker.check_model(str(directory/'model.onnx'));options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(directory/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    mean=np.asarray(manifest['mean'],dtype=np.float32);std=np.asarray(manifest['std'],dtype=np.float32);normalized=(real_windows-mean[None,:,None])/std[None,:,None]
    cases={'synthetic_normalized':sample.numpy(),'real_'+cfg.get('classification_profile','P0')+'_training_normalized':normalized[:,None]};errors={}
    for name,values in cases.items():
        with torch.no_grad(): reference=model(torch.from_numpy(values)).numpy()
        actual=session.run(None,{'eeg':values})[0];error=float(np.max(np.abs(reference-actual)))
        raw_error=float(np.max(np.abs(probabilities_from_logits(reference)-probabilities_from_logits(actual))))
        calibrated_error=float(np.max(np.abs(probabilities_from_logits(reference,manifest['temperature'])-probabilities_from_logits(actual,manifest['temperature']))))
        errors[name]={'windows':len(values),'logits_max_abs_error':error,'raw_probability_max_abs_error':raw_error,'calibrated_probability_max_abs_error':calibrated_error}
        if not np.isfinite([error,raw_error,calibrated_error]).all() or max(error,raw_error,calibrated_error)>1e-4: raise ValueError('ONNX numerical mismatch: '+name)
    error=max(v[k] for v in errors.values() for k in ('logits_max_abs_error','raw_probability_max_abs_error','calibrated_probability_max_abs_error'))
    manifest.update(onnx_sha256=file_hash(directory/'model.onnx'),onnx_max_abs_error=error,onnx_validation={'status':'passed','tolerance':1e-4,'cases':errors},
                    sos_coefficients=filter_sos().tolist(),onnx_file_bytes=(directory/'model.onnx').stat().st_size)
    manifest.update(model_id=model_id,checkpoint_id=directory.name,architecture=ARCHITECTURES[model_id],
                    contract=contract(cfg),publication_policy=publication_policy(cfg))
    manifest['inference_contract_hash']=digest(inference_contract(manifest))
    manifest['software'].update(onnx=onnx.__version__,onnxruntime=ort.__version__);write_json(directory/'contract.json',manifest)
    return {'max_abs_error':error,'sha256':manifest['onnx_sha256'],'cases':errors,'directory':str(directory)}


def evaluate(cfg,split='test',model_dir=None,model_id='eegnet',checkpoint_id=None):
    predictor=ONNXPredictor(cfg,model_dir,model_id,checkpoint_id);cfg=profile_config(cfg,predictor.profile);records,cache=load_cache(cfg,split);trained=predictor.manifest
    if split=='test' and set(trained['normalization_fit_subjects'])&set(SPLITS['test']): raise ValueError('Test subjects used in normalization')
    begin=time.perf_counter();result=evaluate_records(records,predictor.predict_batch,cfg,threshold=predictor.threshold)
    raw=evaluate_records(records,lambda a:probabilities_from_logits(predictor.logits_batch(a)),cfg);timings=[]
    real=next((input_arrays(r,cfg)[0][input_arrays(r,cfg)[1]][0] for r in records if input_arrays(r,cfg)[1].any()),None)
    if real is not None:
        predictor.predict(real.T)
        for _ in range(20):
            start=time.perf_counter();predictor.predict(real.T);timings.append(time.perf_counter()-start)
    result.update(split=split,model_id=model_id,checkpoint_id=predictor.directory.name,model_hash=predictor.model_hash,contract_hash=cache['contract_hash'],
                  threshold=predictor.threshold,temperature=predictor.temperature,raw_probability_0_5=raw,parameter_count=trained['parameter_count'],model_file_bytes=trained['onnx_file_bytes'],
                  elapsed_seconds=time.perf_counter()-begin,inference_p50_seconds=float(np.percentile(timings,50)) if timings else None,
                  inference_p95_seconds=float(np.percentile(timings,95)) if timings else None,
                  note='conditional accepted-path development results; path labels do not validate instantaneous symptoms; Windows figures are not Pi measurements')
    write_json(Path(cfg['output_root'])/'reports'/('evaluation_%s_%s_%s.json'%(model_id,predictor.directory.name,split)),result);return result


def fivefold(cfg,epochs=20,model_id='eegnet'):
    from sklearn.model_selection import StratifiedGroupKFold
    all_records=[]
    for split in SPLITS: all_records.extend(load_cache(cfg,split)[0])
    all_records=[r for r in all_records if r.get('raw_complete',True)]
    y=np.asarray([r['label'] for r in all_records]);groups=np.asarray([r['subject'] for r in all_records]);dummy=np.zeros(len(y));folds=[]
    report_path=Path(cfg['output_root'])/('reports/fivefold_%s.json'%model_id)
    for fold,(development,test_ix) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(dummy,y,groups)):
        inner_train,inner_val=next(StratifiedGroupKFold(4,shuffle=True,random_state=43).split(dummy[development],y[development],groups[development]))
        tr=[all_records[i] for i in development[inner_train]];va=[all_records[i] for i in development[inner_val]]
        trained=train(cfg,epochs,'fivefold_%d'%fold,tr,va,model_id=model_id)
        predictor=ONNXPredictor(cfg,model_id=model_id,checkpoint_id=trained['checkpoint_id']);test_records=[all_records[i] for i in test_ix]
        fold_result=evaluate_records(test_records,predictor.predict_batch,cfg,threshold=predictor.threshold)
        folds.append({'fold':fold,'train_subjects':sorted(set(r['subject'] for r in tr)),'validation_subjects':sorted(set(r['subject'] for r in va)),
                      'test_subjects':sorted(set(groups[test_ix].tolist())),'threshold':predictor.threshold,'temperature':predictor.temperature,'result':fold_result})
        write_json(report_path,{'folds':folds,'status':'running'})
    pooled_paths=[r for fold in folds for r in fold['result']['paths']];pooled=metrics(pooled_paths);accepted=[r for r in pooled_paths if r['status']=='accepted']
    if accepted:
        truth=[r['label'] for r in accepted];predicted=[r['predicted_class'] for r in accepted];matrix=confusion_matrix(truth,predicted,labels=[0,1]);tn,fp,fn,tp=[int(v) for v in matrix.ravel()]
        pooled.update(BACC=float(balanced_accuracy_score(truth,predicted)) if len(set(truth))==2 else None,
                      Macro_F1=float(f1_score(truth,predicted,labels=[0,1],average='macro',zero_division=0)),confusion_matrix=matrix.tolist(),
                      sensitivity=tp/(tp+fn) if tp+fn else None,specificity=tn/(tn+fp) if tn+fp else None,threshold='per-fold frozen validation threshold')
    result={'model_id':model_id,'folds':folds,'pooled_OOF':pooled,'status':'complete','boundary':'prior development on this dataset; internal cross-validation, not untouched independent testing'}
    write_json(report_path,result);return result
