"""Monitoring must use one frozen cleaning contract for detector and evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eeg_agent.config import EEG_CHANNELS, load_config
from eeg_agent.monitor import MonitorTask, _cloud_eog_summary
from eeg_agent.signal import Calibration


class SyntheticSource:
    def __init__(self, *_):
        self.meta={'channels':EEG_CHANNELS+['M1','M2','VEOG','HEOG'],'warnings':[]}
        t=np.arange(36*1024)/1024
        e=np.column_stack([8*np.sin(2*np.pi*1.3*t),6*np.sin(2*np.pi*2.1*t)])
        eeg=np.column_stack([6*np.sin(2*np.pi*(10+j/40)*t)+2*np.sin(2*np.pi*18*t+.1*j)
                             for j in range(30)])
        mixing=np.tile([.45,.2],(30,1))
        self.raw=np.column_stack([eeg+e@mixing.T,np.zeros((len(t),2)),e])
        self.events=[(20*1024,20),(35*1024,22)]

    def chunks(self,block,max_seconds=None,start_sample=0):
        end=min(len(self.raw),int(max_seconds*1024)) if max_seconds else len(self.raw)
        for start in range(start_sample,end,block):
            stop=min(start+block,end)
            yield start,self.raw[start:stop],[(s,m) for s,m in self.events if start<=s<stop]


def make_task(monkeypatch,tmp_path,profile='P0',strategy='deep_only'):
    import eeg_agent.monitor as monitor
    seen=[]
    predictor=SimpleNamespace(profiles=[profile],threshold=.5,temperature=1.,model_hash='fake-model',
        manifest={'architecture':{'version':'fake'}},directory=Path('synthetic-'+profile))
    def predict(x):
        seen.append(x.copy())
        return {'probability_high':.2,'raw_probability':.2,'logits':[1.,0.], 'threshold':.5}
    predictor.predict_details=predict
    monkeypatch.setattr(monitor,'CDTSource',SyntheticSource)
    monkeypatch.setattr(monitor,'ONNXPredictor',lambda *a,**k:predictor)
    cfg=load_config(); cfg['output_root']=str(tmp_path)
    task=MonitorTask(cfg,{'subject':1,'model_id':'eegnet','checkpoint_id':'fake',
        'strategy':strategy,'backend':'mock','speed':1000,'max_seconds':36})
    return task,seen


def test_p0_monitor_preserves_baseline_and_bypasses_eog(monkeypatch,tmp_path):
    task,seen=make_task(monkeypatch,tmp_path)
    task.execute()
    assert task.state=='completed',task.error
    assert len(seen)==3
    assert task.baseline_summary()['windows']==2 and task.baseline_summary()['profile']=='P0'
    assert not task.latest['eog_cleaning']['applied']
    assert task.latest['profile']=='P0'
    assert task.eog_counts['cleaned_windows']==0
    np.testing.assert_array_equal(seen[-1],task.last_window['x'])


def test_p1_monitor_detector_and_auxiliary_share_cleaned_windows(monkeypatch,tmp_path):
    task,seen=make_task(monkeypatch,tmp_path,'P1','full_evidence')
    auxiliary=[]
    def tool(name,x,w,reasons):
        auxiliary.append((name,x.copy(),w['profile']))
        return {'available':True,'profile':w['profile']}
    task.run_tool=tool
    task.execute()
    assert task.state=='completed',task.error
    assert len(seen)==3 and len(auxiliary)==6
    assert task.baseline_summary()['available'] and task.baseline_summary()['profile']=='P1'
    assert task.latest['eog_cleaning']['applied'] and task.latest['prediction_available']
    assert task.eog_counts['cleaned_windows']==3
    assert task.latest['eog_cleaning']['method']=='causal_reference_rls'
    assert task.latest['eog_cleaning']['parameters_frozen']
    assert not task.latest['eog_cleaning']['coefficients_frozen']
    clean_logs=[e['payload'] for e in task.events if e['kind']=='tool_call' and e['payload']['tool']=='EOGArtifactCleaner']
    assert len(clean_logs)==3 and all(e['version']=='causal-eog-rls-v1' for e in clean_logs)
    assert all(e['method']=='causal_reference_rls' for e in clean_logs)
    for i,x in enumerate(seen):
        for _,measured,profile in auxiliary[2*i:2*i+2]:
            np.testing.assert_array_equal(x,measured)
            assert profile=='P1'
    assert not task._prior_baseline_windows
    assert '眼电处理计数' in task.result['markdown']


def test_failed_eog_calibration_keeps_measurements_but_blocks_p1_model(monkeypatch,tmp_path):
    import eeg_agent.eog as eog
    monkeypatch.setattr(eog,'fit_eog_calibration',lambda values,cutoff,cfg:
        Calibration(reason='controlled insufficient baseline',cutoff_sample=cutoff))
    task,seen=make_task(monkeypatch,tmp_path,'P1','full_evidence')
    measured=[]
    task.run_tool=lambda name,x,w,reasons: measured.append(w['profile']) or {'available':True}
    task.execute()
    assert task.state=='completed',task.error
    assert not seen and len(measured)==6 and set(measured)=={'P0'}
    assert task.latest['quality']['valid'] and not task.latest['prediction_available']
    assert task.latest['eog_cleaning']['measurement_fallback']
    assert not task.latest['eog_cleaning']['input_available']
    assert 'eog_calibration_unavailable' in task.latest['reasons']
    assert task.baseline_summary()['profile']=='P0'
    assert task.eog_counts['measurement_fallback_windows']==3
    assert task.eog_counts['calibration_unavailable_windows']==3
    assert task.eog_counts['input_unavailable_windows']==3
    assert task.result['model']['preprocessing_profile']=='P1'


def test_cleaning_rejection_cannot_become_valid_p0_measurement(monkeypatch,tmp_path):
    import eeg_agent.monitor as monitor
    task,_=make_task(monkeypatch,tmp_path,'P1')
    x=np.ones((1280,30)); calls=[]
    def rejected(w,cfg,profile):
        calls.append(profile)
        return {**w,'x':x,'profile':'P1','quality':{'valid':False,'reasons':['residual_eog_correlation']},
            'eog_cleaning':{'applied':True,'applied_profile':'P1','input_available':False,
                'requested_profile':'P1','calibration':{'available':True}}}
    monkeypatch.setattr(monitor,'process_window',rejected)
    w=task._prepare_window({'x':x,'e':np.ones((1280,2)),'quality':{'valid':True}})
    assert calls==['P1'] and not w['quality']['valid']
    assert not w['eog_cleaning'].get('measurement_fallback')
    assert task.eog_counts['calibration_unavailable_windows']==0
    assert task.eog_counts['input_unavailable_windows']==1
    assert task.eog_counts['quality_rejected_windows']==1
    task.executor.shutdown(wait=False,cancel_futures=True)


def test_cloud_cleaning_evidence_excludes_coefficients_and_signal_arrays():
    secret='NOT_FOR_CLOUD'
    value={'applied':True,'pre_frontal_eog_correlation':.8,'post_frontal_eog_correlation':.2,
        'removed_rms_ratio':.3,'raw':[secret],'x':[secret],
        'calibration':{'available':True,'samples':2560,'B':[[secret]],'mean':[secret],'score':90},
        'adaptive':{'updates':1280,'skipped_updates':12,'B':[[secret]],'mean':[secret]}}
    rendered=json.dumps(_cloud_eog_summary(value))
    assert secret not in rendered and 'score' not in rendered
    assert json.loads(rendered)['calibration']['samples']==2560
    assert json.loads(rendered)['adaptive']=={'updates':1280,'skipped_updates':12}


@pytest.mark.parametrize('artifact,prefer,checkpoint,expected_profile',[
    (False,True,None,'P0'),(True,True,None,'P1'),
    (True,False,None,'P0'),(True,True,'fixed_seed42','P0')])
def test_default_model_selects_p1_only_with_exported_artifact(monkeypatch,tmp_path,artifact,prefer,checkpoint,expected_profile):
    import eeg_agent.monitor as monitor
    cfg=load_config();cfg['output_root']=str(tmp_path);cfg['prefer_eog_models']=prefer
    if artifact:
        candidate=tmp_path/'models'/'eegnet'/'eog_rls_seed42'/'model.onnx'
        candidate.parent.mkdir(parents=True);candidate.write_bytes(b'test-marker')
    received=[]
    def predictor(model_cfg,**kwargs):
        received.append((model_cfg.get('classification_profile','P0'),kwargs.get('checkpoint_id')))
        return SimpleNamespace(profiles=[received[-1][0]])
    monkeypatch.setattr(monitor,'ONNXPredictor',predictor)
    monkeypatch.setattr(monitor,'CDTSource',SyntheticSource)
    task=MonitorTask(cfg,{'subject':1,'model_id':'eegnet','checkpoint_id':checkpoint,'backend':'mock'})
    assert received==[(expected_profile,checkpoint)]
    assert task.preprocessing_profile==expected_profile
    task.executor.shutdown(wait=False,cancel_futures=True)
