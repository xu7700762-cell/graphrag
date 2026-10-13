import json
import numpy as np
from fastapi.testclient import TestClient
from eeg_agent.config import load_config,digest,contract,SPLITS
from eeg_agent.agents import Controller,anonymous_summary,CloudProvider,SignalQualityAgent
from eeg_agent.signal import Calibration
from eeg_agent.runtime import aggregate
from eeg_agent.learning import trajectory,metrics
from eeg_agent.service import create_app

def qc(end=100):
    return {'observed_end_sample':end,'unit':'uV','valid':True,'reasons':[],'frontal_eog_correlation':.8,
            'peak_to_peak_uv':20.,'processed_peak_to_peak_uv':15.,'flat_channel_fraction':0.,'jump_fraction':0.,
            'line_power_ratio':.01,'line_warning':False,'eog_std_uv':[5.,6.],'subject_id':'NEVER_SEND','label':1,'score':99}

def test_proposal_contract_expiry_and_calibration():
    cfg=load_config();c=Controller(cfg,'rules','mock',lambda *x:None,('P0','P1'));c.cal=Calibration(available=True);c.latest=qc()
    p={'profile':'P1','observed_end_sample':100,'contract_hash':c.hash}
    assert c.validate(p,101) is None
    assert c.validate(p,20000)=='expired_or_future' and c.validate(p,99)=='expired_or_future'
    assert c.validate({**p,'profile':'ICA'},101)=='unsupported_profile'
    assert c.validate({**p,'contract_hash':'bad'},101)=='contract_version'
    c.cal=Calibration();assert c.validate(p,101)=='calibration_or_quality';c.close()

def test_privacy_and_mock_tools():
    cfg=load_config();cal=Calibration(available=True,cutoff_sample=80)
    payload=anonymous_summary(qc(),cal,digest(contract(cfg)),'P0');serialized=json.dumps(payload)
    assert all(k not in serialized for k in ['NEVER_SEND','subject_id','"label"','"score"'])
    logged=[];provider=CloudProvider('mock',cfg,lambda kind,p:logged.append((kind,p)))
    assert SignalQualityAgent(provider).plan(payload)['profile']=='P1' and provider.summary()['simulated']
    assert len([e for e in logged if e[0]=='agent_tool'])==4

def test_hysteresis_and_hold():
    cfg=load_config();c=Controller(cfg,'rules','mock',lambda *x:None,('P0','P1'));cal=Calibration(available=True)
    c.observe(qc(5120),cal);assert c.choose(5120)=='P0'
    c.observe(qc(10240),cal);assert c.choose(10240)=='P1'
    low=qc(15360);low['frontal_eog_correlation']=.1;c.observe(low,cal);assert c.choose(15360)=='P1'
    low['observed_end_sample']=20480;c.observe(low,cal);assert c.choose(20480)=='P1'
    low['observed_end_sample']=25600;c.observe(low,cal);assert c.choose(25600)=='P0';c.close()

def test_coverage_and_threshold():
    cfg=load_config();assert aggregate([.6]*2,2,10,cfg)['status']=='insufficient'
    assert aggregate([.6]*3,4,20,cfg)['status']=='insufficient'
    assert aggregate([.5]*4,4,20,cfg)['predicted_class']==1
    assert aggregate([],4,20,cfg)['probability_high'] is None

def test_trajectory_past_only():
    cfg=load_config();r={'valid':np.ones(5,bool),'p1':np.ones(5,bool),'start':np.arange(5)*5120,'correlation':np.ones(5)*.8}
    assert trajectory(r,'rules',cfg).tolist()==[0,0,1,1,1]
    changed={**r,'correlation':np.array([.8,.8,.01,.01,.01])}
    assert np.array_equal(trajectory(r,'rules',cfg)[:3],trajectory(changed,'rules',cfg)[:3])

def test_splits_and_rejection_denominator():
    assert all(not set(SPLITS[a])&set(SPLITS[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
    m=metrics([{'status':'insufficient'},{'status':'accepted','label':0,'probability_high':.1},{'status':'accepted','label':1,'probability_high':.9}])
    assert m['total_paths']==3 and m['accepted_paths']==2 and m['BACC']==1

def test_web_validation_and_missing_model(tmp_path):
    cfg=load_config();cfg['output_root']=str(tmp_path);client=TestClient(create_app(cfg))
    assert client.get('/').status_code==200
    assert not client.get('/api/config').json()['model']['available']
    assert client.post('/api/runs',json={'subject':99}).status_code==422
    assert client.post('/api/runs',json={'speed':0}).status_code==422
    assert client.get('/api/runs/missing').status_code==404
    assert client.get('/api/health').json()['status']=='ok'
