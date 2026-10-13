import numpy as np
from eeg_agent.config import load_config, EEG_CHANNELS
from eeg_agent.signal import CausalFilter, StreamPipeline

CHANNELS=EEG_CHANNELS+['M1','M2','VEOG','HEOG','ECG','EGG','Trigger']

def filtered(raw,sizes):
    cfg=load_config(); cfg['warmup_seconds']=0
    f=CausalFilter(CHANNELS,cfg); output=[]; valid=[]; times=[]; start=0; k=0
    while start<len(raw):
        end=min(len(raw),start+sizes[k%len(sizes)])
        x,t,v,*_=f.process(start,raw[start:end]);output.append(x);valid.append(v);times.append(t)
        start=end;k+=1
    return np.concatenate(output),np.concatenate(times),np.concatenate(valid)

def test_nonfinite_partition_and_past_prefix():
    raw=np.random.default_rng(42).normal(size=(16000,37))
    clean,t,_=filtered(raw,[256]);raw[12013,3]=np.nan;raw[12301,31]=np.inf
    a,ta,va=filtered(raw,[256]);b,tb,vb=filtered(raw,[17,513,127])
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ta,tb);np.testing.assert_array_equal(va,vb)
    np.testing.assert_array_equal(a[ta<12013],clean[t<12013])

def test_window_waits_for_all_raw_samples():
    cfg=load_config();cfg['warmup_seconds']=0;windows=[]
    p=StreamPipeline(CHANNELS,cfg,windows.append,lambda _:None)
    raw=np.random.default_rng(4).normal(size=(5120,37))
    p.feed(0,raw[:5117],[(0,20)]);assert not windows
    p.feed(5117,raw[5117:]);assert len(windows)==1
    assert windows[0]['start_sample']==0 and windows[0]['end_sample']==5120

def test_nonfinite_discards_only_interrupted_window():
    cfg=load_config();cfg['warmup_seconds']=0
    raw=np.random.default_rng(4).normal(size=(16000,37));raw[6001,0]=np.nan
    def process(sizes):
        windows=[];p=StreamPipeline(CHANNELS,cfg,windows.append,lambda _:None)
        start=0;k=0
        while start<len(raw):
            end=min(len(raw),start+sizes[k%len(sizes)])
            p.feed(start,raw[start:end],[(0,20)] if start==0 else [])
            start=end;k+=1
        return windows
    a=process([256]);b=process([17,513,127])
    assert [(w['start_sample'],w['end_sample']) for w in a]==[(0,5120),(6004,11124)]
    for wa,wb in zip(a,b): np.testing.assert_array_equal(wa['x'],wb['x'])

def test_raw_and_processed_quality_measurements():
    cfg=load_config();cfg['warmup_seconds']=0;windows=[]
    raw=np.random.default_rng(4).normal(size=(5120,37))
    raw[:,33]=raw[:,0] # raw HEOG is strongly correlated with Fp1
    p=StreamPipeline(CHANNELS,cfg,windows.append,lambda _:None)
    p.feed(0,raw,[(0,20)])
    q=windows[0]['quality']
    assert set(q['raw_metrics'])==set(q['processed_metrics'])
    assert q['raw_metrics']['frontal_eog_correlation']>.99
