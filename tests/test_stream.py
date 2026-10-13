import numpy as np
import pytest
from eeg_agent.config import load_config,EEG_CHANNELS
from eeg_agent.signal import CausalFilter,StreamPipeline,fit_calibration,quality

CHANNELS=EEG_CHANNELS+['M1','M2','VEOG','HEOG','ECG','EGG','Trigger']
def data(n,seed=42): return np.random.default_rng(seed).normal(size=(n,37))
def chunks_filter(raw,sizes):
    f=CausalFilter(CHANNELS,load_config());out=[];times=[];start=0;i=0
    while start<len(raw):
        end=min(start+sizes[i%len(sizes)],len(raw));x,t,*_=f.process(start,raw[start:end]);out.append(x);times.append(t);start=end;i+=1
    return np.concatenate(out),np.concatenate(times)

def test_block_partition_and_future_independence():
    raw=data(18000);a,ta=chunks_filter(raw,[256]);b,tb=chunks_filter(raw,[17,128,513,255])
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ta,tb)
    altered=raw.copy();altered[12000:]*=50;c,_=chunks_filter(altered,[256])
    np.testing.assert_array_equal(a[ta<12000],c[ta<12000])

def test_event_visibility_and_missing_channels():
    cfg=load_config();p=StreamPipeline(CHANNELS,cfg,lambda w:None,lambda p:None)
    with pytest.raises(ValueError,match='not yet arrived'):p.feed(0,data(256),[(512,20)])
    with pytest.raises(ValueError,match='Missing'):CausalFilter(CHANNELS[1:],cfg)

def test_gap_discards_partial_window():
    cfg=load_config();cfg['warmup_seconds']=0;windows=[]
    p=StreamPipeline(CHANNELS,cfg,windows.append,lambda p:None)
    p.feed(0,data(2048),[(0,20)]);p.feed(4096,data(3072),[])
    assert not windows
    p.feed(7168,data(2048),[])
    assert len(windows)==1 and windows[0]['start_sample']==4096

def test_window_profile_locked_and_incomplete_eof():
    cfg=load_config();cfg['warmup_seconds']=0;windows=[];paths=[];selected=[]
    def choose(s):selected.append(s);return 'P0' if len(selected)==1 else 'P1'
    p=StreamPipeline(CHANNELS,cfg,windows.append,paths.append,choose)
    for s in range(0,12288,256):p.feed(s,data(256,seed=s),[(0,20)] if s==0 else [])
    assert [w['profile'] for w in windows]==['P0','P1']
    assert selected==[0,5120,10240] and not paths
    p.eof();assert paths[0]['complete'] is False
    assert all(w['end_sample']<=12288 for w in windows)

def test_calibration_fallback_and_controlled_injection():
    cfg=load_config();c=fit_calibration(np.zeros((100,32)),10000,cfg);assert not c.available
    rng=np.random.default_rng(5);n=4096;e=rng.normal(size=(n,2))*8
    clean=rng.normal(size=(n,30));B=rng.normal(size=(30,2));noisy=clean+e@B.T
    c=fit_calibration(np.column_stack([noisy,e]),n*4,cfg);assert c.available
    held_e=rng.normal(size=(2048,2))*8;held_clean=rng.normal(size=(2048,30));held_noisy=held_clean+held_e@B.T
    corrected=c.clean(held_noisy,held_e)
    assert np.mean((corrected-held_clean)**2)<np.mean((held_noisy-held_clean)**2)*.02
    assert abs(np.var(corrected)/np.var(held_clean)-1)<.1

def test_nonfinite_does_not_poison_filter():
    cfg=load_config();f=CausalFilter(CHANNELS,cfg);raw=data(256);raw[10,0]=np.nan
    out,_,valid,_,gap=f.process(0,raw)
    assert gap and not valid.any() and np.isfinite(out).all()

def test_line_noise_is_separate_from_saturation():
    cfg=load_config();t=np.arange(5120)/1024;rng=np.random.default_rng(1)
    raw=rng.normal(size=(5120,30))+600*np.sin(2*np.pi*50*t)[:,None]
    x=rng.normal(size=(1280,30))*5;e=rng.normal(size=(1280,2))
    q=quality(x,e,raw,cfg,5120)
    assert q['line_warning'] and q['valid']
    shifted=raw+20000;assert quality(x,e,shifted,cfg,5120)['valid']
    raw[100,2]=6000;assert 'raw_range_exceeded' in quality(x,e,raw,cfg,5120)['reasons']
    raw[:,2]=0;assert 'flat_required_channel' in quality(x,e,raw,cfg,5120)['reasons']
