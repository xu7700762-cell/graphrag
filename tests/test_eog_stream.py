"""Replay partition, exact-end snapshots and post-gap EOG state."""
import numpy as np
from eeg_agent.config import EEG_CHANNELS,load_config,profile_config
from eeg_agent.signal import StreamPipeline
from eeg_agent.eog import process_window

CHANNELS=EEG_CHANNELS+['M1','M2','VEOG','HEOG']

def data():
    t=np.arange(55*1024)/1024
    e=np.column_stack([20*np.sin(2*np.pi*1.3*t),16*np.sin(2*np.pi*2.1*t)])
    brain=np.column_stack([6*np.sin(2*np.pi*(10+j/40)*t)+2*np.sin(2*np.pi*18*t+.1*j) for j in range(30)])
    return np.column_stack([brain+e@np.tile([.45,.2],(30,1)).T,np.zeros((len(t),2)),e])

def replay(raw,sizes):
    cfg=profile_config(load_config(),'P1');windows=[];events=[(20*1024+3,20)]
    p=StreamPipeline(CHANNELS,cfg,lambda w:windows.append(process_window(w,cfg)),lambda _:None)
    start=0;k=0
    while start<len(raw):
        end=min(start+sizes[k%len(sizes)],len(raw))
        p.feed(start,raw[start:end],[(s,m) for s,m in events if start<=s<end])
        start=end;k+=1
    p.eof();return windows

def same(a,b):
    assert [(w['start_sample'],w['end_sample']) for w in a]==[(w['start_sample'],w['end_sample']) for w in b]
    for x,y in zip(a,b):
        np.testing.assert_array_equal(x['x'],y['x'])
        assert x['quality']==y['quality']
        assert x['eog_cleaning']==y['eog_cleaning']

def test_rls_no_downsample_raw_chunks_jump_mask_and_exact_window_snapshot():
    raw=data();raw[27*1024+1,0]+=4000 # jump between decimation instants
    a=replay(raw,[256]);b=replay(raw,[1,3,17,777])
    same(a,b)
    assert any(w['eog_cleaning'].get('adaptive',{}).get('skipped_updates',0)>0 for w in a)

def test_rls_future_isolation_and_nonfinite_gap_reset():
    raw=data();altered=raw.copy();cut=40*1024
    altered[cut:,:30]*=1.2
    a=replay(raw,[777]);b=replay(altered,[256])
    same([w for w in a if w['end_sample']<=cut],[w for w in b if w['end_sample']<=cut])
    raw[28*1024+1,0]=np.nan
    a=replay(raw,[256]);b=replay(raw,[3,1,777]);same(a,b)
    assert all(not (w['start_sample']<28*1024+1<w['end_sample']) for w in a)
