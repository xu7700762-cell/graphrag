"""A cleanup profile must select its own inputs, gates and normalization."""
import json
from pathlib import Path
import numpy as np
import pytest

from eeg_agent.config import load_config, profile_config, contract, digest, cache_path
from eeg_agent.learning import input_arrays, training_arrays, evaluate_records, accepted_logit_paths


def record(subject,label,value):
    return {'subject':subject,'label':label,'key':str(subject),'duration':20,'raw_complete':True,
            'x0':np.full((4,30,1280),9999,dtype=np.float32),
            'x1':np.full((4,30,1280),value,dtype=np.float32),
            'valid':np.ones(4,dtype=bool),'valid1':np.ones(4,dtype=bool),'p1':np.ones(4,dtype=bool)}


def test_p0_contract_preserved_and_cleanup_cache_separate():
    cfg=load_config()
    old=Path(cfg['output_root'])/'models/eegnet/fixed_seed42/contract.json'
    if old.exists():
        original=json.loads(old.read_text(encoding='utf-8'))
        assert digest(contract(cfg))==original['contract_hash']
    cleaned=profile_config(cfg,'P1')
    assert digest(contract(cleaned))!=digest(contract(cfg))
    assert cache_path(cleaned)!=cache_path(cfg)
    assert contract(cleaned)['classification_profiles']==['P1']


def test_uncalibrated_p1_cannot_enter_training_or_evaluation():
    cfg=profile_config(load_config(),'P1');r=record(2,0,.2)
    r['p1'][0]=False
    with pytest.raises(ValueError,match='Uncalibrated'):
        input_arrays(r,cfg)
    with pytest.raises(ValueError,match='Uncalibrated'):
        evaluate_records([r],lambda x:np.ones(len(x))*.2,cfg)


def test_p1_normalization_gate_and_validation_select_cleaned_inputs():
    cfg=profile_config(load_config(),'P1')
    rows=[record(2,0,.2),record(3,1,.8),record(4,1,99999)]
    rows[-1]['valid1'][:]=False
    x,y,weights,mean,std,contributing=training_arrays(rows,cfg)
    assert len(x)==8 and len(contributing)==2
    np.testing.assert_allclose(mean,.5)
    assert weights[y==0].sum()==weights[y==1].sum()
    evaluated=evaluate_records(rows,lambda a:a[:,0,0],cfg,threshold=.5)
    assert [p['predicted_class'] for p in evaluated['paths']]==[0,1,None]
    assert evaluated['metrics']['accepted_paths']==2
    seen=[]
    def logits(a):
        seen.append(a);return np.zeros((len(a),2))
    paths=accepted_logit_paths(rows,logits,cfg)
    assert len(paths)==2 and all(a.max()<1 for a in seen)
