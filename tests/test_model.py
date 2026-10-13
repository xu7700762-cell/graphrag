import json
from pathlib import Path
import numpy as np
import pytest
from eeg_agent.config import load_config
from eeg_agent.runtime import ONNXPredictor,ReplayRun
from eeg_agent.learning import load_cache
from eeg_agent.signal import CausalFilter
from tests.test_stream import CHANNELS,data

def test_sample_and_event_order_rejected():
    f=CausalFilter(CHANNELS,load_config());f.process(0,data(256))
    with pytest.raises(ValueError,match='Out-of-order'):f.process(128,data(256))

def test_frozen_model_contract_and_finite_prediction():
    cfg=load_config();directory=Path(cfg['output_root'])/'models/baseline'
    if not (directory/'model.onnx').exists():pytest.skip('Train first for artifact integration check')
    p=ONNXPredictor(cfg);value=p.predict(np.zeros((1280,30)))
    assert 0<=value<=1
    altered=load_config();altered['qc']['peak_to_peak_uv']+=1
    with pytest.raises(ValueError,match='contract mismatch'):ONNXPredictor(altered)
    with pytest.raises(ValueError,match='Invalid model input'):p.predict(np.zeros((30,1280)))
    manifest=json.loads((directory/'contract.json').read_text(encoding='utf-8'))
    assert not set(manifest['normalization_fit_subjects']) & {1,9,12,17,19}
    assert manifest['onnx_max_abs_error']<1e-4

def test_cache_quality_profile_gates():
    cfg=load_config()
    if not (Path(cfg['output_root'])/'cache/manifest.json').exists():pytest.skip('Prepare first')
    records,_=load_cache(cfg,'validation')
    assert len(records)==36
    assert all(not np.any(r['valid1']&~r['valid']) for r in records)
