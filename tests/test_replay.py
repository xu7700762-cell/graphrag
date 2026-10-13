import json
import time
import numpy as np
from eeg_agent.config import load_config
from eeg_agent.runtime import ReplayRun
from eeg_agent.service import create_app
from fastapi.testclient import TestClient

def test_http_start_sse_stop_and_missing_model(tmp_path):
    cfg=load_config();cfg['output_root']=str(tmp_path)
    client=TestClient(create_app(cfg))
    r=client.post('/api/runs',json={'subject':1,'mode':'hierarchical','backend':'mock','speed':100,'max_seconds':36})
    assert r.status_code==200
    run_id=r.json()['id']
    with client.stream('GET','/api/runs/'+run_id+'/events') as response:
        events=[json.loads(line[6:]) for line in response.iter_lines() if line.startswith('data: ')]
    snapshot=client.get('/api/runs/'+run_id).json()
    assert snapshot['state']=='completed'
    assert any(e['kind']=='model' and not e['payload']['available'] for e in events)
    windows=[e['payload'] for e in events if e['kind']=='window']
    assert windows and all(w['probability_high'] is None for w in windows)
    assert any(e['kind']=='agent_call' and e['payload']['simulated'] for e in events)
    assert snapshot['result']['timing_kind']=='accelerated_throughput'
    assert all(p['status']!='accepted' for p in snapshot['result']['paths'])
    assert client.post('/api/runs/'+run_id+'/stop').status_code==200

def test_readme_raw_boundary_exclusion_and_hash():
    from eeg_agent.data import paths
    cfg=load_config();r=paths(cfg)
    assert len(r)==147 and sum(x['raw_complete'] for x in r)==146
    assert next(x for x in r if x['key']=='04:2516402')['raw_complete'] is False
