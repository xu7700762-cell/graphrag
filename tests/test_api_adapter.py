import json
from types import SimpleNamespace
import pytest
from eeg_agent.config import load_config,digest,contract
from eeg_agent.agents import CloudProvider,SignalQualityAgent,anonymous_summary
from eeg_agent.signal import Calibration
from tests.test_contracts import qc

class Message:
    def __init__(self,calls):self.tool_calls=calls
    def model_dump(self,**kwargs):
        return {'role':'assistant','content':None,'tool_calls':[{'id':c.id,'type':'function','function':{'name':c.function.name,'arguments':c.function.arguments}} for c in self.tool_calls]}

def call(name,id,args='{}'):
    return SimpleNamespace(id=id,function=SimpleNamespace(name=name,arguments=args))

def test_real_adapter_function_call_protocol_without_external_api(monkeypatch, tmp_path):
    monkeypatch.setenv('EEG_API_BASE_URL','http://127.0.0.1:9/v1');monkeypatch.setenv('EEG_API_KEY','TEST_ONLY');monkeypatch.setenv('EEG_API_MODEL','test-model')
    requests=[]
    def create(**kwargs):
        requests.append(kwargs)
        tool_calls=[call('EOGArtifactAssessor','e'),call('BadChannelAssessor','b')] if len(requests)==1 else [call('ProposeProfile','p','{"profile":"P1"}')]
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12,completion_tokens=3),choices=[SimpleNamespace(message=Message(tool_calls))])
    import openai
    monkeypatch.setattr(openai,'OpenAI',lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    cfg=load_config();cfg['cloud_config_path']=str(tmp_path / '.env');p=CloudProvider('api',cfg,lambda *args:None)
    payload=anonymous_summary(qc(),Calibration(available=True),digest(contract(cfg)),'P0')
    proposal=SignalQualityAgent(p).plan(payload)
    assert proposal['profile']=='P1' and proposal['observed_end_sample']==100
    assert p.summary()['input_tokens']==24 and p.summary()['calls']==2
    assert 'TEST_ONLY' not in json.dumps(requests)
    assert 'NEVER_SEND' not in json.dumps(requests)
    assert len([m for m in requests[1]['messages'] if m['role']=='tool'])==3

def test_budget_failure_accounted():
    cfg=load_config();cfg['api_max_calls']=1;p=CloudProvider('mock',cfg,lambda *args:None)
    p.call('Supervisor',{})
    with pytest.raises(RuntimeError):p.call('Supervisor',{})
    assert p.summary()['budget_rejections']==1 and p.summary()['calls']==1
