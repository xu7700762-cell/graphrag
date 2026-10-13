"""Exercise the real cloud tool-message path without credentials or network."""
import copy
import json
import platform
import threading
from types import SimpleNamespace

from eeg_agent.monitor import MonitorTask, _cloud_failure_reason


def test_auxiliary_tool_reply_is_anonymous_scalar_summary_and_local_evidence_stays_full():
    secret='NEVER_SEND_ARRAY_OR_IDENTITY'
    local={'status':'success','biomarkers':{
        'available':True,'version':'2.0','unit':'uV^2','baseline_available':True,'reason':None,
        'raw':[secret],'psd':[[secret]],'channels':[secret],'subject':secret,'score':secret,
        'regions':{'frontal':{'absolute_power_uv2':{'theta':2.0,'subject':secret},
                              'ratios':{'theta_alpha':0.5}},'subject':secret},
        'baseline_change':{'per_channel':{'theta':[secret]},'regions':{'frontal':{'theta':0.25}}}},
        'covariance':{'available':False,'reason':'prior baseline covariance unavailable',
                      'airm_distance':None,'matrix':[[secret]],'label':secret,'logits':[secret]}}
    calls=[]

    class Reply:
        def __init__(self, tool_calls=None, content=None):
            self.tool_calls=tool_calls
            self.content=content

        def model_dump(self, **kwargs):
            return {'role':'assistant','content':self.content}

    def call(role, payload, tools, messages):
        calls.append(copy.deepcopy(messages))
        if len(calls)==1:
            return Reply([SimpleNamespace(id='aux',function=SimpleNamespace(
                name='RequestAuxiliaryAnalysis',arguments='{"tools":["BiomarkerCalculator","CovarianceAnalyzer"]}'))])
        return Reply(content=json.dumps({'explanation':'Local evidence remains authoritative.',
                                         'citations':[],'requested_tools':[]}))

    task=MonitorTask.__new__(MonitorTask)
    task.provider=SimpleNamespace(backend='api',call=call)
    task.knowledge=SimpleNamespace(search=lambda *args,**kwargs:[])
    task.lock=threading.RLock(); task.references={}; task.language=[]; task.state='running'
    task.last_window={'end_sample':10240}
    task.evidence=lambda:{'arrived':{}}
    task.emit=lambda *args:None
    task.request_auxiliary=lambda *args:local
    result=task.cloud_review('VRMSAgent','Explain arrived evidence','review')
    assert result['fallback_reason'] is None
    payload=json.loads(next(m['content'] for m in calls[0] if m['role']=='user'))
    assert payload['runtime_environment']=={'os':platform.system(),'architecture':platform.machine()}
    assert 'deployment target' in calls[0][0]['content']
    assert secret not in json.dumps(calls)
    reply=json.loads(next(m['content'] for m in calls[1] if m['role']=='tool'))
    assert reply['status']=='success'
    assert reply['biomarkers']['regions']['frontal']['absolute_power_uv2']['theta']==2.0
    assert reply['biomarkers']['baseline_change']['frontal']['theta']==0.25
    assert reply['covariance']['available'] is False
    assert reply['covariance']['reason']=='prior baseline covariance unavailable'
    assert local['biomarkers']['psd']==[[secret]]
    assert _cloud_failure_reason(ValueError(secret))=='ValueError'
    assert _cloud_failure_reason(ValueError('Missing citation'))=='ValueError: Missing citation'
    assert _cloud_failure_reason(RuntimeError(secret))=='RuntimeError'
