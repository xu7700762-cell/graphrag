import json
import time
from pathlib import Path
from eeg_agent.config import load_config
from eeg_agent.monitor import MonitorTask, TaskManager

def test_real_source_missing_model_and_recovery(tmp_path):
    cfg=load_config();cfg['output_root']=str(tmp_path)
    task=MonitorTask(cfg,{'subject':1,'model_id':'eegconformer','strategy':'adaptive','backend':'mock','speed':1000,'max_seconds':36})
    task.execute()
    assert task.state=='completed',task.error
    assert task.latest['probability_high'] is None
    assert task.baseline.summary()['frozen'] and task.baseline.summary()['windows']==2
    assert task.result['model']['model_id']=='eegconformer' and task.result['model']['available'] is False
    assert task.result['paths'][0]['status']=='incomplete'
    assert (task.directory/'report.md').exists()
    restored=TaskManager(cfg).get(task.id)
    assert restored.state=='completed' and restored.report()['task_id']==task.id

def test_stop_does_not_wait_for_queued_cloud(tmp_path):
    cfg=load_config();cfg['output_root']=str(tmp_path)
    task=MonitorTask(cfg,{'subject':1,'backend':'mock','speed':1,'max_seconds':130})
    def slow_review(*args): time.sleep(1.5)
    task.cloud_review=slow_review;task.start();time.sleep(.05)
    begin=time.perf_counter();task.request_stop();task.thread.join(.7)
    assert task.state=='stopped' and time.perf_counter()-begin<.7

def test_stop_during_final_report_does_not_wait(tmp_path):
    import threading
    cfg=load_config();cfg['output_root']=str(tmp_path)
    task=MonitorTask(cfg,{'subject':1,'backend':'mock','speed':1000,'max_seconds':36})
    final_started=threading.Event()
    def slow_final(role,query,phase):
        if phase=='final': final_started.set();time.sleep(1.5)
    task.cloud_review=slow_final;task.start();assert final_started.wait(2)
    begin=time.perf_counter();task.request_stop();task.thread.join(.7)
    assert task.state=='stopped' and time.perf_counter()-begin<.7

def test_cloud_uses_allowlisted_summary(tmp_path):
    cfg=load_config();cfg['output_root']=str(tmp_path)
    task=MonitorTask(cfg,{'subject':1,'model_id':'eegnet','backend':'mock','speed':1000,'max_seconds':36})
    task.latest={'quality':{'valid':True},'deep':{'logits':[1,2],'probability_high':.7},
                 'temporal':{'history':[1,2,3],'mean_30s':.6},
                 'biomarkers':{'psd':[[999]],'regions':{'all':{}},'baseline_change':{'regions':{},'per_channel':[888]}},
                 'subject_id':'secret','score':90,'label':1,'x':[12345]}
    serialized=json.dumps(task.evidence())
    assert all(v not in serialized for v in ('secret','12345','999','888','logits','history','subject_id','"label"','"score"'))
    task.request_stop();task.executor.shutdown(wait=False,cancel_futures=True)

def test_interrupted_context_is_not_auto_resumed(tmp_path):
    cfg=load_config();cfg['output_root']=str(tmp_path)
    directory=Path(tmp_path)/'tasks'/'example';directory.mkdir(parents=True)
    (directory/'context.json').write_text(json.dumps({'id':'example','state':'running','request':{'start_sample':5000}}),encoding='utf-8')
    manager=TaskManager(cfg);assert manager.get('example').state=='interrupted'
