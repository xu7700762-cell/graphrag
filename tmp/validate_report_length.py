"""Two real report calls using an existing anonymous snapshot, never rerun EEG."""
import copy
import json
from pathlib import Path

from eeg_agent.agents import CloudProvider
from eeg_agent.brain import BrainSession, _generation_failure, _language, _tool_facts
from eeg_agent.config import load_config

cfg = {**load_config(), 'api_timeout_seconds': 60}
saved = json.loads((Path(cfg['output_root']) / 'brain_sessions' / 'bc3789c7250e' / 'conversation.json').read_text(encoding='utf-8'))
reports = copy.deepcopy(saved['result']['agent_reports'])
vrms = next(r for r in reports if r['agent'] == 'vrms')
refs = vrms['references']
tools = vrms['tool_results']
provider = CloudProvider('api', cfg, lambda *args: None)
session = object.__new__(BrainSession)
session.provider = provider
validation = {'specialist': {}, 'supervisor': {}}
try:
    value = session._call('VRMSAgent', {'assigned_task': '简洁说明所选路径的 MIL、融合与云端最终分类，以及主要证据边界。',
        'tool_results': tools, 'knowledge': refs, 'tool_facts': _tool_facts(tools)},
        'Write an evidence-grounded specialist answer in Chinese. '
        'Return {explanation:string,citations:[provided citation ids],measurement_claims:[{source:string,path:string,value:exact_value}]}. '
        'Every measured number or predicted class mentioned must copy its exact tool_facts entry into measurement_claims. '
        'Do not calculate, round or relabel values. The final class comes only from measurements.cloud_judgment.final_class; '
        'MIL and numeric fusion are intermediate results. State missing inputs and limitations briefly.')
    text, citations = _language(value, refs, tools)
    vrms.update(report=text, citations=citations, generation_status='cloud_verified', fallback_reason=None)
    validation['specialist'] = {'ok': True, 'characters': len(text), 'paragraphs': len(text.split('\n\n')), 'value': value}
except Exception as exc:
    validation['specialist'] = {'ok': False, **_generation_failure(exc, 'specialist_report')}
try:
    refs = list({r['id']: r for report in reports for r in report['references']}.values())
    tools = [item for report in reports for item in report['tool_results']]
    value = session._call('Supervisor', {'question': '请解释当前 VRMS 分类，疲劳和情感能否判断？简洁回答。',
        'agent_reports': reports, 'tool_facts': _tool_facts(tools)},
        'You are Supervisor. Return one concise Chinese answer as JSON '
        '{explanation:string,citations:[provided ids],measurement_claims:[{source:string,path:string,value:exact_value}]}. '
        'Answer the question directly using completed tools and specialist reports. '
        'Every measured number or classification mentioned must copy its exact tool_facts entry into measurement_claims. '
        'The final class comes only from measurements.cloud_judgment.final_class, not the MIL or numeric fusion class. '
        'Do not calculate, round or relabel values. State missing/failing specialists honestly. '
        'Fatigue and emotion classifiers are unavailable; no fatigue or emotion diagnosis.')
    text, citations = _language(value, refs, tools)
    validation['supervisor'] = {'ok': True, 'characters': len(text), 'paragraphs': len(text.split('\n\n')), 'value': value}
except Exception as exc:
    validation['supervisor'] = {'ok': False, **_generation_failure(exc, 'supervisor_report')}
finally:
    validation['provider'] = provider.summary()
    provider.client.close()
output = Path(cfg['output_root']) / 'reports' / 'real_report_length_validation.json'
output.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({key: {k: v for k, v in value.items() if k != 'value'}
                  for key, value in validation.items()}, ensure_ascii=False, indent=2))
