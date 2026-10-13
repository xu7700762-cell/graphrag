"""Capture one live report and audit numeric formatting on saved anonymous tools."""
import ast
import json
import re
from pathlib import Path
from eeg_agent.agents import CloudProvider
from eeg_agent.brain import BrainSession, _generation_failure, _language, _numbers, _tool_facts
from eeg_agent.config import load_config

cfg = {**load_config(), 'api_timeout_seconds': 60}
saved = json.loads((Path(cfg['output_root']) / 'brain_sessions' / 'f903ba4b0f2f' / 'conversation.json').read_text(encoding='utf-8'))
report = next(r for r in saved['result']['agent_reports'] if r['agent'] == 'vrms')
plan = next(e['payload'] for e in saved['events'] if e['kind'] == 'plan')
task = next(a['task'] for a in plan['agents'] if a['id'] == 'vrms')
tree = ast.parse(Path('eeg_agent/brain.py').read_text(encoding='utf-8'))
instruction = next(node.args[2].value for node in ast.walk(tree) if isinstance(node, ast.Call) and
    isinstance(node.func, ast.Attribute) and node.func.attr == '_call' and len(node.args) == 3 and
    isinstance(node.args[2], ast.Constant) and node.args[2].value.startswith('Write an evidence-grounded specialist'))
provider = CloudProvider('api', cfg, lambda *args: None)
session = object.__new__(BrainSession)
session.provider = provider
audit = {'source_session': saved['id']}
try:
    value = session._call('VRMSAgent', {'assigned_task': task, 'tool_results': report['tool_results'],
        'knowledge': report['references'], 'prior_report': [],
        'tool_facts': _tool_facts(report['tool_results'])}, instruction)
    audit['received_value'] = value
    facts = {(f['source'], f['path']): f['value'] for f in _tool_facts(report['tool_results'])}
    claims = value.get('measurement_claims', [])
    verified_claims = [c for c in claims if isinstance(c, dict) and
        (c.get('source'), c.get('path')) in facts and facts[(c['source'], c['path'])] == c.get('value')]
    cited_text = '\n'.join(r['text'] for r in report['references'] if r['id'] in value.get('citations', []))
    supported = _numbers(cited_text)
    for claim in verified_claims:
        supported.update(_numbers(claim['value']))
    prose = value['explanation']
    for reference in report['references']:
        prose = prose.replace(reference['id'], '')
    prose = re.sub(r'(?m)^\s*(?:#{1,6}\s*)?\d+[.)、．]\s*', '', prose)
    prose = re.sub(r'\b[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+\b', '', prose)
    unmatched = _numbers(prose) - supported
    audit['unmatched_numeric_values'] = sorted(str(v) for v in unmatched)
    audit['verified_claims'] = verified_claims
    audit['supported_numeric_values'] = sorted(str(v) for v in supported)
    audit['unmatched_context'] = [m.group() for m in re.finditer(r'.{0,24}[0-9]+(?:\.[0-9]+)?.{0,32}', prose)
                                  if _numbers(m.group()) & unmatched]
    text, citations = _language(value, report['references'], report['tool_results'])
    audit.update(ok=True, characters=len(text))
except Exception as exc:
    audit.update(ok=False, **_generation_failure(exc, 'specialist_report'))
finally:
    audit['provider'] = provider.summary()
    provider.client.close()
output = Path(cfg['output_root']) / 'reports' / 'vrms_numeric_capture_f903ba4b0f2f.json'
output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(audit, ensure_ascii=False, indent=2))
