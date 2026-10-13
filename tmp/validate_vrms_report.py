"""One production-prompt VRMS report call on an already saved safe snapshot."""
import ast
import json
from pathlib import Path
from eeg_agent.agents import CloudProvider
from eeg_agent.brain import BrainSession, _generation_failure, _language, _tool_facts
from eeg_agent.config import load_config

cfg = {**load_config(), 'api_timeout_seconds': 60}
saved = json.loads((Path(cfg['output_root']) / 'brain_sessions' / 'bc3789c7250e' / 'conversation.json').read_text(encoding='utf-8'))
report = next(r for r in saved['result']['agent_reports'] if r['agent'] == 'vrms')
tree = ast.parse(Path('eeg_agent/brain.py').read_text(encoding='utf-8'))
instruction = next(node.args[2].value for node in ast.walk(tree) if isinstance(node, ast.Call) and
    isinstance(node.func, ast.Attribute) and node.func.attr == '_call' and len(node.args) == 3 and
    isinstance(node.args[2], ast.Constant) and node.args[2].value.startswith('Write an evidence-grounded specialist'))
provider = CloudProvider('api', cfg, lambda *args: None)
session = object.__new__(BrainSession)
session.provider = provider
audit = {}
try:
    value = session._call('VRMSAgent', {'assigned_task': '简洁解释所选路径的三层分类及证据边界，数值已单独展示，不需重复所有分数。',
        'tool_results': report['tool_results'], 'knowledge': report['references'],
        'prior_report': [], 'tool_facts': _tool_facts(report['tool_results'])}, instruction)
    audit['received_value'] = value
    text, citations = _language(value, report['references'], report['tool_results'])
    audit.update(ok=True, characters=len(text), paragraphs=len(text.split('\n\n')))
except Exception as exc:
    audit.update(ok=False, **_generation_failure(exc, 'specialist_report'))
finally:
    audit['provider'] = provider.summary()
    provider.client.close()
output = Path(cfg['output_root']) / 'reports' / 'vrms_report_compact_validation.json'
output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(audit, ensure_ascii=False, indent=2))
