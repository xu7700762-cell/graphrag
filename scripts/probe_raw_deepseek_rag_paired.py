"""Paired DeepSeek decisions on the previous fixed ten-path error subset.

Twenty judgments, at most forty native API requests, zero retries.
Frozen numerical evidence is reused; only teaching/case RAG is ablated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from scripts import probe_raw_gpt_high_rag as base
from eeg_agent import agents, cloud_config, raw_tool_agent

SOURCE = base.RESEARCH / 'outputs/vrms_rag_probes/20261011_gpt_high_ten_errors_01'
OUT = base.RESEARCH / 'outputs/vrms_rag_probes/20261011_deepseek_rag_paired_01'
ARMS = ('with_rag', 'without_rag')
ACTIVE_ARM = 'with_rag'
VALIDATION_FIELDS = ('policy.bacc', 'raw_policy.bacc', 'calibrated_policy.bacc')


class PairedRuntime(base.ProbeRuntime):
    numerical_execution = 'reused_verified_frozen_numeric_evidence; no_new_waveform_encoding_this_run'

    def initial_evidence(self):
        value = super().initial_evidence()
        if ACTIVE_ARM == 'without_rag':
            value['evidence_teaching'] = {'available': False, 'reason': 'paired_ablation'}
        return value

    def execute(self, name):
        value = super().execute(name)
        if ACTIVE_ARM == 'without_rag' and name == 'case_retrieval':
            value.pop('training_case_context', None)
        self.memo[name] = copy.deepcopy(value)
        assert not base.hidden_keys(value)
        return value


def build_payloads(prediction, teaching, arm):
    global ACTIVE_ARM
    ACTIVE_ARM = arm
    runtime = PairedRuntime(prediction, teaching_context=teaching)
    return runtime.initial_evidence(), runtime.execute('corrective_evidence')


def schema():
    result = copy.deepcopy(raw_tool_agent.SCHEMA)
    reading = {'type': 'object', 'additionalProperties': False, 'properties': {
        'tool': {'type': 'string', 'enum': list(raw_tool_agent.TOOL_NAMES)},
        'raw_decision_score': {'type': 'number'}, 'raw_class': {'type': 'string', 'enum': ['High', 'Low']},
        'calibration_usable': {'type': 'boolean'}, 'calibrated_high_probability': {'type': ['number', 'null']},
        'used_boundary': {'type': 'string', 'enum': ['raw', 'calibrated']},
        'validation_field': {'type': 'string', 'enum': list(VALIDATION_FIELDS) + ['not_provided']},
        'validation_value': {'type': ['number', 'null']},
        'role': {'type': 'string', 'enum': ['supports', 'conflicts', 'not_used']},
        'interpretation': {'type': 'string'}}}
    reading['required'] = list(reading['properties'])
    case = {'type': 'object', 'additionalProperties': False, 'properties': {
        'case_id': {'type': 'string'}, 'training_class': {'type': 'string', 'enum': ['High', 'Low']},
        'mil_raw_high_probability': {'type': 'number'}, 'interpretation': {'type': 'string'}}}
    case['required'] = list(case['properties'])
    audit = {'type': 'object', 'additionalProperties': False, 'properties': {
        'mil_raw_high_probability': {'type': 'number'}, 'mil_raw_class': {'type': 'string', 'enum': ['High', 'Low']},
        'tool_readings': {'type': 'array', 'items': reading, 'minItems': 8, 'maxItems': 8},
        'teaching_citation_ids': {'type': 'array', 'items': {'type': 'string'}},
        'case_evidence': {'type': 'array', 'items': case},
        'dependence_check': {'type': 'string'}, 'decision_basis': {'type': 'string'},
        'effect': {'type': 'string', 'enum': ['supports_final_class', 'conflicts_with_final_class',
                    'insufficient_to_change_decision', 'not_available']}, 'summary': {'type': 'string'}}}
    audit['required'] = list(audit['properties'])
    result['properties']['evidence_use_audit'] = audit
    result['required'].append('evidence_use_audit')
    return result


SYSTEM = base.INDEPENDENT_SYSTEM + """
If evidence_teaching is available, apply its reviewed field definitions and
decision workflow, rather than merely attaching citations. Detailed training
cases, if present, are cached in-sample references, not query truth or OOF
validation. Do not force a particular final class or an accuracy improvement.
Return evidence_use_audit as specified in final_response_schema. Copy the MIL
output and all eight ACTUAL tool readings, including raw class, raw score,
calibration flag/probability, and one actually supplied validation BACC field.
Use not_provided and null only if that metric is absent. used_boundary states
whether you use the raw or calibrated direction; role states support, conflict,
or not_used relative to your final class. Do not invent a usable calibrator.
Briefly explain shared evidence, the strongest support and counterevidence,
and why your final choice follows from them. This is a concise public evidence
audit, not private chain-of-thought. Do not copy hypothetical teaching numbers
as actual query data. If teaching/cases are unavailable, leave their citation
and case arrays empty and use effect=not_available. If available, cite at least
one supplied teaching entry and detailed case, copying its training class and
MIL output. If they do not help, say so. Other numerical evidence is identical
across the experiment. It was computed earlier and is replayed in this run.
"""


def audit_final(final, initial, corrective, arm):
    """Validate reported readings independently of whether the prediction is right."""
    errors = []
    audit = final.get('evidence_use_audit', {})
    def close(actual, claimed):
        if actual is None:
            return claimed is None
        return type(claimed) in (int, float) and abs(actual - claimed) < 5e-6
    p = initial['baseline']['raw_high_probability']
    if not close(p, audit.get('mil_raw_high_probability')) or audit.get('mil_raw_class') != initial['baseline']['raw_class']:
        errors.append('MIL_reading')
    actual_tools = {x['tool']: x for x in corrective['actual_individual_evidence']}
    readings = audit.get('tool_readings', [])
    if (len(readings) != 8 or {x.get('tool') for x in readings} != set(actual_tools)):
        errors.append('eight_tool_coverage')
    for x in readings:
        name = x.get('tool')
        if name not in actual_tools:
            continue
        actual = actual_tools[name]
        if (not close(actual['raw_decision_score'], x.get('raw_decision_score')) or
                actual['raw_learned_class'] != x.get('raw_class')):
            errors.append(name + '.raw_reading')
        if (actual.get('calibration_usable', False) is not x.get('calibration_usable') or
                not close(actual.get('calibrated_high_probability'), x.get('calibrated_high_probability'))):
            errors.append(name + '.calibration_reading')
        field = x.get('validation_field')
        validation = actual.get('internal_validation', {})
        if field in VALIDATION_FIELDS:
            policy, metric = field.split('.')
            value = validation.get(policy, {}).get(metric)
            if value is None or not close(value, x.get('validation_value')):
                errors.append(name + '.validation_reading')
        elif field != 'not_provided' or x.get('validation_value') is not None or any(
                validation.get(f.split('.')[0], {}).get('bacc') is not None for f in VALIDATION_FIELDS):
            errors.append(name + '.missing_validation')
        if x.get('used_boundary') == 'calibrated':
            probability = actual.get('calibrated_high_probability')
            if actual.get('calibration_usable') is not True or probability is None:
                errors.append(name + '.unusable_boundary')
                continue
            direction = 'High' if probability >= .5 else 'Low'
        elif x.get('used_boundary') == 'raw':
            direction = actual['raw_learned_class']
        else:
            errors.append(name + '.boundary')
            continue
        expected_role = 'supports' if direction == final['final_class'] else 'conflicts'
        if x.get('role') not in (expected_role, 'not_used'):
            errors.append(name + '.direction_role')
        if not isinstance(x.get('interpretation'), str) or not x['interpretation'].strip():
            errors.append(name + '.interpretation')
    if not all(isinstance(audit.get(k), str) and audit[k].strip() for k in ('dependence_check', 'decision_basis', 'summary')):
        errors.append('missing_public_interpretation')
    teaching = initial['evidence_teaching']
    if arm == 'with_rag':
        ids = {x['citation_id'] for x in teaching['input_definitions'] + teaching['judgment_principles']}
        if not audit.get('teaching_citation_ids') or not set(audit['teaching_citation_ids']).issubset(ids):
            errors.append('teaching_citation')
        context = actual_tools['case_retrieval']['training_case_context']
        cases = {x['case_id']: x for x in context['detailed_cases']}
        if not audit.get('case_evidence'):
            errors.append('missing_detailed_case')
        for cited in audit.get('case_evidence', []):
            actual = cases.get(cited.get('case_id'))
            if (actual is None or actual['training_class'] != cited.get('training_class') or
                    not close(actual['mil_raw_high_probability'], cited.get('mil_raw_high_probability'))):
                errors.append('case_reading')
    elif audit.get('teaching_citation_ids') or audit.get('case_evidence') or audit.get('effect') != 'not_available':
        errors.append('invented_RAG_in_ablation')
    return {'passed': not errors, 'errors': errors, 'nine_input_readings': len(readings) + 1,
            'effect': audit.get('effect')}


def prepare():
    if OUT.exists():
        raise FileExistsError('Do not overwrite an existing paid experiment')
    selected = base.read(SOURCE / 'selection.json')
    assert len(selected) == 10
    old = {r['path_key']: r for r in csv.DictReader((base.PREVIOUS / 'path_results.csv').open(encoding='utf-8-sig', newline=''))}
    teaching = base.build_teaching_context()
    assert teaching['available'] and len(teaching['input_definitions']) == 9
    assert 'BACC' in json.dumps(teaching, ensure_ascii=False)
    OUT.mkdir(parents=True)
    inputs = {}
    for row in selected:
        row.update(previous_deepseek_class=old[row['path_key']]['deepseek_class'],
                   previous_deepseek_correct=old[row['path_key']]['deepseek_correct'] == 'True')
        src = SOURCE / 'paths' / row['path_key'].replace(':', '_') / 'prediction.json'
        inputs[row['path_key']] = base.read(src)
        full = build_payloads(inputs[row['path_key']], teaching, 'with_rag')
        stripped = build_payloads(inputs[row['path_key']], teaching, 'without_rag')
        check_initial, check_tools = copy.deepcopy(full)
        check_initial['evidence_teaching'] = {'available': False, 'reason': 'paired_ablation'}
        next(x for x in check_tools['actual_individual_evidence'] if x['tool'] == 'case_retrieval').pop('training_case_context')
        assert (check_initial, check_tools) == stripped
        assert not base.hidden_keys(full) and not base.hidden_keys(stripped)
    base.save(OUT / 'selection.json', selected)
    base.save(OUT / 'inputs.json', inputs)
    base.save(OUT / 'teaching.json', teaching)
    protected = [Path(__file__), APP / '.env', APP / 'cloud_profiles.json',
        APP / 'eeg_agent/raw_evidence_teaching.py', APP / 'eeg_agent/raw_tool_agent.py',
        APP / 'eeg_agent/brain_knowledge/vrms/03_nine_evidence_guide.md',
        SOURCE / 'summary.json', SOURCE / 'selection.json', base.PREVIOUS / 'path_results.csv',
        OUT / 'inputs.json', OUT / 'selection.json', OUT / 'teaching.json']
    protected.extend(SOURCE / 'paths' / r['path_key'].replace(':', '_') / 'prediction.json' for r in selected)
    protocol = {'schema': 'deepseek_ten_paths_paired_new_RAG_v1', 'created_at': base.stamp(),
        'paths_per_arm': 10, 'arms': list(ARMS), 'maximum_api_requests': 40,
        'maximum_requests_per_judgment': 2, 'automatic_retries': 0,
        'model_id': 'deepseek-flash', 'thinking': 'disabled', 'temperature': 0,
        'output_tokens_per_request': 8192, 'api_timeout_seconds': 120,
        'numerical_execution': 'reused_verified_frozen_numeric_evidence',
        'numerical_fitting_calls': 0, 'fresh_encoding_calls': 0,
        'query_truth_hidden': True, 'fused_answer_hidden': True,
        'paired_input_invariance_verified': True, 'previous_deepseek_correct': sum(r['previous_deepseek_correct'] for r in selected),
        'protected_file_hashes': {str(f): base.sha(f) for f in protected},
        'limitation': 'Selected old GPT errors; nine old DeepSeek errors and one correct; not full-population ACC.'}
    base.save(OUT / 'protocol.json', protocol)
    base.save(OUT / 'offline_ready.json', {'paths': 10, 'judgments': 20, 'api_requests': 0})
    print(json.dumps({'offline_ready': True, 'paths': 10, 'judgments': 20, 'maximum_requests': 40,
        'teaching_characters': len(json.dumps(teaching, ensure_ascii=False)), 'api_requests': 0}), flush=True)


def case_directory(path_key, arm):
    return OUT / 'paths' / path_key.replace(':', '_') / arm


def one(path_key, arm):
    global ACTIVE_ARM
    ACTIVE_ARM = arm
    target = case_directory(path_key, arm)
    if target.exists():
        raise RuntimeError('Judgment already attempted; do not repeat paid requests')
    target.mkdir(parents=True)
    prediction = base.read(OUT / 'inputs.json')[path_key]
    teaching = base.read(OUT / 'teaching.json')
    initial, corrective = build_payloads(prediction, teaching, arm)
    profile = cloud_config._profile_state(None)['profiles']['deepseek']
    assert profile['model_id'] == 'deepseek-flash' and profile['base_url'].rstrip('/') == 'https://api.deepseek.com'
    assert profile['api_key']
    agents.cloud_settings = lambda path=None: {name: profile[field] for field, name in cloud_config.NAMES.items()}
    cfg = base.load_config()
    cfg.update(api_timeout_seconds=120, api_max_calls=2)
    provider = agents.CloudProvider('api', cfg, lambda kind, payload: None)
    assert provider.thinking_disabled and provider.client.max_retries == 0
    raw_tool_agent.EvidenceRuntime, raw_tool_agent.SYSTEM, raw_tool_agent.SCHEMA = PairedRuntime, SYSTEM, schema()
    raw_tool_agent.TOOLS = copy.deepcopy(raw_tool_agent.TOOLS)
    for tool in raw_tool_agent.TOOLS:
        tool['description'] = ('Read the eight individual frozen tool results; no fused result.'
            if tool['name'] == 'corrective_evidence' else 'Read one immutable numerical result.')
    original_create = provider.client.chat.completions.create
    records = []

    def create(**kwargs):
        sequence = len(records) + 1
        if sequence > 2:
            raise RuntimeError('Two-request judgment budget exhausted')
        kwargs['max_tokens'] = 8192
        kwargs['temperature'] = 0
        kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}
        if sequence == 1:
            kwargs['tool_choice'] = {'type': 'function', 'function': {'name': 'corrective_evidence'}}
        else:
            kwargs.pop('tools', None)
            kwargs.pop('tool_choice', None)
            kwargs['response_format'] = {'type': 'json_object'}
        assert not base.hidden_keys(kwargs['messages'])
        record = {'sequence': sequence, 'started_at': base.stamp(), 'request': copy.deepcopy(kwargs)}
        records.append(record)
        base.save(target / f'request_{sequence}.json', record)
        provider.calls[-1]['output_token_budget'] = 8192
        try:
            response = original_create(**kwargs)
            record.update(finished_at=base.stamp(), response_id=response.id, response_model=response.model,
                          usage=response.usage.model_dump(mode='json') if response.usage else None)
            base.save(target / f'response_{sequence}.json', response.model_dump(mode='json'))
            return response
        except Exception as exc:
            record.update(finished_at=base.stamp(), error_type=type(exc).__name__)
            raise
        finally:
            base.save(target / f'request_{sequence}.json', record)

    provider.client.chat.completions.create = create
    base.save(target / 'api_started.json', {'started_at': base.stamp(), 'maximum_requests': 2, 'retries': 0})
    judgment = raw_tool_agent.classify(provider, prediction, teaching_context=teaching)
    judgment['numerical_execution'] = PairedRuntime.numerical_execution
    base.save(target / 'judgment.json', judgment)
    base.save(target / 'api_usage.json', provider.summary())
    try:
        audit = audit_final(judgment, initial, corrective, arm) if judgment['available'] else {'passed': False, 'errors': ['no_valid_judgment']}
    except (KeyError, TypeError, AttributeError, ValueError):
        audit = {'passed': False, 'errors': ['malformed_evidence_audit']}
    base.save(target / 'evidence_audit.json', audit)
    print(json.dumps({'path': path_key, 'arm': arm, 'available': judgment['available'],
        'class': judgment['final_class'], 'evidence_reading_passed': audit['passed'],
        'requests': len(records)}, ensure_ascii=False), flush=True)


def aggregate(stage):
    selected = base.read(OUT / 'selection.json')
    rows, all_usage = [], []
    for row in selected:
        for arm in ARMS:
            target = case_directory(row['path_key'], arm)
            if not (target / 'evidence_audit.json').exists():
                continue
            j = base.read(target / 'judgment.json')
            audit = base.read(target / 'evidence_audit.json')
            usage = base.read(target / 'api_usage.json')
            all_usage.append(usage)
            rows.append({'path_key': row['path_key'], 'arm': arm, 'true_class': row['true_class'],
                'old_deepseek_class': row['previous_deepseek_class'], 'old_deepseek_correct': row['previous_deepseek_correct'],
                'new_class': j['final_class'] if j['available'] else None, 'available': j['available'],
                'correct': bool(j['available'] and j['final_class'] == row['true_class']),
                'confidence': j.get('confidence'), 'evidence_reading_passed': audit['passed'],
                'audit_errors': ';'.join(audit['errors']), 'explanation': j.get('explanation', ''),
                'api_requests': usage['calls'], 'api_failures': usage['failures']})
    metrics = {}
    for arm in ARMS:
        subset = [r for r in rows if r['arm'] == arm]
        metrics[arm] = {'completed': len(subset), 'total': 10, 'valid_judgments': sum(r['available'] for r in subset),
            'correct': sum(r['correct'] for r in subset), 'acc_fixed_denominator': sum(r['correct'] for r in subset) / 10,
            'corrected_old_errors': sum(r['correct'] and not r['old_deepseek_correct'] for r in subset),
            'damaged_old_correct': sum(not r['correct'] and r['old_deepseek_correct'] for r in subset),
            'evidence_reading_passed': sum(r['evidence_reading_passed'] for r in subset)}
    paired = []
    by_key = {(r['path_key'], r['arm']): r for r in rows}
    for row in selected:
        a, b = by_key.get((row['path_key'], 'with_rag')), by_key.get((row['path_key'], 'without_rag'))
        if a and b:
            paired.append({'path_key': row['path_key'], 'with_rag_class': a['new_class'], 'without_rag_class': b['new_class'],
                'class_changed': a['new_class'] != b['new_class'], 'RAG_corrected': a['correct'] and not b['correct'],
                'RAG_damaged': b['correct'] and not a['correct']})
    summary = {'stage': stage, 'completed_judgments': len(rows), 'total_judgments': 20, 'metrics': metrics,
        'old_deepseek_correct_on_subset': 1, 'old_deepseek_errors_on_subset': 9,
        'api_requests': sum(u['calls'] for u in all_usage), 'api_failures': sum(u['failures'] for u in all_usage),
        'input_tokens': sum(u['input_tokens'] for u in all_usage), 'output_tokens': sum(u['output_tokens'] for u in all_usage),
        'retries': 0, 'paired_completed': len(paired), 'RAG_corrected': sum(r['RAG_corrected'] for r in paired),
        'RAG_damaged': sum(r['RAG_damaged'] for r in paired), 'paired_class_changes': sum(r['class_changed'] for r in paired),
        'updated_at': base.stamp(), 'path_results': rows, 'paired_results': paired,
        'limitation': 'Fixed old GPT error subset; ACC is for ten selected paths, not all146. One run per arm.'}
    base.save(OUT / 'progress.json', summary)
    if stage == 'completed':
        base.save(OUT / 'summary.json', summary)
        with (OUT / 'path_results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return summary


def run():
    base.read(OUT / 'offline_ready.json')
    if (OUT / 'api_started.json').exists():
        raise RuntimeError('Paired batch already attempted; no duplicate paid requests')
    protocol = base.read(OUT / 'protocol.json')
    assert all(base.sha(f) == h for f, h in protocol['protected_file_hashes'].items())
    selected = base.read(OUT / 'selection.json')
    base.save(OUT / 'api_started.json', {'started_at': base.stamp(), 'maximum_requests': 40, 'retries': 0})
    def execute(key, arm):
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--stage', 'one', '--path-key', key, '--arm', arm]
        completed = subprocess.run(command, capture_output=True, timeout=280,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        (case_directory(key, arm) / 'run_process.log').write_bytes(completed.stdout + completed.stderr)
        return key, arm, completed.returncode
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(execute, row['path_key'], arm) for index, row in enumerate(selected)
                for arm in (ARMS if index % 2 == 0 else ARMS[::-1])]
        for future in as_completed(jobs):
            key, arm, code = future.result()
            s = aggregate('running')
            print(json.dumps({'path': key, 'arm': arm, 'process_exit': code,
                'completed': s['completed_judgments'], 'metrics': s['metrics'], 'requests': s['api_requests']}, ensure_ascii=False), flush=True)
    s = aggregate('completed')
    assert s['api_requests'] <= 40
    assert all(base.sha(f) == h for f, h in protocol['protected_file_hashes'].items())
    print(json.dumps({k: v for k, v in s.items() if k not in ('path_results', 'paired_results')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run', 'one'), required=True)
    parser.add_argument('--path-key')
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'one': lambda: one(args.path_key, args.arm)}[args.stage]()
