"""One previously misclassified path, fresh local encoding, high-effort RAG probe.

Prepare is offline. Run makes at most two Responses requests, without retries.
No production configuration or previous evaluation artifacts are changed.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

APP = Path(__file__).resolve().parents[1]
RESEARCH = Path(r'D:\桌面\文献总\EEGAGENT')
sys.path[:0] = [str(APP), str(RESEARCH)]
from eeg_agent import agents, cloud_config, raw_tool_agent
from eeg_agent.config import load_config
from eeg_agent.raw_evidence_teaching import build_teaching_context
from scripts.evaluate_raw_independent_compare import SYSTEM as INDEPENDENT_SYSTEM, hidden_keys

SOURCE = RESEARCH / 'outputs/vrms_raw_gpt_evaluations/20261009_current_allpaths_01_recovered'
PREVIOUS = RESEARCH / 'outputs/vrms_raw_cloud_comparisons/20261010_independent_evidence_01'
OUT = RESEARCH / 'outputs/vrms_rag_probes/20261010_gpt_high_errorpath_01'
PATH_KEY, FOLD, ORDINAL = '08:2047409', 8, 7
WINDOWS = APP / 'outputs/vrms_v2/brain_recordings/inference/1deddcf485d94fb39ec7ac45755d681b/segment-007.npy'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def wsl(path):
    resolved = Path(path).resolve()
    return '/mnt/' + resolved.drive[0].lower() + resolved.as_posix()[2:]


class ProbeRuntime(raw_tool_agent.EvidenceRuntime):
    def initial_evidence(self):
        value = super().initial_evidence()
        value['tool_validation_profiles'] = {name: profile for name, profile in
            value['tool_validation_profiles'].items() if name in raw_tool_agent.TOOL_NAMES}
        value['numerical_execution'] = getattr(self, 'numerical_execution',
            'fresh_encoding_of_same_saved_raw_derived_windows; no_fresh_preprocessing')
        assert not hidden_keys(value)
        return value

    def execute(self, name):
        value = super().execute(name)
        if name == 'corrective_evidence':
            for key in ('raw_combined_class', 'uncalibrated_combined_score',
                        'score_is_calibrated', 'internal_validation'):
                value.pop(key, None)
            value['scope'] = 'whole-path; eight individual results; fused prediction withheld'
        assert not hidden_keys(value)
        self.memo[name] = copy.deepcopy(value)
        return value


def prepare():
    if OUT.exists():
        raise FileExistsError('Preserve the existing probe; never duplicate a paid run')
    saved = read(SOURCE / 'predictions/subject_08.json')['result']
    prior = next(row for row in saved['measurements']['segments'] if row['ordinal'] == ORDINAL)
    old_worker = read(WINDOWS.parent / 'worker_result.json')
    matching = next(row for row in old_worker['segments'] if row['ordinal'] == ORDINAL)
    old_request = read(WINDOWS.parent / 'request.json')
    assert old_request['fold'] == FOLD and old_request['baseline_file'] is False
    assert old_request['signal_sha256'] == saved['signal_sha256']
    assert matching['mil_raw_high_probability'] == prior['mil_raw_high_probability']
    assert sha(WINDOWS) == '969676f7822df699c8890fb2e5482785a8d765240a7e068ae3c3082243e67986'
    OUT.mkdir(parents=True)
    worker_dir = OUT / 'local_worker'
    worker_dir.mkdir()
    shutil.copyfile(WINDOWS, worker_dir / 'segment-007.npy')
    request = {'fold': FOLD, 'signal_sha256': saved['signal_sha256'],
               'segments': [{'ordinal': ORDINAL, 'file': 'segment-007.npy'}], 'baseline_file': False}
    save(worker_dir / 'request.json', request)
    protected = [APP / '.env', APP / 'cloud_profiles.json', PREVIOUS / 'summary.json', PREVIOUS / 'path_results.csv']
    save(OUT / 'protocol.json', {'schema': 'single_errorpath_high_rag_probe_v1', 'created_at': stamp(),
        'path_key': PATH_KEY, 'subject': FOLD, 'ordinal': ORDINAL, 'paths': 1,
        'profile_id': 'cpa_gpt', 'requested_model': 'gpt-6.1-sol', 'reasoning_effort': 'high',
        'maximum_api_requests': 2, 'maximum_output_tokens_per_request': 8192,
        'api_timeout_seconds': 120, 'automatic_retries': 0,
        'fused_class_withheld': True, 'fused_score_withheld': True,
        'query_truth_sent': False, 'previous_cloud_answer_sent': False,
        'fresh_preprocessing': False, 'fresh_local_encoding': True,
        'saved_window_sha256': sha(WINDOWS), 'selection': 'one known error, functional development probe',
        'protected_file_hashes': {str(path): sha(path) for path in protected}})
    command = ['wsl.exe', '-d', 'Ubuntu-22.04-Bio', '--cd', wsl(RESEARCH), '--exec',
               '/opt/miniconda3/envs/pytorch/bin/python', '-B', '-m', 'vrms_cloud.raw_inference',
               wsl(worker_dir / 'request.json')]
    completed = subprocess.run(command, capture_output=True, timeout=240,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    (worker_dir / 'stderr.log').write_bytes(completed.stderr)
    worker = json.loads(completed.stdout.decode('utf-8'))
    save(worker_dir / 'result.json', worker)
    if not worker.get('available') or len(worker['segments']) != 1:
        raise RuntimeError('Single-path local inference unavailable; no API requests made')
    prediction = worker['segments'][0]
    assert prediction['fresh_encoded_windows'] == prior['accepted_windows'] == 26
    assert abs(prediction['mil_raw_high_probability'] - prior['mil_raw_high_probability']) < 1e-8
    assert abs(prediction['numeric_fusion']['raw_combined_score'] - prior['numeric_fusion']['raw_combined_score']) < 1e-8
    context = build_teaching_context()
    runtime = ProbeRuntime(prediction, teaching_context=context)
    initial = runtime.initial_evidence()
    tools = runtime.execute('corrective_evidence')
    cases = next(row for row in tools['actual_individual_evidence'] if row['tool'] == 'case_retrieval')['training_case_context']
    assert context['available'] and len(context['input_definitions']) == 9
    assert cases['available'] and len(cases['cases']) == 12 and len(cases['detailed_cases']) == 3
    save(OUT / 'prediction.json', prediction)
    save(OUT / 'offline_preview.json', {'initial_evidence': initial, 'corrective_evidence': tools})
    save(OUT / 'offline_checks.json', {'paths': 1, 'fresh_encoded_windows': 26,
        'same_mil_and_fusion_within_1e_8': True, 'nine_input_teaching_complete': True,
        'retrieved_internal_subjects': 12, 'detailed_cases': 3,
        'cloud_payload_hidden_keys': [], 'api_requests': 0})
    print(json.dumps({'stage': 'offline_ready', 'path': PATH_KEY, 'fresh_windows': 26,
                      'teaching_inputs': 9, 'cases': 12, 'detailed_cases': 3, 'api_requests': 0}))


def run():
    protocol = read(OUT / 'protocol.json')
    if (OUT / 'api_started.json').exists():
        raise RuntimeError('This probe was already attempted; no automatic duplicate API calls')
    read(OUT / 'offline_checks.json')
    prediction = read(OUT / 'prediction.json')
    teaching = build_teaching_context()
    ProbeRuntime.numerical_execution = ('fresh_preprocessing_and_encoding_before_cloud; callable_immutable_evidence'
        if protocol['fresh_preprocessing'] else
        'fresh_encoding_of_same_saved_raw_derived_windows; no_fresh_preprocessing')
    runtime = ProbeRuntime(prediction, teaching_context=teaching)
    preview = runtime.execute('corrective_evidence')
    cases = next(row for row in preview['actual_individual_evidence'] if row['tool'] == 'case_retrieval')['training_case_context']
    case_map = {row['case_id']: row for row in cases['detailed_cases']}
    citation_ids = [row['citation_id'] for row in teaching['input_definitions'] + teaching['judgment_principles']]
    schema = copy.deepcopy(raw_tool_agent.SCHEMA)
    schema['properties']['evidence_use_audit'] = {'type': 'object', 'additionalProperties': False,
        'properties': {'teaching_citation_ids': {'type': 'array', 'items': {'type': 'string', 'enum': citation_ids}},
            'case_evidence': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                'properties': {'case_id': {'type': 'string', 'enum': list(case_map)},
                    'training_class': {'type': 'string', 'enum': ['High', 'Low']},
                    'mil_raw_high_probability': {'type': 'number'}, 'interpretation': {'type': 'string'}},
                'required': ['case_id', 'training_class', 'mil_raw_high_probability', 'interpretation']}},
            'effect': {'type': 'string', 'enum': ['supports_final_class', 'conflicts_with_final_class',
                                                'insufficient_to_change_decision']},
            'summary': {'type': 'string'}},
        'required': ['teaching_citation_ids', 'case_evidence', 'effect', 'summary']}
    schema['required'].append('evidence_use_audit')
    system = INDEPENDENT_SYSTEM + '''
Read evidence_teaching before judging the nine inputs. It explains score scales,
calibration thresholds, shared features and conflicts. Training_case_context is
an expanded set of anonymous real INTERNAL training references: not query truth,
not OOF validation, and not freshly computed query waveforms. Use detailed case
vectors only as bounded demonstrations. Case labels/distances are not query probabilities.
In evidence_use_audit cite at least one supplied teaching citation and one detailed
case actually inspected; copy its training class and MIL probability, and explain
briefly how it supports, conflicts with, or is insufficient for the final decision.
Be honest if the additional evidence is unhelpful. Do not force either final class.
This audit is a concise evidence-use explanation, not private chain-of-thought.
'''
    profile = cloud_config._profile_state(None)['profiles']['cpa_gpt']
    assert profile['model_id'] == protocol['requested_model'] and profile['api_key']
    agents.cloud_settings = lambda path=None: {name: profile[field] for field, name in cloud_config.NAMES.items()}
    cfg = load_config()
    cfg['api_timeout_seconds'], cfg['api_max_calls'] = 120, 2
    provider = agents.CloudProvider('api', cfg, lambda kind, payload: None)
    raw_tool_agent.EvidenceRuntime, raw_tool_agent.SYSTEM, raw_tool_agent.SCHEMA = ProbeRuntime, system, schema
    raw_tool_agent.TOOLS = copy.deepcopy(raw_tool_agent.TOOLS)
    for tool in raw_tool_agent.TOOLS:
        tool['description'] = ('Read all eight individual tool results and expanded training references; no fused result.'
            if tool['name'] == 'corrective_evidence' else 'Read one immutable local numerical result and its context.')
    original_create = provider.client.responses.create
    request_records = []

    def create(**kwargs):
        sequence = len(request_records) + 1
        if sequence > 2:
            raise RuntimeError('Single-path request budget exhausted')
        kwargs['reasoning'] = {'effort': 'high'}
        kwargs['max_output_tokens'] = 8192
        if sequence == 1:
            kwargs['tool_choice'] = {'type': 'function', 'name': 'corrective_evidence'}
        else:
            kwargs.pop('tools', None)
            kwargs.pop('tool_choice', None)
            kwargs.pop('parallel_tool_calls', None)
        assert not hidden_keys(kwargs['input'])
        record = {'sequence': sequence, 'started_at': stamp(), 'request': copy.deepcopy(kwargs)}
        request_records.append(record)
        save(OUT / f'request_{sequence}.json', record)
        provider.calls[-1].update(output_token_budget=8192, reasoning_effort='high')
        try:
            response = original_create(**kwargs)
            record.update(finished_at=stamp(), response_id=response.id, response_model=response.model,
                          usage=response.usage.model_dump(mode='json') if response.usage else None)
            save(OUT / f'response_{sequence}.json', response.model_dump(mode='json'))
            return response
        except Exception as exc:
            record.update(finished_at=stamp(), error_type=type(exc).__name__)
            raise
        finally:
            save(OUT / f'request_{sequence}.json', record)

    provider.client.responses.create = create
    save(OUT / 'api_started.json', {'started_at': stamp(), 'maximum_requests': 2, 'automatic_retries': 0})
    judgment = raw_tool_agent.classify(provider, prediction, teaching_context=teaching)
    save(OUT / 'judgment.json', judgment)
    save(OUT / 'api_usage.json', provider.summary())
    usage_verified = False
    if judgment['available']:
        audit = judgment['evidence_use_audit']
        assert audit['teaching_citation_ids'] and set(audit['teaching_citation_ids']).issubset(citation_ids)
        assert audit['case_evidence']
        for cited in audit['case_evidence']:
            actual = case_map[cited['case_id']]
            assert cited['training_class'] == actual['training_class']
            assert abs(cited['mil_raw_high_probability'] - actual['mil_raw_high_probability']) < 5e-6
            assert cited['interpretation'].strip()
        usage_verified = True
    prior = next(row for row in csv.DictReader((PREVIOUS / 'path_results.csv').open(encoding='utf-8-sig', newline=''))
                 if row['path_key'] == PATH_KEY)
    protected = all(sha(path) == expected for path, expected in protocol['protected_file_hashes'].items())
    assert protected
    summary = {'status': 'completed' if judgment['available'] else 'failed_without_retry',
        'path_key': PATH_KEY, 'requested_model': provider.model, 'reasoning_effort': 'high',
        'previous_gpt_class': prior['gpt_class'], 'true_class': prior['true_class'],
        'new_gpt_class': judgment['final_class'],
        'corrected_previous_error': bool(judgment['available'] and judgment['final_class'] == prior['true_class']),
        'new_evidence_use_verified': usage_verified, 'teaching_inputs_sent': len(teaching['input_definitions']),
        'training_subject_cases_sent': len(cases['cases']), 'detailed_cases_sent': len(cases['detailed_cases']),
        'api_requests': len(request_records), 'api_failures': provider.summary()['failures'],
        'usage': provider.summary(), 'response_ids': [r.get('response_id') for r in request_records],
        'response_models': [r.get('response_model') for r in request_records],
        'fused_answer_hidden': True, 'query_truth_hidden': True, 'protected_files_unchanged': protected,
        'fresh_encoded_windows': prediction['fresh_encoded_windows'],
        'fresh_preprocessing': protocol['fresh_preprocessing'], 'training_calls': 0,
        'limitation': 'One selected known error; high effort and RAG both changed; not ACC or causal improvement proof.'}
    save(OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run'), required=True)
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else run()
