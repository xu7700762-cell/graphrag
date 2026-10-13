"""Batch scoring/HTTP contracts with a new-file worker double; no cloud calls."""
import copy
import json
import subprocess
import threading
import time

import pytest
from fastapi.testclient import TestClient

from eeg_agent.cloud_config import save_settings
from eeg_agent.config import load_config
from eeg_agent.service import create_app
from eeg_agent.vrms_evaluation import EvaluationManager

SECRET = 'TEST_PROJECT_BATCH_SECRET'


def summary(payload, completed=146, cloud_failed=0):
    valid = completed - cloud_failed
    return {'id': payload['evaluation_id'], 'seed': payload['seed'],
        'pipeline_id': 'femba_mil_tools_gpt_seed' + str(payload['seed']),
        'model_id': payload['provider']['model'], 'state': 'completed', 'phase': 'completed',
        'total': 146, 'completed': completed, 'cloud_completed': valid, 'cloud_failed': cloud_failed,
        'metrics': {name: {'correct': correct, 'total': 146, 'accuracy': correct / 146,
                    **({'coverage': valid / 146, 'conditional_accuracy': correct / valid if valid else None}
                       if name == 'gpt' else {})}
                    for name, correct in [('mil', 89), ('numeric_fusion', 107), ('gpt', min(101, valid))]},
        'api_key': SECRET, 'error': 'provider text ' + SECRET}


class WorkerFactory:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.payloads = []
        self.release = threading.Event()
        self.release.set()
        self.change = None
        self.returncode = 0
        self.publish = True

    def __call__(self, command, **options):
        self.calls.append((command, options))
        factory = self
        class Process:
            pid = 999999
            returncode = None
            closed = False
            payload = None
            class Input:
                def __init__(inner, process):
                    inner.text = ''
                    inner.process = process
                def write(inner, value):
                    inner.text += value
                def close(inner):
                    inner.process.payload = json.loads(inner.text)
                    factory.payloads.append(copy.deepcopy(inner.process.payload))
                    inner.process.closed = True
            def __init__(self):
                self.stdin = self.Input(self)
            def poll(self):
                if self.returncode is not None:
                    return self.returncode
                if not self.closed or not factory.release.is_set():
                    return None
                if factory.publish:
                    value = summary(self.payload)
                    if factory.change:
                        factory.change(value)
                    directory = factory.root / 'outputs' / 'vrms_web_evaluations' / self.payload['evaluation_id']
                    directory.mkdir(parents=True)
                    (directory / 'progress.json').write_text(json.dumps(value), encoding='utf-8')
                    (directory / 'summary.json').write_text(json.dumps(value), encoding='utf-8')
                self.returncode = factory.returncode
                return self.returncode
            def terminate(self):
                self.returncode = -15
            def wait(self, timeout=None):
                return self.returncode
        return Process()


class Tasks:
    def list(self):
        return []
    def get(self, task_id):
        raise KeyError(task_id)


@pytest.fixture
def offline(tmp_path, monkeypatch):
    for name in ('EEG_API_BASE_URL', 'EEG_API_MODEL', 'EEG_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    cfg = load_config()
    cfg['output_root'] = str(tmp_path / 'web')
    cfg['cloud_config_path'] = str(tmp_path / '.env')
    worker_root = tmp_path / 'research'
    worker_root.mkdir()
    python = tmp_path / 'python.exe'
    python.touch()
    factory = WorkerFactory(worker_root)
    evaluations = EvaluationManager(cfg, worker_root, python, factory, poll_interval=.01)
    app = create_app(cfg, task_manager=Tasks(), knowledge_index=object())
    save_settings({'base_url': 'https://api.deepseek.com', 'model_id': 'deepseek-flash', 'api_key': SECRET}, cfg['cloud_config_path'])
    yield TestClient(app), evaluations, factory, cfg
    factory.release.set()
    deadline = time.monotonic() + 2
    while evaluations.cloud_in_use() and time.monotonic() < deadline:
        time.sleep(.01)


def finish(evaluations, evaluation_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        value = evaluations.get(evaluation_id)
        if value['terminal']:
            return value
        time.sleep(.01)
    pytest.fail('batch worker did not finish')


def start(evaluations, seed=2026, **options):
    provider = {'base_url': 'https://api.deepseek.com', 'model': 'deepseek-flash', 'api_key': SECRET}
    return evaluations.start(seed, provider, 'deepseek', **options)['id']


@pytest.mark.parametrize('seed', [2026, 2028])
def test_fresh_batch_routes_fixed_counts_and_secret_only_in_stdin(offline, seed):
    client, evaluations, factory, cfg = offline
    evaluation_id = start(evaluations, seed)
    value = finish(evaluations, evaluation_id)
    assert value['state'] == 'completed' and value['completed'] == 146 and value['cloud_completed'] == 146
    assert value['metrics']['mil']['accuracy'] == 89 / 146
    assert value['metrics']['numeric_fusion']['accuracy'] == 107 / 146
    assert value['metrics']['gpt']['accuracy'] == 101 / 146
    assert evaluations.list()[0]['id'] == evaluation_id
    encoded = json.dumps(evaluations.get(evaluation_id))
    assert SECRET not in encoded and 'api_key' not in encoded
    command, options = factory.calls[0]
    assert command[-2:] == ['-m', 'vrms_cloud.web_evaluation']
    assert SECRET not in str(command) and options['stdout'] == options['stderr'] == subprocess.DEVNULL
    assert options['stdin'] == subprocess.PIPE
    assert factory.payloads[0]['provider']['api_key'] == SECRET
    assert factory.payloads[0]['seed'] == seed
    assert SECRET not in (evaluations.root / evaluation_id / 'evaluation.json').read_text(encoding='utf-8')
    assert SECRET not in json.dumps(value)
    restored = EvaluationManager(cfg, factory.root, evaluations.worker_python, factory)
    assert restored.get(evaluation_id)['metrics'] == value['metrics']
    assert not restored.cloud_in_use()




def test_failed_cloud_paths_keep_full_denominator_and_report_coverage(offline):
    client, evaluations, factory, _ = offline
    def change(value):
        value.update(cloud_completed=144, cloud_failed=2)
        value['metrics']['gpt'].update(coverage=144 / 146, conditional_accuracy=101 / 144)
    factory.change = change
    value = finish(evaluations, start(evaluations))
    assert value['state'] == 'completed' and value['generation_status'] == 'partial_fallback'
    assert value['metrics']['gpt']['total'] == 146 and value['metrics']['gpt']['accuracy'] == 101 / 146
    assert value['cloud_failed'] == 2 and value['metrics']['gpt']['coverage'] == 144 / 146


@pytest.mark.parametrize('failure', ['nonzero', 'missing_summary', 'wrong_id', 'wrong_seed',
                                    'partial', 'wrong_denominator', 'wrong_accuracy', 'wrong_coverage'])
def test_process_or_result_failure_cannot_publish_success_or_final_accuracy(offline, failure):
    client, evaluations, factory, _ = offline
    if failure == 'nonzero':
        factory.returncode = 1
    elif failure == 'missing_summary':
        factory.publish = False
    else:
        def change(value):
            if failure == 'wrong_id':
                value['id'] = 'old-history'
            elif failure == 'wrong_seed':
                value['seed'] = 2028
            elif failure == 'partial':
                value.update(completed=145, cloud_completed=145)
            elif failure == 'wrong_denominator':
                value['metrics']['gpt']['total'] = 144
            elif failure == 'wrong_accuracy':
                value['metrics']['gpt']['accuracy'] = .99
            elif failure == 'wrong_coverage':
                value['metrics']['gpt']['coverage'] = .5
        factory.change = change
    evaluation_id = start(evaluations)
    value = finish(evaluations, evaluation_id)
    assert value['state'] == 'failed' and value['generation_status'] == 'fallback'
    assert all(m['accuracy'] is None for m in value['metrics'].values())
    assert not evaluations.cloud_in_use()
    assert SECRET not in json.dumps(value)




def test_offline_batch_guards_duplicate_jobs_and_withholds_pending_accuracy(offline):
    _, evaluations, factory, _ = offline
    factory.release.clear()
    evaluation_id = start(evaluations)
    assert evaluations.cloud_in_use()
    with pytest.raises(RuntimeError):
        start(evaluations)
    assert all(v['accuracy'] is None for v in evaluations.get(evaluation_id)['metrics'].values())


def test_offline_resume_preserves_original_configuration_and_reference(offline):
    _, evaluations, factory, _ = offline
    factory.returncode = 1
    original = start(evaluations)
    assert finish(evaluations, original)['state'] == 'failed'
    with pytest.raises(ValueError):
        start(evaluations, seed=2028, resume_from=original)
    factory.returncode = 0
    resumed = start(evaluations, resume_from=original)
    assert resumed != original and finish(evaluations, resumed)['state'] == 'completed'
    assert evaluations.get(original)['state'] == 'failed'
    assert factory.payloads[-1]['resume_from'] == original


def test_removed_web_evaluation_and_catalog_routes_cannot_start_a_worker(offline):
    client, _, factory, _ = offline
    for url in ['/api/brain/vrms/catalog', '/api/brain/vrms/evaluations', '/api/brain/vrms/evaluations/missing', '/api/brain/vrms/evaluations/missing/report']:
        assert client.get(url).status_code == 404
    assert client.post('/api/brain/vrms/evaluations', json={'seed': 2026}).status_code == 404
    assert not factory.calls
