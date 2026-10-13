"""Fresh, bounded VRMS batch jobs with immutable provider snapshots.

The research worker owns inference and scoring. The web process only launches
it and verifies its new progress/summary; it never derives ACC from chat text.
"""
import copy
import json
import math
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .monitor import atomic_json
from . import vrms_runtime

TOTAL = 146
ACTIVE = {'queued', 'running'}
STATES = ACTIVE | {'completed', 'failed'}


def _count(value, limit=TOTAL):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= limit:
        raise ValueError('invalid evaluation count')
    return value


def _ratio(value, expected):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or abs(value - expected) > 1e-10):
        raise ValueError('inconsistent evaluation metric')
    return float(value)


def _empty_metrics():
    return {name: {'correct': 0, 'total': TOTAL, 'accuracy': None,
                  **({'coverage': 0, 'conditional_accuracy': None} if name == 'gpt' else {})}
            for name in ('mil', 'numeric_fusion', 'gpt')}


def _worker_result(value, job, final=False):
    """Allowlisted counts only; unknown provider messages never reach the UI."""
    if (not isinstance(value, dict) or value.get('id') != job['id'] or
            value.get('seed') != job['seed'] or value.get('total') != TOTAL or
            value.get('model_id') != job['model_id'] or value.get('pipeline_id') != job['pipeline_id']):
        raise ValueError('evaluation provenance mismatch')
    completed = _count(value.get('completed'))
    cloud_completed = _count(value.get('cloud_completed'))
    cloud_failed = _count(value.get('cloud_failed'))
    if cloud_completed + cloud_failed > completed:
        raise ValueError('inconsistent cloud counts')
    if final and (value.get('state') != 'completed' or completed != TOTAL or
                  cloud_completed + cloud_failed != TOTAL):
        raise ValueError('incomplete evaluation summary')
    metrics = value.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError('missing evaluation metrics')
    selected = {}
    for name in ('mil', 'numeric_fusion', 'gpt'):
        metric = metrics.get(name)
        if not isinstance(metric, dict) or metric.get('total') != TOTAL:
            raise ValueError('incorrect evaluation denominator')
        correct = _count(metric.get('correct'))
        if name == 'gpt' and correct > cloud_completed:
            raise ValueError('correct cloud count exceeds valid replies')
        selected[name] = {'correct': correct, 'total': TOTAL,
                          'accuracy': _ratio(metric.get('accuracy'), correct / TOTAL) if final else None}
        if name == 'gpt':
            selected[name]['coverage'] = cloud_completed / TOTAL
            selected[name]['conditional_accuracy'] = None
            if final:
                _ratio(metric.get('coverage'), cloud_completed / TOTAL)
                if cloud_completed:
                    selected[name]['conditional_accuracy'] = _ratio(metric.get('conditional_accuracy'), correct / cloud_completed)
                elif metric.get('conditional_accuracy') is not None:
                    raise ValueError('undefined conditional accuracy')
    return {'completed': completed, 'cloud_completed': cloud_completed,
            'cloud_failed': cloud_failed, 'metrics': selected,
            'phase': value.get('phase') if value.get('phase') in ('preflight', 'evaluating', 'completed') else 'evaluating'}


def _read_json(path):
    if path.stat().st_size > 4_000_000:
        raise ValueError('evaluation file too large')
    return json.loads(path.read_text(encoding='utf-8'))


def _worker_alive(pid, created=None):
    try:
        import psutil
        process = psutil.Process(pid)
        if created is not None and abs(process.create_time() - created) > 1:
            return False
        return process.is_running() and 'vrms_cloud.web_evaluation' in process.cmdline()
    except Exception:
        return False


def _terminate_worker(process):
    # The evaluation worker owns a process pool. Cleanup must stop that owned
    # tree as well as its parent if the web launcher encounters a fatal error.
    children = []
    try:
        import psutil
        children = psutil.Process(process.pid).children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
    except Exception:
        pass
    process.terminate()
    try:
        process.wait(timeout=5)
    except Exception:
        pass
    if children:
        try:
            import psutil
            _, alive = psutil.wait_procs(children, timeout=2)
            for child in alive:
                child.kill()
        except Exception:
            pass


class EvaluationManager:
    def __init__(self, cfg, worker_root=None, worker_python=None, popen_factory=None, poll_interval=.5):
        self.root = Path(cfg['output_root']) / 'vrms_evaluations'
        self.worker_root = Path(worker_root or vrms_runtime.WORKER_ROOT)
        self.worker_python = Path(worker_python or vrms_runtime.WORKER_PYTHON)
        self.worker_outputs = self.worker_root / 'outputs' / 'vrms_web_evaluations'
        self.popen = popen_factory or subprocess.Popen
        self.poll_interval = poll_interval
        self.lock = threading.RLock()
        self.jobs = {}
        self.root.mkdir(parents=True, exist_ok=True)
        for file in self.root.glob('*/evaluation.json'):
            try:
                job = _read_json(file)
                if (not re.fullmatch(r'[0-9a-f]{32}', file.parent.name) or job.get('id') != file.parent.name or
                        job.get('seed') not in (2026, 2028) or job.get('state') not in STATES):
                    continue
                self.jobs[job['id']] = job
                if job['state'] in ACTIVE:
                    if _worker_alive(job.get('_pid'), job.get('_process_created')):
                        job['state'] = 'running'
                        threading.Thread(target=self._observe_orphan, args=(job['id'],), daemon=True).start()
                    else:
                        self._fail(job['id'], '服务重启后未找到可验证的评估进程，本轮完整ACC未发布。', 'interrupted')
            except (OSError, ValueError, TypeError):
                continue

    def _save(self, job):
        atomic_json(self.root / job['id'] / 'evaluation.json', job)

    def _public(self, job):
        value = copy.deepcopy({k: v for k, v in job.items() if not k.startswith('_')})
        value.update(status=job['state'], processed=job['completed'],
                     terminal=job['state'] not in ACTIVE, progress_percent=100 * job['completed'] / TOTAL)
        if job['state'] == 'completed':
            value['report_url'] = '/api/brain/vrms/evaluations/' + job['id'] + '/report'
        return value

    def list(self):
        with self.lock:
            return [self._public(job) for job in sorted(self.jobs.values(), key=lambda j: j['started_at'], reverse=True)]

    def get(self, evaluation_id):
        with self.lock:
            return self._public(self.jobs[evaluation_id])

    def cloud_in_use(self):
        with self.lock:
            return any(job['state'] in ACTIVE for job in self.jobs.values())

    def start(self, seed, provider, active_profile=None, resume_from=None):
        if seed not in (2026, 2028):
            raise ValueError('批量评估仅支持seed2026或seed2028')
        if (not isinstance(provider, dict) or provider.get('model') not in vrms_runtime.SUPPORTED_MODELS or
                not isinstance(provider.get('api_key'), str) or not provider['api_key'].strip() or
                not isinstance(provider.get('base_url'), str)):
            raise ValueError('请配置已登记的DeepSeek或CPA GPT接口及独立密钥')
        if not self.worker_root.is_dir() or not self.worker_python.is_file():
            raise ValueError('VRMS 模型批量评估工作环境不可用')
        with self.lock:
            if self.cloud_in_use():
                raise RuntimeError('已有批量评估正在运行，请等待当前任务完成')
            if resume_from is not None:
                previous = self.jobs.get(resume_from)
                if (previous is None or previous['seed'] != seed or previous['model_id'] != provider['model'] or
                        previous.get('active_profile') != active_profile or previous['state'] not in ('failed', 'completed') or
                        previous['state'] == 'completed' and not previous['cloud_failed']):
                    raise ValueError('只能续跑同种子和同云端配置的未完成评价')
            evaluation_id = uuid.uuid4().hex
            job = {'id': evaluation_id, 'seed': seed,
                   'pipeline_id': 'femba_mil_tools_gpt_seed' + str(seed),
                   'model_id': provider['model'], 'active_profile': active_profile,
                   'state': 'queued', 'phase': 'preflight', 'total': TOTAL, 'completed': 0,
                   'cloud_completed': 0, 'cloud_failed': 0, 'metrics': _empty_metrics(),
                   'started_at': time.time(), 'generation_status': 'unverified', 'error': None}
            if resume_from:
                job['resume_from'] = resume_from
            self.jobs[evaluation_id] = job
            self._save(job)
            payload = {'seed': seed, 'provider': dict(provider), 'evaluation_id': evaluation_id}
            if resume_from:
                payload['resume_from'] = resume_from
            threading.Thread(target=self._run, args=(evaluation_id, payload),
                             daemon=True, name='vrms-evaluation-' + evaluation_id[:8]).start()
            return self._public(job)

    def _fail(self, evaluation_id, reason, error_code):
        with self.lock:
            job = self.jobs[evaluation_id]
            job.update(state='failed', generation_status='fallback', error=reason,
                       error_code=error_code, finished_at=time.time())
            # Incomplete or unverified runs never publish final accuracy.
            for metric in job['metrics'].values():
                metric['accuracy'] = None
            self._save(job)

    def _progress(self, evaluation_id):
        try:
            value = _read_json(self.worker_outputs / evaluation_id / 'progress.json')
            with self.lock:
                job = self.jobs[evaluation_id]
                fields = _worker_result(value, job)
                if fields['completed'] < job['completed']:
                    return
                if any(job.get(k) != v for k, v in fields.items()):
                    job.update(fields)
                    self._save(job)
        except (OSError, ValueError, TypeError, KeyError):
            # The worker atomically publishes progress. Missing/invalid data
            # is never accepted as success; the final summary is strict.
            pass

    def _run(self, evaluation_id, payload):
        process = None
        try:
            process = self.popen([str(self.worker_python), '-B', '-m', 'vrms_cloud.web_evaluation'],
                cwd=str(self.worker_root), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            with self.lock:
                job = self.jobs[evaluation_id]
                job.update(state='running', _pid=process.pid)
                try:
                    import psutil
                    job['_process_created'] = psutil.Process(process.pid).create_time()
                except Exception:
                    pass
                self._save(job)
            # Credentials only enter the worker through its private stdin.
            process.stdin.write(json.dumps(payload, ensure_ascii=False, allow_nan=False))
            process.stdin.close()
            payload.clear()
            while process.poll() is None:
                self._progress(evaluation_id)
                time.sleep(self.poll_interval)
            self._progress(evaluation_id)
            if process.returncode != 0:
                self._fail(evaluation_id, '批量评估工作进程失败，未发布本轮完整ACC。', 'worker_failed')
                return
            value = _read_json(self.worker_outputs / evaluation_id / 'summary.json')
            with self.lock:
                job = self.jobs[evaluation_id]
                fields = _worker_result(value, job, final=True)
                job.update(fields, state='completed', phase='completed', error=None, _exit_code=0,
                    generation_status='cloud_verified' if fields['cloud_failed'] == 0 else 'partial_fallback',
                    finished_at=time.time())
                self._save(job)
        except Exception:
            self._fail(evaluation_id, '本轮评估未取得可验证的完整结果，请检查本地工作环境与接口。', 'result_unverified')
        finally:
            payload.clear()
            if process is not None and process.poll() is None:
                _terminate_worker(process)

    def _observe_orphan(self, evaluation_id):
        # Keep the provider lock while the original worker still runs. A new
        # web process cannot recover its parent's exit code, so it cannot
        # promote that orphan's report to a verified successful web run.
        while True:
            with self.lock:
                job = self.jobs[evaluation_id]
                alive = _worker_alive(job.get('_pid'), job.get('_process_created'))
            if not alive:
                break
            self._progress(evaluation_id)
            time.sleep(self.poll_interval)
        self._fail(evaluation_id, '网页服务重启后无法验证原评估进程的退出状态，完整ACC未发布。', 'exit_unverified')
