"""Local HTTP entry points. Tasks and CLI share the monitor service."""
import asyncio
import json
import uuid
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .cloud_config import LOCK, ConfigError, public_settings, read_settings, save_settings, test_connection
from .config import SPLITS, load_config
from .runtime import ReplayRun


class RunRequest(BaseModel):
    subject: int = Field(default=1, ge=1, le=26)
    mode: Literal['fixed_p0', 'fixed_p1', 'rules', 'single_agent', 'hierarchical'] = 'hierarchical'
    backend: Literal['mock', 'api'] = 'mock'
    speed: float = Field(default=1, gt=0, le=1000)
    max_seconds: Optional[float] = Field(default=None, gt=0)
    task: str = Field(default='分析当前到达的EEG信号', max_length=2000)


class TaskRequest(BaseModel):
    subject: int = Field(default=1, ge=1, le=26)
    model_id: Literal['eegnet', 'eegconformer'] = 'eegnet'
    checkpoint_id: Optional[str] = Field(default=None, max_length=160)
    strategy: Literal['deep_only', 'adaptive', 'full_evidence'] = 'adaptive'
    backend: Literal['mock', 'api'] = 'mock'
    speed: float = Field(default=1, gt=0, le=1000)
    max_seconds: Optional[float] = Field(default=130, gt=0)
    start_sample: int = Field(default=0, ge=0)
    task: str = Field(default='分析VR晕动症模型输出与辅助EEG证据', max_length=2000)


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class BrainRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    backend: Literal['mock', 'api'] = 'api'
    domains: list[Literal['vrms', 'fatigue', 'emotion']] = Field(
        default_factory=lambda: ['vrms', 'fatigue', 'emotion'], min_length=1, max_length=3)
    task_id: Optional[str] = Field(default=None, max_length=80)
    recording_id: Optional[str] = Field(default=None, pattern=r'^(?:subject-\d{2}|edf-[0-9a-f]{12})$')
    input_policy: Optional[Literal['raw_eeg_only', 'monitor_task']] = None
    rag_mode: Literal['default', 'regular', 'graph', 'semantic_graph'] = 'default'


class KnowledgeDocumentRequest(BaseModel):
    domain: Literal['shared', 'vrms', 'fatigue', 'emotion']
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=80000)
    source: str = Field(min_length=1, max_length=1000)
    doi: str = Field(default='', max_length=200)
    page: str = Field(default='', max_length=100)


TERMINAL = {'completed', 'failed', 'stopped', 'interrupted', 'cancelled'}
ACTIVE = {'created', 'running', 'finalizing', 'stopping'}


def create_app(cfg=None, task_manager=None, knowledge_index=None, cloud_config_path=None,
               brain_manager=None):
    from .brain import BrainManager
    from .knowledge import KnowledgeIndex
    from .models import ONNXPredictor, model_catalog
    from .monitor import TaskManager
    from .tools import TOOL_CATALOG

    cfg = dict(cfg or load_config())
    if cloud_config_path is not None:
        cfg['cloud_config_path'] = str(cloud_config_path)
    config_path = cfg.get('cloud_config_path')
    app = FastAPI(title='VRMSAgent · 双模型本地监测', version='0.2.0')
    manager = task_manager if task_manager is not None else TaskManager(cfg)
    knowledge = knowledge_index if knowledge_index is not None else KnowledgeIndex(cfg)
    brain = brain_manager if brain_manager is not None else BrainManager(cfg, manager)
    runs = {}
    app.state.task_manager = manager
    app.state.knowledge_index = knowledge
    app.state.brain_manager = brain
    app.state.legacy_runs = runs
    app.state.cloud_testing = False
    static = Path(__file__).parent / 'web'
    app.mount('/assets', StaticFiles(directory=str(static)), name='assets')
    from .online.router import mount_online
    online = mount_online(app, cfg)

    def cloud_in_use():
        return (app.state.cloud_testing or brain.cloud_in_use() or online.cloud_in_use() or
                any(t['state'] in ACTIVE and t.get('request', {}).get('backend', t.get('backend')) == 'api'
                    for t in manager.list()) or
                any(run.state in ACTIVE and run.backend == 'api' for run in runs.values()))

    def check_source(subject, backend):
        if backend == 'api' and not public_settings(config_path)['configured']:
            raise HTTPException(400, '请先配置本地API Base URL、Model ID和密钥')
        if backend == 'api' and app.state.cloud_testing:
            raise HTTPException(409, '正在测试云端连接，请稍后启动任务')
        path = Path(cfg['data_root']) / 'data/raw' / ('Acquisition %02d.cdt' % subject)
        if not path.exists():
            raise HTTPException(400, '数据文件不存在')

    def find_task(task_id):
        try:
            task = manager.get(task_id)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, '未知任务')
        if task is None:
            raise HTTPException(404, '未知任务')
        return task

    def event_stream(item, request, after):
        last = request.headers.get('last-event-id')
        if last is not None:
            try:
                last_value = int(last)
            except ValueError:
                raise HTTPException(400, 'Last-Event-ID必须是整数')
            if last_value < -1:
                raise HTTPException(400, '事件ID必须不小于-1')
            after = max(after, last_value)
        if after < -1:
            raise HTTPException(400, '事件ID必须不小于-1')

        async def iterate():
            cursor = after
            while not await request.is_disconnected():
                with item.lock:
                    pending = [(int(e.get('sequence', i)), dict(e)) for i, e in enumerate(item.events)
                               if int(e.get('sequence', i)) > cursor]
                    state = item.state
                for seq, event in pending:
                    event.setdefault('sequence', seq)
                    yield 'id: %d\ndata: %s\n\n' % (seq, json.dumps(event, ensure_ascii=False, allow_nan=False))
                    cursor = seq
                if state in TERMINAL or state == 'idle':
                    break
                yield ': heartbeat\n\n'
                await asyncio.sleep(.25)

        return StreamingResponse(iterate(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.get('/')
    def index():
        return FileResponse(static / 'brain.html')

    @app.get('/monitor')
    def monitor_index():
        return FileResponse(static / 'index.html')

    @app.get('/rag-comparison')
    def rag_comparison_page():
        return FileResponse(static / 'brain.html')

    @app.get('/api/rag/comparison')
    def rag_comparison():
        path = Path(__file__).resolve().parents[1] / 'outputs/rag_assets/comparison_v1/comparison.json'
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {'available': False, 'status': 'pending', 'arms': [],
                    'notes': ['尚未发布普通 RAG 与 GraphRAG 的配对结果。本页只读取已保存结果，不触发判读。']}

    @app.get('/api/rag/version')
    def rag_version():
        from .default_rag import catalog_fields
        return catalog_fields()

    @app.get('/api/brain/catalog')
    def brain_catalog():
        return brain.catalog()

    def find_session(session_id):
        try:
            return brain.get(session_id)
        except KeyError:
            raise HTTPException(404, '未知对话')

    @app.post('/api/brain/sessions')
    def brain_create(request: BrainRequest):
        if len(set(request.domains)) != len(request.domains):
            raise HTTPException(400, '领域选择不能重复')
        with LOCK:
            if request.backend == 'api':
                if not public_settings(config_path)['configured']:
                    raise HTTPException(400, '请先在云端设置中配置接口、模型与密钥')
                if app.state.cloud_testing:
                    raise HTTPException(409, '云端连接测试正在进行，请稍后创建对话')
            try:
                return brain.create(request.model_dump())
            except KeyError:
                raise HTTPException(404, '关联的监测任务不存在')
            except ValueError as exc:
                raise HTTPException(400, str(exc))

    @app.get('/api/brain/sessions')
    def brain_list():
        return brain.list()

    @app.get('/api/brain/recordings')
    def brain_recordings():
        return {'recordings': brain.recordings.catalog(), 'input_policy': 'raw_eeg_only'}

    @app.post('/api/brain/recordings/edf')
    async def brain_upload_edf(request: Request, filename: str = 'recording.edf'):
        from .recordings import MAX_UPLOAD
        check_cloud_origin(request)
        if not filename.lower().endswith('.edf') or len(filename) > 200:
            raise HTTPException(400, '请提供原始 EDF 文件')
        if brain.cloud_in_use():
            raise HTTPException(409, '当前对话正在分析，请结束后再上传原始文件')
        temporary = brain.recordings.root / ('_upload_' + uuid.uuid4().hex + '.edf')
        try:
            size = 0
            with temporary.open('wb') as stream:
                async for block in request.stream():
                    size += len(block)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, 'EDF 文件须不超过512MB')
                    stream.write(block)
            if not size:
                raise HTTPException(400, 'EDF 文件为空')
            return brain.recordings.register_edf(temporary, filename)
        except (ValueError, OSError):
            raise HTTPException(400, '无法读取该 EDF；请检查原始格式和 EEG 通道名称')
        finally:
            temporary.unlink(missing_ok=True)

    @app.get('/api/brain/sessions/{session_id}')
    def brain_get(session_id: str):
        return find_session(session_id).snapshot()

    @app.get('/api/brain/sessions/{session_id}/events')
    def brain_events(session_id: str, request: Request, after: int = -1):
        return event_stream(find_session(session_id), request, after)

    @app.post('/api/brain/sessions/{session_id}/messages')
    def brain_message(session_id: str, request: MessageRequest):
        text = request.text.strip()
        if not text:
            raise HTTPException(400, '消息不能为空')
        with LOCK:
            session = find_session(session_id)
            if session.read_only:
                raise HTTPException(409, '旧研究入口已移除，此历史对话仅供查看；请新建原始 EEG 对话。')
            if session.backend == 'api':
                if app.state.cloud_testing:
                    raise HTTPException(409, '连接测试正在进行，请稍后发送消息')
                if not public_settings(config_path)['configured']:
                    raise HTTPException(400, '云端配置尚不完整')
            try:
                return session.message(text)
            except RuntimeError as exc:
                raise HTTPException(409, str(exc))
            except ValueError as exc:
                # Validation strings are local; provider errors are not exposed.
                raise HTTPException(400, str(exc))

    @app.get('/api/brain/sessions/{session_id}/report')
    def brain_report(session_id: str):
        return find_session(session_id).report() or {'status': 'pending', 'reason': '尚无完成的回答'}

    @app.get('/api/brain/knowledge/search')
    def brain_search(q: str = '', domains: str = 'vrms,fatigue,emotion', limit: int = 5):
        selected = [d.strip() for d in domains.split(',') if d.strip()]
        if any(d not in ('shared', 'vrms', 'fatigue', 'emotion') for d in selected):
            raise HTTPException(400, '未知知识库领域')
        return brain.knowledge.search(q[:2000], domains=selected, limit=max(1, min(limit, 5)))

    @app.post('/api/brain/knowledge/documents')
    def brain_document(request: Request, document: KnowledgeDocumentRequest):
        check_cloud_origin(request)
        try:
            return brain.knowledge.add_document(**document.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except OSError:
            raise HTTPException(500, '无法保存知识资料')

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'version': '0.2.0'}

    @app.get('/api/config')
    def settings():
        try:
            predictor = ONNXPredictor(cfg)
            model = {'available': True, 'hash': predictor.model_hash, 'profiles': predictor.profiles}
        except FileNotFoundError:
            model = {'available': False, 'reason': '未加载有效模型，预测不可用'}
        except Exception as exc:
            model = {'available': False, 'reason': type(exc).__name__}
        return {'model': model, 'models': model_catalog(cfg),
                'api_configured': public_settings(config_path)['configured'],
                'subjects': sorted(sum(SPLITS.values(), [])),
                'defaults': {'block_ms': 250, 'window_seconds': 5, 'score_threshold': 30,
                             'high_rule': '>=30', 'model_id': 'eegnet', 'strategy': 'adaptive'},
                'boundary': '模型输出不是即时症状真值；mock不代表真实云端收益；Pi性能另行实测。'}

    def check_cloud_origin(request):
        origin = request.headers.get('origin')
        if origin and urlsplit(origin).netloc != request.url.netloc:
            raise HTTPException(403, '云端设置仅接受当前网页的请求')

    @app.get('/api/cloud-settings')
    def get_cloud_settings(request: Request, include_keys: bool = False):
        check_cloud_origin(request)
        with LOCK:
            try:
                value = public_settings(config_path, in_use=cloud_in_use(), include_keys=include_keys)
                from fastapi.responses import JSONResponse
                return JSONResponse(value, headers={'Cache-Control': 'no-store'})
            except (OSError, ConfigError):
                raise HTTPException(500, '无法读取本地云端配置')

    @app.put('/api/cloud-settings')
    async def put_cloud_settings(request: Request):
        check_cloud_origin(request)
        if request.headers.get('content-type', '').split(';', 1)[0].strip() != 'application/json':
            raise HTTPException(400, '请使用JSON提交云端配置')
        try:
            body = await request.body()
            if len(body) > 16384:
                raise ValueError()
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            raise HTTPException(400, '云端配置JSON格式无效')
        with LOCK:
            if cloud_in_use():
                raise HTTPException(409, '云端配置正在被任务或连接测试使用，请结束后再修改')
            try:
                return save_settings(payload, config_path)
            except ConfigError as exc:
                raise HTTPException(400, str(exc))
            except OSError:
                raise HTTPException(500, '无法保存本地云端配置')

    @app.post('/api/cloud-settings/test')
    def test_cloud_settings(request: Request):
        check_cloud_origin(request)
        with LOCK:
            if cloud_in_use():
                raise HTTPException(409, '云端配置正在被任务或连接测试使用，请结束后再测试')
            app.state.cloud_testing = True
        try:
            return test_connection(config_path)
        except ConfigError as exc:
            raise HTTPException(400, str(exc))
        except OSError:
            raise HTTPException(500, '无法读取本地云端配置')
        finally:
            with LOCK:
                app.state.cloud_testing = False

    @app.post('/api/cloud-settings/test-input')
    async def test_cloud_input(request: Request):
        check_cloud_origin(request)
        if request.headers.get('content-type', '').split(';', 1)[0].strip() != 'application/json':
            raise HTTPException(400, '请使用JSON提交测试配置')
        try:
            body = await request.body()
            if len(body) > 16384:
                raise ValueError()
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError()
        except (ValueError, UnicodeError):
            raise HTTPException(400, '连接测试JSON格式无效')
        with LOCK:
            if cloud_in_use():
                raise HTTPException(409, '云端配置正在被任务或连接测试使用，请结束后再测试')
            app.state.cloud_testing = True
        try:
            return await asyncio.to_thread(test_connection, config_path, payload)
        except ConfigError as exc:
            raise HTTPException(400, str(exc))
        finally:
            with LOCK:
                app.state.cloud_testing = False

    @app.get('/api/models')
    def models():
        return model_catalog(cfg)

    @app.get('/api/tools')
    def tools():
        return TOOL_CATALOG

    @app.get('/api/knowledge/search')
    def search(q: str = '', limit: int = 5):
        if not q.strip():
            return {'query': q, 'matches': [], 'evidence_available': False}
        raw = knowledge.search(q[:2000], limit=max(1, min(limit, 5)))
        matches = raw.get('results', raw.get('evidence', [])) if isinstance(raw, dict) else raw
        return {'query': q, 'matches': matches, 'evidence_available': bool(matches),
                'status': raw.get('status') if isinstance(raw, dict) else ('matched' if matches else 'insufficient'),
                'reason': raw.get('reason') if isinstance(raw, dict) else None}

    @app.post('/api/tasks')
    def start_task(request: TaskRequest):
        with LOCK:
            check_source(request.subject, request.backend)
            if any(run.state in ACTIVE for run in runs.values()):
                raise HTTPException(409, '已有旧版回放正在运行')
            try:
                return manager.create(request.model_dump())
            except FileNotFoundError as exc:
                raise HTTPException(400, str(exc))
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            except RuntimeError as exc:
                raise HTTPException(409, str(exc))

    @app.get('/api/tasks')
    def task_list():
        return manager.list()

    @app.get('/api/tasks/{task_id}')
    def task_status(task_id: str):
        return find_task(task_id).snapshot()

    @app.get('/api/tasks/{task_id}/events')
    def task_events(task_id: str, request: Request, after: int = -1):
        return event_stream(find_task(task_id), request, after)

    @app.post('/api/tasks/{task_id}/stop')
    def task_stop(task_id: str):
        find_task(task_id).request_stop()
        return {'status': 'stop_requested'}

    @app.get('/api/tasks/{task_id}/report')
    def task_report(task_id: str):
        result = find_task(task_id).report()
        return result if result is not None else {'status': 'pending', 'reason': '报告尚未生成'}

    @app.post('/api/tasks/{task_id}/messages')
    def task_message(task_id: str, request: MessageRequest):
        if not request.text.strip():
            raise HTTPException(400, '追问不能为空')
        return find_task(task_id).message(request.text.strip())

    @app.get('/api/tasks/{task_id}/messages')
    def task_messages(task_id: str):
        task = find_task(task_id)
        with task.lock:
            return [e for e in task.events if e['kind'] in ('message', 'message_response') or
                    (e['kind'] == 'language_report' and e['payload'].get('phase') == 'followup')]

    # Legacy experiments remain available, using the same SSE resume contract.
    @app.post('/api/runs')
    def start(request: RunRequest):
        with LOCK:
            if any(run.state in ACTIVE for run in runs.values()) or any(t['state'] in ACTIVE for t in manager.list()):
                raise HTTPException(409, '已有回放正在运行')
            check_source(request.subject, request.backend)
            run = ReplayRun(cfg, **request.model_dump())
            runs[run.id] = run
            run.start()
            return run.snapshot()

    def legacy(run_id):
        if run_id not in runs:
            raise HTTPException(404, '未知回放')
        return runs[run_id]

    @app.get('/api/runs/{run_id}')
    def get(run_id: str):
        return legacy(run_id).snapshot()

    @app.post('/api/runs/{run_id}/stop')
    def stop(run_id: str):
        legacy(run_id).stop_event.set()
        return {'status': 'stop_requested'}

    @app.get('/api/runs/{run_id}/events')
    def stream(run_id: str, request: Request, after: int = -1):
        return event_stream(legacy(run_id), request, after)

    @app.get('/api/reports')
    def reports():
        root = Path(cfg['output_root']) / 'reports'
        result = {}
        for path in sorted(root.rglob('*.json')):
            try:
                result[str(path.relative_to(root).with_suffix(''))] = json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                continue
        return result

    return app
