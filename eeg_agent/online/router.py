"""Additive FastAPI routes; existing offline request models remain unchanged."""
import asyncio
import json
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .cloud import failure_reason
from .runtime import OnlineManager, OnlineSession, TERMINAL


class OnlineRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    recording_id: str = Field(pattern=r'^subject-\d{2}$')
    speed: Union[float, Literal['max']] = 1
    cloud: bool = True
    rag_mode: Literal['default', 'regular', 'graph'] = 'default'
    max_seconds: Optional[float] = Field(default=None, ge=1 / 1024, allow_inf_nan=False)


class ControlRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['pause', 'resume', 'stop']


def mount_online(app, cfg, manager=None):
    manager = manager or OnlineManager(cfg)
    app.state.online_manager = manager
    router = APIRouter()

    def origin(request):
        value = request.headers.get('origin')
        if value:
            parsed = urlsplit(value)
            if (parsed.scheme, parsed.netloc) != (request.url.scheme, request.url.netloc):
                raise HTTPException(403, '只接受同源在线任务请求')

    def find(identity):
        try:
            return manager.get(identity)
        except KeyError:
            raise HTTPException(404, '未知在线任务')

    def snapshot(item):
        return item.snapshot() if isinstance(item, OnlineSession) else item

    @router.get('/online')
    def page():
        return FileResponse(Path(__file__).parents[1] / 'web/online.html')

    @router.get('/api/online/catalog')
    def catalog():
        return manager.catalog()

    @router.get('/api/online/sessions')
    def sessions(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
        return manager.list(limit=limit, offset=offset)

    @router.post('/api/online/sessions')
    def create(request: Request, document: OnlineRequest):
        origin(request)
        if getattr(app.state, 'cloud_testing', False) and document.cloud:
            raise HTTPException(409, '正在测试云端连接，可先关闭路径结束云端选项')
        try:
            values = document.model_dump()
            values['speed'] = 1000. if values['speed'] == 'max' else values['speed']
            if not 0 < values['speed'] <= 1000:
                raise ValueError('Invalid speed')
            return manager.create(values)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        except (OSError, ValueError):
            raise HTTPException(400, '原始CDT记录或格式不符合在线输入合同')

    @router.get('/api/online/sessions/{identity}')
    def get(identity: str):
        return snapshot(find(identity))

    @router.post('/api/online/sessions/{identity}/control')
    def control(identity: str, request: Request, document: ControlRequest):
        origin(request)
        item = find(identity)
        if not isinstance(item, OnlineSession):
            raise HTTPException(409, '历史或中断任务需重新创建回放')
        try:
            return item.control(document.action)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @router.get('/api/online/sessions/{identity}/events')
    async def events(identity: str, request: Request, after: int = -1):
        item = find(identity)
        try:
            last = int(request.headers.get('last-event-id', '-1'))
            if after < -1 or last < -1:
                raise ValueError()
            cursor = max(after, last)
            if isinstance(item, OnlineSession):
                with item.lock:
                    latest_sequence = item.sequence - 1
            else:
                latest_sequence = item['last_sequence']
            if cursor > latest_sequence:
                raise HTTPException(400, '事件序号尚未产生，请从已有序号继续')
        except ValueError:
            raise HTTPException(400, '事件序号须不小于-1')
        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                if isinstance(item, OnlineSession):
                    packets, terminal = item.sse_batch(cursor)
                else:
                    packets, terminal = [], True
                    if cursor < item['last_sequence']:
                        payload = {key: value for key, value in item.items() if key != 'events'}
                        event = {'id': identity, 'sequence': item['last_sequence'], 'type': 'snapshot', 'payload': payload}
                        packets = [(event['sequence'], 'id: ' + str(event['sequence']) + '\ndata: ' + json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n\n')]
                for sequence, frame in packets:
                    cursor = sequence
                    yield frame
                if terminal:
                    break
                yield ': heartbeat\n\n'
                await asyncio.sleep(.25)
        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @router.get('/api/online/sessions/{identity}/report')
    def report(identity: str):
        item = find(identity)
        return item.report() if isinstance(item, OnlineSession) else item

    @router.get('/api/online/sessions/{identity}/report.md')
    def markdown(identity: str):
        result = report(identity)
        lines = ['# 实时在线分析报告', '', '状态：' + result['state'],
                 '输入：' + result['recording']['title'], '数值后端：mamba-minimal CPU FP32', '']
        rag_policy = result.get('rag_policy') or result.get('request', {}).get('rag_policy')
        lines += ['RAG 模式：' + str(rag_policy.get('rag_mode')) +
                  '；判读协议：' + str(rag_policy.get('judge_profile')) +
                  '；参考清单：' + str(rag_policy.get('reference_manifest_sha256')) if rag_policy else
                  'RAG 版本：历史未记录。', '']
        for row in result.get('paths', []):
            lines += ['## ' + ('路径' if row['scope'] == 'task' else '休息段') + ' ' + str(row['ordinal']),
                      str(row['start_seconds']) + '–' + str(row['end_seconds']) + ' 秒',
                      '有效窗口：' + str(row['accepted_windows']) + '/' + str(row['total_windows']),
                      '数据截止：' + str(row['data_cutoff_seconds']) + ' 秒',
                      '路径状态：' + ('完整' if row['complete'] else '路径未闭合') + ('；' + row['reason'] if row.get('reason') else ''),
                      '本地类别：' + str(row['vrms'].get('numeric_class')),
                      '云端类别：' + str(row['cloud'].get('final_class')), '']
            if row['cloud'].get('explanation'):
                lines.append(row['cloud']['explanation'])
            for title, name in (('VRMSAgent', 'vrms'), ('FatigueAgent', 'fatigue'), ('EmotionAgent', 'emotion'), ('汇总', 'supervisor')):
                agent = row['cloud'].get('reports', {}).get(name, {})
                lines += ['', '### ' + title, agent.get('explanation') or failure_reason(agent) or failure_reason(row['cloud']) or '云端解释未生成；本地结果已保存。']
        lines += ['', '过程读数是未校准的整路径前缀估计；疲劳与情绪领域输出 EEG 指标。']
        return PlainTextResponse('\n'.join(lines), media_type='text/markdown; charset=utf-8')

    app.include_router(router)
    app.add_event_handler('shutdown', manager.shutdown)
    return manager
