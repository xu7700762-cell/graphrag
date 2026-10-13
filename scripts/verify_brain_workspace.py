"""Local-only smoke: real CDT replay, domain measurements and follow-up.

Run against an already-started local server. Does not train or call cloud APIs.
"""
import json
import time
from pathlib import Path

import httpx


def wait(client, url, states, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(url)
        response.raise_for_status()
        value = response.json()
        if value['state'] in states:
            return value
        time.sleep(.1)
    raise RuntimeError('local verification timed out')


def main():
    with httpx.Client(base_url='http://127.0.0.1:8880', timeout=15) as client:
        response = client.post('/api/tasks', json={'subject': 1, 'model_id': 'eegnet',
            'strategy': 'full_evidence', 'backend': 'mock', 'speed': 100, 'max_seconds': 130,
            'task': '多领域工作台真实回放验证：完整频谱测量。'})
        response.raise_for_status()
        task_id = response.json()['id']
        task = wait(client, '/api/tasks/' + task_id, {'completed', 'failed', 'stopped'})
        if task['state'] != 'completed':
            raise RuntimeError('EEG replay did not complete')
        response = client.post('/api/brain/sessions', json={'backend': 'mock',
            'domains': ['vrms', 'fatigue', 'emotion'], 'task_id': task_id})
        response.raise_for_status()
        session_id = response.json()['id']
        response = client.post('/api/brain/sessions/' + session_id + '/messages', json={
            'text': '请合作解释本次EEG窗口的VRMS模型结果、疲劳频谱特征和额区不对称，标明各领域能力。'})
        response.raise_for_status()
        first = wait(client, '/api/brain/sessions/' + session_id, {'idle'})
        assert first['result'] and len(first['result']['agent_reports']) == 3
        results = {item['tool']: item['result'] for report in first['result']['agent_reports']
                   for item in report['tool_results']}
        assert results['fatigue.spectral_ratios']['available']
        assert results['emotion.frontal_asymmetry']['available']
        assert results['fatigue.temporal_evidence']['available']
        assert all(not results[t]['classifier_available'] for t in results if not t.startswith('vrms.'))
        frozen = results['vrms.read_evidence']['measurements']['model']
        assert frozen['model_hash'] == task['model']['model_hash']
        probability = results['vrms.read_evidence']['measurements']['deep']['probability_high']
        assert probability == task['latest']['deep']['probability_high']
        response = client.post('/api/brain/sessions/' + session_id + '/messages', json={
            'text': '这些指标能否区分疲劳和眩晕？还缺哪些验证？'})
        response.raise_for_status()
        second = wait(client, '/api/brain/sessions/' + session_id, {'idle'})
        assert second['turn'] == 2 and len(second['messages']) == 4
        assert client.get('/api/tasks/' + task_id).json()['latest']['deep']['probability_high'] == probability
        record = {'status': 'passed', 'task_id': task_id, 'session_id': session_id,
                  'replay': {'speed': 100, 'max_source_seconds': 130, 'state': task['state'],
                             'profile': task['model']['preprocessing_profile'], 'model': frozen},
                  'tool_results': results, 'followup_turns': second['turn'],
                  'source_prediction_unchanged': True, 'real_cloud_calls': 0,
                  'boundary': 'Real local CDT measurements; accelerated replay is not real-time or Pi latency evidence.'}
        path = Path(__file__).resolve().parents[1] / 'outputs/brain_workspace/real_eeg_smoke.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'status': record['status'], 'session_id': session_id,
                          'tool_availability': {t: r['available'] for t, r in results.items()},
                          'source_prediction_unchanged': True}, ensure_ascii=True))


if __name__ == '__main__':
    main()
