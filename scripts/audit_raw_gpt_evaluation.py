"""Independently recount and verify saved GPT decisions; no new API calls."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_agent.config import file_hash
from eeg_agent.raw_tool_agent import EvidenceRuntime, validate_final, TOOL_NAMES


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def audit(out):
    protocol = read(out / 'protocol.json')
    complete = read(out / 'prediction_complete.json')
    for source, expected in protocol['source_sha256'].items():
        assert file_hash(source) == expected, 'Frozen source changed'
    assert file_hash(protocol['label_file']) == protocol['label_sha256']
    decisions, raw_rows, api_calls, failed_calls, encoded = {}, [], 0, 0, 0
    response_models, actual_function_calls = Counter(), Counter()
    response_ids = set()
    earliest_response = datetime.fromisoformat(protocol['created_at']).timestamp()
    for subject in map(int, protocol['subjects']):
        saved = read(out / 'predictions' / f'subject_{subject:02d}.json')
        result = saved['result']
        assert saved['subject'] == subject
        assert saved['provider']['model_id'] == protocol['provider_model']
        api_calls += saved['provider']['calls']
        failed_calls += saved['provider']['failures']
        assert result['signal_sha256'] == protocol['subjects'][str(subject)]['raw_sha256']
        if result['available']:
            assert result['measurements'].get('baseline_policy') == 'disabled'
            assert result['fresh_raw_encoding'] is True
        for row in result['measurements']['segments']:
            raw_rows.append({'subject': subject, 'ordinal': row['ordinal'], 'complete': row['complete'],
                'start_clock': row['start_clock'], 'end_clock': row['end_clock'],
                'gpt_class': row['predicted_class'], 'reason': row.get('reason', '')})
            encoded += row.get('fresh_encoded_windows', 0)
            judgment = row.get('cloud_judgment', {})
            if not judgment.get('available'):
                assert row['predicted_class'] is None
                continue
            assert judgment['generation_status'] == 'cloud_verified'
            assert row['classification_layer'] == 'gpt_tool_agent'
            runtime = EvidenceRuntime(row)
            final = None
            for step in judgment['rounds']:
                response = step['response']
                assert response['status'] == 'completed'
                response_models[response['model']] += 1
                assert response['model'] == protocol['provider_model']
                assert response['id'] not in response_ids, 'Repeated response reused across queries'
                assert response['created_at'] >= earliest_response, 'Response predates this evaluation'
                response_ids.add(response['id'])
                calls = [item for item in response['output'] if item['type'] == 'function_call']
                if calls:
                    for call in calls:
                        assert json.loads(call['arguments']) == {'query': 'qsingle'}
                        runtime.execute(call['name'])
                        actual_function_calls[call['name']] += 1
                else:
                    text = ''.join(part['text'] for item in response['output'] for part in item.get('content', [])
                                   if part['type'] == 'output_text')
                    final = validate_final(json.loads(text), runtime.calls)
            assert final is not None
            assert final['final_class'] == row['predicted_class'] == judgment['final_class']
            assert runtime.calls == judgment['actual_tool_calls']
            assert runtime.memo == judgment['tool_outputs'], 'Tool evidence replay differs'
            assert set(TOOL_NAMES).issubset(runtime.calls)
            decisions[(subject, row['ordinal'])] = final
    assert api_calls == complete['api_calls'] and failed_calls == complete['api_failed_calls']
    assert encoded == complete['fresh_encoded_windows']
    truths = {}
    with Path(protocol['label_file']).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['path_score_available'].lower() == 'true' and row['path_score']:
                truths[f"{int(row['subject_id']):02d}:{int(row['start_sample_1024'])}"] = (
                    'High' if float(row['path_score']) >= 30 else 'Low')
    correct, scored = 0, []
    for path in protocol['paths']:
        actual = decisions.get((path['subject'], path['ordinal']), {}).get('final_class')
        truth = truths[path['path_key']]
        correct += actual == truth
        scored.append({'subject': path['subject'], 'ordinal': path['ordinal'],
                       'path_key': path['path_key'], 'gpt_class': actual, 'true_class': truth})
    summary = read(out / 'summary.json')
    assert len(scored) == summary['metrics']['total'] == 146
    assert correct == summary['metrics']['correct']
    assert correct / 146 == summary['metrics']['accuracy']
    scored_id = {(row['subject'], row['ordinal']) for row in scored}
    unscored = [row for row in raw_rows if row['complete'] and (row['subject'], row['ordinal']) not in scored_id]
    with (out / 'all_raw_paths.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    result = {'status': 'independently_verified', 'correct': correct, 'total': 146, 'accuracy': correct / 146,
              'raw_event_segments': len(raw_rows),
              'incomplete_raw_paths': [row for row in raw_rows if not row['complete']],
              'raw_complete_paths': sum(row['complete'] for row in raw_rows), 'unscorable_raw_paths': unscored,
              'total_valid_gpt_decisions_including_unscored': len(decisions),
              'response_models': response_models, 'actual_function_call_counts': actual_function_calls,
              'api_calls': api_calls, 'api_failed_calls': failed_calls,
              'fresh_encoded_windows': encoded, 'frozen_sources_verified': True,
              'unique_fresh_response_ids': len(response_ids),
              'original_gpt_json_reparsed': True, 'tool_evidence_replayed': True,
              'labels_loaded_after_predictions': True, 'numerical_fallback_as_gpt': False}
    (out / 'completion_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out.resolve())
