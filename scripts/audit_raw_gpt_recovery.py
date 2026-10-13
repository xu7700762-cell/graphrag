"""Verify that recovery used only originally unavailable decisions and identical evidence."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_agent.config import file_hash
from scripts.evaluate_raw_gpt_all import read, save, verify_protocol


def audit(source, out):
    manifest = read(out / 'recovery_protocol.json')
    protocol = read(source / 'protocol.json')
    verify_protocol(protocol)
    for name, expected in manifest['original_file_sha256'].items():
        assert file_hash(source / name) == expected, 'First-pass artifact changed'
    runner = read(out / 'recovery_runner_sha256.json')
    assert file_hash(runner['path']) == runner['sha256']
    attempts = [read(path) for path in sorted((out / 'recovery_attempts').glob('*.json'))]
    ids, original_valid, recovered_valid = set(), 0, 0
    error_types = Counter()
    for subject in map(int, protocol['subjects']):
        original = read(source / 'predictions' / f'subject_{subject:02d}.json')
        current = read(out / 'predictions' / f'subject_{subject:02d}.json')
        before = {row['ordinal']: row for row in original['result']['measurements']['segments']}
        for row in before.values():
            for step in row.get('cloud_judgment', {}).get('rounds', []):
                response = step['response']
                assert response['id'] not in ids
                ids.add(response['id'])
        subset = [attempt for attempt in attempts if attempt['subject'] == subject]
        for attempt in subset:
            original_row = before[attempt['ordinal']]
            assert not original_row.get('cloud_judgment', {}).get('available'), 'A valid GPT decision was retried'
            assert attempt['labels_loaded'] is False
            assert attempt['attempt'] <= manifest['maximum_recovery_attempts_per_path']
            expected = hashlib.sha256(json.dumps(original_row['numeric_fusion'], sort_keys=True,
                                                ensure_ascii=False).encode('utf-8')).hexdigest()
            assert attempt['fresh_evidence_sha256'] == expected
            judgment = attempt['judgment']
            error_types.update([judgment['error_type']] if judgment.get('error_type') else [])
            for step in judgment['rounds']:
                response = step['response']
                assert response['id'] not in ids, 'A GPT response was reused'
                assert response['model'] == protocol['provider_model']
                assert response['created_at'] >= datetime.fromisoformat(manifest['created_at']).timestamp()
                ids.add(response['id'])
        for key in ('calls', 'failures', 'input_tokens', 'output_tokens'):
            assert current['provider'][key] == original['provider'][key] + sum(a['provider'][key] for a in subset)
        for row in current['result']['measurements']['segments']:
            original_row = before[row['ordinal']]
            if original_row.get('cloud_judgment', {}).get('available'):
                assert row == original_row, 'A valid first-pass result changed'
                original_valid += 1
            else:
                for key, value in original_row.items():
                    if key not in ('cloud_judgment', 'predicted_class', 'classification_layer', 'reason'):
                        assert row[key] == value, 'Frozen EEG evidence changed'
                successful = [a for a in subset if a['ordinal'] == row['ordinal'] and a['judgment']['available']]
                assert len(successful) <= 1
                if successful:
                    assert row['cloud_judgment'] == successful[0]['judgment']
                    recovered_valid += 1
                else:
                    assert row == original_row, 'Unavailable result substituted'
    original_metrics = read(source / 'summary.json')['metrics']
    final_metrics = read(out / 'summary.json')['metrics']
    result = {'status': 'independently_verified', 'original_artifacts_unchanged': True,
        'valid_decisions_never_retried_or_changed': True, 'same_frozen_eeg_evidence': True,
        'no_labels_in_recovery': True, 'no_numerical_fallback_as_gpt': True,
        'recovery_attempts': len(attempts), 'successful_recovery_decisions': recovered_valid,
        'original_valid_decisions_including_unscored': original_valid,
        'unique_responses_including_failed_attempts': len(ids), 'recovery_error_types': error_types,
        'first_pass_metrics': original_metrics, 'final_metrics': final_metrics}
    save(out / 'recovery_audit.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.source.resolve(), args.out.resolve())
