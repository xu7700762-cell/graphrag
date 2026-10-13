"""Post-hoc scoring boundaries, missing-denominator and class thresholds."""
import csv

import pytest

from scripts.evaluate_online_numeric import score_predictions, summarize


def test_missing_predictions_stay_in_accuracy_denominator():
    result = summarize([{'target_high': 0, 'score': .1}, {'target_high': 1, 'score': None},
                        {'target_high': 1, 'score': .9}], 'score')
    assert result['accuracy_available'] == 1
    assert result['accuracy_all_missing_as_incorrect'] == pytest.approx(2 / 3)
    assert result['failure_rate'] == pytest.approx(1 / 3)
    assert result['confusion_low_high'] == [[1, 0], [0, 1]]


def test_posthoc_alignment_and_threshold_do_not_change_predictions(tmp_path):
    label_path = tmp_path / 'labels.csv'
    fields = ['subject_id', 'start_sample_1024', 'end_sample_1024', 'is_complete',
              'path_score_available', 'path_score', 'duration_sec']
    with label_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subject, score in ((1, 29.99), (2, 30)):
            writer.writerow(dict(zip(fields, (subject, 100, 15460, 'true', 'true', score, 15))))
    source_paths = [{'subject': subject, 'start_sample': 46, 'end_sample': 15406, 'complete': True}
                    for subject in (1, 2)]
    predictions = [{**row, 'ordinal': 1, 'accepted_windows': 3, 'total_windows': 3, 'coverage': 1.,
                    'mil_probability': .2 if row['subject'] == 1 else .8,
                    'numeric_score': .3 if row['subject'] == 1 else .7,
                    'failure_reason': '', 'prefixes': []} for row in source_paths]
    paths, prefixes, alignment, excluded, total = score_predictions(predictions, source_paths, label_path)
    assert [row['target_high'] for row in paths] == [0, 1]
    assert alignment['1']['offset_samples'] == -54
    assert not prefixes and not excluded and total == 2
    assert all('target_high' not in row for row in predictions)


def test_ambiguous_offsets_are_rejected(tmp_path):
    label_path = tmp_path / 'labels.csv'
    label_path.write_text('subject_id,start_sample_1024,end_sample_1024,is_complete,path_score_available,path_score,duration_sec\n'
                          '1,100,15460,true,true,30,15\n', encoding='utf-8')
    paths = [{'subject': 1, 'start_sample': stamp, 'end_sample': stamp + 15360, 'complete': True}
             for stamp in (46, 30000)]
    with pytest.raises(ValueError, match='Ambiguous'):
        score_predictions([], paths, label_path)
