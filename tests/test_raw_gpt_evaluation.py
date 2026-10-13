"""Whole-path scoring retains missing decisions in the all-path denominator."""
import pytest
from scripts.evaluate_raw_gpt_all import score_rows


def test_all_path_acc_and_coverage_use_the_full_denominator():
    predictions = [{'path_key': '01:1', 'gpt_class': 'Low'},
                   {'path_key': '02:2', 'gpt_class': 'Low'},
                   {'path_key': '03:3', 'gpt_class': None},
                   {'path_key': '04:4', 'gpt_class': 'High'}]
    truths = {'01:1': {'true_class': 'Low'}, '02:2': {'true_class': 'High'},
              '03:3': {'true_class': 'Low'}, '04:4': {'true_class': 'High'}}
    rows, result = score_rows(predictions, truths)
    assert result['correct'] == 2 and result['total'] == 4
    assert result['accuracy'] == .5 and result['coverage'] == .75
    assert result['conditional_accuracy'] == 2 / 3
    assert result['failed_or_unpublished_paths'] == 1
    assert result['confusion_matrix'] == [[1, 0], [1, 1]]
    assert result['balanced_accuracy'] == .5
    assert not rows[2]['correct']


def test_duplicate_or_unmatched_labels_cannot_silently_change_the_denominator():
    with pytest.raises(ValueError, match='Duplicate'):
        score_rows([{'path_key': 'a'}, {'path_key': 'a'}], {})
    with pytest.raises(ValueError, match='Missing path truth'):
        score_rows([{'path_key': 'a', 'gpt_class': 'High'}], {})
