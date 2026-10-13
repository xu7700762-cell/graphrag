"""Equivalence and arrived-only boundaries for the independent online branch."""
import numpy as np
import pytest

from eeg_agent.config import EEG_CHANNELS
from eeg_agent.online.stream import StreamingProcessor
from eeg_agent.recordings import preprocess, state_segments_from_events, trigger_events


CHANNELS = EEG_CHANNELS + ['M1', 'M2', 'VEOG', 'HEOG', 'ECG', 'EGG', 'Trigger']


def waveform(count=45000, marks=((6001, 20), (21361, 22), (31602, 20))):
    rng = np.random.default_rng(14)
    raw = rng.normal(0, 15, (count, len(CHANNELS))).astype(np.float32)
    raw[:, -1] = 0
    for stamp, mark in marks:
        raw[stamp:stamp + 7, -1] = mark
    return raw


def replay(raw, sizes=(256,), finish=True):
    processor = StreamingProcessor(CHANNELS)
    events, start, index = [], 0, 0
    while start < len(raw):
        stop = min(len(raw), start + sizes[index % len(sizes)])
        events.extend(processor.feed(raw[start:stop], start))
        assert processor.buffered_samples <= 1280
        start, index = stop, index + 1
    if finish:
        events.extend(processor.finish())
    return events, processor


def without_arrays(events):
    return [{key: value for key, value in event.items() if key != 'window'} for event in events]


def offline_windows(raw):
    processed, names, compatible = preprocess(raw, CHANNELS, 1024)
    assert compatible and names == EEG_CHANNELS
    marks = trigger_events(raw, -1)
    tasks, rests, _ = state_segments_from_events(marks, len(raw), 1024)
    result = {}
    for scope, segments in (('task', tasks), ('rest', rests)):
        for segment in segments:
            start, end = segment['start_sample'], segment['end_sample']
            for i in range((end - start) // 5120):
                stamp = start + i * 5120
                output_start = (stamp + 3) // 4
                window = processed[:, output_start:output_start + 1280]
                valid = (window.shape == (30, 1280) and np.isfinite(window).all()
                         and np.ptp(window, axis=1).max() <= 2000
                         and np.count_nonzero(window.std(axis=1) >= .05) >= 28)
                result[scope, segment['ordinal'], stamp] = (window, valid)
    return result


@pytest.mark.parametrize('phase', [0, 1, 2, 3])
def test_partitions_and_marker_phase_match_current_offline(phase):
    raw = waveform(marks=((6144 + phase, 20), (21504 + phase, 22), (31744 + phase, 20)))
    expected = offline_windows(raw)
    events, _ = replay(raw, (256,))
    other, _ = replay(raw, (17, 513, 127, 777))
    assert without_arrays(events) == without_arrays(other)
    windows = [event for event in events if event['type'] == 'window']
    assert len(windows) == len(expected)
    for event, second in zip(windows, [e for e in other if e['type'] == 'window']):
        key = (event['scope'], event['ordinal'], event['start_sample'])
        array, valid = expected[key]
        np.testing.assert_array_equal(event['window'], array)
        np.testing.assert_array_equal(event['window'], second['window'])
        assert event['valid'] == bool(valid)
        assert event['end_sample'] <= event['observed_end_sample']


def test_future_changes_cannot_rewrite_arrived_windows_or_events():
    raw = waveform()
    prefix, _ = replay(raw, (17, 513))
    altered = raw.copy()
    altered[22000:, :30] *= 50
    altered[22000:, -1] = 20
    prefix_changed, _ = replay(altered, (17, 513))
    prefix = [e for e in prefix if e['observed_end_sample'] <= 22000]
    prefix_changed = [e for e in prefix_changed if e['observed_end_sample'] <= 22000]
    assert without_arrays(prefix) == without_arrays(prefix_changed)
    for a, b in zip([e for e in prefix if e['type'] == 'window'],
                    [e for e in prefix_changed if e['type'] == 'window']):
        np.testing.assert_array_equal(a['window'], b['window'])
    assert not any(e['type'] == 'path_start' and e['start_sample'] >= 22000 for e in prefix)


@pytest.mark.parametrize('phase', [0, 1, 2, 3])
def test_window_waits_for_full_raw_interval_and_precedes_end(phase):
    start = 6144 + phase
    end = start + 5120
    raw = waveform(end + 20, ((start, 20), (end, 22)))
    processor = StreamingProcessor(CHANNELS)
    events = processor.feed(raw[:end - 1], 0)
    assert not any(e['type'] == 'window' for e in events)
    events = processor.feed(raw[end - 1:end + 1], end - 1)
    assert [e['type'] for e in events] == ['window', 'path_end', 'rest_start']
    expected = offline_windows(raw)['task', 1, start][0]
    np.testing.assert_array_equal(events[0]['window'], expected)
    assert events[0]['end_sample'] == end


def test_nonfinite_resets_filter_but_retains_task_window_grid():
    raw = waveform(52000, ((6145, 20), (47000, 22)))
    raw[17001:17006, 3] = np.nan
    raw[28517, 30] = np.inf
    events, _ = replay(raw, (256,))
    other, _ = replay(raw, (17, 513, 127))
    assert without_arrays(events) == without_arrays(other)
    expected = offline_windows(raw)
    windows = [e for e in events if e['type'] == 'window']
    assert all((e['start_sample'] - 6145) % 5120 == 0 for e in windows if e['scope'] == 'task')
    assert any(not e['valid'] for e in windows)
    for event in windows:
        array, valid = expected[event['scope'], event['ordinal'], event['start_sample']]
        np.testing.assert_array_equal(event['window'], array)
        assert event['valid'] == bool(valid)


def test_missing_samples_keeps_grid_and_new_warmup():
    raw = waveform(50000, ((6147, 20), (46000, 22)))
    processor = StreamingProcessor(CHANNELS)
    events = processor.feed(raw[:18000], 0)
    events.extend(processor.feed(raw[24000:], 24000))
    events.extend(processor.finish())
    tasks = [e for e in events if e['type'] == 'window' and e['scope'] == 'task']
    assert [e['start_sample'] for e in tasks] == list(range(6147, 46000 - 5119, 5120))
    assert any(e['reason'] == 'missing_samples' for e in tasks)
    assert not next(e for e in tasks if e['start_sample'] == 21507)['valid']
    assert next(e for e in tasks if e['start_sample'] == 31747)['valid']
    padded = raw.copy()
    padded[18000:24000, :32] = np.nan
    expected = offline_windows(padded)
    for event in tasks:
        array, valid = expected['task', 1, event['start_sample']]
        np.testing.assert_array_equal(event['window'], array)
        assert event['valid'] == bool(valid)


def test_duplicate_markers_orphan_rest_eof_and_finish_are_explicit():
    marks = ((5000, 22), (11000, 20), (17000, 20), (28000, 22), (34000, 22), (40000, 20))
    raw = waveform(46000, marks)
    events, processor = replay(raw, (7, 255, 513))
    tasks, rests, _ = state_segments_from_events(list(marks), len(raw), 1024)
    ends = [e for e in events if e['type'] in ('path_end', 'rest_end')]
    expected = sorted([('task', s['ordinal'], s['start_sample'], s['end_sample'], s['complete']) for s in tasks]
                      + [('rest', s['ordinal'], s['start_sample'], s['end_sample'], s['complete']) for s in rests],
                      key=lambda row: row[3])
    assert [(e['scope'], e['ordinal'], e['start_sample'], e['end_sample'], e['complete']) for e in ends] == expected
    assert len([e for e in events if e['type'] == 'warning']) == 3
    assert processor.finish() == []
    with pytest.raises(RuntimeError, match='finished'):
        processor.feed(raw[:10], 0)
    no_events, _ = replay(waveform(8000, ()))
    assert no_events == []


def test_channel_reordering_restart_and_input_contracts():
    raw = waveform()
    expected, _ = replay(raw)
    order = np.random.default_rng(25).permutation(len(CHANNELS))
    processor = StreamingProcessor([CHANNELS[i] for i in order])
    actual = processor.feed(raw[:, order], 0) + processor.finish()
    assert without_arrays(actual) == without_arrays(expected)
    for a, b in zip([e for e in actual if e['type'] == 'window'], [e for e in expected if e['type'] == 'window']):
        np.testing.assert_array_equal(a['window'], b['window'])
    # New processor is the restart boundary, with fresh warmup and ordinals.
    restarted, _ = replay(raw, (513, 17))
    assert without_arrays(restarted) == without_arrays(expected)
    with pytest.raises(ValueError, match='1024'):
        StreamingProcessor(CHANNELS, 256)
    with pytest.raises(ValueError, match='Missing'):
        StreamingProcessor(CHANNELS[:-1])
    with pytest.raises(ValueError, match='unique'):
        StreamingProcessor(CHANNELS + ['M1'])
    processor = StreamingProcessor(CHANNELS)
    with pytest.raises(ValueError, match='samples-by-channel'):
        processor.feed(raw[:10, :-1], 0)
    processor.feed(raw[:10], 0)
    with pytest.raises(ValueError, match='out-of-order'):
        processor.feed(raw[5:20], 5)
    with pytest.raises(ValueError, match='nonnegative'):
        processor.feed(raw[:1], True)
