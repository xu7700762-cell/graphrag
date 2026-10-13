import csv
import re
from pathlib import Path
import numpy as np
from .config import EEG_CHANNELS, SPLITS, file_hash, write_json

def list_block(text, name):
    match = re.search(r'(?ms)^\s*' + re.escape(name) + r' START_LIST[^\n]*\n(.*?)^\s*' + re.escape(name) + r' END_LIST', text)
    return [x.strip() for x in match.group(1).splitlines() if x.strip() and not x.lstrip().startswith('#')] if match else []

def metadata(path):
    path = Path(path)
    text = Path(str(path) + '.dpo').read_text(encoding='utf-8', errors='replace')
    def key(name):
        m = re.search(r'(?m)^\s*' + name + r'\s*=\s*(.*?)\s*$', text)
        if not m: raise ValueError('Missing DPO field: ' + name)
        return m.group(1).strip()
    channels = list_block(text, 'LABELS') + list_block(text, 'LABELS_OTHERS')
    n = int(key('NumChannels'))
    if key('DataByteOrder') != 'INTEL' or key('DataSampOrder') != 'SAMP' or key('DataFormat') != '6':
        raise ValueError('Unsupported CDT byte/sample/format contract')
    if int(float(key('SampleFreqHz'))) != 1024: raise ValueError('Expected 1024 Hz')
    if len(channels) != n or len(set(channels)) != n: raise ValueError('Invalid channel names')
    required = EEG_CHANNELS + ['M1','M2','VEOG','HEOG','Trigger']
    if any(ch not in channels for ch in required): raise ValueError('Missing required channels')
    units = re.findall(r'(?m)^\s*DataUnit\s*=\s*(\S+)', text)
    scales = re.findall(r'(?m)^\s*CommonScale\s*=\s*(\S+)', text)
    offsets = re.findall(r'(?m)^\s*CommonOffset\s*=\s*(\S+)', text)
    if not units or any(u != 'uV' for u in units) or any(float(s) != 1 for s in scales) or any(float(s) != 0 for s in offsets):
        raise ValueError('Unsupported unit/scale/offset')
    size = path.stat().st_size
    if size % (n * 4): raise ValueError('Partial trailing sample frame')
    actual = size // (n * 4); declared = int(key('NumSamples'))
    return {'channels': channels, 'n_channels': n, 'actual_samples': actual, 'declared_samples': declared,
            'warnings': ['DPO sample count differs; read actual complete frames'] if actual != declared else [], 'unit': 'uV', 'sfreq': 1024}

def events(path, meta):
    text = Path(str(path) + '.ceo').read_text(encoding='utf-8', errors='replace')
    ev = []
    for line in list_block(text, 'NUMBER_LIST'):
        fields = line.split()
        if len(fields) >= 3 and int(float(fields[2])) in (20,22):
            ev.append((int(float(fields[0])), int(float(fields[2]))))
    source = 'ceo_number_list'
    if not ev:
        source = 'trigger_channel_fallback'
        raw = np.memmap(path, mode='r', dtype='<f4', shape=(meta['actual_samples'],meta['n_channels']))
        col = meta['channels'].index('Trigger'); previous = 0
        for start in range(0, len(raw), 65536):
            values = np.rint(raw[start:start+65536,col]).astype(np.int64)
            prior = np.r_[previous, values[:-1]]
            ix = np.flatnonzero((values != prior) & np.isin(values,[20,22]))
            ev.extend((start+int(i),int(values[i])) for i in ix)
            previous = int(values[-1])
    return sorted(set((s,m) for s,m in ev if 0 <= s <= meta['actual_samples'])), source

class CDTSource:
    def __init__(self, root, subject):
        self.path = Path(root) / 'data/raw' / ('Acquisition %02d.cdt' % subject)
        self.meta = metadata(self.path)
        self.events, self.event_source = events(self.path, self.meta)
        self.raw = np.memmap(self.path, mode='r', dtype='<f4', shape=(self.meta['actual_samples'],self.meta['n_channels']))

    def chunks(self, block=256, limit_seconds=None, start_sample=0):
        if not isinstance(start_sample,int) or not 0<=start_sample<len(self.raw): raise ValueError('Invalid replay start sample')
        stop = len(self.raw) if limit_seconds is None else min(len(self.raw), start_sample+int(limit_seconds*1024))
        pointer = 0
        for start in range(start_sample,stop,block):
            end = min(start+block,stop); visible = []
            while pointer < len(self.events) and self.events[pointer][0] < end:
                if self.events[pointer][0] >= start: visible.append(self.events[pointer])
                pointer += 1
            yield start, np.asarray(self.raw[start:end], dtype=np.float64), visible
        # EOF events at exactly actual_samples are released only once EOF arrives.
        if stop == len(self.raw):
            final = [e for e in self.events[pointer:] if e[0] == stop]
            if final: yield stop, np.empty((0,self.meta['n_channels'])), final

def paths(cfg):
    p = Path(cfg['data_root']) / 'data/labels/task_segments_with_path_scores.csv'
    with p.open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    out = []
    audit_file=Path(cfg['output_root'])/'reports/data_audit.json'
    excluded=set()
    if audit_file.exists():
        import json
        prior=json.loads(audit_file.read_text(encoding='utf-8'))
        excluded=set(prior.get('event_errors',[]))|set(prior.get('boundary_errors',[]))
    for r in rows:
        if r['is_complete'] == 'true' and r['path_score_available'] == 'true' and r['path_score'] and float(r['duration_sec']) >= 15:
            sid = int(r['subject_id']); start = int(r['start_sample_1024']); end = int(r['end_sample_1024'])
            out.append({'subject': sid, 'start': start, 'end': end, 'key': '%02d:%d' % (sid,start),
                        'label': int(float(r['path_score']) >= 30), 'score': float(r['path_score']), 'duration': (end-start)/1024,
                        'raw_complete':('%02d:%d'%(sid,start)) not in excluded})
    return out

def audit(cfg):
    selected = paths(cfg); files = []; boundary_errors = []; event_errors = []
    for sid in range(1,27):
        src = CDTSource(cfg['data_root'],sid); visible = set(src.events)
        for row in [r for r in selected if r['subject'] == sid]:
            if not 0 <= row['start'] < row['end'] <= src.meta['actual_samples']: boundary_errors.append(row['key'])
            if (row['start'],20) not in visible or (row['end'],22) not in visible: event_errors.append(row['key'])
        files.append({'subject':sid, **src.meta, 'event_source':src.event_source, 'event_count':len(src.events),
                      'sha256':file_hash(src.path), 'dpo_sha256':file_hash(str(src.path)+'.dpo'), 'ceo_sha256':file_hash(str(src.path)+'.ceo')})
    unexpected=set(event_errors)-{'04:2516402'}
    report = {'status':'passed_with_recorded_exclusion' if not boundary_errors and not unexpected else 'failed', 'paths':len(selected),
              'subjects':len(set(r['subject'] for r in selected)), 'low':sum(r['label']==0 for r in selected), 'high':sum(r['label']==1 for r in selected),
              'equal_30':sum(r['score']==30 for r in selected), 'description_claim':154, 'discrepancy':'description 154 versus verified CSV 147; no fabricated labels',
              'boundary_errors':boundary_errors, 'event_errors':event_errors, 'label_sha256':file_hash(Path(cfg['data_root'])/'data/labels/task_segments_with_path_scores.csv'),
              'strict_raw_complete_paths':len(selected)-len(set(event_errors)|set(boundary_errors)),
              'exclusion_reason':'Acquisition 04 final scored task ends at clipped EOF 2626240; actual mark22 is 2645592, not arrived in raw file',
              'splits':{k:{'subjects':v,'paths':sum(r['subject'] in v for r in selected)} for k,v in SPLITS.items()}, 'files':files}
    write_json(Path(cfg['output_root'])/'reports/data_audit.json',report)
    if report['status'] == 'failed' or (report['paths'],report['subjects']) != (147,24): raise ValueError('Data protocol audit failed')
    return report
