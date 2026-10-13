"""Add raw/processed QC measurements without changing P0 arrays or quality gates."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eeg_agent.config import load_config,write_json
from eeg_agent.data import CDTSource
from eeg_agent.signal import StreamPipeline

cfg=load_config();root=Path(cfg['output_root'])/'cache'
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for entry in manifest['subjects']:
    subject=entry['subject'];path=root/('quality_%02d.json'%subject)
    previous={row['start_sample']:row for row in json.loads(path.read_text(encoding='utf-8'))}
    rows=[]
    def observe(w):
        if w['start_sample'] not in previous:return
        old=previous[w['start_sample']]
        assert old['valid']==w['quality']['valid'] and old['reasons']==w['quality']['reasons'],'QC gate changed'
        rows.append({'start_sample':w['start_sample'],**w['quality']})
    source=CDTSource(cfg['data_root'],subject);pipe=StreamPipeline(source.meta['channels'],cfg,observe,lambda p:None)
    for start,raw,events in source.chunks(1024):pipe.feed(start,raw,events)
    pipe.eof();assert len(rows)==len(previous),'QC window boundaries changed'
    write_json(path,rows);print(subject,len(rows),flush=True)
write_json(Path(cfg['output_root'])/'reports/quality_measurements.json',{'status':'passed','subjects':len(manifest['subjects']),
    'raw_and_processed':['amplitude','flatline','jump','line_power','EEG_EOG_correlation'],
    'quality_gates_unchanged':True,'P0_arrays_unchanged':True,'cache_manifest_unchanged':True})
