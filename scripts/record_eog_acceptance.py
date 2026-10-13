"""Record checks for the two browser-started Windows sessions (no cloud calls)."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eeg_agent.config import load_config,write_json,file_hash

def main():
    out=Path(load_config()['output_root']);ids=['be7698761999','0f721c01a3a3'];tasks=[]
    for task_id in ids:
        folder=out/'tasks'/task_id
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        context=json.loads((folder/'context.json').read_text(encoding='utf-8'))
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        windows=[e['payload'] for e in events if e['kind']=='window']
        checks={'completed':context['state']=='completed','real_CDT_130_seconds_1x':report['source_seconds']==130 and report['timing_kind']=='real_time_1x',
                'P1_contract_model':report['model']['preprocessing_profile']=='P1' and report['model']['checkpoint_id']=='eog_rls_seed42',
                'actual_local_cleanup':len(windows)==19 and all(w['eog_cleaning']['applied'] and w['eog_cleaning']['method']=='causal_reference_rls' for w in windows),
                'post_cleaning_gate':report['eog_counts']['quality_rejected_windows']==3 and report['paths'][0]['valid_windows']==16,
                'mock_labelled':report['request']['backend']=='mock' and context['simulated_agents'],
                'no_real_cloud_calls':report['cloud']['backend']=='mock'}
        if not all(checks.values()): raise ValueError('EOG browser-task validation failed: '+task_id)
        tasks.append({'task_id':task_id,'model_id':report['model']['model_id'],'checks':checks,
                      'report_sha256':file_hash(folder/'report.json'),'events_sha256':file_hash(folder/'events.jsonl'),
                      'last_window_cleaning':windows[-1]['eog_cleaning'],
                      'source_seconds':report['source_seconds'],'wall_seconds':report['wall_seconds']})
    record={'status':'passed','scope':'P1 EOG Windows browser workflow; actual CDT and models; mock language responses',
            'actions':{'UI_EEGNet_P1_selection':True,'UI_EEGConformer_P1_selection':True,
                       'UI_local_reference_cleaning_panel':True,'UI_live_quality_probabilities_and_report':True,
                       'UI_page_refresh_restored_completed_task':True},
            'tasks':tasks,'pi_hardware_tested':False,'new_method_real_api_tested':False,
            'observation_note':'Browser actions/AX observations in this task; persisted event reports independently checked. Downloads and in-flight refresh are not newly claimed.'}
    write_json(out/'reports/eog_browser_acceptance.json',record)
    print('Two browser-started P1 sessions verified')

if __name__=='__main__':main()
