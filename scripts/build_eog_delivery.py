"""Build a separate, credential-free Pi bundle from actual EOG acceptance artifacts."""
import datetime
import json
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eeg_agent.config import load_config,profile_config,contract,file_hash,write_json
from eeg_agent.models import MODEL_IDS,ONNXPredictor
from eeg_agent.tools import TOOL_CATALOG

def main():
    cfg=load_config();out=Path(cfg['output_root']);reports=out/'reports'
    effect=reports/'eog_effect_report.json'
    if not effect.exists(): raise ValueError('Complete the traceable EOG effect report first')
    equivalence=json.loads((reports/'eog_online_offline_01.json').read_text(encoding='utf-8'))
    controlled=json.loads((reports/'eog_controlled_validation.json').read_text(encoding='utf-8'))
    if equivalence['status']!='passed' or not controlled['passed']: raise ValueError('EOG acceptance did not pass')
    models={};files=[]
    for model_id in MODEL_IDS:
        for checkpoint in ('fixed_seed42','eog_rls_seed42'):
            predictor=ONNXPredictor(cfg,model_id=model_id,checkpoint_id=checkpoint)
            models[model_id+'/'+checkpoint]=predictor.manifest
            files.extend([predictor.directory/'model.onnx',predictor.directory/'contract.json'])
    write_json(ROOT/'docs/eog_processing_contract.json',contract(profile_config(cfg,'P1')))
    write_json(ROOT/'docs/eog_model_contract_summary.json',models)
    write_json(ROOT/'docs/tool_contracts.json',TOOL_CATALOG)
    for folder in ('eeg_agent','knowledge','docs'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend(ROOT/name for name in ('README.md','pyproject.toml','requirements-runtime.txt',
                 'requirements-train.txt','.env.example','deploy/pi.yaml','deploy/eeg-agent.service'))
    files.extend(ROOT/'scripts'/name for name in ('validate_eog.py','verify_eog_online_offline.py',
                 'verify_online_offline.py','summarize_eog_results.py'))
    for name in ('eog_effect_report.json','eog_effect_report.md','eog_controlled_validation.json',
                 'eog_online_offline_01.json','eog_browser_acceptance.json','eog_pytest.log'):
        p=reports/name
        if p.exists(): files.append(p)
    for p in reports.glob('evaluation_*_eog_rls_seed42_test.json'): files.append(p)
    files=sorted(set(files))
    # Whitelist paths from this project; never include .env, raw streams, caches or training weights.
    for p in files:
        p.relative_to(ROOT)
        if p.name=='.env' or p.suffix in ('.cdt','.npz','.pt') or '__pycache__' in p.parts:
            raise ValueError('Forbidden deployment file: '+p.name)
    archive=ROOT/'deploy/VRMSAgent_Pi_eog_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files: z.write(p,p.relative_to(ROOT).as_posix())
    hashes={p.relative_to(ROOT).as_posix():file_hash(p) for p in files}
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(hashes)
        assert not any(name=='.env' or name.endswith('/.env') for name in z.namelist())
        assert z.testzip() is None
        import hashlib
        assert all(hashlib.sha256(z.read(name)).hexdigest()==h for name,h in hashes.items())
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'zip_sha256':file_hash(archive),'zip_bytes':archive.stat().st_size,'files':hashes,
              'no_raw_data':True,'no_credentials':True,'no_training_checkpoints':True,
              'pi_hardware_tested':False,'windows_eog_validation':True,
              'new_method_real_cloud_tested':False,'boundary':'Local Windows mock reports; Pi hardware and real API require separate validation.'}
    write_json(ROOT/'deploy/bundle_eog_manifest.json',manifest)
    print('Verified EOG Pi bundle:',archive,'files:',len(files),'bytes:',archive.stat().st_size)

if __name__=='__main__':main()
