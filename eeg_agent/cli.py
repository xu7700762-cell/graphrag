import argparse
import json
from .config import load_config

def main():
    parser=argparse.ArgumentParser(description='EEG Agent 本地研究原型'); parser.add_argument('--config')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('audit'); prep=sub.add_parser('prepare');prep.add_argument('--profile',choices=['P0','P1'],default='P0')
    export=sub.add_parser('export'); export.add_argument('--model',choices=['eegnet','eegconformer'],default='eegnet')
    export.add_argument('--profile',choices=['P0','P1'],default='P0');export.add_argument('--checkpoint')
    sub.add_parser('knowledge-index')
    sub.add_parser('models')
    t=sub.add_parser('train'); t.add_argument('--epochs',type=int,default=20);t.add_argument('--model',choices=['eegnet','eegconformer','both'],default='both')
    f=sub.add_parser('fivefold'); f.add_argument('--epochs',type=int,default=20);f.add_argument('--model',choices=['eegnet','eegconformer'],default='eegnet')
    e=sub.add_parser('evaluate'); e.add_argument('--split',choices=['train','validation','test'],default='test');e.add_argument('--model',choices=['eegnet','eegconformer','both'],default='both')
    for command in (t,f,e): command.add_argument('--profile',choices=['P0','P1'],default='P0')
    e.add_argument('--checkpoint')
    r=sub.add_parser('replay'); r.add_argument('--subject',type=int,default=1); r.add_argument('--mode',choices=['fixed_p0','fixed_p1','rules','single_agent','hierarchical'],default='hierarchical')
    r.add_argument('--backend',choices=['mock','api'],default='mock'); r.add_argument('--speed',type=float,default=1); r.add_argument('--max-seconds',type=float)
    r.add_argument('--task',default='分析当前到达的EEG信号')
    c=sub.add_parser('compare'); c.add_argument('--subject',type=int,default=1); c.add_argument('--backend',choices=['mock','api'],default='mock'); c.add_argument('--speed',type=float,default=1); c.add_argument('--max-seconds',type=float,default=130)
    for name in ('monitor','compare-strategies'):
        m=sub.add_parser(name);m.add_argument('--subject',type=int,default=1);m.add_argument('--model',choices=['eegnet','eegconformer'],default='eegnet')
        m.add_argument('--checkpoint');m.add_argument('--strategy',choices=['deep_only','adaptive','full_evidence'],default='adaptive')
        m.add_argument('--backend',choices=['mock','api'],default='mock');m.add_argument('--speed',type=float,default=1)
        m.add_argument('--max-seconds',type=float,default=130);m.add_argument('--start-sample',type=int,default=0);m.add_argument('--task',default='分析VR晕动症路径高低类与时间趋势')
    s=sub.add_parser('serve'); s.add_argument('--port',type=int,default=8880); s.add_argument('--host',default='127.0.0.1')
    args=parser.parse_args(); cfg=load_config(args.config)
    if hasattr(args,'profile'):
        from .config import profile_config
        cfg=profile_config(cfg,args.profile)
    if args.command=='audit':
        from .data import audit
        report=audit(cfg); print(json.dumps({k:v for k,v in report.items() if k!='files'},ensure_ascii=False,indent=2))
    elif args.command=='prepare':
        from .learning import prepare
        report=prepare(cfg); print('Prepared %d subjects'%len(report['subjects']))
    elif args.command in ('train','fivefold','evaluate','export'):
        from .learning import train,fivefold,evaluate,export_onnx
        selected=('eegnet','eegconformer') if getattr(args,'model',None)=='both' else (args.model,)
        results=[]
        for model_id in selected:
            if args.command=='train': result=train(cfg,args.epochs,model_id=model_id)
            elif args.command=='fivefold': result=fivefold(cfg,args.epochs,model_id=model_id)
            elif args.command=='export': result=export_onnx(cfg,model_id=model_id,checkpoint_id=args.checkpoint)
            else: result=evaluate(cfg,args.split,model_id=model_id,checkpoint_id=args.checkpoint)
            results.append(result)
        print(json.dumps(results,ensure_ascii=False,indent=2))
    elif args.command=='knowledge-index':
        from .knowledge import KnowledgeIndex
        print(json.dumps(KnowledgeIndex(cfg).build(),ensure_ascii=False,indent=2))
    elif args.command=='models':
        from .models import model_catalog
        print(json.dumps(model_catalog(cfg),ensure_ascii=False,indent=2))
    elif args.command in ('monitor','compare-strategies'):
        from .monitor import MonitorTask
        from .config import write_json
        from pathlib import Path
        request={'subject':args.subject,'model_id':args.model,'checkpoint_id':args.checkpoint,'strategy':args.strategy,
                 'backend':args.backend,'speed':args.speed,'max_seconds':args.max_seconds,'start_sample':args.start_sample,'task':args.task}
        reports=[]
        for strategy in (('deep_only','full_evidence','adaptive') if args.command=='compare-strategies' else (args.strategy,)):
            run=MonitorTask(cfg,{**request,'strategy':strategy});run.execute()
            if run.state=='failed': raise RuntimeError(run.error)
            reports.append(run.result);print(strategy,run.id,flush=True)
        if args.command=='compare-strategies':
            predictions=[[{k:p[k] for k in ('status','probability_high','predicted_class','coverage','valid_windows')} for p in r['paths']] for r in reports]
            report={'model_id':args.model,'reports':reports,'identical_path_results':all(p==predictions[0] for p in predictions),
                    'boundary':'same classifier; auxiliary tools explain EEG and do not change accuracy'}
            write_json(Path(cfg['output_root'])/'reports'/('strategy_comparison_'+args.model+'.json'),report)
        print(json.dumps([{'task_id':r['task_id'],'paths':r['paths'],'tool_counts':r['tool_counts'],'early_exits':r['early_exits']} for r in reports],ensure_ascii=False,indent=2))
    elif args.command=='replay':
        from .runtime import ReplayRun
        run=ReplayRun(cfg,args.subject,args.mode,args.backend,args.speed,args.max_seconds,args.task); run.execute()
        if run.state=='failed': raise RuntimeError(run.error)
        print(json.dumps(run.result,ensure_ascii=False,indent=2))
    elif args.command=='compare':
        from .runtime import ReplayRun
        from .config import write_json
        from pathlib import Path
        reports=[]
        for mode in ('fixed_p0','fixed_p1','rules','single_agent','hierarchical'):
            run=ReplayRun(cfg,args.subject,mode,args.backend,args.speed,args.max_seconds); run.execute()
            if run.state=='failed': raise RuntimeError(run.error)
            reports.append(run.result); print(mode,run.id,flush=True)
        write_json(Path(cfg['output_root'])/'reports/controller_comparison.json',{'reports':reports,'simulated':args.backend=='mock','speed':args.speed,'not_formal_efficacy_comparison':True})
    elif args.command=='serve':
        import uvicorn
        from .service import create_app
        uvicorn.run(create_app(cfg),host=args.host,port=args.port)

if __name__=='__main__': main()
