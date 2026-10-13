import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit
from .config import digest, contract
from .cloud_config import read_settings, resolve_protocol

MODES = ('fixed_p0','fixed_p1','rules','single_agent','hierarchical')

def anonymous_summary(q, calibration, contract_hash, profile):
    allowed = ('observed_end_sample','unit','valid','reasons','peak_to_peak_uv','processed_peak_to_peak_uv',
               'flat_channel_fraction','jump_fraction','line_power_ratio','line_warning','frontal_eog_correlation','eog_std_uv')
    return {'quality':{k:q[k] for k in allowed if k in q}, 'calibration':calibration.summary(),
            'contract_hash':contract_hash,'current_profile':profile,'allowed_profiles':['P0','P1'],
            'profile_catalog':{'P0':'Continuous fixed causal reference/filter/256Hz/30-channel chain; no extra artifact regression',
                               'P1':'Same P0 chain plus frozen prior-baseline centered EOG linear regression; requires valid calibration and model support'},
            'input_contract':{'channels':30,'unit':'uV','sample_rate':256,'window_seconds':5,'normalization':'shared frozen training statistics'}}

def cloud_settings(path=None):
    return read_settings(path)

class CloudProvider:
    def __init__(self, backend, cfg, emit):
        if backend not in ('mock','api'): raise ValueError('backend must be mock or api')
        self.backend=backend;self.cfg=cfg;self.emit=emit;self.calls=[];self.lock=threading.Lock();self.client=None;self.budget_rejections=0;self.thinking_disabled=False
        if backend=='api':
            settings=cloud_settings(cfg.get('cloud_config_path'))
            if not all(settings.values()): raise ValueError('Configure EEG_API_BASE_URL/MODEL/KEY locally first')
            from openai import OpenAI
            self.client=OpenAI(base_url=settings['EEG_API_BASE_URL'],api_key=settings['EEG_API_KEY'],timeout=cfg['api_timeout_seconds'],max_retries=0)
            self.model=settings['EEG_API_MODEL']
            self.protocol=resolve_protocol(self.model, settings['EEG_API_PROTOCOL'])
            # Native DeepSeek V4 defaults to thinking; this bounded tool/report workflow
            # needs the final structured response within its existing output budget.
            self.thinking_disabled=(urlsplit(settings['EEG_API_BASE_URL']).hostname=='api.deepseek.com'
                                    and self.model in ('deepseek-flash','deepseek-pro'))

    def call(self, role, payload, tools=None, messages=None):
        with self.lock:
            if len(self.calls)>=self.cfg['api_max_calls']:
                self.budget_rejections+=1
                self.emit('agent_error',{'role':role,'reason':'call budget exhausted'}); raise RuntimeError('API call budget exhausted')
            record={'role':role,'backend':self.backend,'simulated':self.backend=='mock','status':'pending','input_tokens':0,'output_tokens':0}
            self.calls.append(record)
        begin=time.perf_counter()
        try:
            if self.backend=='mock':
                if role in ('Supervisor','VRMSAgent') and payload.get('workflow')=='vrms_v2':
                    output={'roles':['VRMSAgent'],'task_type':payload.get('task_type','temporal_monitoring'),
                            'explanation':'模拟语言解释；路径类别和全部数值以本地工具证据为准。',
                            'citations':[r['id'] for r in payload.get('knowledge',[])], 'requested_tools':[]}
                elif role=='Supervisor': output={'roles':['SignalQualityAgent','VRAnalysisAgent'],'instruction':'Analyze arrived EEG only'}
                elif role=='VRAnalysisAgent': output={'summary':'模拟语言汇总；数值以本地冻结工具结果为准。'}
                else:
                    q=payload['quality']; available=payload['calibration']['available']
                    p='P1' if 'P1' in payload['allowed_profiles'] and available and q['valid'] and q['frontal_eog_correlation']>=self.cfg['qc']['eog_enter_correlation'] else 'P0'
                    output={'profile':p,'observed_end_sample':q['observed_end_sample'],'contract_hash':payload['contract_hash']}
                record['status']='success'; return output
            base_messages=messages or [{'role':'system','content':
                'You are '+role+'. Use only supplied arrived anonymous tool measurements. No raw EEG, identity, labels, predictions-as-quality, arbitrary code, or frequency/channel changes. Supervisor returns JSON {roles:[SignalQualityAgent,VRAnalysisAgent],instruction:string}. VRAnalysisAgent returns JSON {summary:string}, describing the frozen result and coverage limits. Numerical results are immutable.'},
                {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
            # Conversation reports need room for evidence and citations; monitor
            # tool calls retain the existing bounded budget.
            phase=payload.get('generation_phase')
            budget={'planning':700,'specialist_report':2000,'supervisor_report':3000,
                    'vrms_tool_agent':4096,'vrms_independent_judge':8192}.get(phase,700)
            record['output_token_budget']=budget
            record['generation_phase']=phase
            protocol=getattr(self,'protocol',resolve_protocol(self.model))
            matched = payload.get('matched_judge_parameters')
            if matched is not None:
                if (phase != 'vrms_independent_judge' or tools or matched.get('model') != 'gpt-6.1-sol'
                        or self.model != matched.get('model')
                        or matched.get('reasoning') != {'effort': 'high'}
                        or matched.get('max_output_tokens') != 8192):
                    raise ValueError('Invalid pinned comparison judge contract')
                # This applies only to an explicitly pinned classification request.
                response = self.client.responses.create(**matched, input=base_messages)
                usage = getattr(response, 'usage', None)
                if usage:
                    record['input_tokens'] = getattr(usage, 'input_tokens', 0)
                    record['output_tokens'] = getattr(usage, 'output_tokens', 0)
                status = getattr(response, 'status', None)
                record.update(api_protocol='responses', finish_reason=status,
                              output_truncated=status == 'incomplete',
                              response_model=getattr(response, 'model', None),
                              judge_profile='matched_decisive_high_v1', reasoning_effort='high')
                if status != 'completed':
                    raise ValueError('Incomplete pinned judge response')
                output = json.loads(response.output_text)
                record['status'] = 'success'
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(output, ensure_ascii=False)}}],
                        'id': getattr(response, 'id', None), 'model': getattr(response, 'model', matched['model']),
                        'usage': usage.model_dump(mode='json') if usage else None}
            if phase=='vrms_tool_agent' and protocol=='responses':
                from .raw_tool_agent import SCHEMA
                kwargs={'model':self.model,'input':base_messages,'max_output_tokens':budget,
                        'store':False,
                        'text':{'format':{'type':'json_schema','name':'raw_eeg_agent_final','schema':SCHEMA,'strict':True}}}
                if self.model.startswith('gpt-'): kwargs['reasoning']={'effort':'medium'}
                if tools:
                    kwargs.update(tools=tools,parallel_tool_calls=True,
                                  tool_choice='required' if payload.get('step')==0 else 'auto')
                response=self.client.responses.create(**kwargs)
                usage=getattr(response,'usage',None)
                if usage:
                    record['input_tokens']=getattr(usage,'input_tokens',0)
                    record['output_tokens']=getattr(usage,'output_tokens',0)
                record['api_protocol']='responses'
                record['finish_reason']=getattr(response,'status',None)
                record['output_truncated']=record['finish_reason']=='incomplete'
                if record['finish_reason']!='completed':
                    raise ValueError('Incomplete GPT tool output')
                record['status']='success'
                return response.model_dump(mode='json')
            if protocol=='responses' and not tools:
                kwargs={'model':self.model,'input':base_messages,'max_output_tokens':budget,
                        'store':False,'text':{'format':{'type':'json_object'}}}
                if self.model.startswith('gpt-'): kwargs['reasoning']={'effort':'low'}
                response=self.client.responses.create(**kwargs)
                usage=getattr(response,'usage',None)
                if usage:
                    record['input_tokens']=getattr(usage,'input_tokens',0)
                    record['output_tokens']=getattr(usage,'output_tokens',0)
                record['api_protocol']='responses'
                status=getattr(response,'status',None)
                record['finish_reason']=status
                record['output_truncated']=status=='incomplete'
                record['text_response_available']=bool(getattr(response,'output_text',None))
                if status not in (None,'completed'):
                    raise ValueError('Incomplete API output')
                output=json.loads(response.output_text)
                record['status']='success'
                if phase=='vrms_independent_judge':
                    return {'choices':[{'finish_reason':'stop',
                        'message':{'content':json.dumps(output, ensure_ascii=False)}}],
                        'id':getattr(response,'id',None), 'model':self.model,
                        'usage':getattr(response,'usage',None).model_dump(mode='json')
                            if getattr(response,'usage',None) else None}
                return output
            kwargs={'model':self.model,'messages':base_messages,'temperature':0,'max_tokens':budget}
            if self.thinking_disabled: kwargs['extra_body']={'thinking':{'type':'disabled'}}
            if tools:
                kwargs['tools']=([{'type':'function','function':{k:t[k] for k in ('name','description','parameters')}}
                                  for t in tools] if phase=='vrms_tool_agent' else tools)
                kwargs['tool_choice']=({'type':'function','function':{'name':'corrective_evidence'}}
                                      if phase=='vrms_tool_agent' and payload.get('step')==0 else 'auto')
            else: kwargs['response_format']={'type':'json_object'}
            response=self.client.chat.completions.create(**kwargs)
            record['api_protocol']='chat_completions'
            record['response_model']=getattr(response,'model',None)
            if response.usage:
                record['input_tokens']=response.usage.prompt_tokens; record['output_tokens']=response.usage.completion_tokens
            reply=response.choices[0]
            finish=getattr(reply,'finish_reason',None)
            record['finish_reason']=finish if finish in ('stop','length','tool_calls','content_filter','function_call') else None
            record['output_truncated']=finish=='length'
            record['thinking_disabled']=self.thinking_disabled
            record['text_response_available']=bool(getattr(reply.message,'content',None))
            if finish=='length': raise ValueError('Incomplete API output')
            if phase=='vrms_tool_agent' and finish not in ('stop','tool_calls'):
                raise ValueError('Incomplete cloud tool output')
            record['status']='success'
            if phase in ('vrms_tool_agent','vrms_independent_judge'):
                return response.model_dump(mode='json')
            return reply.message if tools else json.loads(reply.message.content)
        except Exception as exc:
            record['status']='failed'; record['error']=type(exc).__name__
            self.emit('agent_error',{'role':role,'reason':type(exc).__name__}); raise
        finally:
            record['seconds']=time.perf_counter()-begin
            self.emit('agent_call',dict(record))

    def summary(self):
        with self.lock: records=[dict(c) for c in self.calls]
        return {'backend':self.backend,'simulated':self.backend=='mock','calls':len(records),
                'model_id':getattr(self,'model',None),'thinking_disabled':self.thinking_disabled,
                'input_tokens':sum(c['input_tokens'] for c in records),'output_tokens':sum(c['output_tokens'] for c in records),
                'failures':sum(c['status']=='failed' for c in records),'pending':sum(c['status']=='pending' for c in records),
                'total_call_seconds':sum(c.get('seconds',0) for c in records),
                'retries':0,'budget_rejections':self.budget_rejections,'pricing':'not supplied; no invented currency cost','records':records}

class SignalQualityAgent:
    def __init__(self,provider): self.provider=provider

    def plan(self,payload,role='SignalQualityAgent'):
        q=payload['quality']
        tools_values={'LineNoiseAssessor':{k:q[k] for k in ('line_power_ratio','line_warning')},
                      'EOGArtifactAssessor':{k:q[k] for k in ('frontal_eog_correlation','eog_std_uv')},
                      'BadChannelAssessor':{k:q[k] for k in ('flat_channel_fraction','valid','reasons')},
                      'AmplitudeAssessor':{k:q[k] for k in ('peak_to_peak_uv','jump_fraction')}}
        if self.provider.backend=='mock':
            for name,result in tools_values.items(): self.provider.emit('agent_tool',{'role':role,'name':name,'result':result,'simulated_planner':True})
            return self.provider.call(role,payload)
        tools=[{'type':'function','function':{'name':name,'description':'Read local arrived quality measurement; no mutation.',
                  'parameters':{'type':'object','properties':{},'additionalProperties':False}}} for name in tools_values]
        tools.append({'type':'function','function':{'name':'ProposeProfile','description':'Submit a proposal for a future complete window; local validation is mandatory.',
                      'parameters':{'type':'object','properties':{'profile':{'type':'string','enum':payload['allowed_profiles']}},'required':['profile'],'additionalProperties':False}}})
        messages=[{'role':'system','content':role+': read the profile catalog and input contract. First call EOGArtifactAssessor and BadChannelAssessor, then submit ProposeProfile. Prefer P0 for stable quality; P1 only with valid prior calibration, supported model and credible EOG interference. No frequency/reference/channel edits. Correlation reduction does not prove cleaning validity. Never use confidence or labels as quality.'},
                  {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
        measured=set()
        for _ in range(4):
            response=self.provider.call(role,payload,tools,messages)
            messages.append(response.model_dump(exclude_none=True))
            if not response.tool_calls: raise ValueError('Agent did not submit a registered tool proposal')
            proposal=None
            for call in response.tool_calls:
                name=call.function.name; arguments=json.loads(call.function.arguments or '{}')
                if name in tools_values and arguments=={}:
                    result=tools_values[name]; measured.add(name)
                elif name=='ProposeProfile' and set(arguments)=={'profile'} and {'EOGArtifactAssessor','BadChannelAssessor'}<=measured:
                    result={'status':'proposal only'}; proposal={'profile':arguments['profile'],'observed_end_sample':q['observed_end_sample'],'contract_hash':payload['contract_hash']}
                else: raise ValueError('Illegal or premature tool call')
                self.provider.emit('agent_tool',{'role':role,'name':name,'result':result})
                messages.append({'role':'tool','tool_call_id':call.id,'content':json.dumps(result)})
            if proposal: return proposal
        raise ValueError('Agent tool-step limit')

class VRAnalysisAgent:
    def __init__(self,provider,emit):self.provider=provider;self.emit=emit;self.failures=0
    def predict(self,predictor,x):
        probability=predictor.predict(x)
        self.emit('prediction_tool',{'role':'VRAnalysisAgent','tool':'FrozenEEGPredictor','model_hash':predictor.model_hash})
        return probability
    def report(self,result):
        # Labels, subject/path IDs, and underlying files never enter cloud context.
        safe={k:result[k] for k in ('status','probability_high','predicted_class','coverage','valid_windows','total_windows')}
        self.provider.emit('agent_tool',{'role':'VRAnalysisAgent','name':'PathAggregator','result':safe})
        try:
            result=self.provider.call('VRAnalysisAgent',{'frozen_tool_result':safe})
            if not isinstance(result,dict) or not isinstance(result.get('summary'),str):raise ValueError('Invalid language report')
            self.emit('language_report',{'role':'VRAnalysisAgent','summary':result['summary'][:2000],'immutable_numbers':True})
            return result
        except Exception as exc:
            self.failures+=1;self.emit('agent_error',{'role':'VRAnalysisAgent','phase':'report','reason':type(exc).__name__})

class Controller:
    def __init__(self,cfg,mode,backend,emit,allowed_profiles=('P0',),task='分析当前到达的EEG信号'):
        if mode not in MODES: raise ValueError('Unknown controller mode')
        self.cfg=cfg; self.mode=mode; self.emit=emit; self.allowed=set(allowed_profiles); self.hash=digest(contract(cfg))
        self.provider=CloudProvider(backend,cfg,emit); self.agent=SignalQualityAgent(self.provider); self.vr=VRAnalysisAgent(self.provider,emit)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cloud-agent'); self.future=None
        self.current='P0'; self.changed=-100000000; self.latest=None; self.cal=None; self.last_request=-100000000
        self.enter=0;self.exit=0;self.switches=0;self.rejected=0;self.pending=None;self.planning_failures=0;self.bad_count=0;self.routing_failures=0;self.task_context=task[:2000]
        if mode=='hierarchical':
            self.executor.submit(self.supervise,task)

    def supervise(self,task):
        try:
            result=self.provider.call('Supervisor',{'task':task[:2000],'available_roles':['SignalQualityAgent','VRAnalysisAgent']})
            if set(result.get('roles',[])) != {'SignalQualityAgent','VRAnalysisAgent'}: raise ValueError('Invalid routing')
            if isinstance(result.get('instruction'),str):self.task_context=result['instruction'][:2000]
            self.emit('routing',{'roles':result['roles'],'backend':self.provider.backend,'instruction':self.task_context})
        except Exception as exc:
            self.routing_failures+=1;self.emit('routing',{'status':'failed','reason':type(exc).__name__})

    def observe(self,q,cal,feedback=None):
        self.latest=q; self.cal=cal
        corr=q['frontal_eog_correlation']
        self.enter=self.enter+1 if q['valid'] and corr>=self.cfg['qc']['eog_enter_correlation'] else 0
        self.exit=self.exit+1 if corr<=self.cfg['qc']['eog_exit_correlation'] else 0
        now=q['observed_end_sample']
        self.bad_count=self.bad_count+1 if not q['valid'] else 0
        interval=10 if self.bad_count>=2 else self.cfg['agent_review_seconds']
        if self.mode in ('single_agent','hierarchical') and self.future is None and now-self.last_request>=interval*1024:
            self.last_request=now
            payload=anonymous_summary(q,cal,self.hash,self.current)
            payload['allowed_profiles']=sorted(self.allowed)
            payload['task']=self.task_context
            if feedback: payload['processing_feedback']=feedback
            self.future=self.executor.submit(self.agent.plan,payload,'SingleAgent' if self.mode=='single_agent' else 'SignalQualityAgent')

    def validate(self,proposal,now):
        if not isinstance(proposal,dict) or set(proposal)!={'profile','observed_end_sample','contract_hash'}: return 'schema'
        if proposal['profile'] not in self.allowed: return 'unsupported_profile'
        if proposal['contract_hash']!=self.hash: return 'contract_version'
        stamp=proposal['observed_end_sample']
        if not isinstance(stamp,int) or stamp>now or now-stamp>self.cfg['proposal_ttl_seconds']*1024: return 'expired_or_future'
        if proposal['profile']=='P1' and (not self.cal or not self.cal.available or not self.latest or not self.latest['valid']): return 'calibration_or_quality'
        return None

    def choose(self,now):
        target=self.current
        if self.mode=='fixed_p0': target='P0'
        elif self.mode=='fixed_p1': target='P1' if self.cal and self.cal.available and 'P1' in self.allowed else 'P0'
        elif self.mode=='rules':
            fresh=self.latest and now-self.latest['observed_end_sample']<=self.cfg['proposal_ttl_seconds']*1024
            if not fresh:
                target='P0'; self.enter=0; self.exit=0
            else:
                if self.cal and self.cal.available and 'P1' in self.allowed and self.enter>=2: target='P1'
                if self.exit>=2 or not self.cal or not self.cal.available: target='P0'
        else:
            if self.future and self.future.done():
                try: self.pending=self.future.result()
                except Exception as exc:
                    self.pending=None;self.planning_failures+=1
                    self.emit('agent_error',{'role':'SignalQualityAgent','phase':'plan','reason':type(exc).__name__})
                self.future=None
            if self.pending:
                reason=self.validate(self.pending,now)
                if reason:
                    self.rejected+=1; self.emit('proposal_rejected',{'reason':reason,'proposal':self.pending})
                else: target=self.pending['profile']
                self.pending=None
        if target!=self.current and now-self.changed>=self.cfg['minimum_hold_seconds']*1024:
            self.current=target; self.changed=now; self.switches+=1
            self.emit('profile_switch',{'profile':target,'effective_start_sample':now})
        return self.current

    def close(self):
        self.executor.shutdown(wait=True)
        if self.future and self.future.done():
            # A decision that never reaches a future window is not silently executed.
            if self.future.exception():
                self.planning_failures+=1;self.emit('agent_error',{'phase':'final_plan','reason':type(self.future.exception()).__name__})
            else:self.emit('proposal_unused',{'reason':'stream ended'})
        cloud=self.provider.summary()
        return {'switches':self.switches,'rejected_proposals':self.rejected,'planning_failures':self.planning_failures,
                'routing_failures':self.routing_failures,'report_failures':self.vr.failures,
                'language_workflow_complete':not (cloud['failures'] or cloud['budget_rejections'] or self.planning_failures or self.routing_failures or self.vr.failures),'cloud':cloud}
