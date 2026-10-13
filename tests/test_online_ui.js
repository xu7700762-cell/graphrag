'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ui = require('../eeg_agent/web/online.js');
const windowAt = end => ({scope:'task',ordinal:1,start_seconds:end-5,end_seconds:end,valid:true,coverage:1,
  vrms:{available:true,probability_high:.6,mil_class:'High'},fatigue:{theta_alpha:2},emotion:{workload_index:3},
  waveform:{channels:['F3'],sampling_hz:64,values:[Array(320).fill(2)]}});

test('cloud failure never presents a numerical class as a cloud decision',()=>{
  assert.equal(ui.cloudView({status:'fallback',final_class:'High',numeric_class:'High'}).label,'—');
  assert.equal(ui.cloudView({status:'pending',final_class:'High'}).label,'—');
  assert.equal(ui.cloudView({status:'verified',final_class:'Low'}).label,'Low');
  assert.equal(ui.cloudView({status:'verified',final_class:null}).label,'—');
  assert.equal(ui.cloudView({status:'fallback',reports:{vrms:{error_type:'APITimeoutError'}}}).state,'云端请求超时');
});

test('provider diagnostics cannot contaminate session error or overwrite runtime measurements',()=>{
  const store=ui.restoreStore({state:'running',error:null,metrics:{total_windows:7},cloud:{pending:1}});
  ui.applyEvent(store,{sequence:1,type:'agent_call',payload:{status:'failed',error:'APITimeoutError',metrics:{total_windows:999},cloud:{pending:999}}});
  assert.equal(store.snapshot.error,null);
  assert.equal(store.snapshot.metrics.total_windows,7);
  assert.equal(store.snapshot.cloud.pending,1);
  ui.applyEvent(store,{sequence:2,type:'state',payload:{state:'failed',error:'overrun'}});
  assert.equal(store.snapshot.error,'overrun');
});

test('waveform adapts to sub-microvolt data without changing values',()=>{
  const values=Array.from({length:1920},(_,i)=>.5*Math.sin(i*.12));
  values[33]=null;
  const original=[...values], scale=ui.waveformScale(values);
  assert.ok(scale>=.5 && scale<1,scale);
  assert.deepEqual(values,original);
  assert.equal(ui.waveformScale([0,null,NaN]),.05);
});

test('trend axes keep tiny measurement changes proportional and labels distinct',()=>{
  for(const values of [[1.9701,1.9704,1.9699],[1.8811,1.8807],[.00001,.00002],[2,2]]){
    const axis=ui.trendAxis(values);
    assert.ok(axis.high-axis.low>=.02);
    assert.ok(axis.low<=Math.min(...values) && axis.high>=Math.max(...values));
    assert.notEqual(ui.number(axis.high,axis.digits),ui.number(axis.low,axis.digits));
    assert.ok((Math.max(...values)-Math.min(...values))/(axis.high-axis.low)<.1);
  }
  assert.deepEqual(ui.trendAxis([.4,.9],true),{low:0,high:1,digits:2});
});

test('saved timeout reports render concrete failure and preserve real validation reasons',()=>{
  assert.match(ui.reportText({error_type:'APITimeoutError',reason:'云端解释未通过来源核验'}),/API请求超时/);
  assert.equal(ui.reportText({error_type:'ValueError',reason:'数值描述缺少可核验来源'}),'数值描述缺少可核验来源');
  assert.equal(ui.reportText({available:true,explanation:'有效解释'}),'有效解释');
});

test('API absence permits local replay; real-time is default and accelerated modes stay explicit',()=>{
  assert.deepEqual(ui.requestBody('subject-10','1',true,false),{recording_id:'subject-10',speed:1,cloud:false,max_seconds:null});
  assert.equal(ui.requestBody('subject-10','10',true,true).speed,10);
  assert.equal(ui.requestBody('subject-10','max',false,true).speed,'max');
  assert.throws(()=>ui.requestBody('','1',true,true));
  assert.throws(()=>ui.requestBody('subject-10','2',true,true));
});

test('duplicate and old SSE events cannot duplicate windows or revert a new cloud result',()=>{
  const store=ui.restoreStore({id:'s',state:'running',paths:[],latest:{},events:[]});
  assert.equal(ui.applyEvent(store,{sequence:1,type:'window',payload:windowAt(5)}),true);
  assert.equal(ui.applyEvent(store,{sequence:1,type:'window',payload:windowAt(5)}),false);
  ui.applyEvent(store,{sequence:2,type:'path_completed',payload:{ordinal:1,complete:true,vrms:{mil_class:'High'},cloud:{status:'pending'}}});
  ui.applyEvent(store,{sequence:3,type:'cloud_verified',payload:{ordinal:1,result:{final_class:'Low'}}});
  assert.equal(ui.applyEvent(store,{sequence:2,type:'cloud_fallback',payload:{ordinal:1}}),false);
  assert.equal(store.windows.length,1);
  assert.equal(store.paths.length,1);
  assert.equal(ui.cloudView(store.paths[0].cloud).label,'Low');
  assert.equal(store.paths[0].vrms.mil_class,'High');
});

test('plots retain only the last 30 seconds and break unavailable measurements',()=>{
  const store=ui.emptyStore();
  for(let i=1;i<=30;i++)ui.applyEvent(store,{sequence:i,type:'window',payload:windowAt(i*5)});
  assert.equal(store.windows.length,6);
  assert.equal(store.waveforms.length,6);
  assert.equal(store.windows[0].end_seconds,125);
  const rejected=windowAt(155);rejected.valid=false;
  ui.applyEvent(store,{sequence:31,type:'window',payload:rejected});
  const point=ui.series(store).at(-1);
  assert.equal(point.probability,null);
  assert.equal(point.fatigue,null);
  assert.equal(point.workload,null);
});

test('snapshot restoration respects the event cursor and rebuilds bounded plots once',()=>{
  const latest=windowAt(50);
  const events=Array.from({length:10},(_,i)=>({sequence:i+10,type:'window',payload:windowAt((i+1)*5)}));
  const store=ui.restoreStore({id:'s',state:'paused',latest,paths:[{ordinal:2,cloud:{status:'verified',final_class:'High'}}],events});
  assert.equal(store.sequence,19);
  assert.equal(store.windows.length,6);
  assert.equal(store.waveforms.length,6);
  assert.equal(ui.applyEvent(store,events.at(-1)),false);
  assert.equal(store.paths[0].ordinal,2);
  ui.applyEvent(store,{sequence:20,type:'state',payload:{state:'running'}});
  assert.equal(store.snapshot.state,'running');
});

test('rest and task ordinals stay independent, and persisted last_sequence restores the cursor',()=>{
  const store=ui.restoreStore({id:'s',state:'completed',latest:{},last_sequence:500,events:[],paths:[
    {scope:'task',ordinal:1,vrms:{mil_class:'High'}},{scope:'rest',ordinal:1,vrms:{mil_class:null}}]});
  assert.equal(store.sequence,500);
  assert.equal(store.paths.length,1);
  assert.equal(store.paths[0].scope,'task');
  ui.applyEvent(store,{sequence:501,type:'path_completed',payload:{scope:'rest',ordinal:1}});
  assert.equal(store.paths.length,1);
  assert.equal(store.paths[0].vrms.mil_class,'High');
});

test('online DOM contract contains every bound element and no network chart dependency',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../eeg_agent/web/online.html'),'utf8');
  const javascript=fs.readFileSync(path.join(__dirname,'../eeg_agent/web/online.js'),'utf8');
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(match=>match[1]);
  assert.equal(new Set(ids).size,ids.length);
  for(const match of javascript.matchAll(/(?:\$\(|\btext\()'([^']+)'/g))assert.ok(ids.includes(match[1]),'missing element '+match[1]);
  for(const id of ['waveformCanvas','trendCanvas','pauseReplay','resumeReplay','stopReplay','pathRows'])assert.ok(ids.includes(id));
  assert.match(html,/id="cloudEnabled"[^>]*checked/);
  assert.match(html,/value="1" selected/);
  assert.match(html,/<a id="ragComparisonLink" href="\/rag-comparison">RAG 前后对比<\/a>/);
  assert.match(html,/<p id="ragVersionNote"[^>]*>[^<]*<\/p>/);
  assert.doesNotMatch(html,/<(?:script|link)[^>]+(?:src|href)="https?:/);
  assert.match(html,/src="\/assets\/online\.js\?v=[^"]+"/);
  assert.doesNotMatch(javascript,/\.innerHTML\s*=/);
  assert.match(javascript,/\/api\/online\/sessions/);
  assert.match(javascript,/events\?after=/);
});

test('browser controller starts local replay and restores a dropped SSE connection from snapshot',async()=>{
  const html=fs.readFileSync(path.join(__dirname,'../eeg_agent/web/online.html'),'utf8');
  const source=fs.readFileSync(path.join(__dirname,'../eeg_agent/web/online.js'),'utf8');
  class Element {
    constructor(){this.children=[];this.dataset={};this.listeners={};this.classList={toggle(){}};this.value='';this.checked=false;this.textContent='';}
    append(...nodes){this.children.push(...nodes);}
    replaceChildren(...nodes){this.children=[...nodes];}
    addEventListener(name,handler){this.listeners[name]=handler;}
    getBoundingClientRect(){return {width:0,height:0};}
  }
  const elements=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(match=>[match[1],new Element()]));
  elements.get('speedSelect').value='1';elements.get('cloudEnabled').checked=true;
  const storage=new Map(), requests=[], streams=[], frames=[], timers=[], listeners={};
  let snapshot={id:'session-test',state:'running',recording:{id:'subject-10'},speed:1,position_seconds:0,latest:{},paths:[],events:[],last_sequence:0,metrics:{},cloud:{requested:false}};
  const window={document:{getElementById:id=>elements.get(id),createElement:()=>new Element()},
    localStorage:{getItem:key=>storage.get(key)||null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)},
    addEventListener:(name,handler)=>{listeners[name]=handler;},requestAnimationFrame:handler=>{frames.push(handler);return frames.length;},
    setTimeout:handler=>{timers.push(handler);return timers.length;},clearTimeout(){},
    EventSource:class{constructor(url){this.url=url;streams.push(this);}close(){this.closed=true;}},
    fetch:async(url,options)=>{requests.push({url,options});let data;
      if(url==='/api/online/catalog')data={recordings:[{id:'subject-10',title:'测试记录'}],capabilities:{model_available:true,cloud_configured:false}};
      else if(url==='/api/online/sessions'&&!options)data=[];
      else data=JSON.parse(JSON.stringify(snapshot));
      return {ok:true,json:async()=>data};}
  };
  vm.runInNewContext(source,{window});
  await new Promise(resolve=>setImmediate(resolve));
  while(frames.length)frames.shift()();
  elements.get('recordingSelect').value='subject-10';
  await elements.get('replayForm').listeners.submit({preventDefault(){}});
  const start=requests.find(request=>request.options?.method==='POST');
  assert.equal(JSON.parse(start.options.body).cloud,false);
  assert.equal(JSON.parse(start.options.body).speed,1);
  assert.equal(streams.length,1);
  assert.match(streams[0].url,/events\?after=0$/);
  streams[0].onmessage({data:JSON.stringify({sequence:1,type:'window',payload:{...windowAt(5),
    metrics:{window_processing_p95_seconds:.3,total_windows:1},position_seconds:5}})});
  while(frames.length)frames.shift()();
  assert.equal(elements.get('vrmsProbability').textContent,'0.600');
  assert.equal(elements.get('thetaAlphaValue').textContent,'2.000');
  assert.equal(elements.get('latencyValue').textContent,'0.30 秒（P95）');
  streams[0].onmessage({data:JSON.stringify({sequence:2,type:'agent_call',payload:{status:'failed',error:'APITimeoutError'}})});
  while(frames.length)frames.shift()();
  assert.equal(elements.get('onlineError').hidden,true);
  snapshot={...snapshot,last_sequence:10,position_seconds:10,latest:windowAt(10)};
  streams[0].onerror();
  assert.ok(streams[0].closed);
  await timers.shift()();
  while(frames.length)frames.shift()();
  assert.equal(streams.length,2);
  assert.match(streams[1].url,/events\?after=10$/);
  assert.ok(requests.some(request=>request.url==='/api/online/sessions/session-test'));
  assert.equal(elements.get('positionValue').textContent,'00:10');
  streams[1].onmessage({data:JSON.stringify({sequence:11,type:'path_completed',payload:{scope:'task',ordinal:1,
    start_seconds:0,end_seconds:15,data_cutoff_seconds:15,complete:true,accepted_windows:3,total_windows:3,
    vrms:{mil_class:'High',numeric_class:'High'},cloud:{status:'pending'}}})});
  while(frames.length)frames.shift()();
  streams[1].onmessage({data:JSON.stringify({sequence:12,type:'cloud_verified',payload:{ordinal:1,result:{status:'verified',final_class:'Low',
    reports:{vrms:{explanation:'云端判断 Low'},fatigue:{explanation:'疲劳指标解释'},emotion:{explanation:'额区指标解释'},supervisor:{explanation:'<img src=x> 汇总'}}}}})});
  while(frames.length)frames.shift()();
  const report=elements.get('pathExplanations').children[0];
  assert.ok(report.children.some(node=>node.textContent==='疲劳指标解释'));
  assert.ok(report.children.some(node=>node.textContent==='<img src=x> 汇总'));
  report.open=true;
  streams[1].onmessage({data:JSON.stringify({sequence:13,type:'progress',payload:{position_seconds:16}})});
  while(frames.length)frames.shift()();
  assert.equal(elements.get('pathExplanations').children[0].open,true);
  streams[1].onmessage({data:JSON.stringify({sequence:14,type:'path_completed',payload:{scope:'task',ordinal:2,
    start_seconds:20,end_seconds:25,data_cutoff_seconds:25,complete:false,reason:'记录结束前缺少结束标记',
    accepted_windows:1,total_windows:1,vrms:{},cloud:{status:'not_requested'}}})});
  while(frames.length)frames.shift()();
  const unclosed=elements.get('pathExplanations').children[1];
  assert.match(unclosed.children[0].textContent,/路径未闭合/);
  assert.ok(unclosed.children.some(node=>node.textContent.includes('记录结束前缺少结束标记')));
  snapshot={...snapshot,state:'completed',error:null,paths:[{scope:'task',ordinal:1,complete:true,
    start_seconds:0,end_seconds:15,data_cutoff_seconds:15,accepted_windows:3,total_windows:3,vrms:{mil_class:'High'},
    cloud:{status:'fallback',final_class:null,reports:{vrms:{error_type:'APITimeoutError',reason:'独立 LLM 判断未完成'},
      fatigue:{available:true,explanation:'保留有效疲劳解释'},
      supervisor:{error_type:'APITimeoutError',reason:'云端解释未通过来源核验；保留本地指标'}}}}]};
  const fetches=requests.length;
  streams[1].onmessage({data:JSON.stringify({sequence:15,type:'state',payload:{state:'failed',
    error:'overrun: 在线计算队列已满',metrics:{window_processing_p95_seconds:.4},cloud:{pending:0,failed:1}}})});
  while(frames.length)frames.shift()();
  assert.equal(elements.get('sessionState').textContent,'运行失败');
  assert.equal(elements.get('latencyValue').textContent,'0.40 秒（P95）');
  assert.match(elements.get('onlineError').textContent,/overrun/);
  assert.ok(streams[1].closed);
  assert.equal(requests.length,fetches+1); // Terminal reconciliation is independent of immediate SSE rendering.
  await new Promise(resolve=>setImmediate(resolve));
  while(frames.length)frames.shift()();
  assert.equal(elements.get('onlineError').hidden,true); // A fresh error-free snapshot clears the runtime banner.
  const historical=elements.get('pathExplanations').children[0];
  assert.ok(historical.children.filter(node=>node.textContent.includes('API请求超时')).length===2);
  assert.ok(historical.children.some(node=>node.textContent==='保留有效疲劳解释'));
  assert.equal(elements.get('sessionState').textContent,'已完成');
  elements.get('recordingSelect').value='';
  await elements.get('replayForm').listeners.submit({preventDefault(){}});
  assert.match(elements.get('onlineError').textContent,/请选择/);
  assert.equal(elements.get('onlineError').hidden,false);
  listeners.resize();
  while(frames.length)frames.shift()();
  assert.equal(elements.get('onlineError').hidden,false); // Form errors survive unrelated rendering.
});
