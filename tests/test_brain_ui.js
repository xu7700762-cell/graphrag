/* Workstation state regressions with a DOM double. No server, training, or cloud calls. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const web = path.join(__dirname, '../eeg_agent/web');
const source = fs.readFileSync(path.join(web, 'brain.js'), 'utf8');
const html = fs.readFileSync(path.join(web, 'brain.html'), 'utf8');
const marker = source.indexOf("\n$('questionForm').onsubmit");
assert(marker > 0);

class Node {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.attributes = {};
    this._text = ''; this.value = ''; this.hidden = false; this.disabled = false; this.checked = false;
    this.classes = new Set(); this.style = {};
    this.classList = {add: (...xs) => xs.forEach(x => this.classes.add(x)), remove: (...xs) => xs.forEach(x => this.classes.delete(x)), contains: x => this.classes.has(x), toggle: (x, force) => { const has = force === undefined ? !this.classes.has(x) : force; has ? this.classes.add(x) : this.classes.delete(x); return has; }};
  }
  get className() { return [...this.classes].join(' '); }
  set className(value) { this.classes = new Set(value.split(' ').filter(Boolean)); }
  get textContent() { return this._text + this.children.map(n => n.textContent).join(''); }
  set textContent(value) { this._text = String(value); this.children = []; }
  get childNodes() { return this.children; }
  get firstChild() { return this.children[0]; }
  get options() { return this.children.filter(n => n.tagName === 'OPTION'); }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this._text = ''; this.children = [...items]; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  removeAttribute(key) { delete this.attributes[key]; }
  focus() { this.focused = true; }
  click() { this.clicked = true; }
  remove() { this.removed = true; }
  querySelector() { return this.firstChild; }
}
function harness(hash = '#conversation') {
  const nodes = new Map([...html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"/g)].map(m => [m[2], new Node(m[1])]));
  const element = id => { if (!nodes.has(id)) nodes.set(id, new Node()); return nodes.get(id); };
  element('backend').value = 'api';
  element('backend').append(new Node('option')); element('backend').firstChild.value = 'api';
  const pages = ['conversation', 'toolkits', 'knowledge'].map(element);
  pages.forEach((node, i) => { node.id = ['conversation', 'toolkits', 'knowledge'][i]; });
  const nav = pages.map(page => { const node = new Node('button'); node.dataset.page = page.id; return node; });
  const domains = ['vrms', 'fatigue', 'emotion'].map(value => { const n = new Node('input'); n.value = value; n.checked = true; return n; });
  let handler = async () => { throw Error('Unexpected request in an offline test'); };
  const requests = [], saved = new Map();
  const context = vm.createContext({
    document: {getElementById: element, createElement: tag => new Node(tag), createElementNS: (_, tag) => new Node(tag), createTextNode: text => { const node = new Node(); node.textContent = text; return node; }, body: new Node('body'), querySelectorAll: selector => selector === '.page' ? pages : selector === 'button.nav-item' ? nav : selector.startsWith('input[name=') ? domains.filter(n => !selector.includes(':checked') || n.checked) : []},
    location: {hash}, window: {matchMedia: () => ({matches: false}), scrollTo() {}},
    localStorage: {getItem: key => saved.get(key) || null, setItem: (key, value) => saved.set(key, value), removeItem: key => saved.delete(key)},
    fetch: async (url, options) => { requests.push({url, options}); return {ok: true, json: async () => handler(url, options)}; },
    setTimeout: () => 1, clearTimeout() {}, URL, URLSearchParams, Blob, EventSource: class { close() {} }
  });
  context.history = {pushState: (_, __, value) => { context.location.hash = value; }};
  vm.runInContext(source.slice(0, marker), context);
  return {context, element, pages, nav, domains, requests, run: text => vm.runInContext(text, context), setHandler: fn => { handler = fn; }};
}
function ready(h) {
  h.run("initializing=false; apiConfigured=true; cloudSettings={configured:true,model_id:'gpt-6.1-sol',active_profile:'cpa_gpt',in_use:false}; updateControls();");
}
function check(name, fn) { return Promise.resolve(fn()).then(() => process.stdout.write('PASS ' + name + '\n')); }

(async () => {
  await check('removed evaluation links fall back to the raw EEG conversation', () => {
    const h = harness('#evaluation'); ready(h);
    h.context.switchPage('evaluation');
    assert.deepEqual(h.pages.map(p => p.hidden), [false,true,true]);
    assert.equal(h.context.location.hash, '#conversation');
    for (const id of ['vrmsPath','vrmsSeed','researchSeed','taskEvidence','evaluation']) assert(!html.includes('id="' + id + '"'));
    assert(!source.includes('/api/brain/vrms/'));
  });
  await check('retired history remains downloadable but cannot send or recreate its old path', async () => {
    const h = harness(); ready(h);
    const original = {id:'retired',backend:'api',state:'idle',domains:['vrms'],input_policy:'research_path',vrms_path_id:'path-000',pipeline_id:'femba_mil_tools_gpt_seed2028',messages:[{role:'assistant',text:'历史结果',markdown:'历史报告'}],events:[]};
    const saved = JSON.stringify(original);
    h.context.restoreSnapshot(original); h.element('question').value='继续推理'; h.context.updateControls();
    assert.equal(h.element('sendQuestion').disabled,true);
    assert.equal(h.element('question').disabled,true);
    assert.equal(h.element('downloadReport').disabled,false);
    assert.match(h.element('sessionModeText').textContent,/旧研究入口已移除/);
    await h.context.submitQuestion({preventDefault(){}});
    assert.equal(h.requests.length,0);
    assert.equal(JSON.stringify(original),saved);
    h.setHandler((url,options) => options?.method === 'POST' ? {id:'new-record',backend:'api',state:'idle',input_policy:'raw_eeg_only'} : []);
    h.context.newConversation(); h.element('question').value='分析原始 EEG';
    await h.context.submitQuestion({preventDefault(){}});
    const create = h.requests.find(item => item.url === '/api/brain/sessions' && item.options?.method === 'POST');
    const payload = JSON.parse(create.options.body);
    assert.equal(payload.input_policy,'raw_eeg_only');
    assert(!('pipeline_id' in payload) && !('vrms_path_id' in payload) && !('task_id' in payload));
    assert.equal(h.element('question').disabled,false);
  });
  await check('raw GPT results show the final class without a numerical-fusion result', () => {
    const h = harness();
    h.run(`tools=[{agent:'vrms',tool:'vrms.raw_model',result:{tool:'vrms.raw_model',gpt_requested:true,
      gpt_successful_paths:1,gpt_failed_paths:0,status:'ok',measurements:{segments:[{ordinal:1,
      mil_class:'High',numeric_fusion:{predicted_class:'High',raw_combined_score:.8},predicted_class:'Low',
      cloud_judgment:{available:true,final_class:'Low',confidence:'low',rounds:[{}],tool_outputs:{corrective_evidence:{raw_combined_class:'High'}}}}]}}}];renderTools();`);
    const text = h.element('toolsPanel').textContent;
    assert(text.includes('GPT 最终 Low') && text.includes('置信度 低'));
    assert(!text.includes('numeric_fusion') && !text.includes('raw_combined_score') && !text.includes('数值融合'));
    assert.equal(h.run('tools[0].result.measurements.segments[0].numeric_fusion.raw_combined_score'), .8);
  });
  await check('EEG tools-only agents remain usable without a deployed classifier', () => {
    const h = harness(); ready(h);
    h.context.renderCatalog({agents: [
      {id:'vrms',name:'VRMSAgent',model_status:'femba_mil_fusion_cloud',tools:[]},
      {id:'fatigue',name:'FatigueAgent',model_status:'tools_only',tools:[]},
      {id:'emotion',name:'EmotionAgent',model_status:'tools_only',tools:[]}
    ]});
    for (const card of h.element('toolkitCards').children.slice(1)) {
      const status = card.children.find(node => node.classList.contains('model-status'));
      assert.match(status.textContent, /EEG 工具分析模式.*无需分类模型/);
      assert.equal(status.classList.contains('unavailable'),false);
      assert(!card.textContent.includes('尚未部署'));
    }
    h.domains[0].checked = false;
    h.element('question').value = '请根据 EEG 工具解释指标';
    h.context.updateControls();
    assert.equal(h.element('sendQuestion').disabled,false);
  });
  await check('raw EEG source can be resolved by SSE without a manual path selection', () => {
    const h = harness(); ready(h);
    h.context.renderRecordings([{id:'subject-10',title:'被试 10 · 原始 EEG'}]);
    h.run("session={id:'raw-chat',input_policy:'raw_eeg_only'};sessionId='raw-chat';");
    h.context.handleEvent({sequence:0,kind:'recording_selected',payload:{recording_id:'subject-10',title:'被试 10 · 原始 EEG',input_policy:'raw_eeg_only',turn_id:1}});
    assert.equal(h.element('recordingSource').value,'subject-10');
    assert.match(h.element('sessionContext').textContent,/被试 10.*仅分析原始波形/);
    assert(!html.includes('id="legacyPathField"'));
    assert.match(h.element('vrmsPipelineNote').textContent,/问卷仅用于事后评价/);
  });
  await check('raw recording binding survives history restoration', () => {
    const h = harness(); ready(h);
    h.context.renderRecordings([{id:'subject-10',title:'被试 10 · 原始 EEG'}]);
    h.context.restoreSnapshot({id:'raw-chat',backend:'api',state:'idle',domains:['vrms','fatigue','emotion'],input_policy:'raw_eeg_only',recording_id:'subject-10',messages:[],events:[]});
    assert.equal(h.element('recordingSource').value,'subject-10');
    assert.equal(h.element('recordingSource').disabled,true);
    assert(!html.includes('id="legacyMeasurementOptions"'));
    assert.match(h.element('parameterSummary').textContent,/原始 EEG 已关联/);
    assert.equal(h.context.modelStatus('raw_vrms_model').unavailable,false);
  });
  await check('raw EEG guidance asks for direct path and event-rest comparisons', () => {
    const h=harness(); ready(h);
    assert.match(h.element('vrmsPipelineNote').textContent,/mark20→mark22.*mark22→下一次 mark20.*直接比较/);
    const prompts=[...html.matchAll(/data-prompt="([^"]+)"/g)].map(match=>match[1]);
    assert(prompts.some(prompt=>/第一条与最后一条完整路径.*疲劳和心理负荷/.test(prompt)));
    assert(prompts.some(prompt=>/mark22→下一次 mark20.*分别比较首末完整路径、首末完整休息段/.test(prompt)));
    assert.match(html,/比较路径之间、休息段之间的指标/);
    assert.match(html,/路径与休息状态各自独立比较/);
    assert(!prompts.some(prompt=>/每条完整路径与相邻/.test(prompt)));
    assert(!prompts.some(prompt=>/基线|baseline/i.test(prompt)));
    h.context.renderCatalog({agents:[{id:'fatigue',name:'FatigueAgent',tools:[]},{id:'emotion',name:'EmotionAgent',tools:[]}]});
    assert.match(h.element('toolkitCards').textContent,/mark22→下一次 mark20.*休息段直接比较/);
  });
  await check('retired raw opening-reference answers are marked without changing stored or rendered evidence', () => {
    const h=harness(); ready(h);
    const old={role:'assistant',turn_id:1,text:'疲劳指标相对基线下降50.84%。',markdown:'旧报告：基线3.4012，任务期1.6721。',generation_status:'cloud_verified',
      measurement_context:{source:'raw_recording',input_policy:'raw_eeg_only'},
      agent_reports:[{agent:'fatigue',report:'基线3.4012，任务期1.6721。',tool_results:[{tool:'fatigue.raw_spectrum',result:{measurements:{baseline_value:3.4012,baseline_change:{text:'下降50.84%'}}}}]}]};
    const saved=JSON.stringify(old);
    h.context.restoreSnapshot({id:'old-raw',backend:'api',state:'idle',domains:['fatigue'],messages:[old],events:[{sequence:0,kind:'synthesis',payload:old}]});
    assert.equal(h.element('messages').children.length,1);
    assert.match(h.element('messages').textContent,/旧分析使用记录开头参考，已停用，请重新提问/);
    assert.match(h.element('messages').textContent,/疲劳指标相对基线下降50\.84%/);
    assert.match(h.element('messages').textContent,/基线3\.4012，任务期1\.6721/);
    assert.equal(h.run('latestReport'),old.markdown);
    assert.equal(JSON.stringify(old),saved);
    const inferred=harness(); ready(inferred);
    inferred.context.addMessage({...old,measurement_context:undefined,text:'任务期疲劳指标为1.6721。'});
    assert.match(inferred.element('messages').textContent,/旧分析使用记录开头参考/);
  });
  await check('new event-only answers and independent knowledge or monitoring answers receive no legacy warning', () => {
    const h=harness(); ready(h);
    const tool={agent:'fatigue',report:'逐路径指标',tool_results:[{tool:'fatigue.raw_spectrum',result:{measurements:{baseline_value:3.4012}}}]};
    for (const [index,message] of [
      {text:'疲劳指标直接比较路径与休息段。',measurement_context:{source:'raw_recording',comparison_policy:'event_segments_only',baseline_policy:'disabled'},agent_reports:[tool]},
      {text:'基线用于解释旧方法，当前已停用。',measurement_context:{source:'raw_recording',baseline_policy:'disabled'},agent_reports:[tool]},
      {text:'先行基线是什么？这里介绍它的方法。'},
      {text:'监测结果相对先行基线上升。',measurement_context:{source:'monitor_task'},agent_reports:[{agent:'fatigue',tool_results:[{tool:'fatigue.spectral_ratios',result:{measurements:{baseline_value:2}}}]}]},
      {text:'请解释原始文件方法。',measurement_context:{source:'raw_recording'}}
    ].entries()) h.context.handleEvent({sequence:index,kind:'synthesis',payload:{...message,generation_status:'cloud_verified',turn_id:index+1}});
    assert(!h.element('messages').textContent.includes('旧分析使用记录开头参考'));
    assert.equal(h.element('messages').children.length,5);
  });
  await check('a copied three-agent question includes VRMS despite a restored two-domain selection', async () => {
    const h = harness(); ready(h);
    h.context.restoreSnapshot({id:'two-domain-history',backend:'api',state:'idle',domains:['fatigue','emotion'],messages:[],events:[]});
    h.setHandler((url,options) => url === '/api/brain/sessions' && options?.method === 'POST' ? {id:'all-three',backend:'api',state:'idle',domains:JSON.parse(options.body).domains} : []);
    h.context.newConversation();
    h.element('question').value='请分析被试10的原始 EEG，让三个 Agent 协作回答：哪些路径被 VRMS 模型判为高眩晕类？疲劳和心理负荷指标怎样变化？';
    h.context.updateControls();
    assert.match(h.element('scopeLabel').textContent,/3 个领域.*按问题调整/);
    await h.context.submitQuestion({preventDefault(){}});
    const create=h.requests.find(item=>item.url==='/api/brain/sessions' && item.options?.method==='POST');
    assert.deepEqual(JSON.parse(create.options.body).domains,['vrms','fatigue','emotion']);
    assert.equal(h.domains[0].checked,true);
    assert(!h.element('agentNodes').children[0].textContent.includes('未参与'));
  });
  await check('natural-language VRMS requests supplement manual scope without requiring a three-agent phrase', async () => {
    const h = harness(); ready(h); h.domains[0].checked=false;
    h.element('question').value='被试10的首条高眩晕类路径从哪里开始？疲劳和心理负荷怎样变化？';
    h.setHandler((url,options) => url === '/api/brain/sessions' && options?.method==='POST' ? {id:'domain-words',backend:'api',state:'idle'} : []);
    await h.context.submitQuestion({preventDefault(){}});
    const create=h.requests.find(item=>item.url==='/api/brain/sessions' && item.options?.method==='POST');
    assert.deepEqual(JSON.parse(create.options.body).domains,['vrms','fatigue','emotion']);
  });
  await check('explicit single-agent requests narrow scope and generic path questions retain manual scope', async () => {
    for (const question of ['只让 FatigueAgent 分析被试10的疲劳指标。','分析被试10每条路径的疲劳指标。','请解释刚才的结果。']) {
      const h=harness(); ready(h);
      if (!question.startsWith('只让')) h.domains.forEach(n=>{n.checked=n.value==='fatigue';});
      h.element('question').value=question;
      h.setHandler((url,options) => url === '/api/brain/sessions' && options?.method==='POST' ? {id:'fatigue-only',backend:'api',state:'idle'} : []);
      await h.context.submitQuestion({preventDefault(){}});
      const create=h.requests.find(item=>item.url==='/api/brain/sessions' && item.options?.method==='POST');
      assert.deepEqual(JSON.parse(create.options.body).domains,['fatigue']);
    }
  });
  await check('explicit domain requests are sendable when old manual scope is empty', () => {
    const h=harness(); ready(h); h.domains.forEach(n=>{n.checked=false;});
    h.element('question').value='请让全部 Agent 分析被试10的原始 EEG。'; h.context.updateControls();
    assert.equal(h.element('sendQuestion').disabled,false);
    h.element('question').value='请分析这个文件。'; h.context.updateControls();
    assert.equal(h.element('sendQuestion').disabled,true);
  });
  await check('backend domain selection updates a locked existing conversation and restores its workflow', () => {
    const h=harness(); ready(h);
    h.context.restoreSnapshot({id:'two-domain-history',backend:'api',state:'idle',domains:['fatigue','emotion'],messages:[],events:[]});
    h.context.handleEvent({sequence:0,kind:'domains_selected',payload:{domains:['vrms','fatigue','emotion'],added_domains:['vrms'],turn_id:1}});
    h.context.handleEvent({sequence:1,kind:'agent_status',payload:{agent:'vrms',status:'completed',turn_id:1}});
    assert.equal(h.domains[0].checked,true); assert.equal(h.domains[0].disabled,true);
    assert.equal(h.run('session.domains.join(",")'),'vrms,fatigue,emotion');
    assert.match(h.element('agentNodes').children[0].textContent,/VRMSAgent.*已完成/);
    assert.match(h.element('activityPanel').textContent,/本轮协作领域.*VRMSAgent.*已按问题补齐/);
    h.context.handleEvent({sequence:2,kind:'domains_selected',payload:{domains:['fatigue'],added_domains:[],turn_id:2}});
    assert.equal(h.domains[0].checked,false); assert.equal(h.domains[1].checked,true);
    assert.match(h.element('agentNodes').children[0].textContent,/未参与/);
    assert.equal(h.element('scopeLabel').textContent,'1 个领域');
  });
  await check('Supervisor review shows checked question items and topics before a verified synthesis', () => {
    const h=harness(); ready(h);
    h.context.handleEvent({sequence:0,kind:'synthesis_review',session_id:'raw-chat',payload:{requested_items:7,verified_items:7,topics:['疲劳','心理负荷'],turn_id:1}});
    assert.match(h.element('activityPanel').textContent,/Supervisor 逐项核验.*已核验 7\/7 项.*疲劳、心理负荷/);
    assert.equal(h.element('synthesisStatus').textContent,'核验已完成');
    assert.match(h.element('workingText').textContent,/已核验 7\/7 项.*正在形成综合回答/);
    h.context.handleEvent({sequence:1,kind:'synthesis',payload:{text:'综合回答',generation_status:'cloud_verified',turn_id:1}});
    assert.match(h.element('synthesisDescription').textContent,/已逐项核验问题并形成综合回答.*7\/7 项.*疲劳、心理负荷/);
    assert.equal(h.element('synthesisNode').classList.contains('active'),false);
    assert.equal(h.element('synthesisNode').classList.contains('done'),true);
    h.context.handleEvent({sequence:2,kind:'user_message',payload:{text:'新的问题',turn_id:2}});
    assert.equal(h.element('synthesisStatus').textContent,'待汇总');
    h.context.handleEvent({sequence:3,kind:'synthesis',payload:{text:'普通回答',generation_status:'cloud_verified',turn_id:2}});
    assert(!h.element('synthesisDescription').textContent.includes('逐项核验'));
  });
  await check('saved Supervisor review restores from synthesis payload without claiming verification for fallback', () => {
    const review={requested_items:4,verified_items:4,topics:['心理负荷']};
    for (const status of ['cloud_verified','fallback']) {
      const h=harness(); ready(h);
      h.context.restoreSnapshot({id:'review-history',backend:'api',state:'idle',domains:['emotion'],messages:[],events:[{sequence:0,kind:'synthesis',payload:{text:'历史回答',generation_status:status,synthesis_review:review,turn_id:1}}]});
      if (status==='cloud_verified') assert.match(h.element('synthesisDescription').textContent,/逐项核验.*4\/4 项.*心理负荷/);
      else {
        assert.match(h.element('synthesisDescription').textContent,/云端生成失败.*本地证据回退/);
        assert(!h.element('synthesisDescription').textContent.includes('逐项核验'));
      }
    }
    const h=harness(); ready(h);
    h.context.handleEvent({sequence:0,kind:'synthesis',payload:{text:'普通回答',generation_status:'cloud_verified',synthesis_review:{requested_items:0,verified_items:0,topics:[]},turn_id:1}});
    assert(!h.element('synthesisDescription').textContent.includes('逐项核验'));
  });
  await check('restoring SSE history does not duplicate messages or lose report downloads', () => {
    const h = harness(); ready(h);
    const user={role:'user',turn_id:'turn-1',text:'问题'}, answer={role:'assistant',turn_id:'turn-1',text:'答案',generation_status:'cloud_verified'};
    h.context.restoreSnapshot({id:'history-record',backend:'api',state:'idle',domains:['vrms'],messages:[user,answer],events:[{sequence:0,kind:'user_message',payload:{turn_id:'turn-1',text:'问题'}},{sequence:1,kind:'synthesis',payload:{turn_id:'turn-1',text:'答案',generation_status:'cloud_verified'}}]});
    assert.equal(h.element('messages').children.length,2);
    assert.equal(h.element('downloadReport').disabled,false);
    assert.equal(h.domains.filter(n=>n.checked).length,1);
    assert.match(h.element('backendBadge').textContent,/真实 API 已生成/);
  });
  await check('background history restoration preserves the requested page and address', async () => {
    const h = harness('#knowledge'); ready(h); h.context.switchPage('knowledge');
    h.setHandler(url => url.endsWith('/sessions') ? [] : {id:'history-record',backend:'mock',state:'idle',domains:['vrms'],messages:[{role:'user',text:'问题'},{role:'assistant',text:'旧回答'}],events:[]});
    await h.context.openSession('history-record',false);
    assert.equal(h.context.location.hash,'#knowledge');
    assert.deepEqual(h.pages.map(p=>p.hidden),[true,true,false]);
    assert.equal(h.element('messages').children.length,2);
  });
  await check('Markdown download preserves report content, file identity and encoding', async () => {
    const h = harness(); ready(h); let file;
    h.context.URL = {createObjectURL: blob => {file=blob;return 'blob:local-report';},revokeObjectURL() {}};
    h.run("latestReport='科研报告\\nACC 72.60%\\n来源与证据';sessionId='history-record';downloadReport();");
    assert.equal(await file.text(),'科研报告\nACC 72.60%\n来源与证据');
    assert.equal(file.type,'text/markdown;charset=utf-8');
    const link=h.context.document.body.children.at(-1);
    assert.equal(link.download,'history-record.md');
    assert.equal(link.href,'blob:local-report');
    assert.equal(link.clicked,true); assert.equal(link.removed,true);
  });
  await check('model aliases cover catalog, old answers, citations and tool JSON without changing stored evidence', () => {
    const h = harness(); ready(h);
    const result = {available:true,seed:2028,measurements:{pipeline:{id:'femba_mil_tools_gpt_seed2028',seed:2028},mil:{probability_high:.33},femba_mean:{score:-.4216}},provenance:{training_seed:2028,signal_sha256:'abc'}};
    const snapshot = JSON.stringify(result);
    h.context.renderCatalog({agents:[{id:'vrms',name:'VRMSAgent',tools:[{id:'vrms.femba_pipeline',name:'FEMBA 模型'}]}]});
    h.context.addMessage({role:'assistant',turn_id:'legacy',text:'FEMBA + MIL，seed 2028；概率 0.33。'});
    h.context.renderCitations(h.element('citationsPanel'),[{title:'FEMBA seed2028 路径合同',source:'outputs/vrms_agent_seed_retest/seed2028_20261008',text:'模型输出 0.33。'}]);
    h.context.handleEvent({sequence:1,kind:'tool_result',payload:{agent:'vrms',tool:'vrms.femba_pipeline',result}});
    for (const id of ['toolkitCards','vrmsBoundaryText','messages','citationsPanel','toolsPanel','activityPanel']) {
      assert.doesNotMatch(h.element(id).textContent,/FEMBA|seed/i);
    }
    assert.match(h.element('toolsPanel').textContent,/VRMS 模型推理/);
    assert.match(h.element('toolsPanel').textContent,/0\.33|-0\.4216/);
    assert.match(h.element('citationsPanel').textContent,/来源：本地研究资料（来源已登记）/);
    assert.equal(JSON.stringify(result),snapshot);
  });
  await check('Markdown presentation removes configuration keys while keeping raw evidence and measurements', async () => {
    const h = harness(); ready(h); let file;
    h.context.URL = {createObjectURL: blob => {file=blob;return 'blob:report';},revokeObjectURL(){}};
    const report = 'FEMBA seed2028\n```json\n' + JSON.stringify({seed:2028,probability_high:.33,feature:'femba_mean'}) + '\n```\n- [source-1] outputs/vrms_agent_seed_retest/seed2028_20261008 / 历史对照';
    h.context.reportForTest = report;
    h.run("latestReport=reportForTest;downloadReport();");
    assert.doesNotMatch(await file.text(),/FEMBA|seed/i);
    assert.match(await file.text(),/0\.33/);
    assert.match(await file.text(),/本地研究资料（来源已登记）/);
    assert.equal(h.run('latestReport'),report);
  });
  await check('incomplete comparison hides final metrics while showing completed scope and all anonymous paths', async () => {
    const h = harness(); ready(h);
    assert.match(html.match(/<nav class="primary-nav"[\s\S]*?<\/nav>/)[0],/data-page="comparison"/);
    const saved = {available:true,status:'graph_in_progress',arms:[
      {id:'regular',label:'普通RAG',n:146,valid:146,pending:0,correct:107,accuracy:107/146,bacc:.7321,high_recall:.75,low_recall:.7143,failed:0},
      {id:'graph',label:'GraphRAG',n:146,valid:24,pending:122,correct:19,accuracy:19/146,bacc:.1289,high_recall:.1579,low_recall:.1,failed:0}],
      paired:{corrected:0,damaged:1,net_correct:-1,both_valid:24,fixed_denominator:146},
      rows:Array.from({length:146},(_,i)=>({query_id:'q'+String(i+1).padStart(3,'0'),regular_class:'Low',regular_confidence:'low',graph_class:i<24?'Low':null,graph_confidence:i<24?'medium':null,status:i<24?'valid':'pending'})),notes:[]};
    const original = JSON.stringify(saved);
    h.setHandler(url => {assert.equal(url,'/api/rag/comparison');return saved;});
    await h.context.loadComparison();
    const tables = [];
    const walk = node => {if(node.tagName==='TABLE')tables.push(node);for(const child of node.children||[])walk(child);};
    walk(h.element('comparisonResults'));
    const graphRow = tables[0].children[1].children[1].children;
    assert.equal(graphRow[2].textContent,'24');assert.equal(graphRow[3].textContent,'122');
    for (const i of [4,5,6,7]) assert.equal(graphRow[i].textContent,'待全部完成');
    assert.match(h.element('comparisonResults').textContent,/已完成配对 24 \/ 146 条/);
    assert.doesNotMatch(graphRow.map(cell=>cell.textContent).join(' '),/13\.01%|12\.89%/);
    assert.equal(tables[1].children[1].children.length,146);
    assert.equal(h.requests.length,1);assert.equal(h.requests[0].options,undefined);
    assert.equal(JSON.stringify(saved),original);
    saved.status='completed_below_target';saved.arms[1].pending=0;saved.arms[1].valid=146;
    await h.context.loadComparison();
    assert.match(h.element('comparisonResults').textContent,/13\.01%/);
    assert.doesNotMatch(h.element('comparisonResults').textContent,/待全部完成/);
  });
})().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
