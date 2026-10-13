/* EEG replay: bounded plots, independent local/cloud outputs, recoverable SSE. */
(function (root) {
  'use strict';
  const ACTIVE = new Set(['created', 'queued', 'running', 'paused', 'finalizing', 'stopping']);
  const TERMINAL = new Set(['completed', 'stopped', 'failed', 'interrupted']);
  const SESSION_EVENTS = new Set(['state', 'progress', 'window', 'cloud_pending', 'cloud_verified', 'cloud_fallback']);
  const MAX_SECONDS = 30;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const number = (value, digits = 3) => finite(value) ? value.toFixed(digits) : '—';
  function clock(seconds) {
    if (!finite(seconds) || seconds < 0) return '—';
    const total = Math.floor(seconds);
    const hours = Math.floor(total / 3600);
    return (hours ? String(hours).padStart(2, '0') + ':' : '') + String(Math.floor(total / 60) % 60).padStart(2, '0') + ':' + String(total % 60).padStart(2, '0');
  }
  function className(value) { return /^(high|low)$/i.test(String(value)) ? (String(value).toLowerCase() === 'high' ? 'High' : 'Low') : '—'; }
  function ragModeLabel(mode) { return {regular: '普通 RAG', graph: 'GraphRAG', semantic_graph: '语义 GraphRAG'}[mode] || mode; }
  function validOrdinal(value) { return Number.isInteger(value) && value >= 1; }
  function reportText(report) {
    if (!report) return '';
    if (report.explanation) return report.explanation;
    const reasons = {
      APITimeoutError:'API请求超时；未取得有效云端答复，本地结果已保存',
      TimeoutError:'API请求超时；未取得有效云端答复，本地结果已保存',
      AuthenticationError:'API认证失败，请检查项目API密钥；本地结果已保存',
      PermissionDeniedError:'API接口拒绝访问，请检查服务商权限；本地结果已保存',
      APIConnectionError:'无法连接已配置的API接口；本地结果已保存',
      RateLimitError:'API调用限额或余额不足；本地结果已保存',
      NotFoundError:'API地址或模型ID不存在；本地结果已保存',
      BadRequestError:'API接口不接受请求，请检查模型ID及协议；本地结果已保存',
      InternalServerError:'API服务暂时不可用；本地结果已保存'
    };
    return (Object.prototype.hasOwnProperty.call(reasons, report.error_type) ? reasons[report.error_type] : '') || report.reason || '';
  }
  function waveformScale(values) {
    const sorted = values.filter(finite).map(Math.abs).sort((a,b) => a-b);
    return Math.max(.05, (sorted[Math.floor(sorted.length * .98)] || 0) * 1.1);
  }
  function trendAxis(values, probability = false) {
    if (probability) return {low:0,high:1,digits:2};
    values = values.filter(finite);
    if (!values.length) return {low:0,high:1,digits:2};
    const min = Math.min(...values), max = Math.max(...values), center = min + (max - min) / 2;
    const span = Math.max(max - min, .02, Math.abs(center) * .05) * 1.2;
    let low = center - span / 2, high = center + span / 2;
    if (min >= 0 && low < 0) { high -= low; low = 0; }
    return {low,high,digits:Math.min(6,Math.max(2,Math.ceil(-Math.log10(span)) + 1))};
  }
  function cloudView(cloud) {
    cloud = cloud || {};
    const result = cloud.result || cloud;
    const status = cloud.status || cloud.generation_status || result.generation_status || 'not_requested';
    const verified = ['verified', 'cloud_verified'].includes(status) && className(result.final_class) !== '—';
    if (verified) return {label: className(result.final_class), state: '已验证云端判断', badge: 'verified'};
    if (status === 'pending') return {label: '—', state: '等待云端', badge: 'pending'};
    if (['fallback', 'failed', 'unavailable'].includes(status)) {
      const error = result.reports?.vrms?.error_type || result.error_type;
      return {label: '—', state: ['APITimeoutError', 'TimeoutError'].includes(error) ? '云端请求超时' : '云端未完成', badge: 'fallback'};
    }
    return {label: '—', state: '未请求云端', badge: 'local'};
  }
  function requestBody(recordingId, speed, cloudEnabled, cloudConfigured, ragMode) {
    if (typeof recordingId !== 'string' || !recordingId) throw new Error('请选择原始记录。');
    const parsed = speed === 'max' ? 'max' : Number(speed);
    if (![1, 10, 'max'].includes(parsed)) throw new Error('回放速度无效。');
    const body = {recording_id: recordingId, speed: parsed, cloud: Boolean(cloudEnabled && cloudConfigured), max_seconds: null};
    if (ragMode !== undefined) {
      if (!['default', 'regular', 'graph', 'semantic_graph'].includes(ragMode)) throw new Error('RAG 模式无效。');
      body.rag_mode = ragMode;
    }
    return body;
  }
  function emptyStore() { return {snapshot: null, sequence: -1, windows: [], waveforms: [], paths: []}; }
  function addWindow(store, payload) {
    if (!payload || !finite(payload.end_seconds)) return;
    store.windows = store.windows.filter(item => item.end_seconds !== payload.end_seconds || item.start_seconds !== payload.start_seconds);
    store.windows.push(payload);
    store.windows.sort((a, b) => a.end_seconds - b.end_seconds);
    const cutoff = payload.end_seconds - MAX_SECONDS;
    store.windows = store.windows.filter(item => item.end_seconds > cutoff && item.end_seconds <= payload.end_seconds).slice(-7);
    if (payload.waveform && Array.isArray(payload.waveform.values)) {
      store.waveforms = store.waveforms.filter(item => item.end_seconds !== payload.end_seconds);
      store.waveforms.push({start_seconds: payload.start_seconds, end_seconds: payload.end_seconds, ...payload.waveform});
    }
    store.waveforms = store.waveforms.filter(item => item.end_seconds > cutoff && item.end_seconds <= payload.end_seconds).slice(-7);
  }
  function upsertPath(store, row) {
    if (!row || row.scope === 'rest' || !validOrdinal(row.ordinal)) return;
    const index = store.paths.findIndex(item => item.ordinal === row.ordinal);
    if (index < 0) store.paths.push(row);
    else store.paths[index] = {...store.paths[index], ...row};
    store.paths.sort((a, b) => a.ordinal - b.ordinal);
    if (store.snapshot) store.snapshot.paths = store.paths;
  }
  function restoreStore(snapshot) {
    const store = emptyStore();
    store.snapshot = {...snapshot};
    store.paths = Array.isArray(snapshot.paths) ? snapshot.paths.filter(row => row.scope !== 'rest' && validOrdinal(row.ordinal)).map(row => ({...row})) : [];
    const events = Array.isArray(snapshot.events) ? [...snapshot.events].sort((a, b) => a.sequence - b.sequence) : [];
    for (const event of events) {
      if (Number.isInteger(event.sequence)) store.sequence = Math.max(store.sequence, event.sequence);
      if (event.type === 'window') addWindow(store, event.payload);
    }
    if (snapshot.latest && finite(snapshot.latest.end_seconds)) addWindow(store, snapshot.latest);
    if (Number.isInteger(snapshot.sequence)) store.sequence = Math.max(store.sequence, snapshot.sequence);
    if (Number.isInteger(snapshot.last_sequence)) store.sequence = Math.max(store.sequence, snapshot.last_sequence);
    return store;
  }
  function applyEvent(store, event) {
    if (!event || !Number.isInteger(event.sequence) || event.sequence <= store.sequence) return false;
    store.sequence = event.sequence;
    const payload = event.payload || {};
    const snapshot = store.snapshot || (store.snapshot = {state: 'running', metrics: {}, cloud: {}});
    if (event.type === 'snapshot') {
      const restored = restoreStore(payload);
      Object.assign(store, restored, {sequence: Math.max(event.sequence, restored.sequence)});
    } else if (event.type === 'window') {
      snapshot.latest = payload;
      snapshot.position_seconds = Math.max(snapshot.position_seconds || 0, payload.end_seconds);
      addWindow(store, payload);
    } else if (event.type === 'progress') {
      if (finite(payload.position_seconds)) snapshot.position_seconds = payload.position_seconds;
    } else if (event.type === 'state') {
      if (typeof payload.state === 'string') snapshot.state = payload.state;
    } else if (event.type === 'path_completed') {
      upsertPath(store, payload);
    } else if (['cloud_pending', 'cloud_verified', 'cloud_fallback'].includes(event.type)) {
      const path = store.paths.find(item => item.ordinal === payload.ordinal);
      if (path) upsertPath(store, {...path, cloud: {...path.cloud, ...(payload.result || {}), status: event.type.slice(6)}});
    }
    if (SESSION_EVENTS.has(event.type)) {
      if (finite(payload.position_seconds)) snapshot.position_seconds = payload.position_seconds;
      if (payload.metrics) snapshot.metrics = {...snapshot.metrics, ...payload.metrics};
      if (Object.prototype.hasOwnProperty.call(payload, 'error')) snapshot.error = payload.error;
      if (payload.cloud && event.type !== 'path_completed') snapshot.cloud = {...snapshot.cloud, ...payload.cloud};
    }
    return true;
  }
  function series(store) {
    return store.windows.map(row => ({time: row.end_seconds,
      probability: row.valid !== false && row.vrms?.available ? row.vrms.probability_high : null,
      fatigue: row.valid !== false ? row.fatigue?.theta_alpha : null,
      workload: row.valid !== false ? row.emotion?.workload_index : null}));
  }
  const api = {ACTIVE, TERMINAL, MAX_SECONDS, finite, number, clock, className, cloudView, reportText, waveformScale, trendAxis, requestBody, emptyStore, restoreStore, applyEvent, series};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.EEGOnline = api;
  if (!root.document) return;
  const doc = root.document;
  const $ = id => doc.getElementById(id);
  let store = emptyStore(), catalog = null, sessionId = null, stream = null, reconnectTimer = null, busy = false;
  let renderFrame = null, chartDirty = false, closing = false;
  let uiError = null, uiErrorSource = null;
  const storageKey = 'eeg-online-session-id';
  const stateLabels = {created:'准备中', queued:'等待启动', running:'回放中', paused:'已暂停', finalizing:'正在完成', stopping:'正在停止', completed:'已完成', stopped:'已停止', failed:'运行失败', interrupted:'运行中断'};
  const text = (id, value) => { $(id).textContent = value; };
  function savedId() { try { return root.localStorage.getItem(storageKey); } catch (_) { return null; } }
  function saveId(id) { try { if (id) root.localStorage.setItem(storageKey, id); else root.localStorage.removeItem(storageKey); } catch (_) { /* Browser storage may be unavailable. */ } }
  function renderError() {
    const error = store.snapshot?.error || uiError;
    text('onlineError', error?.message || (error ? String(error) : ''));
    $('onlineError').hidden = !error;
  }
  function showError(error, source = 'action') { uiError = error?.message || String(error); uiErrorSource = source; renderError(); }
  function clearError(source) { if (!source || uiErrorSource === source) { uiError = null; uiErrorSource = null; } renderError(); }
  function connection(value, connected = false) { text('connectionStatus', value); $('connectionStatus').classList.toggle('connected', connected); }
  async function request(url, options) {
    const response = await root.fetch(url, options);
    let value;
    try { value = await response.json(); } catch (_) { throw new Error('服务未返回有效数据，请检查在线服务。'); }
    if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : value.error?.message || '在线请求未完成（' + response.status + '）。');
    return value;
  }
  const post = (url, body) => request(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  function controls() {
    const state = store.snapshot?.state;
    const active = ACTIVE.has(state);
    $('startReplay').disabled = busy || active || !catalog?.recordings?.length;
    for (const id of ['recordingSelect', 'speedSelect', 'cloudEnabled', 'ragMode']) $(id).disabled = busy || active;
    $('pauseReplay').disabled = busy || state !== 'running';
    $('resumeReplay').disabled = busy || state !== 'paused';
    $('stopReplay').disabled = busy || !['created', 'queued', 'running', 'paused', 'finalizing'].includes(state);
  }
  function metric(value) { return finite(value) ? value.toFixed(3) : '—'; }
  function renderLatest(latest) {
    latest = latest || {};
    const vrms = latest.vrms || {}, fatigue = latest.fatigue || {}, emotion = latest.emotion || {};
    const valid = latest.valid !== false;
    const vrmsAvailable = valid && vrms.available === true;
    text('segmentValue', latest.scope ? (latest.scope === 'task' ? '任务路径 ' : '休息段 ') + (latest.ordinal || '') : '等待开始');
    text('coverageValue', finite(latest.coverage) ? (100 * latest.coverage).toFixed(1) + '%' : '—');
    text('vrmsProbability', vrmsAvailable ? metric(vrms.probability_high) : '—');
    text('vrmsClass', vrmsAvailable ? className(vrms.mil_class || vrms.predicted_class) : '—');
    text('numericClass', vrmsAvailable ? className(vrms.numeric_class || vrms.numeric_fusion?.predicted_class) : '—');
    text('vrmsStatus', vrmsAvailable ? (vrms.status === 'final' ? '完整路径' : '前缀估计') : latest.scope === 'rest' ? '休息段不分类' : latest.end_seconds ? '暂不可用' : '等待窗口');
    text('thetaAlphaValue', valid ? metric(fatigue.theta_alpha) : '—');
    text('thetaBetaValue', valid ? metric(fatigue.theta_beta) : '—');
    text('workloadValue', valid ? metric(emotion.workload_index) : '—');
    text('faaValue', valid ? metric(emotion.faa) : '—');
    text('fatigueStatus', valid && finite(fatigue.theta_alpha) ? '已更新' : latest.end_seconds ? '窗口不可用' : '等待窗口');
    text('emotionStatus', valid && finite(emotion.workload_index) ? '已更新' : latest.end_seconds ? '窗口不可用' : '等待窗口');
    const qc = latest.qc || {};
    text('qualityValue', !latest.end_seconds ? '质量状态等待测量' : valid ? '当前窗口通过质量检查' : '当前窗口未通过质量检查' + (qc.reason ? '：' + qc.reason : ''));
    text('windowRange', finite(latest.start_seconds) ? clock(latest.start_seconds) + ' – ' + clock(latest.end_seconds) : '五秒完整窗口');
  }
  function renderPaths() {
    const body = $('pathRows');
    const explanations = $('pathExplanations');
    body.replaceChildren();
    const openPaths = new Set(Array.from(explanations.children).filter(item => item.open).map(item => item.dataset.ordinal));
    explanations.replaceChildren();
    if (!store.paths.length) {
      const row = doc.createElement('tr'), cell = doc.createElement('td');
      cell.colSpan = 7; cell.className = 'table-empty'; cell.textContent = '路径结束后在此查看最终结果。'; row.append(cell); body.append(row); return;
    }
    for (const path of store.paths) {
      const row = doc.createElement('tr'), cloud = cloudView(path.cloud), vrms = path.vrms || {};
      const pathState = path.complete === false ? '路径未闭合' : cloud.state;
      const labels = [String(path.ordinal), clock(path.start_seconds) + ' – ' + clock(path.end_seconds), finite(path.coverage) ? (path.coverage * 100).toFixed(1) + '%' : '—',
        className(vrms.mil_class || vrms.predicted_class), className(vrms.numeric_class || vrms.numeric_fusion?.predicted_class), cloud.label];
      for (const label of labels) { const cell = doc.createElement('td'); cell.textContent = label; row.append(cell); }
      const status = doc.createElement('td'), badge = doc.createElement('span');
      badge.className = 'cloud-badge ' + cloud.badge;
      badge.textContent = pathState;
      if (cloud.badge === 'fallback') badge.title = reportText(path.cloud?.reports?.vrms) || reportText(path.cloud);
      status.append(badge); row.append(status); body.append(row);
      const details = doc.createElement('details'), summary = doc.createElement('summary');
      details.dataset.ordinal = String(path.ordinal); details.open = openPaths.has(String(path.ordinal));
      summary.textContent = '路径 ' + path.ordinal + ' · ' + clock(path.start_seconds) + ' – ' + clock(path.end_seconds) + ' · ' + pathState;
      details.append(summary);
      const local = doc.createElement('p');
      local.textContent = '有效窗口 ' + path.accepted_windows + '/' + path.total_windows + '；数据截止 ' + clock(path.data_cutoff_seconds) + '。本地 MIL：' + className(vrms.mil_class) + '；数值工具：' + className(vrms.numeric_class) + '。' + (path.reason || vrms.reason || '');
      details.append(local);
      const reports = path.cloud?.reports || {};
      for (const [title, report] of [['VRMSAgent', reports.vrms], ['FatigueAgent', reports.fatigue], ['EmotionAgent', reports.emotion], ['汇总', reports.supervisor]]) {
        const heading = doc.createElement('h3'), paragraph = doc.createElement('p');
        heading.textContent = title;
        paragraph.textContent = reportText(report) || reportText(path.cloud) || (path.complete === false ? '路径未闭合，未请求整路径云端报告。' : cloud.badge === 'pending' ? '路径云端报告正在生成。' : '本地指标已保存，云端解释未生成。');
        details.append(heading, paragraph);
      }
      explanations.append(details);
    }
  }
  function render() {
    const snapshot = store.snapshot || {};
    text('sessionState', stateLabels[snapshot.state] || '就绪');
    $('sessionState').classList.toggle('running', snapshot.state === 'running');
    text('positionValue', clock(finite(snapshot.position_seconds) ? snapshot.position_seconds : 0));
    const metrics = snapshot.metrics || {};
    text('latencyValue', finite(metrics.window_processing_p95_seconds) ? number(metrics.window_processing_p95_seconds, 2) + ' 秒（P95）' : '—');
    text('pathCountValue', String(store.paths.filter(path => path.complete !== false).length));
    renderLatest(snapshot.latest);
    const policy = snapshot.rag_policy || snapshot.request?.rag_policy;
    text('ragVersionNote', sessionId ? (policy ? ragModeLabel(policy.rag_mode) + ' · ' + (policy.judge_profile || policy.variant || '已固定版本') : '历史未记录版本') + '；本回放版本已固定，切换请新建回放。' : '创建回放时固定检索版本。默认：' + ragModeLabel(catalog?.default_rag_mode || 'regular'));
    renderPaths();
    const clouds = store.paths.map(path => cloudView(path.cloud));
    const pending = clouds.filter(cloud => cloud.badge === 'pending').length;
    const completed = clouds.filter(cloud => cloud.badge === 'verified').length;
    const failed = clouds.filter(cloud => cloud.badge === 'fallback').length;
    text('cloudNotice', pending && TERMINAL.has(snapshot.state) ? '回放已结束；未完成的云端判断保留空类别。' : pending ? '云端正在处理 ' + pending + ' 条已结束路径，本地回放继续。' : completed || failed ? '云端已完成 ' + completed + ' 条，未完成 ' + failed + ' 条；未完成的云端类别保留为空。' : '云端只接收已结束路径的证据；本地数值推理持续运行。');
    const canDownload = Boolean(sessionId && store.snapshot);
    $('reportLinks').hidden = !canDownload;
    if (canDownload) { $('jsonReport').href = '/api/online/sessions/' + encodeURIComponent(sessionId) + '/report'; $('markdownReport').href = '/api/online/sessions/' + encodeURIComponent(sessionId) + '/report.md'; }
    renderError();
    controls();
    if (chartDirty) { drawCharts(); chartDirty = false; }
  }
  function scheduleRender(charts = false) {
    chartDirty = chartDirty || charts;
    if (renderFrame !== null) return;
    renderFrame = root.requestAnimationFrame(() => { renderFrame = null; render(); });
  }
  function canvasContext(id) {
    const canvas = $(id), bounds = canvas.getBoundingClientRect(), ratio = Math.min(root.devicePixelRatio || 1, 2);
    if (!bounds.width || !bounds.height) return null;
    canvas.width = Math.round(bounds.width * ratio); canvas.height = Math.round(bounds.height * ratio);
    const context = canvas.getContext('2d');
    if (!context) return null;
    context.scale(ratio, ratio); context.clearRect(0, 0, bounds.width, bounds.height);
    context.font = '11px "Microsoft YaHei UI", "Segoe UI", sans-serif'; context.lineWidth = 1;
    return {context, width:bounds.width, height:bounds.height};
  }
  function grid(context, width, height, left, top, bottom, start, end) {
    context.strokeStyle = '#E8EDF4'; context.fillStyle = '#52647B';
    for (let i = 0; i <= 6; i++) {
      const x = left + (width - left - 14) * i / 6;
      context.beginPath(); context.moveTo(x, top); context.lineTo(x, height - bottom); context.stroke();
      context.textAlign = i === 0 ? 'left' : i === 6 ? 'right' : 'center';
      context.fillText(clock(start + (end - start) * i / 6), x, height - 7);
    }
    context.textAlign = 'left';
  }
  function drawWaveform() {
    const canvas = canvasContext('waveformCanvas'); if (!canvas) return;
    const {context, width, height} = canvas;
    const frames = store.waveforms;
    const end = frames.length ? Math.max(...frames.map(frame => frame.end_seconds)) : 30;
    const start = Math.max(0, end - MAX_SECONDS), left = 84, bottom = 26, top = 12;
    grid(context, width, height, left, top, bottom, start, end);
    const latest = frames[frames.length - 1];
    const channels = (latest?.channels || []).slice(0, 6);
    $('waveformEmpty').hidden = channels.length > 0;
    if (!channels.length) return;
    const lane = (height - top - bottom) / channels.length, timeWidth = Math.max(1, end - start);
    for (let channel = 0; channel < channels.length; channel++) {
      const center = top + lane * (channel + .5);
      context.fillStyle = '#52647B'; context.fillText(String(channels[channel]).slice(0, 10), 4, center + 4);
      context.strokeStyle = '#DCE4EF'; context.beginPath(); context.moveTo(left, center); context.lineTo(width - 14, center); context.stroke();
      const amplitudes = frames.flatMap(frame => frame.values[frame.channels?.indexOf(channels[channel])] || []);
      const scale = waveformScale(amplitudes);
      context.font = '9px "Microsoft YaHei UI", sans-serif';
      context.fillText('±' + number(scale, scale < 1 ? 2 : 1) + ' μV', 4, center + 16);
      context.font = '11px "Microsoft YaHei UI", "Segoe UI", sans-serif';
      context.strokeStyle = channel % 2 ? '#007C83' : '#175CD3'; context.lineWidth = 1.05; context.beginPath();
      let pen = false, previousTime = null;
      for (const frame of frames) {
        const index = frame.channels?.indexOf(channels[channel]);
        const values = frame.values[index] || [], fs = frame.sampling_hz || 64;
        for (let i = 0; i < values.length; i++) {
          const time = frame.start_seconds + i / fs, value = values[i];
          if (time < start || time > end || !finite(value)) { pen = false; continue; }
          if (previousTime !== null && time - previousTime > 2 / fs) pen = false;
          const x = left + (time - start) / timeWidth * (width - left - 14);
          const y = center - Math.max(-1, Math.min(1, value / scale)) * lane * .36;
          if (pen) context.lineTo(x,y); else context.moveTo(x,y);
          pen = true; previousTime = time;
        }
      }
      context.stroke(); context.lineWidth = 1;
    }
  }
  function drawTrends() {
    const canvas = canvasContext('trendCanvas'); if (!canvas) return;
    const {context,width,height} = canvas, points = series(store);
    const end = points.length ? points[points.length-1].time : 30, start = Math.max(0,end-MAX_SECONDS);
    const left = 72, top = 10, bottom = 27, lane = (height-top-bottom)/3;
    grid(context,width,height,left,top,bottom,start,end);
    $('trendEmpty').hidden = points.some(point => [point.probability,point.fatigue,point.workload].some(finite));
    const specs = [['probability','High 读数','#175CD3'],['fatigue','θ / α','#007C83'],['workload','负荷指标','#6B52A3']];
    specs.forEach(([key,label,color],index) => {
      const {low,high,digits} = trendAxis(points.map(point=>point[key]), key==='probability');
      const rowTop=top+index*lane+8, rowBottom=top+(index+1)*lane-12;
      context.fillStyle=color;context.fillText(label,4,rowTop+5);context.fillStyle='#52647B';context.font='10px "Microsoft YaHei UI", sans-serif';
      context.fillText(number(high,digits),4,rowTop+20);context.fillText(number(low,digits),4,rowBottom);context.font='11px "Microsoft YaHei UI", sans-serif';
      context.strokeStyle='#DCE4EF';context.beginPath();context.moveTo(left,rowBottom);context.lineTo(width-14,rowBottom);context.stroke();
      if(key==='probability'){const y=rowTop+(rowBottom-rowTop)*.5;context.setLineDash([3,4]);context.beginPath();context.moveTo(left,y);context.lineTo(width-14,y);context.stroke();context.setLineDash([]);}
      context.strokeStyle=color;context.lineWidth=1.7;context.beginPath();let pen=false;
      for(const point of points){if(!finite(point[key])){pen=false;continue;}const x=left+(point.time-start)/Math.max(1,end-start)*(width-left-14), y=rowBottom-(point[key]-low)/(high-low)*(rowBottom-rowTop);if(pen)context.lineTo(x,y);else context.moveTo(x,y);pen=true;}
      context.stroke();context.lineWidth=1;
      for(const point of points){if(!finite(point[key]))continue;const x=left+(point.time-start)/Math.max(1,end-start)*(width-left-14),y=rowBottom-(point[key]-low)/(high-low)*(rowBottom-rowTop);context.fillStyle=color;context.beginPath();context.arc(x,y,2.5,0,Math.PI*2);context.fill();}
    });
  }
  function drawCharts() { drawWaveform(); drawTrends(); }
  function closeStream() { if (stream) stream.close(); stream = null; if (reconnectTimer) root.clearTimeout(reconnectTimer); reconnectTimer = null; }
  async function reconcile(id) {
    const snapshot = await request('/api/online/sessions/' + encodeURIComponent(id));
    if (sessionId !== id) return;
    store = restoreStore(snapshot); clearError('connection'); scheduleRender(true);
    return snapshot;
  }
  function connectEvents() {
    closeStream();
    const id = sessionId;
    if (!id || !ACTIVE.has(store.snapshot?.state)) return;
    const source = new root.EventSource('/api/online/sessions/' + encodeURIComponent(id) + '/events?after=' + store.sequence);
    stream = source;
    source.onopen = () => { if (stream === source) connection('已连接本地服务', true); };
    source.onmessage = event => {
      if (stream !== source || sessionId !== id) return;
      let value; try { value = JSON.parse(event.data); } catch (_) { connection('事件格式错误，正在恢复'); recover(); return; }
      if (!applyEvent(store, value)) return;
      scheduleRender(value.type === 'window' || value.type === 'snapshot');
      if (TERMINAL.has(store.snapshot?.state)) { closeStream(); reconcile(id).then(()=>connection('任务已结束',true)).catch(error=>showError(error,'connection')); }
    };
    function recover() {
      if (stream !== source || reconnectTimer || closing) return;
      source.close(); stream = null;
      connection('连接中断，正在恢复');
      reconnectTimer = root.setTimeout(async () => {
        reconnectTimer = null;
        if (sessionId !== id || closing) return;
        try { await reconcile(id); if (ACTIVE.has(store.snapshot?.state)) connectEvents(); else connection('任务已结束',true); }
        catch (error) { showError(error,'connection'); if (!closing && sessionId === id) { reconnectTimer = root.setTimeout(()=>{reconnectTimer=null;connectEvents();},3000); } }
      },1500);
    }
    source.onerror = recover;
  }
  async function control(action) {
    if (busy || !sessionId) return;
    busy=true;controls();clearError();
    try { const id=sessionId; const snapshot=await post('/api/online/sessions/'+encodeURIComponent(id)+'/control',{action}); if(sessionId!==id)return; store=restoreStore(snapshot);render();connectEvents(); }
    catch(error){showError(error);}finally{busy=false;controls();}
  }
  $('replayForm').addEventListener('submit',async event=>{
    event.preventDefault();if(busy)return;busy=true;controls();clearError();
    try {
      const body=requestBody($('recordingSelect').value,$('speedSelect').value,$('cloudEnabled').checked,catalog?.capabilities?.cloud_configured,$('ragMode').value || 'default');
      const snapshot=await post('/api/online/sessions',body);
      closeStream();sessionId=snapshot.id;saveId(sessionId);store=restoreStore(snapshot);chartDirty=true;render();connectEvents();
    }catch(error){showError(error);}finally{busy=false;controls();}
  });
  $('pauseReplay').addEventListener('click',()=>control('pause'));
  $('resumeReplay').addEventListener('click',()=>control('resume'));
  $('stopReplay').addEventListener('click',()=>control('stop'));
  root.addEventListener('resize',()=>scheduleRender(true));
  root.addEventListener('pagehide',()=>{closing=true;closeStream();});
  async function initialize() {
    try {
      const results=await Promise.allSettled([request('/api/online/catalog'),request('/api/online/sessions')]);
      if(results[0].status!=='fulfilled')throw results[0].reason;
      catalog=results[0].value;
      if(Array.isArray(catalog.rag_modes)){
        $('ragMode').replaceChildren();
        for(const item of catalog.rag_modes){const option=doc.createElement('option');option.value=item.id;option.textContent=item.label||item.id;$('ragMode').append(option);}
      }
      $('ragMode').value='default';
      $('recordingSelect').replaceChildren();
      for(const item of catalog.recordings||[]){const option=doc.createElement('option');option.value=item.id;option.textContent=item.title||item.id;$('recordingSelect').append(option);}
      if(!catalog.recordings?.length){const option=doc.createElement('option');option.value='';option.textContent='没有可回放的原始记录';$('recordingSelect').append(option);}
      const caps=catalog.capabilities||{};
      text('capabilityNote',(caps.model_available?'本地冻结模型已配置。':(caps.model_error||'本地模型未配置')+'，仍可回放适用的描述性 EEG 指标。')+' '+(caps.cloud_configured?'云端在完整路径结束后调用，每次最多等待 '+(caps.online_cloud_timeout_seconds || 120)+' 秒。':'云端未配置，本轮可直接运行本地分析；勾选项在配置后生效。'));
      let id=savedId();
      if(results[1].status==='fulfilled'){
        const list=Array.isArray(results[1].value)?results[1].value:results[1].value.sessions||[];
        const active=list.find(item=>ACTIVE.has(item.state));
        if(active)id=active.id;
        else if(id&&!list.some(item=>item.id===id))id=null;
      }
      if(id){sessionId=id;await reconcile(id);saveId(id);if(store.snapshot?.recording?.id)$('recordingSelect').value=store.snapshot.recording.id;if(store.snapshot?.speed)$('speedSelect').value=store.snapshot.speed>=1000?'max':String(store.snapshot.speed);if(typeof store.snapshot?.cloud?.requested==='boolean')$('cloudEnabled').checked=store.snapshot.cloud.requested;$('ragMode').value=store.snapshot?.rag_mode||store.snapshot?.request?.rag_mode||'default';connectEvents();}
      else saveId(null);
      if(!stream)connection('已连接本地服务',true);
      chartDirty=true;render();
    }catch(error){showError(error,'connection');connection('服务暂不可用');controls();}
  }
  initialize();
})(typeof window !== 'undefined' ? window : globalThis);
