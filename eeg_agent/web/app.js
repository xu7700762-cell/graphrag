'use strict';
const $ = id => document.getElementById(id);
const terminal = new Set(['completed', 'failed', 'stopped', 'interrupted', 'cancelled']);
const stateNames = {created: '启动中', running: '运行中', finalizing: '汇总中', completed: '已完成', failed: '失败', stopped: '已停止', interrupted: '中断待新会话', stopping: '停止中'};
const statusNames = {untrained: '未训练', trained: '训练完成', exported: '导出完成', incompatible: '合同不兼容', error: '加载失败'};
const names = {model: '模型', status: '状态', plan: 'Supervisor计划', agent_call: '语言接口', tool_call: 'EEG工具', agent_tool: 'EEG工具', routing: '分派', window: '完整窗口', path: '路径结果', early_exit: 'Early Exit', rag: 'RAG检索', language_report: 'Agent解释', report: '报告', message: '追问', message_response: '追问回答', error: '异常', eog_calibration: '眼电校准'};
const reasons = {amplitude: '幅值超限', flat_required_channel: '必需通道平直', abrupt_jump: '突变', warmup_or_missing: '预热或缺失', model_unavailable: '模型不可用', unsupported_P1: 'P1不适用', insufficient_temporal_history: '有效历史少于3窗', 'margin_below_0.10': '概率接近阈值', current_mean_threshold_conflict: '当前与近期均值类别冲突', 'temporal_std_above_0.15': '近期概率波动较大', 'adjacent_jump_above_0.20': '相邻窗口跳变较大', deep_temporal_stable: '模型输出与近期趋势稳定', deep_prediction_unavailable: '预测不可用', deep_only_strategy: '仅执行基础检测链', full_evidence_strategy: '完整证据策略', full_assessment_requested: '用户请求完整分析'};
let taskId = null, stream = null, lastSequence = -1, eventCount = 0, calls = 0, inputTokens = 0, outputTokens = 0;
let points = [], catalog = [], threshold = null, currentReport = null, busy = false, messageTimer = null;
let cloudSettings = null, cloudDirty = false, cloudPending = false;
let cloudProfileId = null;
const cloudProfileNames = {deepseek: 'DeepSeek', cpa_gpt: 'CPA GPT', relay: '中转站'};
const cloudProfileDefaults = {deepseek: {base_url: 'https://api.deepseek.com', model_id: 'deepseek-flash'}, cpa_gpt: {base_url: 'http://127.0.0.1:8317/v1', model_id: 'gpt-6.1-sol'}, relay: {base_url: '', model_id: '', protocol: 'chat_completions'}};
const messageEvents = new Set();
const finite = x => typeof x === 'number' && Number.isFinite(x);
const fmt = (x, digits = 3) => finite(x) ? x.toFixed(digits) : '不可用';
const percent = x => finite(x) ? (x * 100).toFixed(1) + '%' : '—';
const reasonText = xs => (Array.isArray(xs) ? xs : [xs]).filter(Boolean).map(x => reasons[x] || x).join('、');
Object.assign(reasons, {eog_calibration_unavailable: '眼电校准不足，P1模型预测不可用', eog_cleaning_input_rejected: '眼电清理输入不适用', measurement_without_classifier: '保留本地测量，分类输入不可用', residual_eog_correlation: '残留眼电相关超限', excessive_signal_removal: '清理幅度超限'});

async function json(url, options) {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw Error(typeof value.detail === 'string' ? value.detail : JSON.stringify(value.detail));
  return value;
}
const post = (url, value) => json(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value)});
function clearError() { $('error').textContent = ''; }
function showError(error) { $('error').textContent = error.message || String(error); }
function setBusy(value) {
  busy = value;
  for (const id of ['task', 'model', 'checkpoint', 'subject', 'strategy', 'backend', 'speed', 'limit']) $(id).disabled = value;
  $('start').disabled = value; $('stop').disabled = !value; $('newTask').disabled = value || !taskId;
  updateCloudControls();
}
function closeStream() { if (stream) stream.close(); stream = null; }

function cloudFeedback(text, error = false) {
  $('cloudFeedback').textContent = text;
  $('cloudFeedback').classList.toggle('error', error);
}
function selectedCloudProfile() {
  return cloudSettings?.profiles?.[cloudProfileId] || (cloudProfileId === (cloudSettings?.active_profile || 'deepseek') ? cloudSettings : cloudProfileDefaults[cloudProfileId]) || {};
}
function updateCloudControls() {
  const inUse = !!cloudSettings?.in_use || (busy && $('backend').value === 'api');
  const locked = !cloudSettings || cloudPending || inUse;
  for (const id of ['cloudProfile', 'cloudBaseUrl', 'cloudModelId', 'cloudProtocol', 'cloudApiKey', 'cloudSave']) $(id).disabled = locked;
  for (const id of ['cloudDeepseekCard', 'cloudCpaCard', 'cloudRelayCard']) $(id).disabled = locked;
  const profile = selectedCloudProfile(), active = cloudSettings?.active_profile || 'deepseek';
  $('cloudTest').disabled = locked || !cloudSettings?.configured || cloudProfileId !== active || cloudDirty;
  $('cloudTestInput').disabled = locked;
  $('cloudActivate').disabled = locked || !profile.configured || cloudProfileId === active || cloudDirty;
  $('cloudApiKey').required = !profile.key_configured;
  $('cloudConfigStatus').textContent = cloudPending ? '正在处理配置' : cloudDirty ? '修改未保存' : profile.configured ? '此档案已保存' : '此档案待完善';
  $('cloudConfigStatus').classList.toggle('amber', cloudDirty || !profile.configured);
  const missing = [['base_url', 'Base URL'], ['model_id', 'Model ID'], ['key_configured', 'API Key']].filter(([field]) => !profile[field]).map(([, label]) => label);
  $('cloudUsage').textContent = cloudPending ? '正在保存或测试，请等待本次操作完成。' : inUse ? '云端配置正在被任务使用。任务结束后可修改和测试连接。' : cloudDirty ? '有未保存内容，可先点击“连接测试 · 当前输入”验证，再保存并启用。' : !profile.configured ? `请填写${missing.join('、')}，可先连接测试再保存。` : cloudProfileId !== active ? '正在查看另一档案，可直接测试当前输入；启用后新任务使用此渠道。' : '可测试当前输入，也可测试已启用配置。';
}
function renderCloudSettings(value) {
  cloudSettings = value;
  if (!cloudProfileId) cloudProfileId = value.active_profile || 'deepseek';
  $('cloudProfile').value = cloudProfileId;
  for (const [id, cardId, summaryId] of [['deepseek', 'cloudDeepseekCard', 'cloudDeepseekSummary'], ['cpa_gpt', 'cloudCpaCard', 'cloudCpaSummary'], ['relay', 'cloudRelayCard', 'cloudRelaySummary']]) {
    const saved = value.profiles?.[id] || (id === value.active_profile ? value : {}), card = $(cardId);
    card.classList.toggle('selected', id === cloudProfileId); card.classList.toggle('active', id === value.active_profile);
    card.setAttribute('aria-pressed', String(id === cloudProfileId));
    $(summaryId).textContent = (id === value.active_profile ? '当前启用 · ' : '') + (saved.configured ? '配置已保存' : '待配置') + (saved.model_id ? ' · ' + saved.model_id : '');
  }
  const profile = selectedCloudProfile(), defaults = cloudProfileDefaults[cloudProfileId] || {};
  if (!cloudDirty) {
    $('cloudBaseUrl').value = profile.base_url || defaults.base_url || '';
    $('cloudModelId').value = profile.model_id || defaults.model_id || '';
    $('cloudProtocol').value = profile.protocol || defaults.protocol || 'auto';
    $('cloudApiKey').value = profile.api_key || '';
  }
  $('cloudActiveProfile').textContent = '当前启用：' + (cloudProfileNames[value.active_profile] || 'DeepSeek') + ' · ' + (value.model_id || '未配置');
  $('cloudProfileNote').textContent = cloudProfileId === 'relay' ? '填写中转站提供的接口地址、模型 ID 和独立密钥，再选择它支持的请求协议。' : cloudProfileId === 'cpa_gpt' ? 'CPA GPT 使用自己的 API Key；首次配置请填写 CPA 密钥。' : 'DeepSeek 使用自己的 API Key；切换不会使用 CPA 密钥。';
  $('cloudConfigStatus').textContent = profile.configured ? '此档案配置完整' : '此档案待完善';
  $('cloudConfigStatus').classList.toggle('amber', !profile.configured);
  $('cloudKeyStatus').textContent = profile.key_configured ? '此档案已配置' : '此档案未配置';
  $('cloudKeyStatus').classList.toggle('amber', !profile.key_configured);
  $('cloudApiKey').placeholder = '填写 ' + cloudProfileNames[cloudProfileId] + ' 独立 API 密钥';
  const sourceNames = {local_file: '本地设置', local_profile: '独立档案', environment: '环境配置', missing: '未配置'};
  $('cloudBaseSource').textContent = `${cloudProfileNames[cloudProfileId]} · ${sourceNames[profile.sources?.base_url] || '档案默认值'}`;
  $('cloudModelSource').textContent = `${cloudProfileNames[cloudProfileId]} · ${sourceNames[profile.sources?.model_id] || '档案默认型号'}`;
  $('cloudKeySource').textContent = `${cloudProfileNames[cloudProfileId]} 独立密钥 · ${sourceNames[profile.sources?.api_key] || '未配置'} · ${profile.key_configured ? '已显示此档案密钥，可直接查看和编辑；保存时留空保留已有密钥。' : '请填写此档案自己的密钥。'}`;
  $('backend').options[1].disabled = !value.configured;
  updateCloudControls();
}
async function refreshCloudSettings() {
  try { renderCloudSettings(await json('/api/cloud-settings?include_keys=true')); }
  catch (error) { cloudFeedback('配置读取失败：' + error.message, true); }
}
for (const id of ['cloudBaseUrl', 'cloudModelId', 'cloudProtocol', 'cloudApiKey']) $(id).addEventListener('input', () => {
  cloudDirty = true;
  cloudFeedback('修改尚未保存。');
  updateCloudControls();
});
$('cloudProfile').onchange = () => {
  cloudProfileId = $('cloudProfile').value; cloudDirty = false;
  renderCloudSettings(cloudSettings); cloudFeedback('正在编辑 ' + cloudProfileNames[cloudProfileId] + ' 档案，保存并启用后生效。');
};
for (const [cardId, profileId] of [['cloudDeepseekCard', 'deepseek'], ['cloudCpaCard', 'cpa_gpt'], ['cloudRelayCard', 'relay']]) $(cardId).onclick = () => {
  if ($('cloudProfile').disabled) return;
  $('cloudProfile').value = profileId; $('cloudProfile').onchange();
};
$('cloudActivate').onclick = async () => {
  if (cloudPending || cloudDirty || cloudSettings?.in_use || !selectedCloudProfile().configured) return;
  cloudPending = true; updateCloudControls(); cloudFeedback('正在切换云端模型档案…');
  try {
    await json('/api/cloud-settings', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({active_profile: cloudProfileId})});
    await refreshCloudSettings();
    cloudFeedback('已启用 ' + cloudProfileNames[cloudProfileId] + '；测试连接将使用此档案的密钥。');
  } catch (error) { cloudFeedback('切换失败：' + error.message, true); }
  finally { cloudPending = false; updateCloudControls(); }
};
$('cloudSettingsForm').onsubmit = async event => {
  event.preventDefault();
  if (cloudPending || cloudSettings?.in_use || (busy && $('backend').value === 'api')) return;
  if (!$('cloudSettingsForm').reportValidity()) return;
  const payload = {profile_id: cloudProfileId, base_url: $('cloudBaseUrl').value.trim(), model_id: $('cloudModelId').value.trim(), protocol: $('cloudProtocol').value, activate: true};
  const apiKey = $('cloudApiKey').value.trim();
  if (apiKey) payload.api_key = apiKey;
  cloudPending = true; updateCloudControls(); cloudFeedback('正在保存配置…');
  try {
    const value = await json('/api/cloud-settings', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    cloudDirty = false; await refreshCloudSettings();
    cloudFeedback(value.configured ? cloudProfileNames[cloudProfileId] + ' 配置已保存并启用；可查看密钥并测试连接。' : '配置已保存，请补充此档案的独立密钥。');
  } catch (error) {
    cloudFeedback('保存失败：' + error.message, true);
    await refreshCloudSettings();
  } finally { cloudPending = false; updateCloudControls(); }
};
$('cloudTestInput').onclick = async () => {
  if (cloudPending || cloudSettings?.in_use || (busy && $('backend').value === 'api')) return;
  if (!$('cloudSettingsForm').reportValidity()) return;
  const payload = {base_url: $('cloudBaseUrl').value.trim(), model_id: $('cloudModelId').value.trim(),
    api_key: $('cloudApiKey').value.trim(), protocol: $('cloudProtocol').value};
  if (!payload.api_key) { cloudFeedback('请填写 API 密钥再测试当前输入。', true); return; }
  cloudPending = true; updateCloudControls(); cloudFeedback('正在测试输入框中的接口配置…');
  try {
    const value = await post('/api/cloud-settings/test-input', payload);
    const timing = finite(value.latency_seconds) ? ` · ${fmt(value.latency_seconds, 2)}秒` : '';
    cloudFeedback(value.ok ? `连接成功 · ${value.model_id}${timing}${value.note ? ' · ' + value.note : ''} · 当前输入未保存`
      : `连接失败：${value.reason || '接口未返回有效响应'}${timing}`, !value.ok);
  } catch (error) { cloudFeedback('连接失败：' + error.message, true); }
  finally { cloudPending = false; updateCloudControls(); }
};
$('cloudTest').onclick = async () => {
  if (cloudDirty) { cloudFeedback('请先保存修改，再测试已保存的配置。', true); return; }
  if (cloudPending) { cloudFeedback('正在处理，请等待本次操作完成。'); return; }
  if (cloudSettings?.in_use || (busy && $('backend').value === 'api')) { cloudFeedback('云端配置正在被任务使用，请结束任务后测试。', true); return; }
  if (cloudProfileId !== (cloudSettings?.active_profile || 'deepseek')) { cloudFeedback('请先使用此已保存配置或保存并启用，再测试当前档案。', true); return; }
  if (!cloudSettings?.configured) { cloudFeedback('请先填写并保存Base URL、Model ID和API Key。', true); return; }
  cloudPending = true; updateCloudControls(); cloudFeedback('正在使用已保存的密钥发送一次固定测试请求…');
  try {
    const value = await json('/api/cloud-settings/test', {method: 'POST'});
    const timing = finite(value.latency_seconds) ? ` · ${fmt(value.latency_seconds, 2)}秒` : '';
    cloudFeedback(value.ok ? `连接成功 · ${value.model_id || cloudSettings.model_id}${timing}${value.note ? ' · ' + value.note : ''}` : `连接失败：${value.reason || '接口未返回有效响应'}${timing}`, !value.ok);
  } catch (error) { cloudFeedback('连接失败：' + error.message, true); }
  finally { await refreshCloudSettings(); cloudPending = false; updateCloudControls(); }
};

function modelChoice() {
  const selected = catalog.find(m => (m.model_id || m.id) === $('model').value);
  const saved = $('checkpoint').value;
  $('checkpoint').textContent = '';
  const checkpoints = selected?.checkpoints || [];
  const valid = checkpoints.filter(c => c.available);
  if (!valid.length) {
    const option = document.createElement('option'); option.value = ''; option.textContent = '无有效检查点 · 仅质量监测'; $('checkpoint').append(option);
  }
  for (const c of checkpoints) {
    const option = document.createElement('option'); option.value = c.checkpoint_id || c.id;
    option.textContent = `${c.name || option.value} · ${c.preprocessing_profile || 'P0'}${c.preprocessing_profile === 'P1' ? '眼电回归' : ''} · ${statusNames[c.status] || c.status || '未知状态'}`;
    option.disabled = !c.available; $('checkpoint').append(option);
  }
  if (valid.length) {
    const preferred = valid.find(c => (c.checkpoint_id || c.id) === saved) || valid.find(c => (c.checkpoint_id || c.id) === selected.default_checkpoint_id) || valid[0];
    $('checkpoint').value = preferred.checkpoint_id || preferred.id;
  }
  updateModelDetail();
}
function updateModelDetail() {
  const selected = catalog.find(m => (m.model_id || m.id) === $('model').value);
  const checkpoint = selected?.checkpoints?.find(c => (c.checkpoint_id || c.id) === $('checkpoint').value);
  const available = !!checkpoint?.available;
  $('modelBadge').textContent = available ? `${selected.name || $('model').value} · 冻结模型可用` : '预测不可用 · 仅质量监测';
  $('modelBadge').classList.toggle('amber', !available);
  const state = statusNames[checkpoint?.status || selected?.status] || '未训练';
  const windows = selected?.device?.windows === 'verified' ? '已验证' : '待验收';
  const pi = selected?.device?.pi === 'verified' ? '已实测' : '待实测';
  const profile = checkpoint?.preprocessing_profile || selected?.preprocessing_profile || 'P0';
  $('inputProfile').textContent = profile === 'P1' ? 'P1 · 因果眼电回归' : 'P0 · 常规因果处理';
  $('classificationProfile').textContent = profile;
  $('modelDetail').textContent = `${state} · ${profile === 'P1' ? 'P1眼电回归模型' : 'P0模型'} · Windows ${windows} · Pi ${pi}` + (available ? ` · 温度 ${fmt(checkpoint.temperature)} / 阈值 ${fmt(checkpoint.threshold, 2)}` : ` · ${checkpoint?.reason || selected?.reason || '未加载合法模型'}`);
}
async function refreshCatalog() { catalog = await json('/api/models'); modelChoice(); renderModelCatalog(); }
$('model').onchange = modelChoice; $('checkpoint').onchange = updateModelDetail;

function resetView() {
  closeStream(); lastSequence = -1; eventCount = 0; calls = 0; inputTokens = 0; outputTokens = 0;
  points = []; threshold = null; currentReport = null; messageEvents.clear();
  if (messageTimer) clearTimeout(messageTimer);
  for (const id of ['logs', 'messages', 'subreport']) $(id).textContent = '';
  $('quality').textContent = '—'; $('qualityDetail').textContent = '等待完整窗口'; $('coverage').textContent = '有效覆盖 —';
  $('probability').textContent = '—'; $('threshold').textContent = '冻结阈值 —'; $('rawProbability').textContent = '原始概率 —';
  $('latency').textContent = '—'; $('calls').textContent = '0 / 50'; $('eventCount').textContent = '0条';
  $('plan').textContent = '等待Supervisor安排分析'; $('calibration').textContent = '先行基线尚未冻结';
  $('policy').textContent = '分析策略等待合法窗口'; $('pathResult').textContent = '路径结束后依据有效覆盖发布结果';
  $('temporal').textContent = '至少3个有效窗口才发布趋势结论'; $('candidate').textContent = '尚无模型持续高类候选段';
  $('biomarkers').textContent = '等待按需调用'; $('covariance').textContent = '等待调用；偏离基线不等于高类支持';
  $('eogStatus').textContent = '等待校准'; $('eogStatus').classList.add('amber');
  $('eogDetails').textContent = '眼电清理与分类配置随所选冻结检查点确定。';
  $('eogMeasures').textContent = '等待同一窗口清理前后测量'; $('eogCalibration').textContent = '先行眼电校准尚未冻结';
  $('eogCounts').textContent = '清理与拒绝统计等待完整窗口';
  $('citations').textContent = '尚未检索'; $('report').textContent = '报告尚未生成';
  $('downloadReport').disabled = true; $('downloadJson').disabled = true; draw();
}

function log(event) {
  const node = document.createElement('div'); node.className = 'log';
  const title = document.createElement('b'); title.textContent = names[event.kind] || event.kind; node.append(title);
  const p = event.payload || {}; let text;
  if (event.kind === 'window') text = `${fmt(p.end_sample / 1024, 1)}s · ${p.profile || '处理不可用'}${p.requested_profile && p.requested_profile !== p.profile ? '测量回退' : ''} · ${p.prediction_available ? '已预测' : reasonText(p.reasons) || '预测不可用'}`;
  else if (event.kind === 'tool_call') text = `${p.tool} · ${p.status || '已执行'}${p.reasons?.length ? ' · ' + reasonText(p.reasons) : ''}${finite(p.seconds) ? ' · ' + fmt(p.seconds * 1000, 1) + 'ms' : ''}`;
  else if (event.kind === 'agent_call') text = `${p.role} · ${p.simulated ? 'mock' : 'API'} · ${p.status || '已返回'}`;
  else if (event.kind === 'language_report') text = `${p.role} · ${p.fallback_reason ? '确定性模板 / ' + p.fallback_reason : p.simulated ? 'mock解释' : '云端解释'}`;
  else if (event.kind === 'rag') text = `${p.query || ''} · ${(p.evidence || []).length}条引用`;
  else if (event.kind === 'report') text = '报告数值和类别已由本地证据渲染';
  else text = JSON.stringify(p);
  node.append(document.createTextNode(text)); $('logs').prepend(node);
  while ($('logs').children.length > 180) $('logs').lastChild.remove();
  $('eventCount').textContent = ++eventCount + '条';
}
function metricTable(container, rows, headers) {
  container.textContent = ''; const table = document.createElement('table');
  if (headers) { const row = table.insertRow(); headers.forEach(text => { const th = document.createElement('th'); th.textContent = text; row.append(th); }); }
  rows.forEach(values => { const row = table.insertRow(); values.forEach(text => { row.insertCell().textContent = text; }); });
  container.append(table);
}
function renderPlan(p) {
  const root = $('plan'); root.textContent = '';
  const heading = document.createElement('p'); heading.className = 'muted'; heading.textContent = `${p.simulated ? 'mock规划' : '云端规划'} · ${p.strategy || $('strategy').value}`; root.append(heading);
  const list = document.createElement('ol'); (p.steps || []).forEach(step => { const node = document.createElement('li'); node.textContent = typeof step === 'string' ? step : JSON.stringify(step); list.append(node); }); root.append(list);
}
function renderTemporal(p) {
  if (!p) return;
  metricTable($('temporal'), [['近期均值', percent(p.mean_30s)], ['线性斜率', fmt(p.slope_probability_per_minute ?? p.slope) + ' 概率/分钟'], ['波动 / 最大相邻跳变', fmt(p.std_probability ?? p.volatility) + ' / ' + fmt(p.max_jump)], ['有效窗口 / 时长 / 覆盖', `${p.valid_windows || 0}窗 / ${fmt(p.analysis_duration_seconds, 1)}秒 / ${percent(p.coverage)}`]], null);
  const hint = document.createElement('p'); hint.className = 'muted'; hint.textContent = p.available ? '最近30秒内已到达窗口的模型输出统计' : '有效窗口少于3个，趋势结论不足'; $('temporal').append(hint);
  for (const event of p.events || []) $('candidate').textContent = `模型持续高类候选段：${fmt(event.first_high_start_seconds, 1)}s开始，${fmt(event.confirmation_seconds, 1)}s确认（第三窗完成）。真实症状onset尚未验证。`;
}
function renderBiomarkers(p) {
  if (!p) return;
  if (!p.available) { $('biomarkers').textContent = p.reason || '工具输入不足'; return; }
  const labels = {delta: 'δ', theta: 'θ', alpha: 'α', beta: 'β', gamma_low: '30–45 Hz'};
  const region = p.regions?.all;
  const changes = p.baseline_change?.regions?.all;
  const rows = Object.entries(region?.absolute_power_uv2 || {}).map(([band, power]) => [labels[band] || band, fmt(power), percent(region.relative_power?.[band]), changes ? percent(changes[band]) : '基线不可用']);
  metricTable($('biomarkers'), rows, ['频带 / 全通道均值', '绝对功率 μV²', '相对功率', '相对基线变化']);
  for (const key of ['frontal', 'occipital']) {
    const r = p.regions?.[key]; if (!r) continue;
    const line = document.createElement('p'); line.className = 'muted'; line.textContent = `${key === 'frontal' ? '额区F3/Fz/F4' : '枕区O1/Oz/O2'}：θ/α ${fmt(r.ratios?.theta_alpha)}，θ/β ${fmt(r.ratios?.theta_beta)}，慢/快 ${fmt(r.ratios?.slow_fast)}`; $('biomarkers').append(line);
  }
  const hint = document.createElement('p'); hint.className = 'muted'; hint.textContent = '这些是EEG测量，不是已验证的晕动症方向性生物标志物。'; $('biomarkers').append(hint);
}
function renderCovariance(p) {
  if (!p) return;
  if (!p.available) { $('covariance').textContent = p.reason || '工具输入不足'; return; }
  metricTable($('covariance'), [['与先行基线的黎曼距离', fmt(p.airm_distance)], ['参考基线', `${p.baseline_windows || 0}个合格窗口`], ['协方差迹 / 收缩', `${fmt(p.trace_uv2)} μV² / ${percent(p.shrinkage)}`]], null);
  const note = document.createElement('p'); note.className = 'muted'; note.textContent = p.baseline_available ? '距离表示空间与幅值状态变化，不决定高低类别。' : '基线不足：距离不可用，绝对协方差测量仍可用。'; $('covariance').append(note);
}
function renderEOG(value, counts, calibration, profile) {
  const p = value || {}, c = p.calibration || calibration || {};
  const applied = !!p.applied, fallback = !!p.measurement_fallback, rejected = applied && p.input_available === false;
  $('eogStatus').textContent = rejected ? '眼电回归已执行 · 质量拒绝' : applied ? '已执行因果眼电回归' : fallback ? 'P0测量回退 · 分类不可用' : profile === 'P0' || p.requested_profile === 'P0' ? 'P0 · 未执行眼电回归' : c.available ? '先行校准已冻结' : '眼电校准不足';
  $('eogStatus').classList.toggle('amber', !applied || rejected);
  $('eogDetails').textContent = rejected ? '清理后质量未通过，当前窗口不进入分类和辅助工具。' + (p.reason ? ' ' + p.reason : '') : applied ? '同一清理窗口供模型、频谱与协方差使用，清理后重新检查质量。' : fallback ? '先行校准或输入不满足P1要求，持续显示P0质量与辅助测量，不向P1模型送入未清理脑电。' : '分类输入与所选检查点合同一致。' + (p.reason ? ' ' + p.reason : '');
  const seconds = finite(c.samples) ? c.samples / 256 : c.seconds;
  $('eogCalibration').textContent = `眼电校准：${c.available ? '可用并冻结' : c.reason || '等待先行基线'}${finite(seconds) ? ' · ' + fmt(seconds, 1) + '秒' : ''}${finite(c.cutoff_sample) ? ' · 冻结于' + fmt(c.cutoff_sample / 1024, 1) + '秒' : ''}`;
  if (Object.keys(p).length) metricTable($('eogMeasures'), [['额区最大|EEG–EOG相关|', fmt(p.pre_frontal_eog_correlation), fmt(p.post_frontal_eog_correlation)], ['最大峰峰值 / μV', fmt(p.pre_peak_to_peak_uv, 1), fmt(p.post_peak_to_peak_uv, 1)], ['移除分量RMS比例', '—', percent(p.removed_rms_ratio)]], ['同一完整窗口', '回归前P0', applied ? '回归后P1' : '未清理测量']);
  if (counts) $('eogCounts').textContent = `已清理 ${counts.cleaned_windows || 0} / ${counts.total_windows || 0}窗 · 质量拒绝 ${counts.quality_rejected_windows || 0}窗 · 分类输入不可用 ${counts.input_unavailable_windows ?? counts.calibration_unavailable_windows ?? 0}窗`;
  if (p.method && applied) {
    const line = document.createElement('p'); line.className = 'muted';
    line.textContent = p.method === 'causal_reference_rls' ? `方法：因果自适应EOG回归 · 算法参数冻结 · 系数按已到达样本更新${finite(p.adaptive?.updates) ? ' · ' + p.adaptive.updates + '次更新' : ''}${p.adaptive?.skipped_updates ? ' · 安全跳过' + p.adaptive.skipped_updates + '次更新' : ''}` : '方法：先行基线冻结EOG回归';
    $('eogDetails').append(line);
  }
}
function renderCitations(container, evidence) {
  container.textContent = '';
  if (!evidence?.length) { container.textContent = '没有匹配证据，不能以知识库补造结论。'; return; }
  for (const item of evidence) {
    const node = document.createElement('details'); const title = document.createElement('summary');
    title.textContent = `[${item.id || item.reference_id || '引用'}] ${item.title || item.source || ''} · ${item.section || ''}`;
    node.append(title); const p = document.createElement('p'); p.textContent = item.text || item.content || ''; node.append(p);
    const source = document.createElement('small'); source.textContent = `来源：${item.source || '未知'} · 版本 ${item.version || '—'} · ${(item.sha256 || item.source_sha256 || item.hash || '').slice(0, 12)}`; node.append(source); container.append(node);
  }
}
function renderWindow(p, addPoint = true) {
  const q = p.quality || {};
  $('quality').textContent = q.valid ? '有效' : '不足';
  $('qualityDetail').textContent = q.valid ? '峰峰值 ' + fmt(q.peak_to_peak_uv, 1) + ' μV' : reasonText(p.reasons || q.reasons) || '质量不合格';
  $('coverage').textContent = '当前路径有效覆盖 ' + percent(p.coverage);
  $('probability').textContent = percent(p.probability_high); $('rawProbability').textContent = '原始概率 ' + percent(p.raw_probability ?? p.deep?.raw_probability);
  threshold = p.threshold ?? p.deep?.threshold ?? threshold;
  $('threshold').textContent = '冻结阈值 ' + fmt(threshold, 2) + ' · 校准概率';
  $('latency').textContent = finite(p.completion_lag_seconds) ? Math.round(p.completion_lag_seconds * 1000) + ' ms' : '—';
  const baseline = p.calibration || p.baseline || {};
  $('calibration').textContent = baseline.available ? `先行基线 ${baseline.seconds || 0}s / ${baseline.windows || 0}窗 · ${baseline.profile || 'P0'}` : baseline.frozen ? '先行基线不足' : `基线收集中 ${baseline.seconds || 0}s`;
  $('classificationProfile').textContent = p.requested_profile || p.profile || '—';
  renderEOG(p.eog_cleaning, p.eog_counts, p.eog_calibration, p.requested_profile || p.profile);
  if (p.policy) $('policy').textContent = `${p.policy.early_exit ? 'Early Exit：无需额外工具' : '辅助工具：' + (p.policy.tools || []).join('、')} · ${reasonText(p.policy.reasons)}`;
  renderTemporal(p.temporal); renderBiomarkers(p.biomarkers); renderCovariance(p.covariance);
  if (addPoint) points.push({time: p.end_sample / 1024, value: p.probability_high ?? null}); draw();
}
function renderReport(value) {
  if (!value || value.status === 'pending') { $('report').textContent = value?.reason || '报告尚未生成'; return; }
  currentReport = value; $('report').textContent = value.markdown || value.report_markdown || JSON.stringify(value, null, 2);
  $('downloadReport').disabled = false; $('downloadJson').disabled = false;
}
function renderMessage(event) {
  const key = event.sequence ?? JSON.stringify(event.payload);
  if (messageEvents.has(key)) return; messageEvents.add(key);
  const p = event.payload || {}, node = document.createElement('div'); node.className = 'message';
  node.textContent = (p.role === 'user' || p.type === 'user' || (event.kind === 'message' && p.text && !p.explanation) ? '用户：' : 'VRMSAgent：') + (p.text || p.explanation || p.answer || JSON.stringify(p)); $('messages').append(node);
}
function handle(event) {
  const seq = Number(event.sequence);
  if (Number.isInteger(seq) && seq <= lastSequence) return;
  if (Number.isInteger(seq)) lastSequence = seq;
  log(event); const p = event.payload || {};
  if (event.kind === 'window') renderWindow(p);
  if (event.kind === 'eog_calibration') renderEOG(null, null, p.calibration, p.requested_profile);
  if (event.kind === 'plan') renderPlan(p);
  if (event.kind === 'rag') renderCitations($('citations'), p.evidence || p.matches || []);
  if (event.kind === 'language_report') { const node = document.createElement('p'); node.textContent = `${p.role || 'Agent'}${p.simulated ? '（mock）' : ''}：${p.explanation || ''}${p.fallback_reason ? ' · 回退原因：' + p.fallback_reason : ''}`; $('subreport').append(node); if (p.phase === 'followup') renderMessage(event); }
  if (event.kind === 'auxiliary') { renderBiomarkers(p.biomarkers); renderCovariance(p.covariance); }
  if (event.kind === 'report') renderReport(p);
  if (event.kind === 'message' || event.kind === 'message_response') renderMessage(event);
  if (event.kind === 'agent_call') { calls++; inputTokens += p.input_tokens || 0; outputTokens += p.output_tokens || 0; $('calls').textContent = calls + ' / 50'; $('tokens').textContent = p.simulated ? 'mock不计真实token' : `输入 ${inputTokens} / 输出 ${outputTokens} token`; }
  if (event.kind === 'path') $('pathResult').textContent = p.status === 'accepted' ? `路径估计：${p.predicted_class ? '高类' : '低类'} · 概率 ${percent(p.probability_high)} · 覆盖 ${percent(p.coverage)} · ${p.valid_windows}个有效窗` : `路径输出不足（${p.status}）· 覆盖 ${percent(p.coverage)} · ${p.valid_windows || 0}个有效窗`;
  if (event.kind === 'status') $('state').textContent = stateNames[p.state] || p.state;
  if (event.kind === 'error') { showError(p.reason || p.error || '运行失败'); $('state').textContent = '失败'; }
}
function draw() {
  const canvas = $('chart'), dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = 210;
  if (!w) return;
  canvas.width = w * dpr; canvas.height = h * dpr; const c = canvas.getContext('2d'); c.scale(dpr, dpr); c.clearRect(0, 0, w, h);
  const left = 35, top = 12, bottom = 25, y = value => top + (1 - value) * (h - top - bottom);
  c.font = '10px Segoe UI';
  for (let v = 0; v <= 100; v += 25) { c.strokeStyle = '#edf0e8'; c.beginPath(); c.moveTo(left, y(v / 100)); c.lineTo(w - 10, y(v / 100)); c.stroke(); c.fillStyle = '#81938a'; c.fillText(v, 3, y(v / 100) + 3); }
  if (finite(threshold)) { c.save(); c.setLineDash([5, 4]); c.strokeStyle = '#c19148'; c.beginPath(); c.moveTo(left, y(threshold)); c.lineTo(w - 10, y(threshold)); c.stroke(); c.restore(); c.fillStyle = '#a67b3c'; c.fillText('τ=' + threshold.toFixed(2), w - 60, y(threshold) - 5); }
  if (!points.length) { c.fillStyle = '#9aa69d'; c.fillText('等待完整窗口输出', w / 2 - 50, h / 2); return; }
  const max = Math.max(points[points.length - 1].time, 1); c.strokeStyle = '#288b80'; c.lineWidth = 2; c.beginPath(); let moving = false, previousTime = null;
  for (const p of points) { if (!finite(p.value)) { moving = false; previousTime = p.time; continue; } if (previousTime !== null && Math.abs(p.time - previousTime - 5) > .001) moving = false; const x = left + p.time / max * (w - left - 10); if (!moving) c.moveTo(x, y(p.value)); else c.lineTo(x, y(p.value)); moving = true; previousTime = p.time; }
  c.stroke(); c.fillStyle = '#81938a'; c.fillText('0 s', left, h - 3); c.fillText(max.toFixed(0) + ' s', w - 45, h - 3);
}

async function finish() {
  const id = taskId; if (!id) return;
  const snapshot = await json('/api/tasks/' + id); if (id !== taskId) return;
  if (!terminal.has(snapshot.state)) return;
  closeStream(); setBusy(false); $('state').textContent = stateNames[snapshot.state] || snapshot.state;
  await loadReport(); await refreshHistory(); await refreshCloudSettings();
}
function connectEvents() {
  closeStream(); const id = taskId;
  stream = new EventSource(`/api/tasks/${id}/events?after=${lastSequence}`);
  stream.onmessage = e => { if (id === taskId) handle(JSON.parse(e.data)); };
  stream.onerror = async () => { if (id !== taskId) return; try { await finish(); } catch (error) { showError(error); } };
}
async function openTask(id) {
  const snapshot = await json('/api/tasks/' + id); resetView(); taskId = id; localStorage.setItem('vrms-task-id', id);
  const request = snapshot.request || {};
  for (const field of ['subject', 'strategy', 'backend', 'speed']) if (request[field] !== undefined) $(field).value = request[field];
  if (request.model_id) $('model').value = request.model_id; modelChoice();
  if (request.checkpoint_id) $('checkpoint').value = request.checkpoint_id; updateModelDetail();
  $('task').value = request.task || $('task').value; $('limit').value = request.max_seconds ?? '';
  $('taskId').textContent = id; $('state').textContent = stateNames[snapshot.state] || snapshot.state;
  $('backendBadge').textContent = request.backend === 'api' ? '真实云端 API' : 'mock · 模拟云端 Agent';
  $('timingLabel').textContent = request.speed === 1 ? '1×真实时间发布延迟' : '加速回放：仅用于吞吐测试';
  setBusy(!terminal.has(snapshot.state)); $('loadReport').disabled = false; $('ask').disabled = false;
  if (snapshot.plan) renderPlan(snapshot.plan); if (snapshot.latest) renderWindow(snapshot.latest, false);
  connectEvents(); await loadReport();
}
async function refreshHistory() {
  const tasks = await json('/api/tasks'), select = $('history'); const selected = select.value; select.textContent = '';
  const empty = document.createElement('option'); empty.value = ''; empty.textContent = '选择已有任务'; select.append(empty);
  for (const item of tasks) { const option = document.createElement('option'); option.value = item.id; option.textContent = `${item.id} · ${item.request?.model_id || ''} · ${stateNames[item.state] || item.state}`; select.append(option); }
  select.value = tasks.some(item => item.id === selected) ? selected : taskId || '';
}
$('start').onclick = async () => {
  try {
    clearError(); setBusy(true); resetView();
    const snapshot = await post('/api/tasks', {subject: Number($('subject').value), model_id: $('model').value, checkpoint_id: $('checkpoint').value || null, strategy: $('strategy').value, backend: $('backend').value, speed: Number($('speed').value), max_seconds: $('limit').value ? Number($('limit').value) : null, task: $('task').value});
    await openTask(snapshot.id); await refreshHistory(); await refreshCloudSettings();
  } catch (error) { showError(error); setBusy(false); }
};
$('stop').onclick = async () => { try { if (taskId) { await post(`/api/tasks/${taskId}/stop`, {}); $('stop').disabled = true; $('state').textContent = '停止中'; } } catch (error) { showError(error); } };
$('newTask').onclick = () => { closeStream(); resetView(); taskId = null; localStorage.removeItem('vrms-task-id'); $('taskId').textContent = '原始 CDT'; $('state').textContent = '待机'; setBusy(false); $('newTask').disabled = true; $('loadReport').disabled = true; $('ask').disabled = true; refreshCatalog().catch(showError); };
$('restore').onclick = () => { if (busy && $('history').value !== taskId) { showError('请先停止当前任务再打开其他任务'); return; } if ($('history').value) openTask($('history').value).catch(showError); };
async function loadReport() { if (taskId) renderReport(await json(`/api/tasks/${taskId}/report`)); }
$('loadReport').onclick = () => loadReport().catch(showError);
function download(text, filename, type) { const url = URL.createObjectURL(new Blob([text], {type})), a = document.createElement('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
$('downloadReport').onclick = () => { if (currentReport) download(currentReport.markdown || $('report').textContent, taskId + '.md', 'text/markdown;charset=utf-8'); };
$('downloadJson').onclick = () => { if (currentReport) download(JSON.stringify(currentReport, null, 2), taskId + '.json', 'application/json;charset=utf-8'); };
$('questionForm').onsubmit = async e => {
  e.preventDefault(); const text = $('question').value.trim(); if (!text || !taskId) return;
  try {
    $('ask').disabled = true; const response = await post(`/api/tasks/${taskId}/messages`, {text}); $('question').value = '';
    if (response.answer) { renderMessage({kind: 'message', payload: {role: 'user', text}}); renderMessage({kind: 'message_response', payload: {answer: response.answer}}); $('ask').disabled = false; return; }
    const id = taskId;
    const poll = async remaining => {
      if (id !== taskId) return;
      const events = await json(`/api/tasks/${id}/messages`); events.forEach(renderMessage);
      const newestQuestion = Math.max(-1, ...events.filter(event => event.kind === 'message').map(event => event.sequence));
      if (events.some(event => event.sequence > newestQuestion && (event.kind === 'message_response' || event.payload?.phase === 'followup'))) remaining = 0;
      const snapshot = await json(`/api/tasks/${id}`);
      if (snapshot.latest) renderWindow(snapshot.latest, false);
      // Completed tasks close SSE; drain their new follow-up events explicitly.
      if (terminal.has(snapshot.state)) {
        const response = await fetch(`/api/tasks/${id}/events?after=${lastSequence}`);
        if (!response.ok) throw Error('追问事件读取失败');
        (await response.text()).split('\n').filter(line => line.startsWith('data: ')).forEach(line => handle(JSON.parse(line.slice(6))));
      }
      if (remaining > 0) messageTimer = setTimeout(() => poll(remaining - 1).catch(error => { showError(error); $('ask').disabled = false; }), 1000);
      else $('ask').disabled = false;
    };
    await poll(12);
  } catch (error) { showError(error); $('ask').disabled = false; }
};

function renderModelCatalog() {
  const rows = catalog.map(m => [m.name || m.model_id, statusNames[m.status] || m.status, m.available ? '可预测' : '预测不可用', m.device?.windows === 'verified' ? 'Windows已验证' : 'Windows待验收', m.device?.pi === 'verified' ? 'Pi已实测' : 'Pi待实测']);
  metricTable($('modelCatalog'), rows, ['模型', '产物状态', '可用性', 'Windows', '树莓派']);
}
function collectMetrics(value, location = '', output = [], depth = 0) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || depth > 5) return output;
  if ('BACC' in value || 'bacc' in value) output.push({location, metrics: value});
  else for (const [key, child] of Object.entries(value)) if (!['paths', 'windows', 'history'].includes(key)) collectMetrics(child, location ? location + ' / ' + key : key, output, depth + 1);
  return output;
}
async function loadResults() {
  try {
    await refreshCatalog(); const reports = await json('/api/reports'), root = $('resultsContent'); root.textContent = '';
    let count = 0;
    for (const [key, report] of Object.entries(reports)) {
      const found = collectMetrics(report); if (!found.length) continue; count++;
      const heading = document.createElement('h3'); heading.textContent = key; root.append(heading);
      const block = document.createElement('div'); root.append(block);
      metricTable(block, found.map(({location, metrics: m}) => [location, `${m.accepted_paths ?? '—'} / ${m.total_paths ?? '—'}`, fmt(m.BACC ?? m.bacc, 4), fmt(m.AUROC ?? m.auroc, 4), fmt(m.Macro_F1 ?? m.macro_f1, 4), fmt(m.sensitivity, 4), fmt(m.specificity, 4), percent(m.path_coverage), fmt(m.Brier ?? m.brier, 4), fmt(m.ECE_10_equal_width ?? m.ECE ?? m.ece, 4)]), ['模型 / 划分 / 输出', '接受/全部路径', 'BACC', 'AUROC', 'Macro-F1', '敏感度', '特异度', '路径接受率', 'Brier', 'ECE']);
    }
    if (!count) root.textContent = '尚无双模型评价报告，请通过命令行完成训练与评价。';
  } catch (error) { $('resultsContent').textContent = error.message; }
}
$('refresh').onclick = loadResults;
$('knowledgeForm').onsubmit = async e => { e.preventDefault(); try { const result = await json('/api/knowledge/search?q=' + encodeURIComponent($('knowledgeQuery').value)); renderCitations($('knowledgeResults'), result.matches); } catch (error) { $('knowledgeResults').textContent = error.message; } };
document.querySelectorAll('button.nav').forEach(button => button.onclick = () => { document.body.dataset.page = button.dataset.page; document.querySelectorAll('.nav').forEach(b => b.classList.toggle('active', b === button)); document.querySelectorAll('.page').forEach(page => page.classList.toggle('hidden', page.id !== button.dataset.page)); if (button.dataset.page === 'results') loadResults(); if (button.dataset.page === 'console') draw(); if (button.dataset.page === 'cloud') refreshCloudSettings(); if (location.hash !== '#' + button.dataset.page) location.hash = button.dataset.page; });
window.addEventListener('hashchange', () => { const page = location.hash.slice(1) || 'console'; const button = document.querySelector('button.nav[data-page="' + page.replace(/[^a-z]/g, '') + '"]'); if (button && document.body.dataset.page !== page) button.click(); });
window.addEventListener('resize', draw);
(async () => {
  try {
    const cfg = await json('/api/config');
    for (const sid of cfg.subjects) { const option = document.createElement('option'); option.value = sid; option.textContent = 'Acquisition ' + String(sid).padStart(2, '0'); $('subject').append(option); }
    $('backend').options[1].disabled = !cfg.api_configured;
    catalog = cfg.models || await json('/api/models'); modelChoice(); await refreshHistory(); draw();
    const id = localStorage.getItem('vrms-task-id');
    if (id) { try { await openTask(id); } catch (error) { localStorage.removeItem('vrms-task-id'); showError('历史任务无法恢复：' + error.message); } }
  } catch (error) { showError(error); }
  await refreshCloudSettings();
  if (location.hash === '#cloud') document.querySelector('[data-page="cloud"]').click();
})();
