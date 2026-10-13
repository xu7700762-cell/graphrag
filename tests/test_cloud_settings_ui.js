/* Browser-state regression checks. No real credentials, server, or cloud calls. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../eeg_agent/web/app.js'), 'utf8');
const marker = source.indexOf('\nfunction modelChoice()');
assert(marker > 0, 'Cloud settings code must precede model initialization');
const secret = 'fake-ui-private-key';
const complete = () => ({base_url: 'https://cloud.example/v1', model_id: 'ui-model',
  key_configured: true, configured: true, in_use: false,
  sources: {base_url: 'local_file', model_id: 'local_file', api_key: 'local_file'}});
const cpaDraft = () => ({base_url: 'http://127.0.0.1:8317/v1', model_id: 'gpt-6.1-sol',
  key_configured: false, configured: false, sources: {api_key: 'missing'}});
function profileSettings(initial) {
  const active = initial.active_profile || 'deepseek';
  const profiles = initial.profiles || {deepseek: {...initial}, cpa_gpt: cpaDraft()};
  return {...profiles[active], profiles: structuredClone(profiles), active_profile: active, in_use: !!initial.in_use};
}

function harness(initial = complete()) {
  const elements = new Map(), requests = [];
  let saved = profileSettings(initial), handler = null;
  const keys = new Map(Object.entries(saved.profiles).filter(([, profile]) => profile.key_configured)
    .map(([id]) => [id, id + '-previous-independent-key']));
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      value: id === 'backend' ? 'mock' : '', textContent: '', placeholder: '', disabled: false,
      options: [{disabled: false}, {disabled: false}], listeners: {},
      classList: {toggle() {}}, reportValidity: () => true,
      setAttribute(name, value) { this[name] = value; },
      addEventListener(name, callback) { this.listeners[name] = callback; }
    });
    return elements.get(id);
  }
  const context = vm.createContext({
    document: {getElementById: element},
    // Any attempt to persist a credential in the browser fails this test.
    localStorage: {setItem() { throw Error('Cloud settings must not use localStorage'); }},
    fetch: async (url, options = {}) => {
      requests.push({url, options});
      if (handler) return handler(url, options);
      if (options.method === 'PUT') {
        const payload = JSON.parse(options.body);
        if (payload.active_profile) {
          if (!saved.profiles[payload.active_profile].configured) return {ok: false, json: async () => ({detail: '目标档案缺少独立密钥'})};
          saved.active_profile = payload.active_profile;
        } else {
          const id = payload.profile_id || saved.active_profile;
          if (payload.api_key) keys.set(id, payload.api_key);
          const profile = {...saved.profiles[id], base_url: payload.base_url, model_id: payload.model_id, protocol: payload.protocol,
            key_configured: keys.has(id)};
          profile.configured = !!(profile.base_url && profile.model_id && profile.key_configured);
          if (payload.activate && !profile.configured) return {ok: false, json: async () => ({detail: '此档案缺少独立API密钥'})};
          saved.profiles[id] = profile;
          if (payload.activate !== false) saved.active_profile = id;
        }
        saved = {...saved, ...saved.profiles[saved.active_profile], in_use: false};
      }
      const value = url.endsWith('/test') ? {ok: true, model_id: saved.model_id,
        latency_seconds: .01, note: null} : {...saved};
      if (url.endsWith('/test-input')) return {ok: true, json: async () => ({ok: true,
        model_id: JSON.parse(options.body).model_id, latency_seconds: .02})};
      if (url.includes('include_keys=true')) value.profiles = Object.fromEntries(
        Object.entries(saved.profiles).map(([id, profile]) => [id, {...profile, api_key: keys.get(id) || ''}]));
      return {ok: true, json: async () => value};
    }
  });
  vm.runInContext(source.slice(0, marker), context, {filename: 'app.js:cloud-settings'});
  context.renderCloudSettings(profileSettings(initial));
  return {context, element, requests,
    keys,
    setHandler(value) { handler = value; },
    async save() { await element('cloudSettingsForm').onsubmit({preventDefault() {}}); },
    async test() { await element('cloudTest').onclick(); },
    async testInput() { await element('cloudTestInput').onclick(); },
    select(id) { element('cloudProfile').value = id; element('cloudProfile').onchange(); },
    async activate() { await element('cloudActivate').onclick(); },
    input(id, value) { element(id).value = value; element(id).listeners.input(); }
  };
}

async function check(name, run) {
  await run();
  process.stdout.write('PASS ' + name + '\n');
}

(async () => {
  await check('relay saves an independent endpoint, model, key and selected protocol', async () => {
    const h = harness();
    h.select('relay');
    assert.equal(h.element('cloudBaseUrl').value, '');
    assert.equal(h.element('cloudProtocol').value, 'chat_completions');
    assert.equal(h.element('cloudApiKey').required, true);
    h.input('cloudBaseUrl', 'https://relay.example/v1');
    h.input('cloudModelId', 'gpt-relay');
    h.input('cloudProtocol', 'responses');
    h.input('cloudApiKey', secret);
    await h.save();
    const payload = JSON.parse(h.requests[0].options.body);
    assert.equal(payload.profile_id, 'relay');
    assert.equal(payload.protocol, 'responses');
    assert.equal(h.keys.get('relay'), secret);
    assert.equal(h.keys.get('deepseek'), 'deepseek-previous-independent-key');
    assert.equal(h.element('cloudApiKey').value, secret);
    assert.equal(h.element('cloudProtocol').value, 'responses');
    await h.test();
    assert.match(h.element('cloudFeedback').textContent, /连接成功/);
  });
  await check('save displays the key and retains a usable saved configuration', async () => {
    const h = harness({base_url: '', model_id: '', key_configured: false, configured: false, in_use: false});
    h.input('cloudBaseUrl', 'https://cloud.example/v1');
    h.input('cloudModelId', 'ui-model');
    h.input('cloudApiKey', secret);
    await h.save();
    assert.equal(h.element('cloudApiKey').value, secret);
    assert.match(h.element('cloudKeySource').textContent, /可直接查看和编辑/);
    assert.match(h.element('cloudFeedback').textContent, /密钥/);
    assert.equal(h.element('cloudTest').disabled, false);
    assert.equal(h.element('backend').options[1].disabled, false);
    assert.equal(JSON.parse(h.requests[0].options.body).api_key, secret);
    await h.test();
    const testRequest = h.requests.find(r => r.url.endsWith('/test'));
    assert.equal(testRequest.options.method, 'POST');
    assert.equal(testRequest.options.body, undefined);
    assert(!JSON.stringify(testRequest).includes(secret));
    assert.match(h.element('cloudFeedback').textContent, /连接成功/);
    assert.equal(h.element('cloudTest').disabled, false);
    // Saving again with an empty field must preserve the stored key.
    h.input('cloudApiKey', '');
    h.input('cloudModelId', 'next-model');
    await h.save();
    const puts = h.requests.filter(r => r.options.method === 'PUT');
    assert(!Object.hasOwn(JSON.parse(puts[1].options.body), 'api_key'));
    assert.match(h.element('cloudKeyStatus').textContent, /此档案已配置/);
    assert.equal(h.element('cloudTest').disabled, false);
  });

  await check('refresh during a test cannot leave stale in-use controls locked', async () => {
    const h = harness();
    let release, testing = true;
    const delayed = new Promise(resolve => { release = resolve; });
    h.setHandler(async (url) => {
      if (url.endsWith('/test')) {
        await delayed;
        testing = false;
        return {ok: true, json: async () => ({ok: true, model_id: 'ui-model', latency_seconds: .1})};
      }
      return {ok: true, json: async () => ({...complete(), in_use: testing})};
    });
    const pending = h.test();
    await h.context.refreshCloudSettings();
    assert.equal(h.element('cloudTest').disabled, true);
    release();
    await pending;
    assert.equal(h.element('cloudTest').disabled, false);
    assert.equal(h.element('cloudApiKey').disabled, false);
    assert(h.requests.filter(r => r.url.startsWith('/api/cloud-settings?')).length >= 2);
  });

  await check('unconfigured and unsaved settings give actionable feedback', async () => {
    const missing = harness({...complete(), key_configured: false, configured: false});
    await missing.test();
    assert.match(missing.element('cloudFeedback').textContent, /保存|完整|密钥/);
    assert.equal(missing.requests.length, 0);
    const dirty = harness();
    dirty.input('cloudModelId', 'unsaved-model');
    await dirty.test();
    assert.match(dirty.element('cloudFeedback').textContent, /先保存/);
    assert.equal(dirty.requests.length, 0);
  });

  await check('a failed test releases controls and preserves its failure feedback', async () => {
    const h = harness();
    h.setHandler(async url => {
      if (url.endsWith('/test')) throw Error('fake network failure');
      return {ok: true, json: async () => complete()};
    });
    await h.test();
    assert.equal(h.element('cloudTest').disabled, false);
    assert.equal(h.element('cloudSave').disabled, false);
    assert.match(h.element('cloudFeedback').textContent, /连接失败/);
    assert(h.requests.some(r => r.url.startsWith('/api/cloud-settings?')));
  });

  await check('limited reasoning response shows a qualified connectivity result', async () => {
    const h = harness();
    h.setHandler(async url => ({ok: true, json: async () => url.endsWith('/test') ? {
      ok: true, model_id: 'ui-model', latency_seconds: .1,
      note: '接口已响应，测试输出达到上限；完整报告能力需实际任务验证'
    } : complete()}));
    await h.test();
    assert.match(h.element('cloudFeedback').textContent, /连接成功/);
    assert.match(h.element('cloudFeedback').textContent, /上限.*验证/);
    assert.equal(h.element('cloudTest').disabled, false);
  });

  await check('CPA draft selects separate key and cannot test DeepSeek by accident', async () => {
    const h = harness();
    h.select('cpa_gpt');
    assert.equal(h.element('cloudProfile').disabled, false);
    assert.equal(h.element('cloudBaseUrl').value, 'http://127.0.0.1:8317/v1');
    assert.equal(h.element('cloudModelId').value, 'gpt-6.1-sol');
    assert.equal(h.element('cloudApiKey').value, '');
    assert.equal(h.element('cloudApiKey').required, true);
    assert.match(h.element('cloudApiKey').placeholder, /CPA GPT.*独立/);
    assert.equal(h.element('cloudActivate').disabled, true);
    assert.equal(h.element('cloudTest').disabled, true);
    await h.test();
    assert.equal(h.requests.length, 0);
    assert.match(h.element('cloudFeedback').textContent, /先使用.*配置|保存并启用/);
  });

  await check('saving CPA carries profile identity and preserves both independent secrets', async () => {
    const h = harness(), deepseekKey = h.keys.get('deepseek');
    h.select('cpa_gpt');
    h.input('cloudApiKey', secret);
    await h.save();
    const payload = JSON.parse(h.requests[0].options.body);
    assert.equal(payload.profile_id, 'cpa_gpt');
    assert.equal(payload.activate, true);
    assert.equal(payload.api_key, secret);
    assert.equal(h.keys.get('deepseek'), deepseekKey);
    assert.equal(h.keys.get('cpa_gpt'), secret);
    assert.equal(h.element('cloudApiKey').value, secret);
    assert.equal(h.element('cloudApiKey').required, false);
    assert.match(h.element('cloudActiveProfile').textContent, /CPA GPT.*gpt-6.1-sol/);
    h.input('cloudModelId', 'gpt-6.1-sol');
    h.input('cloudApiKey', '');
    await h.save();
    assert(!Object.hasOwn(JSON.parse(h.requests.filter(r => r.options.method === 'PUT')[1].options.body), 'api_key'));
    assert.equal(h.keys.get('cpa_gpt'), secret);
    assert.equal(h.keys.get('deepseek'), deepseekKey);
  });

  await check('switching a configured profile sends identity only and tests the active model', async () => {
    const h = harness({...complete(), active_profile: 'deepseek', profiles: {
      deepseek: complete(), cpa_gpt: {...cpaDraft(), key_configured: true, configured: true}}});
    h.select('cpa_gpt');
    assert.equal(h.element('cloudActivate').disabled, false);
    await h.activate();
    assert.deepEqual(JSON.parse(h.requests[0].options.body), {active_profile: 'cpa_gpt'});
    assert.equal(h.element('cloudTest').disabled, false);
    await h.test();
    const call = h.requests.find(r => r.url.endsWith('/test'));
    assert.equal(call.options.body, undefined);
    assert.match(h.element('cloudFeedback').textContent, /连接成功.*gpt-6.1-sol/);
    assert.equal(h.keys.get('deepseek'), 'deepseek-previous-independent-key');
    assert.equal(h.keys.get('cpa_gpt'), 'cpa_gpt-previous-independent-key');
  });
  await check('input connection test works on unsaved relay and retains typed values', async () => {
    const h = harness();
    h.select('relay');
    h.input('cloudBaseUrl', 'https://draft.example/v1');
    h.input('cloudModelId', 'draft-model');
    h.input('cloudApiKey', secret);
    h.input('cloudProtocol', 'chat_completions');
    assert.equal(h.element('cloudTestInput').disabled, false);
    await h.testInput();
    assert.equal(h.requests.length, 1);
    assert.equal(h.requests[0].url, '/api/cloud-settings/test-input');
    assert.deepEqual(JSON.parse(h.requests[0].options.body), {base_url: 'https://draft.example/v1',
      model_id: 'draft-model', api_key: secret, protocol: 'chat_completions'});
    assert.equal(h.element('cloudApiKey').value, secret);
    assert.match(h.element('cloudFeedback').textContent, /连接成功.*未保存/);
    assert.equal(h.element('cloudTestInput').disabled, false);
    assert.equal(h.element('cloudTest').disabled, true);
  });
})().catch(error => { process.stderr.write(error.stack + '\n'); process.exitCode = 1; });
