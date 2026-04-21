// services/DatabaseService.js — REMOVED (upgrade 4)
// All persistence now lives in Python SQLite backend (backend/main.py)
// This shim keeps existing import statements working without errors.

const API = '/api';

export class DatabaseService {
  constructor() { this._ready = Promise.resolve(); }

  async loadModel(id) {
    // Models are loaded directly by AppController via /api/nn/load/:id
    // Return null so AppController falls through to its own fetch call
    return null;
  }
  async saveModel(id, data) {
    await fetch(`${API}/nn/save/${id}`, { method: 'POST' });
  }
  async clearModels() {
    await fetch(`${API}/nn/reset/static`, { method: 'POST' });
    await fetch(`${API}/nn/reset/dynamic`, { method: 'POST' });
  }
  async saveStaticSamples(gesture, samples) {
    await fetch(`${API}/samples/save`, { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ gesture, samples, sample_type: 'static' }) });
  }
  async saveDynamicSamples(gesture, samples) {
    await fetch(`${API}/samples/save`, { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ gesture, samples, sample_type: 'dynamic' }) });
  }
  async getAllStaticSamples() {
    const r = await fetch(`${API}/samples/load`); const d = await r.json();
    const out = {};
    for (const [g, v] of Object.entries(d)) { if (v.static) out[g] = v.static; }
    return out;
  }
  async getAllDynamicSamples() {
    const r = await fetch(`${API}/samples/load`); const d = await r.json();
    const out = {};
    for (const [g, v] of Object.entries(d)) { if (v.dynamic) out[g] = v.dynamic; }
    return out;
  }
  async getAllMeta() {
    const r = await fetch(`${API}/samples/meta`);
    return r.json();
  }
  async clearAllSamples() {
    await fetch(`${API}/samples/clear`, { method: 'DELETE' });
  }
  async getSamplePreview(gesture, n = 3) {
    const r = await fetch(`${API}/samples/preview/${encodeURIComponent(gesture)}`);
    return r.json();
  }
  async loadSetting(key) {
    const r = await fetch(`${API}/settings/${key}`);
    const d = await r.json();
    return d.value ? JSON.parse(d.value) : null;
  }
  async saveSetting(key, value) {
    await fetch(`${API}/settings`, { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ key, value: JSON.stringify(value) }) });
  }
  async getStats() {
    const [meta, status] = await Promise.all([
      fetch(`${API}/samples/meta`).then(r => r.json()),
      fetch(`${API}/nn/status`).then(r => r.json()),
    ]);
    const gestures = Object.keys(meta);
    return {
      gestureCount: gestures.length,
      totalStatic:  gestures.reduce((s, g) => s + (meta[g].static || 0), 0),
      totalDynamic: gestures.reduce((s, g) => s + (meta[g].dynamic || 0), 0),
    };
  }
}
