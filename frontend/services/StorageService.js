// services/StorageService.js — Routes persistent storage to Python backend
const API = '/api';

export class StorageService {
  static save(k, d) {
    fetch(`${API}/settings`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({key: k, value: JSON.stringify(d)})
    }).catch(() => {});
    // Also mirror to localStorage as instant fallback
    try { localStorage.setItem('signlens_' + k, JSON.stringify(d)); } catch(e) {}
    return true;
  }

  static load(k, fb = null) {
    // Sync read from localStorage (Python backend is async)
    try {
      const r = localStorage.getItem('signlens_' + k);
      return r ? JSON.parse(r) : fb;
    } catch(e) { return fb; }
  }

  static async loadAsync(k, fb = null) {
    try {
      const res = await fetch(`${API}/settings/${k}`);
      const data = await res.json();
      if (data.value) {
        const parsed = JSON.parse(data.value);
        // Sync back to localStorage
        try { localStorage.setItem('signlens_' + k, data.value); } catch(e) {}
        return parsed;
      }
    } catch(e) {}
    return fb;
  }

  static remove(k) {
    localStorage.removeItem('signlens_' + k);
    fetch(`${API}/settings/${k}`, {method: 'DELETE'}).catch(() => {});
  }

  static saveApiKey(k) { this.save('apiKey', k); }
  static loadApiKey() { return this.load('apiKey', ''); }
  static saveCalibration(c) { this.save('calibration', c); }
  static loadCalibration() { return this.load('calibration', null); }
  static savePin(p) { this.save('adminPin', p); }
  static loadPin() { return this.load('adminPin', '1234'); }
}
