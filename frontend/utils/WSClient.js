// utils/WSClient.js — WebSocket client for Python event bus
// Safari-safe: all errors caught, never blocks module loading, protocol auto-detected

class WSClient {
  constructor() {
    this._handlers = {};
    this._ws       = null;
    this._alive    = false;
    // Defer connection so module parsing always completes first
    setTimeout(() => this._connect(), 500);
  }

  _connect() {
    try {
      // Match ws/wss to the page protocol — Safari blocks mixed content
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws    = new WebSocket(`${proto}://${location.host}/ws`);

      ws.onopen = () => {
        this._alive = true;
        console.log('[WS] connected to Python event bus');
        try { ws.send('ping'); } catch(e) {}
      };

      ws.onclose = () => {
        this._alive = false;
        // Reconnect after 3s, silently
        setTimeout(() => this._connect(), 3000);
      };

      ws.onerror = () => {
        // Silently ignore — onclose fires next and will reconnect
      };

      ws.onmessage = (e) => {
        try {
          const data      = JSON.parse(e.data);
          const { event, ...payload } = data;
          (this._handlers[event] || []).forEach(cb => {
            try { cb(payload); } catch(err) { console.warn('[WS handler]', err); }
          });
          (this._handlers['*'] || []).forEach(cb => {
            try { cb(data); } catch(err) {}
          });
        } catch(err) {}
      };

      this._ws = ws;
    } catch(err) {
      // WebSocket not available or blocked — retry silently
      setTimeout(() => this._connect(), 5000);
    }
  }

  on(event, cb) {
    if (!this._handlers[event]) this._handlers[event] = [];
    this._handlers[event].push(cb);
    return () => {
      this._handlers[event] = (this._handlers[event] || []).filter(h => h !== cb);
    };
  }

  send(data) {
    try {
      if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        this._ws.send(typeof data === 'string' ? data : JSON.stringify(data));
      }
    } catch(e) {}
  }
}

export const wsClient = new WSClient();
