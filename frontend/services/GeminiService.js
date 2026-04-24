// services/GeminiService.js
// All Gemini calls are proxied through the backend — API key never touches the browser.
import {eventBus,Events} from '../utils/EventBus.js';

var GEMINI_API = '/api/gemini';

export class GeminiService {
  constructor() {
    this.enabled = false;
    this._checked = false;
  }

  // Called once on init — asks backend whether a key is available (env or DB).
  async loadServerKey() {
    if (this._checked) return;
    this._checked = true;
    try {
      var r = await fetch('/api/settings/gemini_key');
      var d = await r.json();
      if (d.available) {
        this.enabled = true;
        console.log('[Gemini] Server-side proxy ready');
      }
    } catch(e) {}
  }

  setApiKey(k) { /* key lives server-side only — no-op */ }

  async getSuggestions(sentence, lastWord, context) {
    if (!this.enabled) return null;
    try {
      var r = await fetch(GEMINI_API + '/suggestions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence: sentence || '', last_word: lastWord || '', context: context || '' }),
      });
      var d = await r.json();
      return d.suggestions || null;
    } catch(e) { return null; }
  }

  async completeSentence(s) {
    if (!this.enabled || !s) return null;
    try {
      var r = await fetch(GEMINI_API + '/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence: s }),
      });
      var d = await r.json();
      return d.completion || null;
    } catch(e) { return null; }
  }

  async correctGrammar(s) {
    if (!this.enabled || !s) return null;
    try {
      var r = await fetch(GEMINI_API + '/grammar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence: s }),
      });
      var d = await r.json();
      return d.corrected || null;
    } catch(e) { return null; }
  }
}
