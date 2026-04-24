// services/TTSService.js
// Human-sounding TTS: best-voice selection + word-queue buffering
// Word queue: accumulates recognised words for TTS_BUFFER_MS before speaking,
// so "Hello" + "World" signed quickly are spoken as "Hello World" not two
// separate choppy utterances.
export class TTSService {
  constructor() {
    this.enabled   = true;
    this.autoSpeak = true;
    this.rate      = 1.0;   // user-controlled multiplier
    this._unlocked = false;
    this._voice    = null;

    // Word-queue state
    this._wordQueue  = [];
    this._wordTimer  = null;
    this._bufferMs   = (APP_CONFIG.RECOGNITION && APP_CONFIG.RECOGNITION.TTS_BUFFER_MS) || 2000;

    var self = this;

    // iOS / Android require a user gesture before speechSynthesis works.
    var unlock = function() {
      if (self._unlocked || !window.speechSynthesis) return;
      try {
        var u = new SpeechSynthesisUtterance('');
        u.volume = 0;
        window.speechSynthesis.speak(u);
        setTimeout(function() {
          try { window.speechSynthesis.cancel(); } catch(e) {}
        }, 100);
      } catch(e) {}
      self._unlocked = true;
      document.removeEventListener('touchend', unlock);
      document.removeEventListener('click',    unlock);
    };
    document.addEventListener('touchend', unlock, { once: true, passive: true });
    document.addEventListener('click',    unlock, { once: true });

    // Voice selection — fires after voices list loads (async on Chrome)
    function _pickVoice() {
      if (!window.speechSynthesis) return;
      var voices = window.speechSynthesis.getVoices();
      if (!voices || voices.length === 0) return;

      // Priority list — neural / natural voices first
      var prefer = [
        'Google US English',
        'Microsoft Aria Online (Natural)',
        'Microsoft Jenny Online (Natural)',
        'Microsoft Guy Online (Natural)',
        'Samantha',          // macOS / iOS
        'Alex',              // macOS
        'Google UK English Female',
        'Google UK English Male',
        'Microsoft David',
        'Microsoft Zira',
      ];

      var picked = null;
      for (var pi = 0; pi < prefer.length && !picked; pi++) {
        for (var vi = 0; vi < voices.length; vi++) {
          if (voices[vi].name.indexOf(prefer[pi]) !== -1) {
            picked = voices[vi];
            break;
          }
        }
      }

      // Fall back: any English voice
      if (!picked) {
        for (var fi = 0; fi < voices.length; fi++) {
          if (voices[fi].lang && voices[fi].lang.indexOf('en') === 0) {
            picked = voices[fi];
            break;
          }
        }
      }

      if (!picked && voices.length > 0) picked = voices[0];
      self._voice = picked;
    }

    if (window.speechSynthesis) {
      if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = _pickVoice;
      }
      // Also try immediately (Firefox returns voices synchronously)
      setTimeout(_pickVoice, 80);
    }
  }

  // ── Speak a text string immediately ───────────────────────────────
  speak(text) {
    if (!this.enabled || !text || !window.speechSynthesis) return;

    // Clean up the text a bit — capitalise first letter, add period if missing
    var t = text.trim();
    if (t.length > 0) {
      t = t.charAt(0).toUpperCase() + t.slice(1);
      if (!/[.!?]$/.test(t)) t += '.';
    }

    window.speechSynthesis.cancel();

    var u = new SpeechSynthesisUtterance(t);

    // Human-sounding parameters
    u.rate  = this.rate * 0.92;   // ~8% slower than default — gives listener time to follow
    u.pitch = 1.08;               // slightly above neutral — sounds less robotic
    u.volume = 1.0;

    if (this._voice) u.voice = this._voice;

    var self = this;
    if (!self._unlocked) {
      setTimeout(function() {
        try { window.speechSynthesis.cancel(); window.speechSynthesis.speak(u); } catch(e) {}
      }, 300);
    } else {
      try { window.speechSynthesis.speak(u); } catch(e) {}
    }
  }

  // ── Auto-speak: buffer words, speak as a phrase after silence ────
  // If another word arrives within TTS_BUFFER_MS, it joins the queue.
  // This prevents choppy word-by-word speech when signing quickly.
  speakIfAuto(word) {
    if (!this.autoSpeak || !word) return;
    var self = this;

    this._wordQueue.push(word);

    if (this._wordTimer) clearTimeout(this._wordTimer);
    this._wordTimer = setTimeout(function() {
      var phrase = self._wordQueue.join(' ');
      self._wordQueue  = [];
      self._wordTimer  = null;
      self.speak(phrase);
    }, this._bufferMs);
  }

  // ── Flush the word queue immediately (called by speakSentence) ───
  flushQueue() {
    if (this._wordTimer) {
      clearTimeout(this._wordTimer);
      this._wordTimer = null;
    }
    if (this._wordQueue.length > 0) {
      var phrase = this._wordQueue.join(' ');
      this._wordQueue = [];
      this.speak(phrase);
    }
  }

  stop()     { if (window.speechSynthesis) try { window.speechSynthesis.cancel(); } catch(e) {} }
  setRate(r) { this.rate = Math.max(0.5, Math.min(2, r)); }
}
