// services/TTSService.js
export class TTSService {
  constructor() {
    this.enabled    = true;
    this.autoSpeak  = true;
    this.rate       = 1;
    this._unlocked  = false;
    // iOS / Android require a user gesture before speechSynthesis will work.
    // We fire a silent utterance on the first touch/click to unlock the API.
    var self = this;
    var unlock = function() {
      if (self._unlocked || !window.speechSynthesis) return;
      try {
        var u = new SpeechSynthesisUtterance('');
        u.volume = 0;
        window.speechSynthesis.speak(u);
        // Cancel after a tick — just needed to warm up the engine
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
  }

  speak(t) {
    if (!this.enabled || !t || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(t);
    u.rate = this.rate;
    // On some mobile browsers speak() must be called in a microtask
    // after user-gesture unlock — a short delay makes it reliable
    var self = this;
    if (!self._unlocked) {
      // Queue and wait for unlock; will fire once user taps
      setTimeout(function() {
        try { window.speechSynthesis.cancel(); window.speechSynthesis.speak(u); } catch(e) {}
      }, 300);
    } else {
      try { window.speechSynthesis.speak(u); } catch(e) {}
    }
  }

  speakIfAuto(t) { if (this.autoSpeak) this.speak(t); }
  stop()         { if (window.speechSynthesis) try { window.speechSynthesis.cancel(); } catch(e) {} }
  setRate(r)     { this.rate = Math.max(0.5, Math.min(2, r)); }
}
