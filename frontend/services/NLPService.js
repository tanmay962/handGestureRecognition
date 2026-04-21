// services/NLPService.js — v6.1 Phase 2B
// Adaptive NLP: personal corpus, gesture-aware suggestions, offline grammar
// No optional-chaining — Safari 12 safe
var API = '/api';

export class NLPService {
  constructor(geminiService) {
    this.gemini            = geminiService;
    this._personalCorpusCount = 0;   // sentences learned so far
    this._sentencesSinceRetrain = 0; // trigger retrain every N sentences
  }

  // ── Next-word suggestions ─────────────────────────────────────
  // Priority: Gemini AI → personal n-gram → base n-gram
  async getSuggestions(sentenceModel, recentGestures) {
    recentGestures = recentGestures || [];

    // 1. Try Gemini first
    if (this.gemini && this.gemini.enabled) {
      var ai = await this.gemini.getSuggestions(
        sentenceModel.getSentence(),
        sentenceModel.getLastWord(),
        sentenceModel.getContextString()
      );
      if (ai) return ai;
    }

    // 2. Try gesture-aware personal backend suggestions
    try {
      var res = await fetch(API + '/nlp/context_suggestions', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          words:           sentenceModel.words,
          recent_gestures: recentGestures.slice(-5), // last 5 gestures
          personal:        true,
        })
      });
      if (res.ok) {
        var data = await res.json();
        if (data.suggestions && data.suggestions.length > 0) return data.suggestions;
      }
    } catch(e) {}

    // 3. Fall back to base n-gram
    try {
      var res2 = await fetch(API + '/nlp/suggestions', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ words: sentenceModel.words })
      });
      var data2 = await res2.json();
      return data2.suggestions || [];
    } catch(e) {
      return ['hello','I','please','help','thank'];
    }
  }

  // ── Sentence completion ───────────────────────────────────────
  async getCompletion(sm) {
    if (!this.gemini || !this.gemini.enabled || sm.getWordCount() < 2) return null;
    return this.gemini.completeSentence(sm.getSentence());
  }

  // ── Grammar correction — Gemini first, offline fallback ───────
  async correctGrammar(sm) {
    // Try Gemini
    if (this.gemini && this.gemini.enabled && sm.getWordCount() >= 3) {
      var aiResult = await this.gemini.correctGrammar(sm.getSentence());
      if (aiResult) return aiResult;
    }
    // Offline fallback
    return this._offlineGrammar(sm.getSentence());
  }

  // ── Offline grammar correction (no Gemini needed) ─────────────
  // Handles most common sign language grammar patterns
  _offlineGrammar(sentence) {
    if (!sentence || sentence.trim().length === 0) return null;
    var words = sentence.trim().toLowerCase().split(/\s+/);
    var corrected = words.slice();
    var changed = false;

    // Rule 1: Add "I" at start if sentence starts with action verb
    var actionVerbs = ['want','need','like','love','hate','go','see','hear','feel','know','think'];
    if (corrected.length > 0 && actionVerbs.indexOf(corrected[0]) >= 0) {
      corrected.unshift('i');
      changed = true;
    }

    // Rule 2: "me" at start → "I"
    if (corrected[0] === 'me') { corrected[0] = 'i'; changed = true; }

    // Rule 3: Capitalise first word
    if (corrected.length > 0) {
      corrected[0] = corrected[0].charAt(0).toUpperCase() + corrected[0].slice(1);
    }

    // Rule 4: "i " → "I " (anywhere in sentence)
    for (var i = 1; i < corrected.length; i++) {
      if (corrected[i] === 'i') { corrected[i] = 'I'; changed = true; }
    }

    // Rule 5: Remove duplicate consecutive words
    var deduped = [corrected[0]];
    for (var j = 1; j < corrected.length; j++) {
      if (corrected[j].toLowerCase() !== corrected[j-1].toLowerCase()) {
        deduped.push(corrected[j]);
      } else {
        changed = true;
      }
    }
    corrected = deduped;

    // Rule 6: Add period at end if missing
    var last = corrected[corrected.length - 1];
    if (last && '.!?'.indexOf(last.slice(-1)) < 0) {
      corrected[corrected.length - 1] = last + '.';
      changed = true;
    }

    return changed ? corrected.join(' ') : null;
  }

  // ── Learn from accepted sentence (Phase 2B) ───────────────────
  async learnFromSentence(sentence, gestureSequence) {
    if (!sentence || sentence.trim().length === 0) return;
    gestureSequence = gestureSequence || [];

    try {
      await fetch(API + '/nlp/learn', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          sentence:         sentence.trim(),
          gesture_sequence: gestureSequence,
        })
      });
      this._personalCorpusCount++;
      this._sentencesSinceRetrain++;

      // Trigger background retrain every N sentences
      var RETRAIN_EVERY = (window.APP_CONFIG && window.APP_CONFIG.NLP && window.APP_CONFIG.NLP.RETRAIN_AFTER_SENTENCES) || 3;
      if (this._sentencesSinceRetrain >= RETRAIN_EVERY) {
        this._sentencesSinceRetrain = 0;
        this._triggerRetrain();
      }
    } catch(e) {}
  }

  // ── Trigger background NLP retrain ───────────────────────────
  _triggerRetrain() {
    fetch(API + '/nlp/retrain', { method:'POST' })
      .then(function(r){ return r.json(); })
      .then(function(d){ console.log('[NLP] personal model retrained:', d.sentences, 'sentences'); })
      .catch(function(){});
  }

  // ── Word prefix suggestions for spelling ──────────────────────
  async getWordSuggestions(prefix) {
    if (!prefix || prefix.length < 1) return [];
    try {
      var res = await fetch(API + '/nlp/word_suggestions', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ prefix: prefix })
      });
      var data = await res.json();
      return data.suggestions || [];
    } catch(e) {
      return [];
    }
  }

  // Sync shim for inline use
  getWordSuggestionsSync(prefix) {
    if (!prefix) return [];
    var COMMON = 'the,be,to,of,and,a,in,that,have,I,it,for,not,on,with,you,do,at,this,help,please,thank,yes,no,stop,go,sorry,hello,water,food,good,need,home,here,come,open,close'.split(',');
    return COMMON.filter(function(w){ return w.startsWith(prefix.toLowerCase()) && w !== prefix.toLowerCase(); }).slice(0, 5);
  }

  getPersonalCorpusCount() { return this._personalCorpusCount; }
}
