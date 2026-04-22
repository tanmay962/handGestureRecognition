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

  // ── Grammar correction — Gemini first, backend fallback ──────
  async correctGrammar(sm) {
    // Try Gemini
    if (this.gemini && this.gemini.enabled && sm.getWordCount() >= 3) {
      var aiResult = await this.gemini.correctGrammar(sm.getSentence());
      if (aiResult) return aiResult;
    }
    // Backend fallback (Python rule-based, same logic as before)
    try {
      var res = await fetch(API + '/nlp/grammar', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ sentence: sm.getSentence() })
      });
      var data = await res.json();
      return data.corrected || null;
    } catch(e) {
      return null;
    }
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
