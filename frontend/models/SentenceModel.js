// models/SentenceModel.js — v6.1 Phase 2B
// Tracks gesture sequence alongside sentence for NLP personalisation
// No optional-chaining — Safari 12 safe
export class SentenceModel {
  constructor() {
    this.words           = [];
    this.spelling        = '';
    this.context         = [];
    this.suggestions     = ['hello','I','please','help','thank'];
    this.wordSuggestions = [];
    this.completion      = null;
    this.gestureSequence = []; // Phase 2B: tracks gestures used to build sentence
  }

  addWord(w) {
    this.words.push(w.toLowerCase());
    this.context.push(w.toLowerCase());
    if (this.context.length > 50) this.context = this.context.slice(-30);
    this.spelling = ''; this.wordSuggestions = [];
    this._syncToServer('add_word', w);
  }

  // Phase 2B: log the gesture that produced this word
  addWordFromGesture(w, gestureName) {
    this.addWord(w);
    if (gestureName) this.gestureSequence.push(gestureName);
    if (this.gestureSequence.length > 50) this.gestureSequence = this.gestureSequence.slice(-30);
  }

  removeLastWord() {
    this.words.pop();
    if (this.gestureSequence.length > 0) this.gestureSequence.pop();
    this._syncToServer('remove_last');
  }

  clear() {
    this.words = []; this.spelling = ''; this.completion = null;
    this.wordSuggestions = []; this.gestureSequence = [];
    this._syncToServer('clear');
  }

  // Phase 2B: called when user accepts/speaks a sentence — learns from it
  acceptAndLearn(nlpService) {
    var sentence = this.getSentence();
    if (sentence && nlpService && sentence.split(' ').length >= 2) {
      nlpService.learnFromSentence(sentence, this.gestureSequence.slice());
    }
  }

  getSentence()     { return this.words.join(' '); }
  getLastWord()     { return this.words[this.words.length - 1] || null; }
  getWordCount()    { return this.words.length; }
  getContextString(){ return this.context.slice(-20).join(' '); }
  getRecentGestures(n){ return this.gestureSequence.slice(-(n || 5)); }
  setSuggestions(s) { this.suggestions     = s || []; }
  setCompletion(t)  { this.completion      = t; }
  setWordSuggestions(s){ this.wordSuggestions = s || []; }

  addLetter(ch)  { this.spelling += ch.toLowerCase(); this._syncToServer('add_letter', ch); }
  removeLetter() { this.spelling = this.spelling.slice(0, -1); }
  clearSpelling(){ this.spelling = ''; this.wordSuggestions = []; }
  getSpelling()  { return this.spelling; }

  acceptWordSuggestion(word) {
    this.addWord(word); this.spelling = ''; this.wordSuggestions = [];
    this._syncToServer('accept_word_suggestion', word);
  }

  acceptCompletion() {
    if (!this.completion) return false;
    this.words = this.completion.split(' ').map(function(w){ return w.toLowerCase(); });
    this.completion = null;
    this._syncToServer('accept_completion');
    return true;
  }

  replaceWithCorrected(c) {
    this.words = c.split(' ').map(function(w){ return w.toLowerCase(); });
    this._syncToServer('replace_corrected', null, c);
  }

  getDisplayText() {
    var t = this.getSentence();
    if (this.spelling) t += (t ? ' ' : '') + this.spelling + '_';
    return t;
  }

  _syncToServer(action, word, text) {
    word = word || null; text = text || null;
    fetch('/api/sentence', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ action:action, word:word, text:text })
    }).catch(function(){});
  }
}
