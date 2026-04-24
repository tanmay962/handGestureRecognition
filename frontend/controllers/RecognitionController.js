// controllers/RecognitionController.js — Gesture Detection v1.0
// Optimisations:
//   1. Rate limiting: predict every Nth frame (not every frame)
//   2. Confidence smoothing: rolling average over last N frames
//   3. Time-based stability: hold gesture for N ms (not N frames)
//   4. Separate cooldowns: letter vs word vs same-letter
//   5. Motion detection: LSTM only activates on movement
//   6. NLP debounce: wait 300ms before fetching suggestions
'use strict';

var RecognitionController = (function() {
  function RecognitionController(staticNN, dynamicNN, sensorModel, sentenceModel, nlpService, ttsService) {
    this.sNN        = staticNN;
    this.dNN        = dynamicNN;
    this.sensor     = sensorModel;
    this.sentence   = sentenceModel;
    this.nlp        = nlpService;
    this.tts        = ttsService;
    this.running    = false;
    this.confThresh = APP_CONFIG.RECOGNITION.CONFIDENCE_THRESHOLD;
    this.log        = [];

    // ── Rate limiting ────────────────────────────────────────
    this._frameCount   = 0;
    this._predictEvery = APP_CONFIG.RECOGNITION.PREDICT_EVERY_N;

    // ── Probability-vector ensemble ──────────────────────────
    this._probHistory    = []; // [{probs:[], name:'', idx:0}, ...]
    this._ensembleWindow = APP_CONFIG.RECOGNITION.ENSEMBLE_WINDOW || 5;

    // ── Time-based stability ─────────────────────────────────
    this._stableName      = null;
    this._stableStartTime = null;

    // ── Cooldowns ────────────────────────────────────────────
    this._lastConfirmedGesture = null;
    this._lastConfirmedTime    = 0;

    // ── Motion detection for LSTM ────────────────────────────
    this._frameBuffer    = [];
    this._lastFeatures   = null;
    this._motionActive   = false;
    this._motionThresh   = APP_CONFIG.RECOGNITION.MOTION_THRESHOLD;
    this._dynamicActive  = false;

    // ── Duplicate prediction guard ────────────────────────────
    this._predictPending = false;

    // ── Hysteresis ────────────────────────────────────────────
    this._hysteresisGesture = null;

    // ── Streak gate ───────────────────────────────────────────
    this._streakName  = null;
    this._streakCount = 0;

    // ── Prediction trail ─────────────────────────────────────
    this._predTrail = [];

    // ── Top-3 histogram ──────────────────────────────────────
    this._topPredictions = [];

    // ── Input mode ('camera' | 'glove' | 'both') ─────────────
    this.inputMode = 'camera';

    // ── Adaptive cooldown ─────────────────────────────────────
    this._adaptiveCooldownUntil = 0;

    // ── Aux-hand independent tracking (word-level gestures only) ─
    this._streakNameAux      = null;
    this._streakCountAux     = 0;
    this._stableNameAux      = null;
    this._stableStartTimeAux = null;
    this._lastConfTimeAux    = 0;
    this._probHistoryAux     = [];
    this._auxPrediction      = null; // { name, conf } for live display

    // ── NLP debounce ─────────────────────────────────────────
    this._nlpTimer = null;

    // ── Dwell + spelling ────────────────────────────────────
    this._dwellFingers  = -1;
    this._dwellStart    = 0;
    this._dwellActive   = false;
    this._autoAcceptTimer = null;
    this._lastLetterTime  = 0;

    // ── System gesture ───────────────────────────────────────
    this._sysGestStart = 0;
    this._sysGestName  = null;

    this.contextState = 'IDLE';

    var self = this;
    // Always update sensor — regardless of recognition running state
    eventBus.on(Events.FEATURES_EXTRACTED, function(d) {
      self.sensor.setFromFeatures(d.features, { poseDetected: d.poseDetected });
      if (self.running) self._onFeatures(d.features);
    });
  }

  RecognitionController.prototype.start = function() {
    if (!this.sNN.trained && !this.dNN.trained) return false;
    this.running = true;
    this.contextState = 'RECOGNIZING';
    eventBus.emit(Events.RECOG_STARTED);
    return true;
  };

  RecognitionController.prototype.stop = function() {
    this.running = false;
    this.contextState = 'IDLE';
    this._clearAuxPrediction();
    eventBus.emit(Events.RECOG_STOPPED);
  };

  // ── Main frame handler ────────────────────────────────────
  RecognitionController.prototype._onFeatures = function(features) {
    // Rate limiting — skip frames
    this._frameCount++;
    if (this._frameCount % this._predictEvery !== 0) return;

    this._processFrame(features);
  };

  RecognitionController.prototype._processFrame = function(features) {
    var self = this;

    // Duplicate prediction guard — skip if previous fetch hasn't resolved yet
    if (this._predictPending) return;

    // System gestures check first
    if (this._checkSystemGestures(features)) return;

    // Dwell selection
    if (this.sentence.suggestions.length > 0 || this.sentence.wordSuggestions.length > 0) {
      this._checkDwellSelection();
    }

    // Motion detection for LSTM
    var motionDetected = this._detectMotion(features);
    if (motionDetected) {
      this._frameBuffer.push(features.slice());
      this._motionActive = true;
    } else if (this._motionActive && this._frameBuffer.length >= 15) {
      // Motion stopped — pad to 45 frames and predict dynamic
      this._dynamicActive = true;
      this._motionActive  = false;
    }
    if (this._frameBuffer.length > APP_CONFIG.NN.DYNAMIC_FRAMES) {
      this._frameBuffer.shift();
    }

    // Build aux-swapped feature vector when aux hand is present (features[39] = auxPresent)
    var auxSwapped = this._buildAuxSwapped(features);

    // Static prediction
    if (!this.sNN.trained) return;
    this._predictPending = true;
    fetch('/api/nn/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: features, aux_features: auxSwapped || [], model_type: 'static' }),
    }).then(function(r) {
      return r.json();
    }).then(function(staticResult) {
      // Ensemble: average last N probability vectors, take argmax of the mean
      var ensembled = self._ensembleVote(staticResult);

      // Handle aux-hand result independently
      if (staticResult.aux) {
        self._handleAuxResult(staticResult.aux);
      } else {
        self._clearAuxPrediction();
      }

      // Broadcast live prediction to canvas overlay on every frame
      if (ensembled && ensembled.name && ensembled.conf > 0.25) {
        eventBus.emit(Events.GESTURE_PREDICTING, {
          gesture:    ensembled.name,
          conf:       ensembled.conf,
          model:      'static',
          auxGesture: self._auxPrediction ? self._auxPrediction.name : null,
          auxConf:    self._auxPrediction ? self._auxPrediction.conf : 0,
        });
      }

      // Dynamic prediction if motion detected and buffer ready
      if (self.dNN.trained && self._dynamicActive && self._frameBuffer.length >= 20) {
        var traj = self._trimmedBuffer();
        self._frameBuffer   = []; // clear immediately so next gesture starts fresh
        self._dynamicActive = false;
        fetch('/api/nn/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features: traj, model_type: 'dynamic' }),
        }).then(function(r) { return r.json(); })
          .then(function(dynResult) {
            self._predictPending = false;
            if (dynResult && dynResult.conf > APP_CONFIG.RECOGNITION.DYNAMIC_CONF_THRESH) {
              eventBus.emit(Events.GESTURE_PREDICTING, {
                gesture: dynResult.name, conf: dynResult.conf, model: 'dynamic'
              });
            }
            self._mergeAndStabilise(ensembled, ensembled.conf, dynResult, features);
          }).catch(function() {
            self._predictPending = false;
            self._mergeAndStabilise(ensembled, ensembled.conf, null, features);
          });
      } else {
        self._predictPending = false;
        self._mergeAndStabilise(ensembled, ensembled.conf, null, features);
      }
    }).catch(function() { self._predictPending = false; });

    this._lastFeatures = features.slice();
  };

  // ── Aux-hand feature swap ────────────────────────────────
  // Moves aux hand features (11-21) into dom slot (0-10), zeros dom slot.
  // Result: model predicts what the *left* hand is doing, independently.
  RecognitionController.prototype._buildAuxSwapped = function(features) {
    if (!features || !features[39]) return null; // no aux hand present
    var v = features.slice();
    for (var i = 0; i < 11; i++) {
      v[i]      = features[i + 11]; // aux → dom slot
      v[i + 11] = 0;                // clear aux slot
    }
    v[38] = features[39]; // auxPresent → domPresent flag
    v[39] = 0;
    return v;
  };

  // ── Aux-hand ensemble (mirrors _ensembleVote but uses separate history) ─
  RecognitionController.prototype._ensembleAux = function(result) {
    if (!result || !result.probs || result.probs.length === 0) return result;
    this._probHistoryAux.push({ probs: result.probs.slice() });
    if (this._probHistoryAux.length > this._ensembleWindow) this._probHistoryAux.shift();
    if (this._probHistoryAux.length < 2) return result;

    var n = result.probs.length;
    var avg = new Array(n).fill(0);
    for (var h = 0; h < this._probHistoryAux.length; h++) {
      var pv = this._probHistoryAux[h].probs;
      for (var j = 0; j < n && j < pv.length; j++) avg[j] += pv[j] / this._probHistoryAux.length;
    }
    var bestIdx = 0, bestConf = 0;
    for (var k = 0; k < avg.length; k++) {
      if (avg[k] > bestConf) { bestConf = avg[k]; bestIdx = k; }
    }
    var bestName = (this.sNN && this.sNN.getName) ? this.sNN.getName(bestIdx) : result.name;
    if (!bestName || bestName === 'Unknown') bestName = result.name;
    return { name: bestName, conf: bestConf, idx: bestIdx, probs: avg };
  };

  // ── Aux-hand prediction handler ──────────────────────────
  // Aux hand fires word-level gestures only — letters stay on dom hand
  // to avoid collision with ongoing spelling.
  RecognitionController.prototype._handleAuxResult = function(result) {
    var ensembled = this._ensembleAux(result);
    var isLetterOrDigit = ensembled.name && ensembled.name.length === 1 && /[A-Z0-9]/.test(ensembled.name);

    if (!ensembled || ensembled.conf < this.confThresh || ensembled.below_threshold || isLetterOrDigit) {
      this._clearAuxPrediction();
      return;
    }

    // Update live display (shown on canvas + DOM badge)
    this._auxPrediction = { name: ensembled.name, conf: ensembled.conf };
    this._updateAuxDisplay(ensembled.name, ensembled.conf);

    // Streak gate
    if (ensembled.name === this._streakNameAux) {
      this._streakCountAux++;
    } else {
      this._streakNameAux      = ensembled.name;
      this._streakCountAux     = 1;
      this._stableNameAux      = null;
      this._stableStartTimeAux = null;
    }
    var minStreak = APP_CONFIG.RECOGNITION.MIN_STREAK_FRAMES || 2;
    if (this._streakCountAux < minStreak) return;

    // Time-based stability
    var now = Date.now();
    if (ensembled.name === this._stableNameAux) {
      if (!this._stableStartTimeAux) this._stableStartTimeAux = now;
      if (now - this._stableStartTimeAux >= APP_CONFIG.RECOGNITION.STABLE_MS_WORD) {
        if (now - this._lastConfTimeAux >= APP_CONFIG.RECOGNITION.COOLDOWN_WORD) {
          this._lastConfTimeAux    = now;
          this._stableStartTimeAux = null;
          this._streakCountAux     = 0;
          this._onGestureConfirmed(ensembled.name, ensembled.conf, 'aux-hand');
        }
      }
    } else {
      this._stableNameAux      = ensembled.name;
      this._stableStartTimeAux = now;
    }
  };

  RecognitionController.prototype._clearAuxPrediction = function() {
    if (this._auxPrediction) {
      this._auxPrediction      = null;
      this._stableNameAux      = null;
      this._stableStartTimeAux = null;
      this._streakNameAux      = null;
      this._streakCountAux     = 0;
      this._probHistoryAux     = [];
      this._updateAuxDisplay(null, 0);
    }
  };

  RecognitionController.prototype._updateAuxDisplay = function(name, conf) {
    var el = document.getElementById('auxGestDisp');
    var en = document.getElementById('auxGestName');
    var ec = document.getElementById('auxGestConf');
    if (!el) return;
    if (!name) {
      el.style.display = 'none';
      return;
    }
    el.style.display = 'block';
    el.style.opacity = (0.4 + conf * 0.6).toFixed(2);
    if (en) en.textContent = name;
    if (ec) ec.textContent = (conf * 100).toFixed(1) + '% [aux]';
  };

  // ── Probability-vector ensemble ──────────────────────────
  // Averages last N full softmax outputs and argmaxes the mean.
  // More accurate than per-name confidence averaging because it
  // considers all classes simultaneously.
  RecognitionController.prototype._ensembleVote = function(result) {
    if (!result || !result.probs || result.probs.length === 0) {
      return result || { name: 'Unknown', conf: 0, idx: -1, probs: [] };
    }
    this._probHistory.push({ probs: result.probs.slice(), name: result.name, idx: result.idx });
    if (this._probHistory.length > this._ensembleWindow) {
      this._probHistory.shift();
    }
    if (this._probHistory.length < 2) return result;

    // Element-wise average across history
    var n = result.probs.length;
    var avg = new Array(n).fill(0);
    for (var h = 0; h < this._probHistory.length; h++) {
      var pv = this._probHistory[h].probs;
      for (var j = 0; j < n && j < pv.length; j++) {
        avg[j] += pv[j] / this._probHistory.length;
      }
    }
    var bestIdx = 0, bestConf = 0;
    for (var k = 0; k < avg.length; k++) {
      if (avg[k] > bestConf) { bestConf = avg[k]; bestIdx = k; }
    }
    // Look up name from the frontend gesture map
    var bestName = (this.sNN && this.sNN.getName) ? this.sNN.getName(bestIdx) : result.name;
    if (!bestName || bestName === 'Unknown') bestName = result.name;

    // Extract top-3 for confidence histogram display
    var indexed = [];
    for (var ti = 0; ti < avg.length; ti++) indexed.push({ idx: ti, conf: avg[ti] });
    indexed.sort(function(a, b) { return b.conf - a.conf; });
    this._topPredictions = [];
    for (var t2 = 0; t2 < Math.min(3, indexed.length); t2++) {
      var tn = (this.sNN && this.sNN.getName) ? this.sNN.getName(indexed[t2].idx) : String(indexed[t2].idx);
      if (tn && tn !== 'Unknown') this._topPredictions.push({ name: tn, conf: indexed[t2].conf });
    }

    return { name: bestName, conf: bestConf, idx: bestIdx, probs: avg };
  };

  // ── Motion detection ─────────────────────────────────────
  RecognitionController.prototype._detectMotion = function(features) {
    if (!this._lastFeatures) return false;
    var diff = 0;
    // Check orientation features (indices 5-10, 16-21) for both hands
    var checkIdx = [5,6,7,8,9,10, 16,17,18,19,20,21];
    for (var i = 0; i < checkIdx.length; i++) {
      diff += Math.abs(features[checkIdx[i]] - (this._lastFeatures[checkIdx[i]] || 0));
    }
    return (diff / checkIdx.length) > this._motionThresh;
  };

  // ── Trim buffer to active motion frames ──────────────────
  RecognitionController.prototype._trimmedBuffer = function() {
    var buf = this._frameBuffer;
    // Pad to 45 frames
    var padded = buf.slice();
    while (padded.length < APP_CONFIG.NN.DYNAMIC_FRAMES) {
      padded.push(padded[padded.length - 1] || new Array(41).fill(0));
    }
    padded = padded.slice(0, APP_CONFIG.NN.DYNAMIC_FRAMES);
    var flat = [];
    for (var i = 0; i < padded.length; i++) {
      for (var j = 0; j < padded[i].length; j++) {
        flat.push(padded[i][j]);
      }
    }
    return flat;
  };

  // ── Merge static + dynamic, then time-based stability ────
  RecognitionController.prototype._mergeAndStabilise = function(staticResult, smoothedConf, dynResult, features) {
    var finalName = null, finalConf = 0, finalModel = 'none';
    var dynThresh   = APP_CONFIG.RECOGNITION.DYNAMIC_CONF_THRESH;
    var minReject   = APP_CONFIG.RECOGNITION.MIN_REJECT_CONF      || 0.15;
    var hystExit    = APP_CONFIG.RECOGNITION.HYSTERESIS_EXIT       || 0.45;
    var dynMargin   = APP_CONFIG.RECOGNITION.DYNAMIC_PRIORITY_MARGIN || 0.05;
    var minStreak   = APP_CONFIG.RECOGNITION.MIN_STREAK_FRAMES     || 3;

    // ── 1. Pick winner: dynamic gets priority with margin tie-break ──
    if (dynResult && dynResult.conf > dynThresh) {
      finalName  = dynResult.name;
      finalConf  = dynResult.conf;
      finalModel = 'dynamic';
    } else if (dynResult && dynResult.conf > smoothedConf - dynMargin && dynResult.conf > dynThresh * 0.85) {
      // Dynamic within margin of static — prefer dynamic for motion gestures
      finalName  = dynResult.name;
      finalConf  = dynResult.conf;
      finalModel = 'dynamic';
    } else if (staticResult && smoothedConf >= this.confThresh) {
      finalName  = staticResult.name;
      finalConf  = smoothedConf;
      finalModel = 'static';
    } else if (staticResult && smoothedConf >= hystExit && this._hysteresisGesture === staticResult.name) {
      // ── 2. Hysteresis: hold current gesture at lower exit threshold ──
      finalName  = staticResult.name;
      finalConf  = smoothedConf;
      finalModel = 'static';
    }

    // ── 3. Rejection zone: discard anything below minimum confidence ──
    if (finalName && finalConf < minReject) {
      finalName = null; finalConf = 0;
    }

    // ── Update hysteresis state ──
    this._hysteresisGesture = finalName || null;

    // ── On empty frame: reset stability but keep ensemble history for smoothing ──
    if (!finalName) {
      this._stableName        = null;
      this._stableStartTime   = null;
      this._streakName        = null;
      this._streakCount       = 0;
      this._hysteresisGesture = null;
      // Decay ensemble history rather than wiping it (prevents jitter on brief occlusions)
      if (this._probHistory.length > 2) this._probHistory.shift();
      eventBus.emit(Events.GESTURE_PREDICTING, { gesture: null, conf: 0, model: '' });
      return;
    }

    // ── 4. Streak gate: require MIN_STREAK_FRAMES consecutive same-gesture frames ──
    if (finalName === this._streakName) {
      if (this._streakCount < minStreak) this._streakCount++;
    } else {
      // Gesture changed — reset streak and stability timer
      this._streakName      = finalName;
      this._streakCount     = 1;
      this._stableName      = null;
      this._stableStartTime = null;
    }
    // Streak not yet met — update live display but don't start stability timer
    if (this._streakCount < minStreak) {
      var gnS = document.getElementById('gestName');
      var gcS = document.getElementById('gestConf');
      var gdS = document.getElementById('gestDisp');
      if (gdS) { gdS.style.display = 'block'; gdS.style.opacity = (0.2 + finalConf * 0.4).toFixed(2); }
      if (gnS) gnS.textContent = finalName;
      if (gcS) gcS.textContent = (finalConf * 100).toFixed(1) + '% [building…]';
      return;
    }

    // ── 5. Update display with confidence brightness (full opacity once streak met) ──
    var gn = document.getElementById('gestName');
    var gc = document.getElementById('gestConf');
    var gd = document.getElementById('gestDisp');
    if (gd) { gd.style.display = 'block'; gd.style.opacity = (0.4 + finalConf * 0.6).toFixed(2); }
    if (gn) gn.textContent = finalName;
    if (gc) gc.textContent = (finalConf * 100).toFixed(1) + '% [' + finalModel + ']';

    // ── 6. Time-based stability check ──
    var now = Date.now();
    if (finalName === this._stableName) {
      if (!this._stableStartTime) this._stableStartTime = now;
      var holdTime = this._getHoldTime(finalName);
      if (now - this._stableStartTime >= holdTime) {
        var cooldown = this._getCooldown(finalName, this._lastConfirmedGesture);
        if (now - this._lastConfirmedTime >= cooldown) {
          this._lastConfirmedGesture = finalName;
          this._lastConfirmedTime    = now;
          this._stableStartTime      = null;
          this._streakCount          = 0;
          this._onGestureConfirmed(finalName, finalConf, finalModel);
        }
      }
    } else {
      this._stableName      = finalName;
      this._stableStartTime = now;
    }
  };

  // ── Hold time per gesture type ────────────────────────────
  RecognitionController.prototype._getHoldTime = function(name) {
    if (name.length === 1 && /[A-Z]/.test(name)) return APP_CONFIG.RECOGNITION.STABLE_MS_LETTER;
    if (name.length === 1 && /[0-9]/.test(name)) return APP_CONFIG.RECOGNITION.STABLE_MS_NUMBER;
    return APP_CONFIG.RECOGNITION.STABLE_MS_WORD;
  };

  // ── Cooldown per gesture type ─────────────────────────────
  RecognitionController.prototype._getCooldown = function(name, last) {
    var isLetter = name.length === 1 && /[A-Z]/.test(name);
    var isNumber = name.length === 1 && /[0-9]/.test(name);
    var base;
    if (!isLetter && !isNumber) base = APP_CONFIG.RECOGNITION.COOLDOWN_WORD;
    else if (name === last)     base = APP_CONFIG.RECOGNITION.COOLDOWN_SAME_LETTER;
    else                        base = APP_CONFIG.RECOGNITION.COOLDOWN_DIFF_LETTER;

    // Adaptive cooldown: reduce after user manually interacts (suggestion, quickTest)
    if (this._adaptiveCooldownUntil && Date.now() < this._adaptiveCooldownUntil) {
      var factor = APP_CONFIG.RECOGNITION.ADAPTIVE_COOLDOWN_FACTOR || 0.65;
      return Math.round(base * factor);
    }
    return base;
  };

  // ── Activate adaptive cooldown window ────────────────────
  RecognitionController.prototype._activateAdaptiveCooldown = function() {
    var win = APP_CONFIG.RECOGNITION.ADAPTIVE_COOLDOWN_WINDOW || 3000;
    this._adaptiveCooldownUntil = Date.now() + win;
  };

  // ── Gesture confirmed ─────────────────────────────────────
  RecognitionController.prototype._onGestureConfirmed = function(name, conf, model) {
    var self     = this;
    var isLetter = name.length === 1 && /[A-Z]/.test(name);
    var isDigit  = name.length === 1 && /[0-9]/.test(name);

    if (isLetter || isDigit) {
      this.contextState = 'SPELLING';
      this.sentence.addLetter(name);
      this._lastLetterTime = Date.now();
      var wordSuggs = this.nlp.getWordSuggestionsSync(this.sentence.getSpelling());
      this.sentence.setWordSuggestions(wordSuggs);
      eventBus.emit(Events.SPELLING_UPDATED, {
        spelling: this.sentence.getSpelling(),
        suggestions: wordSuggs,
      });
      this._startAutoAccept();
      // Do NOT speakIfAuto for individual letters — speak only when the word is confirmed
      // (via auto-accept, dwell selection, or acceptWordSuggestion). This prevents
      // "H E L L O" being spoken instead of "hello".
    } else {
      this.contextState = 'RECOGNIZING';
      if (this.sentence.getSpelling()) {
        this.sentence.addWord(this.sentence.getSpelling());
        this.sentence.clearSpelling();
      }
      this.sentence.addWordFromGesture(name.toLowerCase(), name);
      this.nlp.learnWord(name.toLowerCase()); // build personal vocab from confirmed gestures
      this.tts.speakIfAuto(name);
      // NLP debounce — wait before fetching
      this._scheduleNLP();
    }

    this.log.unshift({ gesture: name, conf: conf, time: new Date(), model: model });
    if (this.log.length > 50) this.log.pop();

    // Prediction trail (last N confirmed gestures for UI display)
    var trailSize = APP_CONFIG.RECOGNITION.PRED_TRAIL_SIZE || 5;
    this._predTrail.unshift({ name: name, conf: conf, model: model });
    if (this._predTrail.length > trailSize) this._predTrail.pop();

    eventBus.emit(Events.GESTURE_RECOGNIZED, { gesture: name, conf: conf, model: model });
    eventBus.emit(Events.SENTENCE_UPDATED);
  };

  // ── NLP debounce ─────────────────────────────────────────
  RecognitionController.prototype._scheduleNLP = function() {
    var self = this;
    if (this._nlpTimer) clearTimeout(this._nlpTimer);
    this._nlpTimer = setTimeout(function() {
      self.nlp.getSuggestions(self.sentence, self.sentence.getRecentGestures(5))
        .then(function(suggs) {
          self.sentence.setSuggestions(suggs);
          eventBus.emit(Events.SENTENCE_UPDATED);
        });
      self.nlp.getCompletion(self.sentence)
        .then(function(comp) {
          self.sentence.setCompletion(comp);
          eventBus.emit(Events.SENTENCE_UPDATED);
        });
    }, APP_CONFIG.RECOGNITION.NLP_DEBOUNCE_MS);
  };

  // ── System gestures ───────────────────────────────────────
  RecognitionController.prototype._checkSystemGestures = function(features) {
    var curls   = features.slice(0, 5);
    var detected = null;
    if (curls.every(function(c) { return c > 0.85; }))                              detected = 'speak';
    else if (curls.every(function(c) { return c < 0.15; }))                         detected = 'clear';
    else if (curls[0] < 0.2 && curls.slice(1).every(function(c) { return c > 0.8; })) detected = 'backspace';

    if (detected) {
      if (this._sysGestName === detected) {
        var elapsed = Date.now() - this._sysGestStart;
        var needed  = detected === 'speak' ? 2000 : 1500;
        var prog    = document.getElementById('sysGestProg');
        if (prog) prog.style.width = Math.min(100, elapsed / needed * 100) + '%';
        if (elapsed >= needed) {
          this._executeSysGesture(detected);
          this._sysGestName = null;
          this._sysGestStart = 0;
          return true;
        }
      } else {
        this._sysGestName  = detected;
        this._sysGestStart = Date.now();
      }
    } else {
      this._sysGestName  = null;
      this._sysGestStart = 0;
      var p = document.getElementById('sysGestProg');
      if (p) p.style.width = '0%';
    }
    return false;
  };

  RecognitionController.prototype._executeSysGesture = function(action) {
    if (action === 'speak') {
      // Learn from this sentence before speaking (builds personal NLP corpus)
      this.sentence.acceptAndLearn(this.nlp);
      this.tts.speak(this.sentence.getSentence() || this.sentence.getDisplayText());
      eventBus.emit(Events.SYSTEM_GESTURE, { action: 'speak' });
    } else if (action === 'clear') {
      this.sentence.clear();
      eventBus.emit(Events.SENTENCE_UPDATED);
      eventBus.emit(Events.SYSTEM_GESTURE, { action: 'clear' });
    } else if (action === 'backspace') {
      if (this.sentence.getSpelling()) this.sentence.removeLetter();
      else this.sentence.removeLastWord();
      eventBus.emit(Events.SENTENCE_UPDATED);
      eventBus.emit(Events.SYSTEM_GESTURE, { action: 'backspace' });
    }
  };

  // ── Dwell selection ───────────────────────────────────────
  RecognitionController.prototype._checkDwellSelection = function() {
    var count = this.sensor.countExtendedFingers();
    if (count >= 1 && count <= 5) {
      if (this._dwellFingers === count) {
        if (!this._dwellActive) { this._dwellActive = true; this._dwellStart = Date.now(); }
        if (Date.now() - this._dwellStart >= APP_CONFIG.RECOGNITION.DWELL_TIME) {
          this._selectSuggestion(count - 1);
          this._dwellActive  = false;
          this._dwellFingers = -1;
        }
      } else {
        this._dwellFingers = count;
        this._dwellActive  = true;
        this._dwellStart   = Date.now();
      }
    } else {
      this._dwellActive  = false;
      this._dwellFingers = -1;
    }
  };

  RecognitionController.prototype._selectSuggestion = function(index) {
    this._activateAdaptiveCooldown(); // User interacted via dwell — speed up next recognition
    if (this.sentence.wordSuggestions.length > 0 && index < this.sentence.wordSuggestions.length) {
      var word = this.sentence.wordSuggestions[index];
      this.sentence.acceptWordSuggestion(word);
      this.nlp.learnWord(word);
      this.tts.speakIfAuto(word);
      eventBus.emit(Events.SUGGESTION_SELECTED, { word: word, index: index });
      eventBus.emit(Events.SENTENCE_UPDATED);
    } else if (this.sentence.suggestions.length > 0 && index < this.sentence.suggestions.length) {
      var w = this.sentence.suggestions[index];
      this.sentence.addWord(w);
      this.nlp.learnWord(w);
      this.tts.speakIfAuto(w);
      eventBus.emit(Events.SUGGESTION_SELECTED, { word: w, index: index });
      eventBus.emit(Events.SENTENCE_UPDATED);
    }
  };

  // ── Auto-accept spelling ──────────────────────────────────
  RecognitionController.prototype._startAutoAccept = function() {
    var self = this;
    if (this._autoAcceptTimer) clearTimeout(this._autoAcceptTimer);
    this._autoAcceptTimer = setTimeout(function() {
      if (self.sentence.getSpelling() && self.sentence.wordSuggestions.length > 0) {
        var top = self.sentence.wordSuggestions[0];
        self.sentence.acceptWordSuggestion(top);
        self.nlp.learnWord(top);
        self.tts.speakIfAuto(top);
      } else if (self.sentence.getSpelling()) {
        var spelled = self.sentence.getSpelling();
        self.sentence.addWord(spelled);
        self.nlp.learnWord(spelled);
        self.tts.speakIfAuto(spelled);
        self.sentence.clearSpelling();
      }
      eventBus.emit(Events.SENTENCE_UPDATED);
    }, APP_CONFIG.RECOGNITION.SPELL_PAUSE);
  };

  // ── Public API ────────────────────────────────────────────
  RecognitionController.prototype.addSuggestionWord = function(word) {
    var self = this;
    this._activateAdaptiveCooldown(); // User is actively choosing — speed up next recognition
    this.sentence.addWord(word);
    return this.nlp.getSuggestions(this.sentence).then(function(s) {
      self.sentence.setSuggestions(s);
      return self.nlp.getCompletion(self.sentence);
    }).then(function(c) {
      self.sentence.setCompletion(c);
      eventBus.emit(Events.SENTENCE_UPDATED);
    });
  };

  RecognitionController.prototype.clearSentence = function() {
    var self = this;
    this.sentence.clear();
    return this.nlp.getSuggestions(this.sentence).then(function(s) {
      self.sentence.setSuggestions(s);
      eventBus.emit(Events.SENTENCE_UPDATED);
    });
  };

  RecognitionController.prototype.undoWord = function() {
    var self = this;
    if (this.sentence.getSpelling()) this.sentence.removeLetter();
    else this.sentence.removeLastWord();
    return this.nlp.getSuggestions(this.sentence).then(function(s) {
      self.sentence.setSuggestions(s);
      self.sentence.setCompletion(null);
      eventBus.emit(Events.SENTENCE_UPDATED);
    });
  };

  RecognitionController.prototype.fixGrammar = function() {
    var self = this;
    return this.nlp.correctGrammar(this.sentence).then(function(c) {
      if (c) { self.sentence.replaceWithCorrected(c); eventBus.emit(Events.SENTENCE_UPDATED); return true; }
      return false;
    });
  };

  RecognitionController.prototype.speakSentence = function() {
    this.tts.speak(this.sentence.getDisplayText());
  };

  RecognitionController.prototype.acceptCompletion = function() {
    var self = this;
    if (this.sentence.acceptCompletion()) {
      return this.nlp.getSuggestions(this.sentence).then(function(s) {
        self.sentence.setSuggestions(s);
        eventBus.emit(Events.SENTENCE_UPDATED);
      });
    }
    return Promise.resolve();
  };

  RecognitionController.prototype.quickTest = function(gestureName, gestureModel) {
    var self = this;
    return gestureModel.simulateStaticAsync(gestureName, 1).then(function(samples) {
      var features = (samples && samples[0]) ? samples[0] : [];
      while (features.length < 41) features.push(0);
      self.sensor.setFromFeatures(features);
      self._onGestureConfirmed(gestureName, 0.95, 'demo');
    }).catch(function() {
      self._onGestureConfirmed(gestureName, 0.95, 'demo');
    });
  };

  return RecognitionController;
})();
