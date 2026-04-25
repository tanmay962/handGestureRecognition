// controllers/AppController.js — Gesture Detection v1.0
// Camera only, Holistic, no BLE/VR
import {APP_CONFIG} from '../config/app.config.js';
import {eventBus,Events} from '../utils/EventBus.js';
import {wsClient} from '../utils/WSClient.js';
import {NeuralNetwork} from '../models/NeuralNetwork.js';
import {GestureModel} from '../models/GestureModel.js';
import {SentenceModel} from '../models/SentenceModel.js';
import {SensorModel} from '../models/SensorModel.js';
import {CameraService} from '../services/CameraService.js';
import {GeminiService} from '../services/GeminiService.js';
import {NLPService} from '../services/NLPService.js';
import {TTSService} from '../services/TTSService.js';
import {StorageService} from '../services/StorageService.js';
import {MQTTService} from '../services/MQTTService.js';
import {RecognitionController} from './RecognitionController.js';
import {TrainingController} from './TrainingController.js';
import {AppView} from '../views/AppView.js';

var API = '/api';

export class AppController {
  constructor(root) {
    this.root      = root;
    this.activeTab = 'detect';
    this.appMode   = 'user';
    this._dbReady  = false;

    this.staticNN     = new NeuralNetwork('static');
    this.dynamicNN    = new NeuralNetwork('dynamic');
    this.gestureModel = new GestureModel(APP_CONFIG.DEFAULT_GESTURES);
    this.sentenceModel= new SentenceModel();
    this.sensorModel  = new SensorModel();

    var cal = StorageService.loadCalibration();
    if (cal) this.sensorModel.calibration = cal;

    this.camera = new CameraService();
    this.mqttService = new MQTTService();
    this.gemini = new GeminiService();
    this.nlp    = new NLPService(this.gemini);
    this.tts    = new TTSService();

    this.recogCtrl = new RecognitionController(this.staticNN, this.dynamicNN,
                       this.sensorModel, this.sentenceModel, this.nlp, this.tts);
    this.trainCtrl = new TrainingController(this.staticNN, this.dynamicNN,
                       this.gestureModel, this.sensorModel);
    this.view      = new AppView(root, this);
    // Wire camera reference to TrainingController for mirror augmentation
    this.trainCtrl.setCamera(this.camera);

    var key = StorageService.loadApiKey();
    if (key) this.gemini.setApiKey(key);

    this.simFlex = [0,0,0,0,0];
    this.simIMU  = {ax:0,ay:0,az:0,gx:0,gy:0,gz:0};
    this._cameraStarting = false;

    // Global state consumed by TrainView (no optional chaining needed)
    window._trainMeta      = {};
    window._readiness      = {};
    window._histStats      = {};
    window._confThresh     = 0.65;
    window._perGestAcc     = {static:{}, dynamic:{}};
    window._trainSection   = 'alphabet';
    window._guidedGesture  = null;
    window._livePrediction = null;
    window._countdown      = 0;
    window._customGestures = [];
    window._nlpStats       = {};
    window._trainSource    = 'camera'; // 'camera' | 'glove'
    this._cameraError      = null;

    var self = this;
    var rr = function() { self.view.render(); };
    eventBus.on(Events.GESTURE_RECOGNIZED, function(d) {
      if (self.mqttService.enabled) {
        self.mqttService.publishGesture(d.gesture, d.conf, d.model);
      }
      rr();
    });
    // Live prediction → camera canvas overlay + top-3 histogram DOM update
    eventBus.on(Events.GESTURE_PREDICTING, function(d) {
      self.camera.setPrediction(d.gesture || null, d.conf || 0, d.model || '', d.auxGesture || null, d.auxConf || 0);
      // Update top-3 confidence histogram directly in DOM (no full re-render needed)
      var top3 = self.recogCtrl._topPredictions || [];
      for (var i = 0; i < 3; i++) {
        var nameEl = document.getElementById('top3n' + i);
        var barEl  = document.getElementById('top3b' + i);
        var pctEl  = document.getElementById('top3p' + i);
        var entry  = top3[i];
        if (nameEl) nameEl.textContent = entry ? entry.name : '';
        if (barEl)  barEl.style.width  = entry ? Math.round(entry.conf * 100) + '%' : '0%';
        if (pctEl)  pctEl.textContent  = entry ? Math.round(entry.conf * 100) + '%' : '';
      }
    });
    // Clear overlay when recognition stops; show status hint
    eventBus.on(Events.RECOG_STOPPED, function() {
      self.camera.setPrediction(null, 0, '');
      var trained = self.staticNN.trained || self.dynamicNN.trained;
      self.camera.setStatus(trained ? 'TAP  ▶  TO RECOGNIZE' : 'TRAIN MODEL FIRST');
      rr();
    });
    // Clear status when recognition starts
    eventBus.on(Events.RECOG_STARTED, function() {
      self.camera.setStatus('DETECTING...');
    });
    eventBus.on(Events.TRAIN_PROGRESS,     rr);
    eventBus.on(Events.TRAIN_COMPLETE,     rr);
    eventBus.on(Events.SENTENCE_UPDATED,   rr);
    eventBus.on(Events.SYSTEM_GESTURE,     rr);
    eventBus.on(Events.SPELLING_UPDATED,   rr);
    eventBus.on(Events.RECORDING_DONE,     rr);
    eventBus.on(Events.STATE_UPDATED,      rr);
    eventBus.on(Events.SAMPLES_COLLECTED,  function() { self._refreshTrainMeta(); rr(); });

    // Update finger curl bars for BOTH hands — dom (feat 0-4) + aux (feat 11-15)
    // UI bars: indices 0-4 = dominant hand, 5-9 = auxiliary hand
    eventBus.on(Events.FEATURES_EXTRACTED, function(d) {
      var f = d.features;
      if (!f || f.length < 16) return;
      var srcIdx = [0,1,2,3,4, 11,12,13,14,15]; // dom curls then aux curls
      for (var i = 0; i < 10; i++) {
        var pct = Math.round(f[srcIdx[i]] * 100);
        // Detect tab — vertical bars
        var fb = document.getElementById('fb' + i);
        var fv = document.getElementById('fv' + i);
        if (fb) fb.style.height = pct + '%';
        if (fv) fv.textContent  = pct + '%';
        // Train tab — horizontal bars
        var tb = document.getElementById('trainFb' + i);
        var tv = document.getElementById('trainFv' + i);
        if (tb) tb.style.width  = pct + '%';
        if (tv) tv.textContent  = pct + '%';
      }
    });

    // Update recording progress bar directly in DOM on each frame tick
    eventBus.on(Events.RECORDING_TICK, function(d) {
      var bar   = document.getElementById('recProgBar');
      var label = document.getElementById('recProgLabel');
      if (!d) return;
      var pct = Math.round((d.frame / d.total) * 100);
      if (bar)   bar.style.width = pct + '%';
      if (label) label.textContent = d.frame + ' / ' + d.total + ' frames';
    });

    // Show / hide recording trail on canvas
    eventBus.on(Events.RECORDING_START, function() {
      self.camera._recordingTrail = [];
      self.camera._isRecording    = true;
    });
    eventBus.on(Events.RECORDING_DONE, function() {
      self.camera._isRecording    = false;
      self.camera._recordingTrail = [];
    });
  }

  // ── Boot ──────────────────────────────────────────────────────────────────
  async start() {
    this.view.render();

    try { await this.gestureModel.loadFromDB(); } catch(e) {}

    // Load unified model (single BiLSTM for static + dynamic)
    try {
      var src = window._trainSource || 'camera';
      var ud = await (await fetch(API + '/nn/load/unified?source=' + src, {method:'POST'})).json();
      if (ud.ok) {
        this.staticNN.trained       = true;
        this.staticNN.accuracy      = ud.accuracy     || 0;
        this.staticNN.val_accuracy  = ud.val_accuracy || 0;
        this.staticNN.loss          = ud.loss         != null ? ud.loss : 1;
        this.staticNN.epochs        = ud.epochs       || 0;
        this.dynamicNN.trained      = true;
        this.dynamicNN.accuracy     = ud.accuracy     || 0;
        this.dynamicNN.val_accuracy = ud.val_accuracy || 0;
        this.dynamicNN.loss         = ud.loss         != null ? ud.loss : 1;
        this.dynamicNN.epochs       = ud.epochs       || 0;
      }
    } catch(e) {}

    // Check whether backend has a Gemini key available (env var or user-stored in DB).
    // The key itself never comes to the browser — all Gemini calls are proxied.
    await this.gemini.loadServerKey();

    // Load conf threshold
    try {
      var ct = await (await fetch(API + '/settings/conf_threshold')).json();
      if (ct.threshold !== undefined && ct.threshold !== null) window._confThresh = ct.threshold;
    } catch(e) {}

    // Load per-gesture accuracy
    try {
      var pa = await (await fetch(API + '/nn/per_gesture_accuracy')).json();
      window._perGestAcc = pa;
    } catch(e) {}

    this._dbReady = true;

    await this._registerDefaultGestures();
    await this._loadCustomGestures();
    await this._refreshTrainMeta();
    this.view.render();

    // WebSocket handlers
    var self = this;
    wsClient.on('train_progress', function(d) {
      // Update live progress bar via trainCtrl so TrainView reflects it
      if (d.progress_pct !== undefined && self.trainCtrl.isTraining) {
        self.trainCtrl.progress = d.progress_pct;
      }
      // Final completion event carries accuracy/loss/epochs
      if (d.complete) {
        if (d.model === 'static' || d.model === 'unified') {
          self.staticNN.accuracy     = d.accuracy;
          self.staticNN.val_accuracy = d.val_accuracy || 0;
          self.staticNN.loss         = d.loss;
          self.staticNN.epochs       = d.epochs;
          self.staticNN.trained      = true;
          if (d.per_gesture_acc) { if (!window._perGestAcc) window._perGestAcc = {}; window._perGestAcc.static = d.per_gesture_acc; }
        }
        if (d.model === 'dynamic' || d.model === 'unified') {
          self.dynamicNN.accuracy     = d.accuracy;
          self.dynamicNN.val_accuracy = d.val_accuracy || 0;
          self.dynamicNN.loss         = d.loss;
          self.dynamicNN.epochs       = d.epochs;
          self.dynamicNN.trained      = true;
          if (d.per_gesture_acc) { if (!window._perGestAcc) window._perGestAcc = {}; window._perGestAcc.dynamic = d.per_gesture_acc; }
        }
      }
      eventBus.emit(Events.TRAIN_PROGRESS, d);
    });
    wsClient.on('model_saved',   function() { self.view.render(); });
    wsClient.on('model_reset',   function() { self.view.render(); });
    wsClient.on('samples_cleared', function() { self._refreshTrainMeta(); self.view.render(); });
    wsClient.on('gesture_samples_deleted', function() { self._refreshTrainMeta(); self.view.render(); });
    wsClient.on('gesture_deleted', function() { self._loadCustomGestures(); self._refreshTrainMeta(); self.view.render(); });
    wsClient.on('sentence_updated', function(d) { if (d.words) self.sentenceModel.words = d.words; self.view.render(); });

    this._startLivePredictionLoop();
    await this._autoCamera();
    console.log('[SignLens v4.5] Ready — Phase 2 (two-hand + adaptive NLP)');
  }

  // ── Internal helpers ──────────────────────────────────────────────────────
  async _registerDefaultGestures() {
    var DYN = {J:1,Z:1};
    var batch = [];
    var i, ch, d;
    var alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    for (i = 0; i < alpha.length; i++) {
      ch = alpha[i];
      batch.push({name:ch, gesture_type: DYN[ch] ? 'dynamic' : 'static', category:'alphabet'});
    }
    for (d = 0; d <= 9; d++) {
      batch.push({name:String(d), gesture_type:'static', category:'number'});
    }
    for (i = 0; i < batch.length; i++) {
      fetch(API + '/gestures/register', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(batch[i])}).catch(function(){});
    }
  }

  async _loadCustomGestures() {
    try {
      var rows = await (await fetch(API + '/gestures')).json();
      var out = [];
      for (var i = 0; i < rows.length; i++) {
        if (rows[i].category === 'custom') out.push(rows[i].name);
      }
      window._customGestures = out;
    } catch(e) {}
  }

  async _refreshTrainMeta() {
    var self = this;
    // Each request is independent — one failure never blocks the others
    var metaP = fetch(API + '/samples/meta').then(function(r){ return r.json(); }).catch(function(){ return null; });
    var readP = fetch(API + '/gestures/readiness?source=' + (window._trainSource||'camera')).then(function(r){ return r.json(); }).catch(function(){ return null; });
    var histP = fetch(API + '/history/stats').then(function(r){ return r.json(); }).catch(function(){ return null; });
    var results = await Promise.all([metaP, readP, histP]);
    if (results[0]) window._trainMeta = results[0];
    if (results[1]) window._readiness = results[1];
    if (results[2]) window._histStats = results[2];
    self._refreshNlpStats();
  }

  async _refreshNlpStats() {
    try {
      var data = await fetch(API + '/nlp/stats').then(function(r){ return r.json(); });
      window._nlpStats = data;
    } catch(e) {}
  }

  _startLivePredictionLoop() {
    var self = this;
    setInterval(function() {
      if (self.activeTab !== 'train') return;
      if (!self.sensorModel.handDetected) { window._livePrediction = null; return; }
      if (!self.staticNN.trained && !self.dynamicNN.trained) return;
      var features = self.camera.lastFeatures || self.sensorModel.getFeatureVector();
      fetch(API + '/nn/predict', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({features:features, model_type:'static'})
      }).then(function(r){ return r.json(); }).then(function(data) {
        if (data.conf > 0.4) {
          window._livePrediction = {name:data.name, conf:data.conf, model:'static'};
          self.view.render();
        } else {
          window._livePrediction = null;
        }
      }).catch(function(){});
    }, 900);
  }

  // ── Navigation ────────────────────────────────────────────────────────────
  switchTab(id) {
    this.activeTab = id;
    this.view.render();
    var self = this;
    if (id === 'detect' || id === 'train') this._autoCamera();
    if (id === 'train') this._refreshTrainMeta();
  }
  switchMode(mode) { this.appMode = mode; this.view.render(); this._autoCamera(); }
  checkAdminPin(pin) { return pin === StorageService.loadPin(); }
  setTrainSection(s) { window._trainSection = s; this.view.render(); }
  setTrainSource(s)  { window._trainSource  = s; this._refreshTrainMeta(); this.view.render(); }

  connectMQTT() {
    var self = this;
    this.mqttService.connect(function() { self.view.render(); });
    this.view.render();
  }
  disconnectMQTT() { this.mqttService.disconnect(); this.view.render(); }


  async _autoCamera() {
    // Guard: skip if already starting (prevents concurrent starts causing play() abort)
    if (!this.camera.active && !this._cameraStarting) {
      if (typeof Holistic === 'undefined') {
        this._cameraError = 'MediaPipe failed to load from CDN. Check your internet connection and reload.';
        this.view.render();
        return;
      }
      this._cameraStarting = true;
      try {
        this._cameraError = null;
        await this.camera.start();
        var trained = this.staticNN.trained || this.dynamicNN.trained;
        if (trained && !this.recogCtrl.running) {
          this.recogCtrl.start();
          // RECOG_STARTED event will set status to 'DETECTING...'
        } else {
          this.camera.setStatus(trained ? 'TAP  ▶  TO RECOGNIZE' : 'TRAIN MODEL FIRST');
        }
        this.view.render();
      } catch(e) {
        console.warn('[App] Camera:', e);
        if (e && e.name === 'NotAllowedError') {
          this._cameraError = 'Camera permission denied. Allow camera access in your browser settings and reload.';
        } else if (e && e.name === 'NotFoundError') {
          this._cameraError = 'No camera found. Connect a webcam and try again.';
        } else {
          this._cameraError = 'Camera error: ' + (e && e.message ? e.message : String(e));
        }
        this.view.render();
      } finally {
        this._cameraStarting = false;
      }
    }
    this._mountCamera();
  }
  async startCamera() { await this._autoCamera(); this.trainCtrl.setCamera(this.camera); }
  stopCamera()       { this.camera.stop(); this.view.render(); }
  async switchCamera() { await this.camera.switchCamera(); this.view.render(); }
  _mountCamera() {
    if (!this.camera.active) return;
    var main  = document.getElementById('vidContainer');
    var train = document.getElementById('trainVidContainer');
    if (main)  this.camera.mountInto(main);
    else if (train) this.camera.mountInto(train);
    // Ensure status text matches current state
    if (!this.camera._statusText && !this.camera._currentPrediction) {
      if (this.recogCtrl.running) {
        this.camera.setStatus('DETECTING...');
      } else {
        var trained = this.staticNN.trained || this.dynamicNN.trained;
        this.camera.setStatus(trained ? 'TAP  ▶  TO RECOGNIZE' : 'TRAIN MODEL FIRST');
      }
    }
  }

  // ── Recognition ───────────────────────────────────────────────────────────
  startRecognition() { this.recogCtrl.start();  this.view.render(); }
  stopRecognition()  { this.recogCtrl.stop();   this.view.render(); }
  async quickTest(g) { await this.recogCtrl.quickTest(g, this.gestureModel); this.view.render(); }

  // ── Training ──────────────────────────────────────────────────────────────
  async trainModel() {
    if (this.trainCtrl.isTraining) return;
    // Capture source NOW before training starts — user may switch toggle mid-training (BUG #18 fix)
    var trainedSource = window._trainSource || 'camera';
    await this.trainCtrl.train();
    // Re-sync samples from DB using the source that was active when training began
    await this.gestureModel.loadFromDB(trainedSource);
    await this._refreshTrainMeta();
    this.view.render();
  }

  async deleteModel() {
    if (!confirm('Delete trained model? You will need to retrain.')) return;
    await this.trainCtrl.deleteModel();
    this.view.render();
  }

  async resetModel() {
    if (!confirm('Delete ALL samples and model? This cannot be undone.')) return;
    await this.trainCtrl.resetAll();
    await this._refreshTrainMeta();
    this.view.render();
  }

  async setConfThreshold(val) {
    window._confThresh = val;
    fetch(API + '/settings/conf_threshold', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({threshold: val})
    }).catch(function(){});
    // Also update recognitionController
    if (this.recogCtrl) this.recogCtrl.confThresh = val;
  }

  // ── Sample capture ────────────────────────────────────────────────────────
  async addSample(gestureName, type) {
    var self = this;
    var statusEl = document.getElementById('trainStatus');
    function showStatus(msg, color, dur) {
      if (!statusEl) return;
      statusEl.style.display    = 'block';
      statusEl.textContent      = msg;
      statusEl.style.background = color;
      statusEl.style.color      = '#000000';
      statusEl.style.fontWeight = '700';
      clearTimeout(statusEl._t);
      statusEl._t = setTimeout(function(){ statusEl.style.display='none'; }, dur || 2800);
    }

    if (type === 'static') {
      showStatus('Get ready — capturing "' + gestureName + '" in 3s…', '#ffffff', 3500);
      var result = await this.trainCtrl.collectStaticSample(gestureName, false);
      if (result.error) {
        showStatus('Error: ' + result.error, '#fb7185', 3500);
      } else {
        // Refresh meta to get DB-confirmed count, then show it in the toast
        await this._refreshTrainMeta();
        var meta = window._trainMeta || {};
        var dbCount = (meta[gestureName] && meta[gestureName].static) || '?';
        showStatus(
          result.live
            ? 'Saved "' + gestureName + '" — ' + dbCount + ' samples in DB'
            : 'Simulated "' + gestureName + '" (no hand) — ' + dbCount + ' in DB',
          result.live ? '#4ade80' : '#fbbf24',
          2800
        );
      }
      window._lastSampledGesture = gestureName;
      this.view.render();
      setTimeout(function(){ window._lastSampledGesture = null; self.view.render(); }, 1600);

    } else if (type === 'dynamic') {
      if (!this.camera.active) {
        showStatus('Start camera first for dynamic recording!', '#fb7185', 3000);
        return;
      }
      showStatus('Get ready — recording "' + gestureName + '" in 3s…', '#ffffff', 3500);
      await this.trainCtrl.runCountdown(3);
      showStatus('Recording "' + gestureName + '"… perform gesture NOW!', '#c084fc', 6000);
      this.trainCtrl.startDynamicRecording(gestureName);
      this.view.render();

      var unsub = eventBus.on(Events.FEATURES_EXTRACTED, function(data) {
        self.trainCtrl.pushRecordingFrame(data.features, data.handCount || 1)
          .then(function(result) {
            if (result === 'aborted') {
              unsub();
              showStatus('Hand lost during recording — please try again', '#fb7185', 3000);
              window._lastSampledGesture = null;
              self.view.render();
              return;
            }
            if (result === 'done') {
              unsub();
              self._refreshTrainMeta().then(function() {
                var meta2 = window._trainMeta || {};
                var dbCount2 = (meta2[gestureName] && meta2[gestureName].dynamic) || '?';
                showStatus('Saved "' + gestureName + '" dynamic — ' + dbCount2 + ' in DB', '#4ade80', 2800);
                window._lastSampledGesture = gestureName;
                self.view.render();
                setTimeout(function(){ window._lastSampledGesture = null; self.view.render(); }, 1600);
              });
            }
          })
          .catch(function(e) {
            unsub();
            console.error('[AppController] dynamic save failed:', e);
            showStatus('Failed to save gesture — please try again', '#fb7185', 3000);
            window._lastSampledGesture = null;
            self.view.render();
          });
      });
    }
  }

  async addBulkSamples(name, count) {
    var statusEl = document.getElementById('trainStatus');
    function showStatus(msg, color, dur) {
      if (!statusEl) return;
      statusEl.style.display    = 'block';
      statusEl.textContent      = msg;
      statusEl.style.background = color;
      statusEl.style.color      = '#000000';
      statusEl.style.fontWeight = '700';
      clearTimeout(statusEl._t);
      statusEl._t = setTimeout(function(){ statusEl.style.display='none'; }, dur || 3000);
    }
    showStatus('Get ready — capturing ' + count + 'x "' + name + '" in 3s…', '#ffffff', 4000);
    await this.trainCtrl.collectBulkStatic(name, count);
    await this._refreshTrainMeta();
    var meta = window._trainMeta || {};
    var dbCount = (meta[name] && meta[name].static) || '?';
    showStatus('Saved ' + count + 'x "' + name + '" — ' + dbCount + ' total in DB', '#4ade80', 3000);
    this.view.render();
  }

  // ── Gesture management ────────────────────────────────────────────────────
  addGesture(name)    { this.trainCtrl.addGesture(name);    this.view.render(); }
  removeGesture(name) { this.trainCtrl.removeGesture(name); this.view.render(); }

  async addCustomGesture(name, gestureType) {
    var t = name ? name.trim() : '';
    if (!t) return;
    gestureType = gestureType || 'static';
    this.gestureModel.addGesture(t);
    try {
      await fetch(API + '/gestures/register', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name:t, gesture_type:gestureType, category:'custom'})
      });
    } catch(e) {}
    await this._loadCustomGestures();
    await this._refreshTrainMeta();
    this.view.render();
  }

  async deleteGesture(name) {
    if (!confirm('Delete gesture "' + name + '" and all its samples?')) return;
    try { await fetch(API + '/gestures/' + encodeURIComponent(name), {method:'DELETE'}); } catch(e) {}
    this.gestureModel.removeGesture(name);
    await this._loadCustomGestures();
    await this._refreshTrainMeta();
    this.view.render();
  }

  async deleteGestureSamples(gesture, sampleType) {
    sampleType = sampleType || 'all';
    if (!confirm('Delete ' + sampleType + ' samples for "' + gesture + '"?')) return;
    try {
      await fetch(API + '/samples/gesture', {
        method:'DELETE', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({gesture:gesture, sample_type:sampleType, source: window._trainSource||'all'})
      });
    } catch(e) {}
    if (sampleType === 'static' || sampleType === 'all') {
      delete this.gestureModel.staticSamples[gesture];
      if (this.gestureModel.sampleCounts[gesture]) this.gestureModel.sampleCounts[gesture].static = 0;
    }
    if (sampleType === 'dynamic' || sampleType === 'all') {
      delete this.gestureModel.dynamicSamples[gesture];
      if (this.gestureModel.sampleCounts[gesture]) this.gestureModel.sampleCounts[gesture].dynamic = 0;
    }
    await this._refreshTrainMeta();
    this.view.render();
  }

  // ── Guided capture mode ───────────────────────────────────────────────────
  async startGuidedMode(section) {
    // Check if two-hand model — warn user
    if (this.camera && this.camera.handCount !== undefined) {
      var statusEl = document.getElementById('trainStatus');
      if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.textContent = '✋✋ Position both hands in frame if your gestures use two hands. Guided mode starting…';
        statusEl.style.background = 'var(--p)';
        statusEl.style.color = 'var(--bg)';
      }
      await new Promise(function(r){ setTimeout(r, 2000); });
    }
    section = section || 'alphabet';
    var list = section === 'alphabet'
      ? 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
      : '0123456789'.split('');
    var DYN = {J:1, Z:1};
    window._trainSection = section;
    var self = this;
    for (var i = 0; i < list.length; i++) {
      var name = list[i];
      window._guidedGesture = name;
      self.view.render();
      await new Promise(function(r){ setTimeout(r, 400); });
      if (!DYN[name]) {
        await self.addBulkSamples(name, 5);
      }
      await new Promise(function(r){ setTimeout(r, 200); });
    }
    window._guidedGesture = null;
    this.view.render();
  }

  // ── Session history ───────────────────────────────────────────────────────
  async loadSessionHistory() {
    var panel = document.getElementById('historyPanel');
    if (!panel) return;
    if (panel.style.display === 'block') { panel.style.display = 'none'; return; }
    try {
      var rows = await (await fetch(API + '/history?limit=30')).json();
      panel.style.display = 'block';
      if (!rows.length) { panel.innerHTML = '<div style="font-size:10px;color:var(--dm)">No history yet</div>'; return; }
      var out = '';
      for (var i = 0; i < rows.length; i++) {
        var r = rows[i];
        out += '<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--brd);font-size:9px;gap:8px">' +
          '<span style="color:var(--mx)">' + r.event_type + '</span>' +
          '<span style="color:var(--tx);flex:1;text-align:center">' + (r.gesture||'') + ' ' + (r.detail||'') + '</span>' +
          '<span style="color:var(--dm);white-space:nowrap">' + new Date(r.created_at*1000).toLocaleTimeString() + '</span>' +
        '</div>';
      }
      panel.innerHTML = out;
    } catch(e) { console.warn('[History]', e); }
  }

  // ── Export / Import ───────────────────────────────────────────────────────
  async viewSampleDetails(gestureName) {
    var preview = await fetch(API + '/samples/preview/' + encodeURIComponent(gestureName))
      .then(function(r){ return r.json(); }).catch(function(){ return []; });
    window._samplePreview = preview;
    window._samplePreviewGesture = gestureName;
    this.view.render();
  }

  // ── NLP ───────────────────────────────────────────────────────────────────
  async addSuggestionWord(w) { await this.recogCtrl.addSuggestionWord(w); this.view.render(); }
  async clearSentence()      { await this.recogCtrl.clearSentence();      this.view.render(); }
  async undoWord()           { await this.recogCtrl.undoWord();           this.view.render(); }
  async acceptCompletion()   { await this.recogCtrl.acceptCompletion();   this.view.render(); }
  async fixGrammar() {
    // Try AI grammar first, then offline fallback
    var done = await this.recogCtrl.fixGrammar();
    if (!done && this.sentenceModel.getWordCount() >= 2) {
      // Offline fallback via backend
      try {
        var res = await fetch(API + '/nlp/grammar', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ sentence: this.sentenceModel.getSentence() })
        });
        var data = await res.json();
        if (data.corrected) {
          this.sentenceModel.replaceWithCorrected(data.corrected);
        }
      } catch(e) {}
    }
    this.view.render();
  }
  speakSentence() {
    // Phase 2B: learn from this sentence before speaking
    this.sentenceModel.acceptAndLearn(this.nlp);
    this.recogCtrl.speakSentence();
  }




  // ── Settings ──────────────────────────────────────────────────────────────
  setTTSEnabled(v)    { this.tts.enabled   = v; this.view.render(); }
  setAutoSpeak(v)     { this.tts.autoSpeak = v; this.view.render(); }
  setTTSRate(v)       { this.tts.setRate(v);    this.view.render(); }
  setSeqMode(v)       { this.view.render(); }
  setInputMode(m)     { this.recogCtrl.inputMode = m; this.view.render(); }
  setAdminPin(p)      { StorageService.savePin(p); }

  exportDataset() {
    fetch(API + '/export/json')
      .then(function(r) { return r.blob(); })
      .then(function(blob) {
        var url = URL.createObjectURL(blob);
        var a   = document.createElement('a');
        var ts  = new Date().toISOString().slice(0, 10);
        a.href     = url;
        a.download = 'signlens_backup_' + ts + '.json';
        a.click();
        URL.revokeObjectURL(url);
      })
      .catch(function(err) { alert('Export failed: ' + err.message); });
  }

  importDataset(file) {
    if (!file) return;
    var self = this;
    var reader = new FileReader();
    reader.onload = function(e) {
      var body;
      try { body = JSON.parse(e.target.result); }
      catch(ex) { alert('Invalid JSON file'); return; }
      fetch(API + '/import/json', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      .then(function(r) { return r.json(); })
      .then(function(d) {
        if (d.error) { alert('Import failed: ' + d.error); return; }
        var imp = d.imported || {};
        alert('Imported: ' + (imp.static || 0) + ' static samples, ' + (imp.dynamic || 0) + ' dynamic samples, ' + (imp.gestures || 0) + ' gestures');
        self.view.render();
      })
      .catch(function(err) { alert('Import failed: ' + err.message); });
    };
    reader.readAsText(file);
  }

  async collectSamples(name) {
    if (this.sensorModel.handDetected) {
      await this.trainCtrl.collectLiveBurst(name, 5);
    } else {
      await this.trainCtrl.collectStaticSamples(name, 5);
    }
    this.view.render();
  }
  startRecording(name) { this.trainCtrl.startDynamicRecording(name); this.view.render(); }

  // ── State for views ───────────────────────────────────────────────────────
  getState() {
    return {
      tab:             this.activeTab,
      mode:            this.appMode,
      camActive:       this.camera.active,
      cameraError:     this._cameraError || null,
      fps:             this.camera.fps,
      geminiEnabled:   this.gemini.enabled,
      staticTrained:   this.staticNN.trained,
      dynamicTrained:  this.dynamicNN.trained,
      running:         this.recogCtrl.running,
      inputMode:       this.recogCtrl.inputMode,
      sentence:        this.sentenceModel.getSentence(),
      displayText:     this.sentenceModel.getDisplayText(),
      spelling:        this.sentenceModel.getSpelling(),
      suggestions:     this.sentenceModel.suggestions,
      wordSuggestions: this.sentenceModel.wordSuggestions,
      completion:      this.sentenceModel.completion,
      log:             this.recogCtrl.log,
      gestures:        this.gestureModel.gestures,
      sampleCounts:    this.gestureModel.sampleCounts,
      trainStats:      this.trainCtrl.getStats(),
      combos:          [],
      sensor:          this.sensorModel,
      tts:             {enabled:this.tts.enabled, auto:this.tts.autoSpeak, rate:this.tts.rate},
      confThresh:      this.recogCtrl.confThresh,
      apiKey:          '',  // key lives server-side only — never sent to browser
      apiStatus:       this.gemini.enabled ? 'ok' : 'off',
      mqttConnected:   this.mqttService.connected,
      mqttEnabled:     this.mqttService.enabled,
      mqttBroker:      this.mqttService.broker,
      mqttTopic:       this.mqttService.topicGesture,
      contextState:    this.recogCtrl.contextState,
      recording:       this.gestureModel.recording,
      predTrail:       this.recogCtrl._predTrail       || [],
      topPredictions:  this.recogCtrl._topPredictions  || [],
    };
  }
}
