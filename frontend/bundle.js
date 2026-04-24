// Gesture Detection v1.0 — Production Bundle
// Built: 2026-04-24T16:29:50.677Z
// MediaPipe Holistic: hands + face + body = 41 features
// MLP static + LSTM dynamic + adaptive NLP + Gemini + PWA
// Optimised: rate limiting, confidence smoothing, time-based stability
'use strict';
var API = '/api';


// ═══ config/app.config.js ═══
// config/app.config.js — Gesture Detection v1.0
// MediaPipe Holistic: hands + face + body = 41 features
// Optimised: rate limiting, confidence smoothing, time-based stability, separate cooldowns
var APP_CONFIG = {
  name: 'Gesture Detection',
  version: '1.0',
  DEFAULT_GESTURES: ['Hello','Thank You','Yes','No','Help','Please','Sorry','Stop','Go','Water'],

  NN: {
    STATIC_HIDDEN:       [256, 128, 64],
    DYNAMIC_HIDDEN:      [256, 128, 64],
    LEARNING_RATE:       0.008,
    TRAINING_EPOCHS:     200,
    SAMPLES_PER_GESTURE: 35,
    STATIC_INPUT:        41,
    DYNAMIC_INPUT:       1845,
    DYNAMIC_FRAMES:      45,
    FEATURE_VERSION:     '1.0',
  },

  RECOGNITION: {
    CONFIDENCE_THRESHOLD:  0.60,
    // Stability hold — tuned for responsive demo
    STABLE_MS_LETTER:      500,
    STABLE_MS_WORD:        750,
    STABLE_MS_NUMBER:      500,
    // Cooldowns — prevent duplicate confirmations
    COOLDOWN_SAME_LETTER:  1200,
    COOLDOWN_DIFF_LETTER:  350,
    COOLDOWN_WORD:         1500,
    // Prediction rate limiting — every 2 frames for faster response
    PREDICT_EVERY_N:       2,
    // Confidence smoothing — 5 frames balances speed + stability
    CONF_SMOOTH_WINDOW:    5,
    // Motion detection threshold for LSTM activation
    MOTION_THRESHOLD:      0.06,
    DYNAMIC_CONF_THRESH:   0.70,
    DWELL_TIME:            1500,
    SPELL_PAUSE:           1800,
    // NLP debounce
    NLP_DEBOUNCE_MS:       300,
    // Ensemble voting
    ENSEMBLE_WINDOW:       5,
    // TTS word-queue buffer — words signed within this window are spoken as one phrase
    TTS_BUFFER_MS:         2000,
    // Hysteresis — keep active gesture alive during brief dips (raised from 0.40 to reduce false drops)
    HYSTERESIS_EXIT:          0.50,
    // Streak gate — require 3 consecutive same-gesture frames before starting hold timer
    MIN_STREAK_FRAMES:        3,
    // Rejection zone — discard anything below this confidence outright
    MIN_REJECT_CONF:          0.15,
    // Dynamic priority margin
    DYNAMIC_PRIORITY_MARGIN:  0.05,
    // Prediction trail
    PRED_TRAIL_SIZE:          5,
    // Adaptive cooldown
    ADAPTIVE_COOLDOWN_FACTOR: 0.60,
    ADAPTIVE_COOLDOWN_WINDOW: 3000,
  },

  MEDIAPIPE: {
    CDN_BASE:                'https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629',
    MODEL_COMPLEXITY:        1,
    MIN_DETECTION_CONFIDENCE:0.6,
    MIN_TRACKING_CONFIDENCE: 0.5,
    MIRROR_DISPLAY:          false,
    DOMINANT_HAND:           'Right',
    HAND_LOSS_ABORT_FRAMES:  3,
    SMOOTH_LANDMARKS:        true,
  },

  MQTT: {
    DEFAULT_BROKER:  'wss://broker.hivemq.com:8884/mqtt',
    TOPIC_FEATURES:  'gesture-detection/sensor/features',
    TOPIC_RESULTS:   'gesture-detection/results/gesture',
    TOPIC_HEARTBEAT: 'gesture-detection/sensor/heartbeat',
    TOPIC_PROGRESS:  'gesture-detection/training/progress',
    TOPIC_STATUS:    'gesture-detection/status',
    RECONNECT_MS:    5000,
  },

  GEMINI: {
    ENDPOINT:           'https://generativelanguage.googleapis.com/v1beta/models',
    MODEL:              'gemini-2.0-flash',
    SUGGESTION_TEMP:    0.4,
    COMPLETION_TEMP:    0.3,
    GRAMMAR_TEMP:       0.1,
    MAX_TOKENS_SUGGEST: 50,
    MAX_TOKENS_COMPLETE:30,
    MAX_TOKENS_GRAMMAR: 50,
  },

  NLP: {
    PERSONAL_CORPUS_MIN:    5,
    CONTEXT_GESTURE_WINDOW: 5,
    RETRAIN_AFTER_SENTENCES:3,
  },

  // Tabs — Sequences removed (low usage)
  TABS_ADMIN: [
    { id:'detect',   label:'Detect'   },
    { id:'train',    label:'Train'    },
    { id:'settings', label:'Settings' },
  ],

  // Both hands: DOM hand fingers first (0–4), then AUX hand (5–9) → feature indices 0–4, 11–15
  FINGER_NAMES:  ['T','I','M','R','P', 'T₂','I₂','M₂','R₂','P₂'],
  FINGER_COLORS: ['#ffffff','#ffffff','#ffffff','#ffffff','#ffffff',
                  'rgba(255,255,255,0.55)','rgba(255,255,255,0.55)','rgba(255,255,255,0.55)','rgba(255,255,255,0.55)','rgba(255,255,255,0.55)'],
  ADMIN_PIN: '1234',
};

var GESTURE_TEMPLATES = {
  'Hello':    {curls:[.1,.1,.1,.1,.1],ori:[0,-.8,.1,.02,-.01,.01],type:'dynamic'},
  'Thank You':{curls:[.2,.3,.7,.7,.5],ori:[.1,-.6,.3,.01,.02,-.01],type:'dynamic'},
  'Yes':      {curls:[.4,.8,.8,.8,.7],ori:[0,-.9,0,0,.05,0],type:'dynamic'},
  'No':       {curls:[.3,.2,.8,.8,.7],ori:[0,-.3,.1,.05,0,0],type:'dynamic'},
  'Help':     {curls:[.1,.1,.1,.1,.8],ori:[-.3,-.5,.5,.01,-.02,.02],type:'dynamic'},
  'Please':   {curls:[.6,.2,.2,.2,.6],ori:[.1,-.5,-.2,-.01,.01,.03],type:'dynamic'},
  'Sorry':    {curls:[.5,.5,.5,.5,.5],ori:[0,-.3,.6,-.01,-.02,.01],type:'dynamic'},
  'Stop':     {curls:[0,0,0,0,0],ori:[0,-1,0,0,0,0],type:'static'},
  'Go':       {curls:[.7,.1,.8,.8,.8],ori:[.3,-.4,.1,.02,-.03,.01],type:'dynamic'},
  'Water':    {curls:[.3,.7,.7,.7,.7],ori:[.1,-.5,-.2,.01,.01,-.02],type:'dynamic'},
  'A':{curls:[.3,1,1,1,1],ori:[0,-.7,0,0,0,0],type:'static'},
  'B':{curls:[.8,0,0,0,0],ori:[0,-1,0,0,0,0],type:'static'},
  'C':{curls:[.4,.4,.4,.4,.4],ori:[.2,-.7,.1,0,0,0],type:'static'},
  'D':{curls:[.8,0,.9,.9,.9],ori:[0,-.8,0,0,0,0],type:'static'},
  'E':{curls:[.6,.7,.7,.7,.7],ori:[0,-.8,0,0,0,0],type:'static'},
  'F':{curls:[.7,.7,0,0,0],ori:[0,-.8,0,0,0,0],type:'static'},
  'G':{curls:[.5,0,.9,.9,.9],ori:[.4,-.3,0,.03,0,0],type:'static'},
  'H':{curls:[.8,0,0,.9,.9],ori:[.4,-.3,0,.03,0,0],type:'static'},
  'I':{curls:[.8,.9,.9,.9,0],ori:[0,-.8,0,0,0,0],type:'static'},
  'J':{curls:[.8,.9,.9,.9,0],ori:[0,-.8,0,0,0,.1],type:'dynamic'},
  'K':{curls:[.3,0,0,.9,.9],ori:[0,-.8,.1,0,0,0],type:'static'},
  'L':{curls:[0,0,.9,.9,.9],ori:[0,-.8,0,0,0,0],type:'static'},
  'M':{curls:[.6,.8,.8,.8,.9],ori:[0,-.5,-.3,0,0,0],type:'static'},
  'N':{curls:[.6,.8,.8,.9,.9],ori:[0,-.5,-.3,0,0,0],type:'static'},
  'O':{curls:[.6,.6,.6,.6,.6],ori:[0,-.8,0,0,0,0],type:'static'},
  'P':{curls:[.3,0,0,.9,.9],ori:[.3,-.3,-.3,.02,0,0],type:'static'},
  'Q':{curls:[.3,0,.9,.9,.9],ori:[.3,-.2,-.4,.02,0,0],type:'static'},
  'R':{curls:[.8,0,0,.9,.9],ori:[0,-.8,0,0,0,0],type:'static'},
  'S':{curls:[.5,1,1,1,1],ori:[0,-.8,0,0,0,0],type:'static'},
  'T':{curls:[.4,.9,.9,.9,.9],ori:[0,-.8,0,0,0,0],type:'static'},
  'U':{curls:[.8,0,0,.9,.9],ori:[0,-1,0,0,0,0],type:'static'},
  'V':{curls:[.8,0,0,.9,.9],ori:[0,-.9,.15,0,0,0],type:'static'},
  'W':{curls:[.8,0,0,0,.9],ori:[0,-1,0,0,0,0],type:'static'},
  'X':{curls:[.8,.5,.9,.9,.9],ori:[0,-.8,0,0,0,0],type:'static'},
  'Y':{curls:[0,.9,.9,.9,0],ori:[0,-.8,0,0,0,0],type:'static'},
  'Z':{curls:[.8,0,.9,.9,.9],ori:[0,-.7,0,.04,0,.05],type:'dynamic'},
  '0':{curls:[.6,.6,.6,.6,.6],ori:[0,-.8,0,0,0,0],type:'static'},
  '1':{curls:[.8,0,.9,.9,.9],ori:[0,-.9,0,0,0,0],type:'static'},
  '2':{curls:[.8,0,0,.9,.9],ori:[0,-.9,0,0,0,0],type:'static'},
  '3':{curls:[.8,0,0,0,.9],ori:[0,-.9,0,0,0,0],type:'static'},
  '4':{curls:[.8,0,0,0,0],ori:[0,-.9,0,0,0,0],type:'static'},
  '5':{curls:[0,0,0,0,0],ori:[0,-.9,0,0,0,0],type:'static'},
  '6':{curls:[.7,.9,.9,0,0],ori:[0,-.8,0,0,0,0],type:'static'},
  '7':{curls:[.7,.9,0,.9,0],ori:[0,-.8,0,0,0,0],type:'static'},
  '8':{curls:[.7,0,.9,0,0],ori:[0,-.8,0,0,0,0],type:'static'},
  '9':{curls:[.7,0,.9,.9,.9],ori:[0,-.8,.1,0,0,0],type:'static'},
};

var DEFAULT_COMBOS = [
  {seq:['Hello','Thank You'],action:'Hello, thank you!',timeout:3000},
  {seq:['Help','Please'],action:'Please help me!',timeout:3000},
  {seq:['Yes','Yes'],action:'Absolutely yes!',timeout:2000},
  {seq:['No','No','No'],action:'Definitely not!',timeout:4000},
  {seq:['Water','Please'],action:'Can I have water please?',timeout:3000},
  {seq:['Help','Help'],action:'I need help urgently!',timeout:2500},
  {seq:['Go','Go'],action:"Let's go now!",timeout:2500},
  {seq:['Stop','Please'],action:'Please stop!',timeout:3000},
];

var NLP_PHRASES = [
  "hello how are you","I need help","thank you very much","please give me","I want to go",
  "can you help me","good morning everyone","good night","see you later","what is your name",
  "nice to meet you","I am fine thank you","where is the bathroom","I would like to",
  "excuse me please","I do not understand","can you repeat that","yes please","no thank you",
  "I am hungry","I am thirsty","help me please","call the doctor","I feel sick",
  "what time is it","please stop that","I am sorry about that","let us go now",
  "water please","I need water please","thank you for your help","hello my name is",
  "goodbye see you tomorrow","please wait here","I want to eat","where are we going",
  "how much does it cost","can I have some water","please call my family",
  "I want to go home","the weather is nice today","please open the door","speak slowly please",
];


// ═══ utils/EventBus.js ═══
// utils/EventBus.js — Pure pub-sub, no browser side effects at import time
// WebSocket bridge lives in WSClient.js (imported separately, deferred)

class EventBus {
  constructor() { this._l = new Map(); }
  on(e, cb) {
    if (!this._l.has(e)) this._l.set(e, []);
    this._l.get(e).push(cb);
    return () => this.off(e, cb);
  }
  off(e, cb) {
    const l = this._l.get(e);
    if (l) this._l.set(e, l.filter(c => c !== cb));
  }
  emit(e, d = null) {
    (this._l.get(e) || []).forEach(cb => {
      try { cb(d); } catch (err) { console.error(`[Event:${e}]`, err); }
    });
  }
  once(e, cb) { const w = d => { cb(d); this.off(e, w); }; this.on(e, w); }
}

const eventBus = new EventBus();

const Events = {
  TAB_CHANGED: 'app:tab', STATE_UPDATED: 'app:state', MODE_CHANGED: 'app:mode',
  CAMERA_STARTED: 'cam:start', CAMERA_STOPPED: 'cam:stop',
  HAND_DETECTED: 'cam:hand', HAND_LOST: 'cam:lost', FEATURES_EXTRACTED: 'cam:features',
  BLE_CONNECTED: 'ble:conn', BLE_DISCONNECTED: 'ble:disc', BLE_DATA: 'ble:data',
  MQTT_CONNECTED: 'mqtt:conn', MQTT_DISCONNECTED: 'mqtt:disc', MQTT_DATA: 'mqtt:data',
  RECOG_STARTED: 'rec:start', RECOG_STOPPED: 'rec:stop',
  GESTURE_RECOGNIZED: 'rec:gesture', GESTURE_PREDICTING: 'rec:predict', COMBO_DETECTED: 'rec:combo',
  TRAIN_STARTED: 'tr:start', TRAIN_PROGRESS: 'tr:prog',
  TRAIN_COMPLETE: 'tr:done', SAMPLES_COLLECTED: 'tr:samples',
  WORD_ADDED: 'nlp:word', SENTENCE_UPDATED: 'nlp:sent',
  SUGGESTIONS_UPDATED: 'nlp:sugg', COMPLETION_READY: 'nlp:comp',
  SPELLING_UPDATED: 'spell:update', WORD_PREDICTED: 'spell:predict',
  GEMINI_CONNECTED: 'gem:conn', GEMINI_ERROR: 'gem:err', GEMINI_LOADING: 'gem:load',
  TTS_SPEAK: 'tts:speak', VR_POSE: 'vr:pose', VR_INIT: 'vr:init',
  SUGGESTION_SELECTED: 'ui:suggsel', SYSTEM_GESTURE: 'ui:sysgest',
  RECORDING_START: 'rec:recstart', RECORDING_TICK: 'rec:rectick', RECORDING_DONE: 'rec:recdone',
};


// ═══ utils/WSClient.js ═══
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

const wsClient = new WSClient();


// ═══ utils/MathUtils.js ═══
// utils/MathUtils.js — browser-side math helpers used by CameraService
// Heavy ML math (softmax, matrix ops) runs in Python backend

const noise  = (s = .1) => (Math.random() - .5) * s;
const clamp  = (v, mn, mx) => Math.max(mn, Math.min(mx, v));
const dist3D = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
const norm3D = v => { const l = Math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) || .001; return {x: v.x / l, y: v.y / l, z: v.z / l}; };


// ═══ utils/DOMHelper.js ═══
// utils/DOMHelper.js
const $ = id => document.getElementById(id);
const clearEl = el => { while (el && el.firstChild) el.removeChild(el.firstChild); };


// ═══ views/components/index.js ═══
// views/components/index.js
// Reusable UI primitives. Small functions that just return HTML strings.
// Kept simple — no state, no lifecycle, just string → string.
const Badge = (text, variant) =>
  `<span class="bg bg-${variant || 'g'}">${text}</span>`;

const DotBadge = (text, variant, pulse) => {
  variant = variant || 'g';
  return `<span class="bg bg-${variant}">` +
    `<span class="dot dot-${variant}${pulse ? ' dot-pulse' : ''}"></span>` +
    `${text}</span>`;
};

// variant: 'g' | 'p' | 'r' | 'a' | 'o' | 'ghost' (default: outline 'o')
const Btn = (label, onclick, variant, size, disabled) =>
  `<button class="btn${variant ? ' btn-' + variant : ' btn-o'}${size ? ' btn-' + size : ''}" ` +
  `onclick="${onclick}"${disabled ? ' disabled' : ''}>${label}</button>`;

// title is optional — pass '' or null to get a plain container
const Card = (title, body, style) =>
  `<div class="cd"${style ? ` style="${style}"` : ''}>` +
    (title ? `<div class="cd-label">${title}</div>` : '') +
    body +
  `</div>`;

const Toggle = (on, onclick) =>
  `<div class="toggle" onclick="${onclick}">` +
    `<div class="knob" style="left:${on ? '21px' : '3px'}"></div>` +
  `</div>`;

const Bar = (pct, color) =>
  `<div class="bar-wrap"><div class="bar-fill" style="width:${pct}%;background:rgba(255,255,255,0.5)"></div></div>`;

const StatBox = (value, label, color) =>
  `<div class="stat">` +
    `<div class="stat-v" style="color:#ffffff">${value}</div>` +
    `<div class="stat-l">${label}</div>` +
  `</div>`;

const SettingRow = (label, desc, control) =>
  `<div class="srow">` +
    `<div><div class="srow-label">${label}</div>${desc ? `<div class="srow-desc">${desc}</div>` : ''}</div>` +
    control +
  `</div>`;

const LogEntry = (entry, opacity) => {
  opacity = opacity !== undefined ? opacity : 1;
  return `<div class="log-entry" style="opacity:${opacity}">` +
    `<div class="fr g5">` +
      Badge(entry.gesture, 'd') +
      (entry.combo ? Badge(entry.combo, 'd') : '') +
      (entry.model ? `<span style="font-size:8px;color:var(--dm)">[${entry.model}]</span>` : '') +
    `</div>` +
    `<div class="fr g5">` +
      `<span style="font-size:12px;font-weight:700;color:#ffffff">${(entry.conf * 100).toFixed(1)}%</span>` +
      `<span style="font-size:9px;color:var(--dm)">${entry.time.toLocaleTimeString()}</span>` +
    `</div>` +
  `</div>`;
};

// Ten vertical bars showing finger curl 0–100% for both hands.
// IDs fb0–fb9 and fv0–fv9 updated directly by AppController on each frame.
// Bars 0–4 = dominant hand, 5–9 = auxiliary hand; divided by a thin separator.
const FingerBars = (names, colors) => {
  var half = Math.floor(names.length / 2);
  var cols = names.map(function(name, i) {
    var color = (colors && colors[i]) ? colors[i] : 'rgba(255,255,255,0.5)';
    return `<div class="fcol">` +
      `<div class="fbar-w"><div class="fbar-f" id="fb${i}" style="height:0%;background:${color}"></div></div>` +
      `<div class="fname" style="color:${color};font-size:9px">${name}</div>` +
      `<div class="fval" id="fv${i}" style="font-size:8px">0%</div>` +
    `</div>`;
  });
  // Insert visual divider between the two hands
  if (names.length > 5) {
    cols.splice(half, 0, '<div class="fgrid-div"></div>');
  }
  return `<div style="font-size:8px;color:var(--mx);margin-bottom:4px">` +
    `<span style="margin-right:4px">Dom hand</span>` +
    `<span style="opacity:.5">·</span>` +
    `<span style="margin-left:4px">Aux hand</span>` +
  `</div>` +
  `<div class="fgrid">${cols.join('')}</div>`;
};


// ═══ models/NeuralNetwork.js ═══
// models/NeuralNetwork.js — State container for trained model metadata
// Training and inference happen in Python; this holds the result for the UI.

class NeuralNetwork {
  constructor(id) {
    this.id       = id || 'default';
    this.trained  = false;
    this.gestures = new Map();
    this.accuracy = 0;
    this.val_accuracy = 0;
    this.loss     = 1;
    this.epochs   = 0;
  }

  // Called by TrainingController before training to record expected shape
  initialize(inputSize, hiddenSizes, outputSize) {
    this._inputSize   = inputSize;
    this._hiddenSizes = hiddenSizes;
    this._outputSize  = outputSize;
  }

  addGesture(name, id) { this.gestures.set(id, name); }
  getName(id)          { return this.gestures.get(id) || 'Unknown'; }
  getInputSize()       { return this._inputSize || 0; }

  reset() {
    this.trained  = false;
    this.accuracy = 0;
    this.loss     = 1;
    this.epochs   = 0;
    this.gestures = new Map();
    fetch('/api/nn/reset/' + this.id, { method: 'POST' }).catch(function(){});
  }

  toJSON() {
    return {
      id:       this.id,
      gestures: [...this.gestures.entries()],
      accuracy: this.accuracy,
      loss:     this.loss,
      epochs:   this.epochs,
    };
  }

  fromJSON(d) {
    this.gestures = new Map(d.gestures);
    this.accuracy = d.accuracy;
    this.loss     = d.loss;
    this.epochs   = d.epochs;
    this.trained  = true;
    this.id       = d.id || this.id;
  }
}


// ═══ models/GestureModel.js ═══
// models/GestureModel.js — v6.1 Phase 2 + Source tracking
// Each sample tagged with source: 'camera' | 'glove'
// Training loads only samples matching active source
// (API already defined in bundle header)

class GestureModel {
  constructor(defaultGestures) {
    this.gestures       = [...defaultGestures];
    this.staticSamples  = {};
    this.dynamicSamples = {};
    this.sampleCounts   = {};
    this.templates      = GESTURE_TEMPLATES;
    this.recording      = false;
    this.recordTarget   = null;
    this.frameBuffer    = [];
    this.trainSource    = 'camera'; // 'camera' | 'glove' — set by AppController
  }

  async loadFromDB() {
    try {
      var url = API + '/samples/meta';
      var loadUrl = API + '/samples/load?source=' + this.trainSource;
      var metaUrl = API + '/samples/meta';

      var results = await Promise.all([
        fetch(metaUrl).then(function(r){ return r.json(); }),
        fetch(loadUrl).then(function(r){ return r.json(); }),
      ]);
      var meta = results[0]; var samples = results[1];

      this.sampleCounts = {};
      for (var g in meta) {
        this.sampleCounts[g] = {
          static:         meta[g].static         || 0,
          dynamic:        meta[g].dynamic         || 0,
          static_camera:  meta[g].static_camera   || 0,
          static_glove:   meta[g].static_glove    || 0,
          dynamic_camera: meta[g].dynamic_camera  || 0,
          dynamic_glove:  meta[g].dynamic_glove   || 0,
        };
        if (this.gestures.indexOf(g) < 0) this.gestures.push(g);
      }
      for (var g in samples) {
        if (samples[g].static)  this.staticSamples[g]  = samples[g].static;
        if (samples[g].dynamic) this.dynamicSamples[g] = samples[g].dynamic;
      }
      console.log('[GestureModel] loaded from Python DB:', Object.keys(meta).length, 'gestures, source:', this.trainSource);
    } catch(e) { console.warn('[GestureModel] load failed:', e); }
  }

  addGesture(name) {
    var t = name.trim();
    if (t && this.gestures.indexOf(t) < 0) { this.gestures.push(t); return true; }
    return false;
  }
  removeGesture(name) {
    this.gestures = this.gestures.filter(function(g){ return g !== name; });
    delete this.staticSamples[name];
    delete this.dynamicSamples[name];
    delete this.sampleCounts[name];
  }

  getType(name) {
    var t = this.templates[name];
    return (t ? t.type : null) || 'static';
  }

  async simulateStaticAsync(name, count) {
    var res = await fetch(API + '/samples/simulate', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({gesture: name, count: count || 1, sample_type: 'static'})
    });
    var data = await res.json();
    return data.samples;
  }

  // source param: 'camera' | 'glove' — defaults to this.trainSource
  async addStaticSamples(name, vectors, isMirror, source) {
    source = source || this.trainSource;
    if (!this.staticSamples[name]) this.staticSamples[name] = [];
    this.staticSamples[name].push.apply(this.staticSamples[name], vectors);
    if (!this.sampleCounts[name]) this.sampleCounts[name] = {static:0,dynamic:0,static_camera:0,static_glove:0,dynamic_camera:0,dynamic_glove:0};
    this.sampleCounts[name].static++;
    if (source === 'camera') this.sampleCounts[name].static_camera++;
    else if (source === 'glove') this.sampleCounts[name].static_glove++;

    await fetch(API + '/samples/save', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({gesture: name, samples: vectors, sample_type: 'static', source: source})
    });
  }

  async addDynamicSamples(name, trajectories, source) {
    source = source || this.trainSource;
    if (!this.dynamicSamples[name]) this.dynamicSamples[name] = [];
    this.dynamicSamples[name].push.apply(this.dynamicSamples[name], trajectories);
    if (!this.sampleCounts[name]) this.sampleCounts[name] = {static:0,dynamic:0,static_camera:0,static_glove:0,dynamic_camera:0,dynamic_glove:0};
    this.sampleCounts[name].dynamic++;
    if (source === 'camera') this.sampleCounts[name].dynamic_camera++;
    else if (source === 'glove') this.sampleCounts[name].dynamic_glove++;

    await fetch(API + '/samples/save', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({gesture: name, samples: trajectories, sample_type: 'dynamic', source: source})
    });
  }

  pushFrame(features) {
    if (!this.recording) return false;
    this.frameBuffer.push([...features]);
    if (this.frameBuffer.length >= APP_CONFIG.NN.DYNAMIC_FRAMES) {
      var trajectory = this.frameBuffer.flat ? this.frameBuffer.flat() :
        [].concat.apply([], this.frameBuffer);
      this.addDynamicSamples(this.recordTarget, [trajectory], this.trainSource);
      this.frameBuffer = []; this.recording = false;
      return true;
    }
    return false;
  }

  startRecording(gestureName) { this.recording = true; this.recordTarget = gestureName; this.frameBuffer = []; }
  stopRecording()              { this.recording = false; this.frameBuffer = []; }

  getStaticTrainingData() {
    var inputs = [], labels = [], names = [], idx = 0;
    for (var i = 0; i < this.gestures.length; i++) {
      var name = this.gestures[i];
      if (!this.staticSamples[name] || !this.staticSamples[name].length) continue;
      names.push(name);
      for (var j = 0; j < this.staticSamples[name].length; j++) {
        inputs.push(this.staticSamples[name][j]); labels.push(idx);
      }
      idx++;
    }
    return {inputs: inputs, labels: labels, names: names};
  }

  getDynamicTrainingData() {
    var inputs = [], labels = [], names = [], idx = 0;
    for (var i = 0; i < this.gestures.length; i++) {
      var name = this.gestures[i];
      if (!this.dynamicSamples[name] || !this.dynamicSamples[name].length) continue;
      names.push(name);
      for (var j = 0; j < this.dynamicSamples[name].length; j++) {
        inputs.push(this.dynamicSamples[name][j]); labels.push(idx);
      }
      idx++;
    }
    return {inputs: inputs, labels: labels, names: names};
  }

  getSampleCount(name) { return this.sampleCounts[name] || {static:0,dynamic:0,static_camera:0,static_glove:0,dynamic_camera:0,dynamic_glove:0}; }

  async resetAll() {
    await fetch(API + '/samples/clear', {method: 'DELETE'});
    this.staticSamples = {}; this.dynamicSamples = {}; this.sampleCounts = {};
    this.gestures = [...APP_CONFIG.DEFAULT_GESTURES];
  }
}


// ═══ models/SentenceModel.js ═══
class SentenceModel {
  constructor() {
    this.words = [];
    this.spelling = '';
    this.context = [];
    this.suggestions = ['hello', 'I', 'please', 'help', 'thank'];
    this.wordSuggestions = [];
    this.completion = null;
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

  getSentence() { return this.words.join(' '); }
  getLastWord() { return this.words[this.words.length - 1] || null; }
  getWordCount() { return this.words.length; }
  getContextString() { return this.context.slice(-20).join(' '); }
  getRecentGestures(n) { return this.gestureSequence.slice(-(n || 5)); }
  setSuggestions(s) { this.suggestions = s || []; }
  setCompletion(t) { this.completion = t; }
  setWordSuggestions(s) { this.wordSuggestions = s || []; }

  addLetter(ch) { this.spelling += ch.toLowerCase(); this._syncToServer('add_letter', ch); }
  removeLetter() { this.spelling = this.spelling.slice(0, -1); }
  clearSpelling() { this.spelling = ''; this.wordSuggestions = []; }
  getSpelling() { return this.spelling; }

  acceptWordSuggestion(word) {
    this.addWord(word); this.spelling = ''; this.wordSuggestions = [];
    this._syncToServer('accept_word_suggestion', word);
  }

  acceptCompletion() {
    if (!this.completion) return false;
    this.words = this.completion.split(' ').map(function (w) { return w.toLowerCase(); });
    this.completion = null;
    this._syncToServer('accept_completion');
    return true;
  }

  replaceWithCorrected(c) {
    this.words = c.split(' ').map(function (w) { return w.toLowerCase(); });
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
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: action, word: word, text: text })
    }).catch(function () { });
  }
}


// ═══ models/SensorModel.js ═══
// models/SensorModel.js — Gesture Detection v1.0
// Camera-only sensor state. Stores full 41-feature Holistic vector.
var SensorModel = (function() {
  function SensorModel() {
    this.handDetected    = false;
    this.handCount       = 0;
    this.faceDetected    = false;
    this.poseDetected    = false;
    this.source          = 'camera';
    this._fullVector     = null; // 41-feature vector from Holistic
    // For finger bar display (first 5 features are curls)
    this.flex            = [0, 0, 0, 0, 0];
    this.calibration     = null;
  }

  // Called by RecognitionController on every FEATURES_EXTRACTED event
  SensorModel.prototype.setFromFeatures = function(features, meta) {
    meta = meta || {};
    if (!features || features.length < 41) return;
    this._fullVector  = features.slice(0, 41);
    this.flex         = features.slice(0, 5); // curls for finger bar display
    this.handDetected = features[38] > 0.5 || features[39] > 0.5; // domPresent || auxPresent
    this.handCount    = (features[38] > 0.5 ? 1 : 0) + (features[39] > 0.5 ? 1 : 0);
    this.faceDetected = features[40] > 0.5;
    this.poseDetected = meta.poseDetected || false;
    this.source       = 'camera';
  };

  SensorModel.prototype.getFeatureVector = function() {
    if (this._fullVector && this._fullVector.length === 41) {
      return this._fullVector.slice();
    }
    // No data yet — return zeros
    return new Array(41).fill(0);
  };

  // Count extended fingers (features 0-4 are curls, low curl = extended)
  SensorModel.prototype.countExtendedFingers = function() {
    var count = 0;
    for (var i = 0; i < 5; i++) {
      if (this.flex[i] < 0.3) count++;
    }
    return count;
  };

  SensorModel.prototype.reset = function() {
    this.handDetected = false;
    this.handCount    = 0;
    this.faceDetected = false;
    this._fullVector  = null;
    this.flex         = [0, 0, 0, 0, 0];
  };

  return SensorModel;
})();


// ═══ services/StorageService.js ═══
// services/StorageService.js — Routes persistent storage to Python backend
class StorageService {
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


// ═══ services/CameraService.js ═══
// services/CameraService.js — Gesture Detection v1.0
// MediaPipe Holistic: hands + face + body = 41 features
// [hand1x11] + [hand2x11] + [face x10] + [pose x6] + [flags x3] = 41
// Face (10): nose_x, nose_y, eye_scale, dom_wrist_dx, dom_wrist_dy,
//            aux_wrist_dx, aux_wrist_dy, tilt, mouth_open, eye_open
var CameraService = (function() {
  function CameraService() {
    this.holistic     = null;
    this.active       = false;
    this.fps          = 0;
    this._fc          = 0;
    this._ft          = Date.now();
    this._stream      = null;
    this._rafId       = null;
    this.handCount    = 0;
    this.faceDetected = false;
    this.poseDetected = false;
    this.lastFeatures = null;
    this._lastHandsData = [];
    this._lastFaceLM    = null;
    this._lastPoseLM    = null;

    // Recording trail — filled by AppController event handlers
    this._isRecording    = false;
    this._recordingTrail = []; // [{x, y}] canvas-normalised positions

    // Camera facing mode: 'user' (front) or 'environment' (back)
    this.facingMode = 'user';
    // Current live prediction for canvas overlay (dom hand)
    this._currentPrediction = null;
    // Independent aux-hand prediction for canvas overlay
    this._auxPrediction     = null;
    // Status text shown on canvas when no prediction is active
    this._statusText = null;

    // Persistent DOM elements — never destroyed by re-renders
    this.videoEl = document.createElement('video');
    this.videoEl.setAttribute('autoplay', '');
    this.videoEl.setAttribute('playsinline', '');
    this.videoEl.muted = true;
    this.videoEl.style.cssText = 'width:100%;display:block';

    this.canvasEl = document.createElement('canvas');
    this.canvasEl.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%';
  }

  CameraService.prototype.initialize = function() {
    if (typeof Holistic === 'undefined') {
      console.warn('[Camera] MediaPipe Holistic not loaded');
      return null;
    }
    if (this.holistic) return Promise.resolve(true);
    var c = APP_CONFIG.MEDIAPIPE;
    var self = this;
    this._holisticReady = false;
    this.holistic = new Holistic({
      locateFile: function(f) { return c.CDN_BASE + '/' + f; }
    });
    this.holistic.setOptions({
      modelComplexity:          c.MODEL_COMPLEXITY,
      smoothLandmarks:          c.SMOOTH_LANDMARKS,
      minDetectionConfidence:   c.MIN_DETECTION_CONFIDENCE,
      minTrackingConfidence:    c.MIN_TRACKING_CONFIDENCE,
    });
    this.holistic.onResults(function(r) { self._onResults(r); });
    // Wait for all model files to finish downloading before allowing send()
    return this.holistic.initialize().then(function() {
      self._holisticReady = true;
      console.log('[Camera] Holistic v1.0 initialized and ready');
      return true;
    });
  };

  CameraService.prototype.start = function() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera API unavailable. Open the app at http://localhost:8000 (not 0.0.0.0) — browsers block camera access on non-localhost addresses.');
    }
    var initPromise = this.initialize();
    if (!initPromise) throw new Error('MediaPipe Holistic not available');
    var self = this;
    return Promise.all([
      initPromise,
      navigator.mediaDevices.getUserMedia({
        video: { facingMode: self.facingMode, width: { ideal: 640 }, height: { ideal: 480 } }
      }).then(function(stream) {
        self._stream = stream;
        self.videoEl.srcObject = stream;
        self._applyMirror();
        return self.videoEl.play();
      })
    ]).then(function() {
      self.active = true;
      self._startLoop();
      eventBus.emit(Events.CAMERA_STARTED);
      console.log('[Camera] started facingMode=' + self.facingMode);
    });
  };

  // ── Mirror transform — no flip so video shows natural orientation ──
  CameraService.prototype._applyMirror = function() {
    this.videoEl.style.transform  = 'none';
    this.canvasEl.style.transform = 'none';
  };

  // ── Switch between front / back camera ───────────────────────
  CameraService.prototype.switchCamera = function() {
    var self = this;
    this.facingMode = this.facingMode === 'user' ? 'environment' : 'user';
    this._applyMirror();
    this._currentPrediction = null;
    if (!this.active || !this._stream) return Promise.resolve();
    // Stop existing tracks then open new stream
    this._stream.getTracks().forEach(function(t) { t.stop(); });
    this._stream = null;
    // Try 'exact' first (needed on some Android devices), fall back on failure
    return navigator.mediaDevices.getUserMedia({
      video: { facingMode: { exact: self.facingMode }, width: { ideal: 640 }, height: { ideal: 480 } }
    }).catch(function() {
      return navigator.mediaDevices.getUserMedia({
        video: { facingMode: self.facingMode, width: { ideal: 640 }, height: { ideal: 480 } }
      });
    }).then(function(stream) {
      self._stream = stream;
      self.videoEl.srcObject = stream;
      return self.videoEl.play();
    }).catch(function(err) {
      // Roll back if new camera unavailable
      self.facingMode = self.facingMode === 'user' ? 'environment' : 'user';
      self._applyMirror();
      console.warn('[Camera] switchCamera failed, reverting:', err);
    });
  };

  // ── Set live prediction for canvas overlay ───────────────────
  CameraService.prototype.setPrediction = function(name, conf, model, auxName, auxConf) {
    this._currentPrediction = name ? { name: name, conf: conf || 0, model: model || '' } : null;
    this._auxPrediction = (auxName && auxConf > 0.3) ? { name: auxName, conf: auxConf || 0 } : null;
  };

  // ── Set status text shown when no prediction is active ───────
  CameraService.prototype.setStatus = function(text) {
    this._statusText = text || null;
  };

  CameraService.prototype.stop = function() {
    this.active = false;
    if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
    if (this._stream) {
      this._stream.getTracks().forEach(function(t) { t.stop(); });
      this._stream = null;
    }
    this.videoEl.srcObject = null;
    this.handCount = 0;
    eventBus.emit(Events.CAMERA_STOPPED);
  };

  CameraService.prototype.mountInto = function(container) {
    if (!container) return;
    if (this.videoEl.parentNode !== container)  container.appendChild(this.videoEl);
    if (this.canvasEl.parentNode !== container) container.appendChild(this.canvasEl);
    this.videoEl.style.display  = this.active ? 'block' : 'none';
    this.canvasEl.style.display = this.active ? 'block' : 'none';
  };

  CameraService.prototype._startLoop = function() {
    var self = this;
    function loop() {
      if (!self.active) return;
      self._rafId = requestAnimationFrame(loop);
      if (self.videoEl.readyState >= 2 && self.holistic && self._holisticReady) {
        self.holistic.send({ image: self.videoEl }).catch(function() {});
      }
      // FPS counter
      self._fc++;
      var now = Date.now();
      if (now - self._ft >= 1000) {
        self.fps = self._fc;
        self._fc = 0;
        self._ft = now;
        var el = document.getElementById('trainFpsB') || document.getElementById('recogFpsB');
        if (el) el.textContent = self.fps + ' FPS';
      }
    }
    loop();
  };

  // ── Holistic results handler ─────────────────────────────────
  CameraService.prototype._onResults = function(results) {
    // Draw skeleton on canvas
    this._drawResults(results);

    // Collect hands data
    var handsData = [];
    if (results.leftHandLandmarks) {
      handsData.push({ label: 'Left',  lm: results.leftHandLandmarks });
    }
    if (results.rightHandLandmarks) {
      handsData.push({ label: 'Right', lm: results.rightHandLandmarks });
    }

    this._lastHandsData = handsData;
    this._lastFaceLM    = results.faceLandmarks    || null;
    this._lastPoseLM    = results.poseLandmarks    || null;
    this.handCount      = handsData.length;
    this.faceDetected   = !!results.faceLandmarks;
    this.poseDetected   = !!results.poseLandmarks;

    // Build 41-feature vector
    var features = this._buildFeatureVector(handsData, results.faceLandmarks, results.poseLandmarks);
    this.lastFeatures = features;

    // Record wrist trail during dynamic gesture recording
    if (this._isRecording && handsData.length > 0) {
      var wrist = handsData[0].lm[0]; // landmark 0 = wrist
      this._recordingTrail.push({ x: wrist.x, y: wrist.y });
      if (this._recordingTrail.length > 60) this._recordingTrail.shift();
    }

    // Update hand badge
    var hb = document.getElementById('trainHandB') || document.getElementById('recogHandB');
    if (hb) {
      if (this.handCount === 0) hb.textContent = 'No Hand';
      else if (this.handCount === 1) hb.textContent = '1 Hand';
      else hb.textContent = '2 Hands';
    }

    eventBus.emit(Events.FEATURES_EXTRACTED, {
      features:  features,
      handCount: this.handCount,
      faceDetected: this.faceDetected,
      poseDetected: this.poseDetected,
    });
  };

  // ── 41-feature vector builder ────────────────────────────────
  CameraService.prototype._buildFeatureVector = function(handsData, faceLM, poseLM) {
    var ZEROS_11 = [0,0,0,0,0, 0,0,0,0,0,0];
    var dom = null, aux = null, domPresent = 0, auxPresent = 0;

    for (var i = 0; i < handsData.length; i++) {
      var label = this._correctHandedness(handsData[i].label);
      var feat  = this._extractOneHand(handsData[i].lm);
      if (label === APP_CONFIG.MEDIAPIPE.DOMINANT_HAND) {
        dom = feat; domPresent = 1;
      } else {
        aux = feat; auxPresent = 1;
      }
    }

    // Face features (10): nose, scale, wrist→nose offsets, tilt, mouth open, eye open
    var faceFeat = this._extractFaceFeatures(faceLM, poseLM, dom, aux);
    // Pose features (6)
    var poseFeat = this._extractPoseFeatures(poseLM, dom, aux);
    // Presence flags (3)
    var flags = [domPresent, auxPresent, this.faceDetected ? 1 : 0];

    // 11 + 11 + 10 + 6 + 3 = 41 exactly
    var vec = (dom || ZEROS_11).concat(aux || ZEROS_11)
              .concat(faceFeat)
              .concat(poseFeat)
              .concat(flags);

    // Ensure exactly 41 features
    while (vec.length < 41) vec.push(0);
    return vec.slice(0, 41);
  };

  // ── Single hand: 11 features ─────────────────────────────────
  CameraService.prototype._extractOneHand = function(lm) {
    var tips  = [4, 8, 12, 16, 20];
    var pips  = [3, 6, 10, 14, 18];
    var mcps  = [2, 5,  9, 13, 17];
    var curls = [];
    for (var i = 0; i < 5; i++) {
      var t2m = dist3D(lm[tips[i]], lm[mcps[i]]);
      var p2m = dist3D(lm[pips[i]], lm[mcps[i]]);
      curls.push(clamp(1 - (p2m > 0.001 ? t2m / (p2m * 2.5) : 0), 0, 1));
    }
    // Hand direction: wrist(0) → mid-finger MCP(9)
    var hd = norm3D({ x: lm[9].x-lm[0].x, y: lm[9].y-lm[0].y, z: lm[9].z-lm[0].z });
    // Side direction: pinky MCP(17) → index MCP(5)
    var sd = norm3D({ x: lm[17].x-lm[5].x, y: lm[17].y-lm[5].y, z: lm[17].z-lm[5].z });
    return curls.concat([hd.x, hd.y, hd.z, sd.x*0.1, sd.y*0.1, sd.z*0.1]);
  };

  // ── Face features: 10 values ─────────────────────────────────
  // [0,1] nose position  [2] face scale  [3,4] dom wrist→nose  [5,6] aux wrist→nose
  // [7] face tilt  [8] mouth openness  [9] eye openness
  CameraService.prototype._extractFaceFeatures = function(faceLM, poseLM, dom, aux) {
    if (!faceLM || faceLM.length < 468) return [0,0,0, 0,0, 0,0, 0, 0,0];
    // Nose tip = landmark 1
    var nose = faceLM[1];
    // Eye distance: left eye outer (33) to right eye outer (263)
    var eyeDist = dist3D(faceLM[33], faceLM[263]);
    var eyeScale = eyeDist > 0.001 ? eyeDist : 0.1;
    // Face tilt: angle of line from left to right eye
    var dx = faceLM[263].x - faceLM[33].x;
    var dy = faceLM[263].y - faceLM[33].y;
    var tilt = Math.atan2(dy, dx);

    // Wrist-to-nose offsets using actual pose wrist landmarks
    // Pose: 16 = right wrist (dominant), 15 = left wrist (aux)
    var h1nx = 0, h1ny = 0, h2nx = 0, h2ny = 0;
    if (poseLM && poseLM.length >= 17) {
      var rWrist = poseLM[16];
      var lWrist = poseLM[15];
      if (dom && rWrist.visibility > 0.1) {
        h1nx = clamp(rWrist.x - nose.x, -1, 1);
        h1ny = clamp(rWrist.y - nose.y, -1, 1);
      }
      if (aux && lWrist.visibility > 0.1) {
        h2nx = clamp(lWrist.x - nose.x, -1, 1);
        h2ny = clamp(lWrist.y - nose.y, -1, 1);
      }
    }

    // Mouth openness: upper lip center (13) to lower lip center (14)
    var mouthOpen = clamp(dist3D(faceLM[13], faceLM[14]) / eyeScale, 0, 1);

    // Eye openness: average of left eye (159→145) and right eye (386→374) vertical span
    var leftEyeH  = dist3D(faceLM[159], faceLM[145]);
    var rightEyeH = dist3D(faceLM[386], faceLM[374]);
    var eyeOpen   = clamp((leftEyeH + rightEyeH) / 2 / eyeScale, 0, 1);

    return [
      nose.x, nose.y,                // [0,1] nose position
      clamp(eyeDist * 5, 0, 1),      // [2]   face scale
      h1nx, h1ny,                    // [3,4] dom wrist → nose
      h2nx, h2ny,                    // [5,6] aux wrist → nose
      clamp(tilt / Math.PI, -1, 1),  // [7]   face tilt
      mouthOpen,                     // [8]   mouth openness
      eyeOpen,                       // [9]   eye openness
    ];
  };

  // ── Pose features: 6 values ──────────────────────────────────
  CameraService.prototype._extractPoseFeatures = function(poseLM, dom, aux) {
    if (!poseLM || poseLM.length < 33) return [0,0, 0,0, 0,0];
    // Shoulders: left=11, right=12
    var lSh = poseLM[11], rSh = poseLM[12];
    var shoulderMidX = (lSh.x + rSh.x) / 2;
    var shoulderMidY = (lSh.y + rSh.y) / 2;

    // Hand height relative to shoulder (using wrist landmarks 15=left, 16=right)
    var lWr = poseLM[15], rWr = poseLM[16];
    var h1ShDy = dom ? clamp(shoulderMidY - rWr.y, -1, 1) : 0;
    var h2ShDy = aux ? clamp(shoulderMidY - lWr.y, -1, 1) : 0;

    // Elbow angle: right elbow = landmark 14, right shoulder = 12, right wrist = 16
    var rEl = poseLM[14];
    var v1x = rSh.x - rEl.x, v1y = rSh.y - rEl.y;
    var v2x = rWr.x - rEl.x, v2y = rWr.y - rEl.y;
    var mag = Math.sqrt(v1x*v1x+v1y*v1y) * Math.sqrt(v2x*v2x+v2y*v2y);
    var elbowAngle = mag > 0.001 ? Math.acos(clamp((v1x*v2x+v1y*v2y)/mag,-1,1)) / Math.PI : 0;

    var bodyVisible = (poseLM[11].visibility > 0.5 && poseLM[12].visibility > 0.5) ? 1 : 0;

    return [
      clamp(shoulderMidX, 0, 1),  // [0]
      clamp(shoulderMidY, 0, 1),  // [1]
      h1ShDy,                     // [2]
      h2ShDy,                     // [3]
      elbowAngle,                 // [4]
      bodyVisible,                // [5]
    ];
  };

  // ── Mirror augmentation ──────────────────────────────────────
  CameraService.prototype.getMirroredFeatures = function(handsData) {
    if (!handsData || handsData.length === 0) return null;
    var mirrored = handsData.map(function(h) {
      return {
        label: h.label === 'Left' ? 'Right' : 'Left',
        lm: h.lm.map(function(p) {
          return { x: 1 - p.x, y: p.y, z: -p.z };
        }),
      };
    });
    return this._buildFeatureVector(mirrored, this._lastFaceLM, this._lastPoseLM);
  };

  CameraService.prototype._correctHandedness = function(label) {
    // MediaPipe identifies hands anatomically (Right = right hand, Left = left hand)
    // regardless of camera orientation — use labels as-is so the dominant (Right) hand
    // always maps to feature slots 0-10 and the finger curl bars update correctly.
    return label;
  };

  // ── Skeleton drawing ─────────────────────────────────────────
  CameraService.prototype._drawResults = function(results) {
    var canvas = this.canvasEl;
    var video  = this.videoEl;
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw hand connections
    var handColor = 'rgba(255,255,255,0.75)';
    var connections = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],
      [0,13],[13,14],[14,15],[15,16],
      [0,17],[17,18],[18,19],[19,20],
      [5,9],[9,13],[13,17],
    ];

    var self = this;
    function drawHand(lm) {
      ctx.strokeStyle = handColor;
      ctx.lineWidth   = 2;
      connections.forEach(function(c) {
        var a = lm[c[0]], b = lm[c[1]];
        ctx.beginPath();
        ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
        ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
        ctx.stroke();
      });
      lm.forEach(function(p) {
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, Math.PI*2);
        ctx.fillStyle = handColor;
        ctx.fill();
      });
    }

    if (results.leftHandLandmarks)  drawHand(results.leftHandLandmarks);
    if (results.rightHandLandmarks) drawHand(results.rightHandLandmarks);

    // Draw hand movement trail during dynamic recording
    if (self._isRecording && self._recordingTrail.length > 1) {
      var trail = self._recordingTrail;
      for (var ti = 1; ti < trail.length; ti++) {
        var alpha = ti / trail.length;
        ctx.beginPath();
        ctx.moveTo(trail[ti - 1].x * canvas.width, trail[ti - 1].y * canvas.height);
        ctx.lineTo(trail[ti].x * canvas.width, trail[ti].y * canvas.height);
        ctx.strokeStyle = 'rgba(255,255,255,' + (alpha * 0.7) + ')';
        ctx.lineWidth   = 3 * alpha;
        ctx.stroke();
      }
      // Draw a dot at the latest position
      var last = trail[trail.length - 1];
      ctx.beginPath();
      ctx.arc(last.x * canvas.width, last.y * canvas.height, 5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.fill();
    }

    // Draw pose skeleton (just shoulders + arms)
    if (results.poseLandmarks) {
      var pose = results.poseLandmarks;
      var poseConns = [[11,12],[11,13],[13,15],[12,14],[14,16]];
      ctx.strokeStyle = 'rgba(255,255,255,0.45)';
      ctx.lineWidth   = 2;
      poseConns.forEach(function(c) {
        var a = pose[c[0]], b = pose[c[1]];
        if (a.visibility > 0.5 && b.visibility > 0.5) {
          ctx.beginPath();
          ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
          ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
          ctx.stroke();
        }
      });
    }

    // Draw face tracking: oval contour, eyes, nose, mouth
    if (results.faceLandmarks && results.faceLandmarks.length >= 468) {
      var fl = results.faceLandmarks;
      var fw = canvas.width, fh = canvas.height;

      function drawPolyline(indices, close) {
        if (indices.length < 2) return;
        ctx.beginPath();
        ctx.moveTo(fl[indices[0]].x * fw, fl[indices[0]].y * fh);
        for (var pi = 1; pi < indices.length; pi++) {
          ctx.lineTo(fl[indices[pi]].x * fw, fl[indices[pi]].y * fh);
        }
        if (close) ctx.closePath();
        ctx.stroke();
      }

      // Face oval
      ctx.strokeStyle = 'rgba(255,255,255,0.20)';
      ctx.lineWidth   = 1;
      drawPolyline([10,338,297,332,284,251,389,356,454,323,361,288,397,365,
                    379,378,400,377,152,148,176,149,150,136,172,58,132,93,
                    234,127,162,21,54,103,67,109,10], false);

      // Eyes
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.lineWidth   = 1.2;
      drawPolyline([33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33], true);
      drawPolyline([362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362], true);

      // Outer lips
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.lineWidth   = 1.2;
      drawPolyline([61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61], true);

      // Nose bridge + tip
      ctx.strokeStyle = 'rgba(255,255,255,0.35)';
      ctx.lineWidth   = 1;
      drawPolyline([168,6,197,195,5,4,1], false);
      drawPolyline([98,240,64,235,236,3,237,238,241,125,327], false);

      // Nose tip dot
      ctx.beginPath();
      ctx.arc(fl[1].x * fw, fl[1].y * fh, 3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fill();
    }

    // ── Status text overlay (when no prediction is active) ───
    if (!this._currentPrediction && this._statusText) {
      var cw2 = canvas.width, ch2 = canvas.height;
      var sFontSize = Math.min(18, Math.max(11, cw2 * 0.033));
      var bgGrad2 = ctx.createLinearGradient(0, ch2 * 0.78, 0, ch2);
      bgGrad2.addColorStop(0, 'rgba(6,8,13,0)');
      bgGrad2.addColorStop(1, 'rgba(6,8,13,0.75)');
      ctx.fillStyle = bgGrad2;
      ctx.fillRect(0, ch2 * 0.78, cw2, ch2 * 0.22);
      ctx.save();
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.font        = 'bold ' + sFontSize + 'px "IBM Plex Mono",monospace';
      ctx.shadowColor = 'rgba(0,0,0,0.9)';
      ctx.shadowBlur  = 10;
      ctx.fillStyle   = 'rgba(180,185,200,0.85)';
      ctx.fillText(this._statusText, cw2 / 2, ch2 - 12);
      ctx.restore();
    }

    // ── Prediction overlay drawn directly on canvas ───────────
    if (this._currentPrediction || this._auxPrediction) {
      var pred = this._currentPrediction;
      var aux  = this._auxPrediction;
      var cw = canvas.width, ch = canvas.height;
      var nameFontSize = Math.min(60, Math.max(28, cw * 0.11));
      var confFontSize = Math.round(nameFontSize * 0.28);
      var auxFontSize  = Math.round(nameFontSize * 0.55);

      var alpha = pred ? Math.min(1, 0.35 + pred.conf * 0.65) : 0.5;

      // Fade gradient — taller when aux label is also showing, normal otherwise
      var gradStart = aux ? 0.50 : 0.58;
      var bgGrad = ctx.createLinearGradient(0, ch * gradStart, 0, ch);
      bgGrad.addColorStop(0, 'rgba(6,8,13,0)');
      bgGrad.addColorStop(1, 'rgba(6,8,13,' + (0.88 * alpha) + ')');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, ch * gradStart, cw, ch * (1 - gradStart));

      ctx.save();
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'alphabetic';

      if (pred) {
        ctx.globalAlpha = alpha;
        // Dom gesture name
        ctx.font        = 'bold ' + nameFontSize + 'px "IBM Plex Mono",monospace';
        ctx.shadowColor = 'rgba(0,0,0,0.95)';
        ctx.shadowBlur  = 18;
        ctx.fillStyle   = '#ffffff';
        ctx.fillText(pred.name, cw / 2, ch - 30);
        // Confidence + model tag
        ctx.font      = 'bold ' + confFontSize + 'px "IBM Plex Mono",monospace';
        ctx.fillStyle = 'rgba(220,224,236,0.80)';
        ctx.shadowBlur = 6;
        var confLabel = (pred.conf * 100).toFixed(1) + '%';
        if (pred.model) confLabel += '  [' + pred.model + ']';
        ctx.fillText(confLabel, cw / 2, ch - 8);
      }

      // Aux-hand prediction — smaller badge above dom prediction
      if (aux) {
        var auxAlpha = Math.min(0.85, 0.3 + aux.conf * 0.55);
        ctx.globalAlpha = auxAlpha;
        ctx.font        = 'bold ' + auxFontSize + 'px "IBM Plex Mono",monospace';
        ctx.shadowColor = 'rgba(0,0,0,0.9)';
        ctx.shadowBlur  = 10;
        ctx.fillStyle   = 'rgba(180,220,255,0.9)'; // tinted blue to distinguish from dom
        var auxY = pred ? ch - 30 - nameFontSize - 10 : ch - 30;
        ctx.fillText(aux.name, cw / 2, auxY);
        ctx.font      = 'bold ' + confFontSize + 'px "IBM Plex Mono",monospace';
        ctx.fillStyle = 'rgba(160,200,240,0.70)';
        ctx.fillText((aux.conf * 100).toFixed(1) + '% [aux]', cw / 2, auxY + confFontSize + 4);
      }

      ctx.restore();
    }
  };

  return CameraService;
})();


// ═══ services/MQTTService.js ═══
// services/MQTTService.js — Browser MQTT client (camera mode publish)
// Uses mqtt.js over WSS to publish recognized gestures to HiveMQ
var MQTTService = (function() {
  function MQTTService() {
    this._client    = null;
    this.connected  = false;
    this.enabled    = false;
    this.broker     = APP_CONFIG.MQTT.DEFAULT_BROKER;   // wss://broker.hivemq.com:8884/mqtt
    this.topicGesture  = APP_CONFIG.MQTT.TOPIC_RESULTS;   // gesture-detection/results/gesture
    this.topicStatus   = APP_CONFIG.MQTT.TOPIC_STATUS;    // gesture-detection/status
    this._clientId  = 'gesture-browser-' + Math.random().toString(36).slice(2, 9);
    this._onStatusChange = null;
  }

  MQTTService.prototype.connect = function(onStatus) {
    if (typeof mqtt === 'undefined') {
      console.warn('[MQTT] mqtt.js not loaded');
      return;
    }
    if (this._client) return;
    this._onStatusChange = onStatus || null;
    var self = this;

    this._client = mqtt.connect(this.broker, {
      clientId: this._clientId,
      clean:    true,
      reconnectPeriod: 5000,
    });

    this._client.on('connect', function() {
      self.connected = true;
      self.enabled   = true;
      console.log('[MQTT] Browser client connected to ' + self.broker);
      self._publish(self.topicStatus, JSON.stringify({ status: 'online', source: 'camera', ts: Date.now() }));
      if (self._onStatusChange) self._onStatusChange('connected');
    });

    this._client.on('reconnect', function() {
      console.log('[MQTT] Reconnecting…');
      if (self._onStatusChange) self._onStatusChange('reconnecting');
    });

    this._client.on('disconnect', function() {
      self.connected = false;
      if (self._onStatusChange) self._onStatusChange('disconnected');
    });

    this._client.on('error', function(err) {
      console.warn('[MQTT] Error:', err.message || err);
      self.connected = false;
      if (self._onStatusChange) self._onStatusChange('error');
    });

    this._client.on('offline', function() {
      self.connected = false;
      if (self._onStatusChange) self._onStatusChange('offline');
    });
  };

  MQTTService.prototype.disconnect = function() {
    if (!this._client) return;
    this._client.end(true);
    this._client   = null;
    this.connected = false;
    this.enabled   = false;
    if (this._onStatusChange) this._onStatusChange('disconnected');
  };

  MQTTService.prototype.publishGesture = function(gesture, conf, model) {
    if (!this.connected || !this.enabled) return;
    var payload = JSON.stringify({
      gesture:    gesture,
      confidence: Math.round(conf * 100) / 100,
      model:      model || 'static',
      source:     'camera',
      ts:         Date.now(),
    });
    this._publish(this.topicGesture, payload);
  };

  MQTTService.prototype._publish = function(topic, payload) {
    if (!this._client || !this.connected) return;
    this._client.publish(topic, payload, { qos: 0, retain: false });
  };

  return MQTTService;
})();


// ═══ services/GeminiService.js ═══
// services/GeminiService.js
// All Gemini calls are proxied through the backend — API key never touches the browser.
var API = '/api/gemini';

class GeminiService {
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
      var r = await fetch(API + '/suggestions', {
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
      var r = await fetch(API + '/complete', {
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
      var r = await fetch(API + '/grammar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence: s }),
      });
      var d = await r.json();
      return d.corrected || null;
    } catch(e) { return null; }
  }
}


// ═══ services/NLPService.js ═══
// services/NLPService.js — v6.1 Phase 2B
// Adaptive NLP: personal corpus, gesture-aware suggestions, offline grammar
// No optional-chaining — Safari 12 safe
class NLPService {
  constructor(geminiService) {
    this.gemini               = geminiService;
    this._personalCorpusCount    = 0;
    this._sentencesSinceRetrain  = 0;
    this._userVocab              = this._loadUserVocab(); // word → use-count
  }

  // ── User vocabulary (localStorage) ───────────────────────
  _loadUserVocab() {
    try { return JSON.parse(localStorage.getItem('nlp_user_vocab') || '{}'); } catch(e) { return {}; }
  }
  _saveUserVocab() {
    try { localStorage.setItem('nlp_user_vocab', JSON.stringify(this._userVocab)); } catch(e) {}
  }
  learnWord(word) {
    if (!word || word.length < 2) return;
    var w = word.toLowerCase();
    this._userVocab[w] = (this._userVocab[w] || 0) + 1;
    this._saveUserVocab();
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
      if (ai && ai.length > 0) return ai;
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
        body: JSON.stringify({ prefix: prefix, user_vocab: this._userVocab })
      });
      var data = await res.json();
      return data.suggestions || [];
    } catch(e) {
      return this.getWordSuggestionsSync(prefix);
    }
  }

  // Sync shim — fallback and fast-path for inline use
  getWordSuggestionsSync(prefix) {
    if (!prefix) return [];
    var p    = prefix.toLowerCase();
    var self = this;
    var userMatches = Object.keys(this._userVocab)
      .filter(function(w){ return w.startsWith(p) && w !== p; })
      .sort(function(a, b){ return (self._userVocab[b] || 0) - (self._userVocab[a] || 0); });
    var COMMON = 'the,be,to,of,and,a,in,that,have,I,it,for,not,on,with,you,do,at,this,help,please,thank,yes,no,stop,go,sorry,hello,water,food,good,need,home,here,come,open,close,am,are,is,hungry,tired,sick,happy,sad,pain,doctor,family,sleep,want,more,call'.split(',');
    var base = COMMON.filter(function(w){ return w.toLowerCase().startsWith(p) && w.toLowerCase() !== p && userMatches.indexOf(w.toLowerCase()) === -1; });
    return userMatches.concat(base).slice(0, 5);
  }

  getPersonalCorpusCount() { return this._personalCorpusCount; }
}


// ═══ services/TTSService.js ═══
// services/TTSService.js
// Human-sounding TTS: best-voice selection + word-queue buffering
// Word queue: accumulates recognised words for TTS_BUFFER_MS before speaking,
// so "Hello" + "World" signed quickly are spoken as "Hello World" not two
// separate choppy utterances.
class TTSService {
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


// ═══ views/AppView.js ═══
// views/AppView.js
var AppView = (function () {
  function AppView(root, ctrl) {
    this.root = root;
    this.ctrl = ctrl;
  }

  AppView.prototype.render = function () {
    var state = this.ctrl.getState();
    var html = '';

    if (state.mode === 'user') {
      html = renderUserMode(state);
    } else {
      html += _renderHeader(state);
      html += _renderTabs(state);
      html += _renderTabContent(state);
      html += '<div class="footer">gesture-detection · MLP + LSTM · holistic · gemini</div>';
    }

    this.root.innerHTML = html;
    this.ctrl._mountCamera();
  };

  return AppView;
})();

function _renderHeader(state) {
  var trained = state.staticTrained || state.dynamicTrained;
  var modelBadge = trained
    ? '<span class="bg bg-g">model ready</span>'
    : '';
  var camBadge = state.camActive
    ? '<span class="bg bg-g"><span class="dot dot-g dot-pulse"></span> live</span>'
    : '';
  var geminiBadge = state.geminiEnabled ? '<span class="bg bg-a">gemini</span>' : '';

  return '<div class="hdr">' +
    '<div>' +
    '<div class="hdr-brand">gesture detection</div>' +
    '<div class="hdr-title">Sign Language <em>Recognition</em></div>' +
    '</div>' +
    '<div class="hdr-badges">' + camBadge + modelBadge + geminiBadge +
    '<button class="btn btn-o btn-sm" onclick="window._app.switchMode(\'user\')" title="Back to user view" ' +
      'style="font-size:11px;padding:4px 9px;border-color:var(--brd);color:var(--mx)">User View</button>' +
    '</div>' +
    '</div>';
}

function _renderTabs(state) {
  var tabs = APP_CONFIG.TABS_ADMIN;
  var html = '<div class="tabs">';
  for (var i = 0; i < tabs.length; i++) {
    var t = tabs[i];
    var active = state.tab === t.id ? ' active' : '';
    html += '<button class="tab' + active + '" onclick="window._app.switchTab(\'' + t.id + '\')">' + t.label + '</button>';
  }
  html += '</div>';
  return html;
}

function _renderTabContent(state) {
  switch (state.tab) {
    case 'detect':   return renderDetectTab(state);
    case 'train':    return renderTrainTab(state);
    case 'settings': return renderSettingsTab(state);
    default:         return renderDetectTab(state);
  }
}


// ═══ views/DetectView.js ═══
// views/DetectView.js
function renderDetectTab(state) {
  var {
    camActive, cameraError, running, displayText, spelling,
    suggestions, wordSuggestions, completion, log, gestures,
    geminiEnabled, staticTrained, dynamicTrained, inputMode, contextState
  } = state;

  var trained = staticTrained || dynamicTrained;

  return (
    _renderCamera(state, camActive, cameraError, running, trained, inputMode, contextState) +
    (running ? _renderTop3Histogram() : '') +
    Card('Finger Curl Sensor', FingerBars(APP_CONFIG.FINGER_NAMES, APP_CONFIG.FINGER_COLORS)) +
    _renderSentenceBuilder(state, displayText, spelling, suggestions, wordSuggestions, completion, geminiEnabled) +
    (trained ? _renderQuickTest(gestures) : '') +
    ((state.predTrail && state.predTrail.length) ? _renderPredTrail(state.predTrail) : '') +
    (log.length ? _renderLog(log) : '')
  );
}

function _renderCamera(state, camActive, cameraError, running, trained, inputMode, contextState) {
  var cameraContent =
    '<div class="vid-wrap" style="min-height:260px">' +
      '<div id="vidContainer" style="width:100%;height:100%"></div>' +
      '<div class="vid-badges">' +
        '<span class="bg bg-g" id="fpsB">-- FPS</span>' +
        '<span class="bg bg-d" id="handB">No Hand</span>' +
      '</div>' +
      (!camActive ? '<div class="vid-overlay"><div style="font-size:11px;color:var(--mx);margin-top:4px">tap start camera</div></div>' : '') +
    '</div>' +

    '<div style="height:3px;background:var(--s2);position:relative;margin-bottom:10px">' +
      '<div id="sysGestProg" style="height:100%;width:0%;background:rgba(255,255,255,0.5);transition:width .1s"></div>' +
    '</div>' +

    '<div class="fr fr-center mb8" style="gap:8px;flex-wrap:wrap">' +
      (camActive
        ? Btn('Stop', 'window._app.stopCamera()', 'r', 'sm')
        : Btn(cameraError ? 'Retry Camera' : 'Start Camera', 'window._app.startCamera()', 'o', 'sm')) +
      (camActive ? Btn('Flip', 'window._app.switchCamera()', 'o', 'sm') : '') +
      (!running
        ? Btn('Recognize', 'window._app.startRecognition()', camActive ? 'g' : 'o', '', !trained)
        : Btn('Stop', 'window._app.stopRecognition()', 'r')) +
      '<select onchange="window._app.setInputMode(this.value)" style="background:var(--s2);color:var(--tx);border:1px solid var(--brd);border-radius:6px;padding:6px 9px;font-family:inherit;font-size:11px">' +
        '<option value="camera"' + (inputMode === 'camera' ? ' selected' : '') + '>Camera</option>' +
        '<option value="glove"' + (inputMode === 'glove' ? ' selected' : '') + '>Glove</option>' +
        '<option value="both"' + (inputMode === 'both' ? ' selected' : '') + '>Both</option>' +
      '</select>' +
    '</div>' +

    (cameraError ? '<div style="font-size:11px;color:#ffffff;padding:8px 12px;background:rgba(255,255,255,0.04);border-radius:6px;border:1px solid rgba(255,255,255,0.2);margin-bottom:8px">' + cameraError + '</div>' : '');

  var statusBadge = running || camActive
    ? DotBadge('Active', 'g', running || camActive)
    : DotBadge('Idle', 'd', false);

  var contextBadge = contextState !== 'IDLE' ? ' ' + Badge(contextState, 'p') : '';

  return Card(
    '<span>Live Recognition</span><span class="flex"></span>' + statusBadge + contextBadge,
    cameraContent,
    'position:relative;overflow:hidden'
  );
}

function _renderSentenceBuilder(state, displayText, spelling, suggestions, wordSuggestions, completion, geminiEnabled) {
  var titleRight =
    (geminiEnabled ? Badge('Gemini AI', 'a') : Badge('Local NLP', 'd')) +
    ' ' + Btn('Speak', 'window._app.speakSentence()', 'o', 'sm', !state.sentence);

  var body =
    '<div class="sent-box' + (displayText ? '' : ' empty') + '">' +
      (displayText || 'Signs appear here…') +
      '<span class="cursor"></span>' +
    '</div>' +

    (spelling ? '<div style="font-size:11px;color:#ffffff;margin-bottom:8px">Spelling: <strong>' + spelling.toUpperCase() + '_</strong></div>' : '') +

    (wordSuggestions.length
      ? '<div class="mb8"><div style="font-size:10px;color:var(--mx);margin-bottom:5px">Word matches</div>' +
        '<div class="suggs">' +
        wordSuggestions.map(function(w, i) {
          return '<span class="sugg ai" onclick="window._app.addSuggestionWord(\'' + w + '\')">' +
            '<span style="opacity:.5;font-size:9px">' + (i + 1) + '</span> ' + w +
            '</span>';
        }).join('') +
        '</div></div>'
      : '') +

    (completion && completion !== state.sentence
      ? '<div class="completion" onclick="window._app.acceptCompletion()">' +
          '<div class="completion-label">Gemini suggests</div>' +
          '<div class="completion-text">"' + completion + '"</div>' +
        '</div>'
      : '') +

    '<div style="font-size:10px;color:var(--mx);margin-bottom:7px">' +
      (geminiEnabled ? 'AI' : 'Local') + ' next word predictions' +
    '</div>' +
    '<div class="suggs">' +
      suggestions.map(function(w, i) {
        return '<span class="sugg' + (geminiEnabled ? ' ai' : '') + '" onclick="window._app.addSuggestionWord(\'' + w + '\')">' +
          '<span style="opacity:.5;font-size:9px">' + (i + 1) + '</span> ' + w +
          '</span>';
      }).join('') +
    '</div>' +

    '<div class="fr fr-end g6 mt8">' +
      (geminiEnabled && state.sentence ? Btn('Fix Grammar', 'window._app.fixGrammar()', 'ghost', 'sm') : '') +
      Btn('Undo', 'window._app.undoWord()', 'ghost', 'sm') +
      Btn('Clear', 'window._app.clearSentence()', 'ghost', 'sm') +
    '</div>' +

    '<div style="font-size:10px;color:var(--dm);margin-top:10px;line-height:1.7">' +
      'Fist = Speak &nbsp;·&nbsp; Open palm = Clear &nbsp;·&nbsp; Thumbs down = Backspace' +
    '</div>';

  return Card(
    '<span>Sentence Builder</span><span class="flex"></span>' + titleRight,
    body
  );
}

function _renderQuickTest(gestures) {
  return Card(
    'Quick Test',
    '<div class="fr g5" style="flex-wrap:wrap">' +
      gestures.slice(0, 20).map(function(g) {
        return Btn(g, 'window._app.quickTest(\'' + g + '\')', 'o', 'sm');
      }).join('') +
    '</div>'
  );
}

function _renderLog(log) {
  return Card(
    'Recent Detections',
    '<div style="max-height:180px;overflow-y:auto">' +
      log.slice(0, 15).map(function(entry, i) {
        return LogEntry(entry, 1 - i * 0.05);
      }).join('') +
    '</div>'
  );
}

function _renderTop3Histogram() {
  var rows = '';
  for (var i = 0; i < 3; i++) {
    rows +=
      '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">' +
      '<span id="top3n' + i + '" style="font-size:10px;font-weight:700;width:40px;font-family:var(--mono);color:#ffffff;white-space:nowrap;overflow:hidden"></span>' +
      '<div style="flex:1;height:5px;background:var(--s2);border-radius:3px;overflow:hidden">' +
        '<div id="top3b' + i + '" style="height:100%;width:0%;background:rgba(255,255,255,0.5);border-radius:3px;transition:width 0.15s"></div>' +
      '</div>' +
      '<span id="top3p' + i + '" style="font-size:9px;color:var(--mx);width:28px;text-align:right;font-family:var(--mono)"></span>' +
      '</div>';
  }
  return Card(
    'Top Predictions',
    '<div style="font-size:8px;color:var(--dm);margin-bottom:6px">Live confidence from ensemble — updates every frame</div>' +
    rows
  );
}

function _renderPredTrail(trail) {
  var items = trail.map(function(p) {
    return '<span style="display:inline-flex;align-items:center;gap:4px;padding:3px 8px;' +
      'background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);' +
      'border-radius:4px;font-size:10px;font-family:var(--mono);font-weight:700;color:#ffffff">' +
      p.name +
      '<span style="font-size:8px;opacity:0.6">' + Math.round(p.conf * 100) + '%</span>' +
      '</span>';
  }).join('<span style="color:var(--dm);margin:0 2px;font-size:10px">→</span>');

  return Card(
    'Prediction Trail',
    '<div style="font-size:8px;color:var(--dm);margin-bottom:6px">Last ' + trail.length + ' confirmed gestures in order</div>' +
    '<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center">' + items + '</div>'
  );
}


// ═══ views/TrainView.js ═══
// views/TrainView.js
// No optional-chaining (?.), no nullish-coalescing (??) — Safari 12 safe.

var ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
var NUMS = '0123456789'.split('');
var DYN_LETTERS = { J: 1, Z: 1 };

// ── Tiny helpers ──────────────────────────────────────────────────────────────

function _get(obj, key, fallback) {
  return (obj && obj[key] !== undefined && obj[key] !== null) ? obj[key] : fallback;
}

function qColor(q) { return '#ffffff'; }
function qLabel(q) { return q >= 0.8 ? 'Good' : q >= 0.5 ? 'Fair' : 'Low'; }

function pBar(val, max) {
  var pct = Math.min(100, Math.round((val / max) * 100));
  return '<div style="height:5px;background:var(--s1);border-radius:3px;overflow:hidden;flex:1;min-width:40px">' +
    '<div style="width:' + pct + '%;height:100%;background:rgba(255,255,255,0.5);border-radius:3px;transition:width .4s"></div></div>';
}

function esc(s) {
  return String(s).replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

// ── Main export ───────────────────────────────────────────────────────────────

function renderTrainTab(state) {
  var trainStats  = state.trainStats || {};
  var meta        = window._trainMeta  || {};
  var readiness   = window._readiness  || {};
  var histStats   = window._histStats  || {};
  var confThresh  = (window._confThresh !== undefined && window._confThresh !== null) ? window._confThresh : 0.65;
  var perGestAcc  = window._perGestAcc || { static: {}, dynamic: {} };
  var section     = window._trainSection || 'alphabet';
  var guided      = window._guidedGesture || null;
  var liveP       = window._livePrediction || null;
  var countdown   = window._countdown || 0;
  var customs     = window._customGestures || [];
  var nlpStats    = window._nlpStats || {};
  var trainSource = window._trainSource || 'camera';

  var isTraining = _get(trainStats, 'isTraining', false);
  var progress   = _get(trainStats, 'progress', 0);
  var sTrained   = _get(trainStats, 'staticTrained',  false);
  var dTrained   = _get(trainStats, 'dynamicTrained', false);

  var needsRetrain = _get(histStats, 'needsRetrain', false) && (sTrained || dTrained);
  var newSince     = _get(histStats, 'newSamplesSinceLastTrain', 0);

  var readyCount = 0;
  var totalCount = 0;
  var totalSamples = 0;
  var k;
  for (k in readiness) {
    totalCount++;
    if (readiness[k] && readiness[k].ready) readyCount++;
  }
  for (k in meta) {
    totalSamples += (_get(meta[k], 'static', 0) + _get(meta[k], 'dynamic', 0));
  }

  var html = _renderCameraStrip(state, liveP, countdown);
  html += '<div id="trainStatus" style="display:none;padding:8px 14px;border-radius:8px;margin-bottom:10px;font-size:11px;font-weight:600;text-align:center"></div>';

  // ── TRAIN PANEL ──────────────────────────────────────────────────────────────
  html += _renderTrainPanel(trainStats, isTraining, progress, readyCount, totalCount,
    totalSamples, needsRetrain, newSince, confThresh, perGestAcc, sTrained, dTrained);

  // ── DIVIDER — capture section ─────────────────────────────────────────────
  html += '<div style="display:flex;align-items:center;gap:10px;margin:16px 0 10px">' +
    '<div style="font-size:9px;font-weight:700;color:var(--mx);letter-spacing:.14em;white-space:nowrap">CAPTURE SAMPLES</div>' +
    '<div style="flex:1;height:1px;background:var(--brd)"></div>' +
    '</div>';

  html += _renderSourceSelector(trainSource);
  html += _renderSectionTabs(section);

  if      (section === 'alphabet') html += _renderAlpha(meta, readiness, guided);
  else if (section === 'numbers')  html += _renderNums(meta, readiness);
  else if (section === 'words')    html += _renderWords(meta, readiness, state.gestures, customs);
  else                             html += _renderCustom(meta, readiness, customs);

  html += _renderSessionHistory(histStats);
  html += _renderNlpCard(nlpStats);

  return html;
}

// ── Train panel ───────────────────────────────────────────────────────────────
// Single, prominent card: sample summary → Train button → progress → results

function _renderTrainPanel(trainStats, isTraining, progress,
    readyCount, totalCount, totalSamples, needsRetrain, newSince,
    confThresh, perGestAcc, sTrained, dTrained) {

  var sAcc   = _get(trainStats, 'staticAccuracy',  0);
  var sValAc = _get(trainStats, 'valAccuracy',      0);
  var sLoss  = _get(trainStats, 'staticLoss',       1);
  var sEp    = _get(trainStats, 'staticEpochs',     0);
  var trained = sTrained || dTrained;

  // ── Sample readiness summary ──
  var summaryColor = readyCount === totalCount && totalCount > 0
    ? 'rgba(255,255,255,0.12)'
    : 'rgba(255,255,255,0.04)';

  var html = '<div class="cd" style="margin-bottom:12px">';
  html += '<div class="cd-label" style="font-size:11px;font-weight:800;letter-spacing:.12em">TRAIN MODEL</div>';

  // Sample summary row
  html += '<div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap">';
  html += '<div style="flex:1;min-width:90px;padding:10px;background:' + summaryColor + ';border:1px solid var(--brd);border-radius:8px;text-align:center">';
  html += '<div style="font-size:22px;font-weight:800;color:#ffffff">' + totalSamples + '</div>';
  html += '<div style="font-size:8px;color:var(--mx);letter-spacing:.08em;margin-top:2px">TOTAL SAMPLES</div>';
  html += '</div>';
  html += '<div style="flex:1;min-width:90px;padding:10px;background:' + summaryColor + ';border:1px solid var(--brd);border-radius:8px;text-align:center">';
  html += '<div style="font-size:22px;font-weight:800;color:#ffffff">' + readyCount + '<span style="font-size:14px;color:var(--mx)">/' + totalCount + '</span></div>';
  html += '<div style="font-size:8px;color:var(--mx);letter-spacing:.08em;margin-top:2px">GESTURES READY</div>';
  html += '</div>';
  if (trained) {
    html += '<div style="flex:1;min-width:90px;padding:10px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.15);border-radius:8px;text-align:center">';
    html += '<div style="font-size:22px;font-weight:800;color:#ffffff">' + Math.round(sAcc * 100) + '%</div>';
    html += '<div style="font-size:8px;color:var(--mx);letter-spacing:.08em;margin-top:2px">ACCURACY</div>';
    html += '</div>';
  }
  html += '</div>';

  // Retrain alert
  if (needsRetrain) {
    html += '<div style="padding:8px 12px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.18);border-radius:7px;margin-bottom:12px;font-size:10px;color:#ffffff">';
    html += '<strong>' + newSince + ' new samples</strong> added since last training — retrain to include them.';
    html += '</div>';
  }

  // Training progress bar (shown during training)
  if (isTraining) {
    html += '<div style="margin-bottom:12px">';
    html += '<div style="display:flex;justify-content:space-between;margin-bottom:5px">';
    html += '<span style="font-size:9px;color:var(--mx);letter-spacing:.08em">TRAINING IN PROGRESS</span>';
    html += '<span style="font-size:9px;font-weight:700;color:#ffffff">' + Math.round(progress) + '%</span>';
    html += '</div>';
    html += Bar(progress);
    html += '</div>';
  }

  // ── Train button (big, full width) ──
  var btnLabel, btnDisabled;
  if (isTraining) {
    btnLabel    = 'Training… ' + Math.round(progress) + '%';
    btnDisabled = true;
  } else if (!trained) {
    btnLabel    = 'Train on ' + totalSamples + ' Samples';
    btnDisabled = totalSamples === 0;
  } else if (needsRetrain) {
    btnLabel    = 'Retrain (' + newSince + ' new samples)';
    btnDisabled = false;
  } else {
    btnLabel    = 'Retrain Model';
    btnDisabled = totalSamples === 0;
  }

  html += '<button ' +
    'style="width:100%;padding:13px;font-size:12px;font-weight:800;font-family:inherit;' +
    'background:' + (btnDisabled ? 'var(--s2)' : 'rgba(255,255,255,0.1)') + ';' +
    'color:' + (btnDisabled ? 'var(--dm)' : '#ffffff') + ';' +
    'border:1px solid ' + (btnDisabled ? 'var(--brd)' : 'rgba(255,255,255,0.35)') + ';' +
    'border-radius:10px;cursor:' + (btnDisabled ? 'not-allowed' : 'pointer') + ';' +
    'letter-spacing:.06em;margin-bottom:10px;transition:opacity .15s" ' +
    (btnDisabled ? 'disabled ' : '') +
    'onclick="window._app.trainModel()">' +
    btnLabel +
    '</button>';

  // Post-training stats (shown only when trained)
  if (trained && !isTraining) {
    html += '<div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap">';
    html += '<div style="flex:1;min-width:70px;text-align:center;padding:7px 4px;background:var(--s1);border-radius:7px;border:1px solid var(--brd)">';
    html += '<div style="font-size:13px;font-weight:700;color:#ffffff">' + (Math.round(sAcc * 100)) + '%</div>';
    html += '<div style="font-size:7px;color:var(--mx)">Train Acc</div>';
    html += '</div>';
    if (sValAc > 0) {
      html += '<div style="flex:1;min-width:70px;text-align:center;padding:7px 4px;background:var(--s1);border-radius:7px;border:1px solid var(--brd)">';
      html += '<div style="font-size:13px;font-weight:700;color:#ffffff">' + Math.round(sValAc * 100) + '%</div>';
      html += '<div style="font-size:7px;color:var(--mx)">Val Acc</div>';
      html += '</div>';
    }
    html += '<div style="flex:1;min-width:70px;text-align:center;padding:7px 4px;background:var(--s1);border-radius:7px;border:1px solid var(--brd)">';
    html += '<div style="font-size:13px;font-weight:700;color:#ffffff">' + sEp + '</div>';
    html += '<div style="font-size:7px;color:var(--mx)">Epochs</div>';
    html += '</div>';
    html += '<div style="flex:1;min-width:70px;text-align:center;padding:7px 4px;background:var(--s1);border-radius:7px;border:1px solid var(--brd)">';
    html += '<div style="font-size:13px;font-weight:700;color:#ffffff">' + _get(trainStats, 'staticLoss', 1).toFixed(3) + '</div>';
    html += '<div style="font-size:7px;color:var(--mx)">Loss</div>';
    html += '</div>';
    html += '</div>';
  }

  // Confidence threshold
  html += '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px">' +
    '<span style="font-size:9px;color:var(--mx);white-space:nowrap;letter-spacing:.08em">CONFIDENCE</span>' +
    '<input type="range" min="0.3" max="0.95" step="0.05" value="' + confThresh + '" ' +
    'style="flex:1;min-width:80px;accent-color:#ffffff" ' +
    'oninput="window._app.setConfThreshold(parseFloat(this.value));document.getElementById(\'confLbl\').textContent=Math.round(parseFloat(this.value)*100)+\'%\'">' +
    '<span id="confLbl" style="font-size:10px;font-weight:700;color:#ffffff;width:32px">' + Math.round(confThresh * 100) + '%</span>' +
    '</div>';

  // Secondary actions row
  html += '<div style="display:flex;gap:6px;flex-wrap:wrap">';
  html += Btn('Delete Model', 'window._app.deleteModel()', 'r', 'sm', isTraining);
  html += Btn('Export', 'window._app.exportDataset()', 'o', 'sm');
  html += '<label class="btn btn-o btn-sm" style="cursor:pointer">Import' +
    '<input type="file" accept=".json" style="display:none" onchange="window._app.importDataset(this.files[0])">' +
    '</label>';
  html += '</div>';

  html += '</div>'; // close .cd

  // Per-gesture accuracy (shown below the train card when trained)
  html += _renderPerGestureAccuracy(perGestAcc, sTrained, dTrained);

  return html;
}

// ── Section renderers ─────────────────────────────────────────────────────────

function _renderCameraStrip(state, liveP, countdown) {
  var fingerBars = '';
  for (var fi = 0; fi < APP_CONFIG.FINGER_NAMES.length; fi++) {
    fingerBars +=
      '<div style="display:flex;align-items:center;gap:5px;margin-bottom:2px">' +
      '<span style="font-size:8px;width:36px;color:#ffffff;font-weight:600">' + APP_CONFIG.FINGER_NAMES[fi] + '</span>' +
      '<div class="bar-wrap"><div class="bar-fill" id="trainFb' + fi + '" style="width:0%;background:rgba(255,255,255,0.5)"></div></div>' +
      '<span style="font-size:8px;width:26px;text-align:right" id="trainFv' + fi + '">0%</span>' +
      '</div>';
  }

  var livePredHTML;
  if (liveP && liveP.name && liveP.name !== 'Unknown') {
    livePredHTML =
      '<div class="live-pred-box">' +
      '<div style="font-size:8px;color:var(--mx);letter-spacing:.08em;margin-bottom:2px">LIVE PREDICTION</div>' +
      '<div style="display:flex;align-items:center;gap:6px">' +
      '<span style="font-size:18px;font-weight:800;color:#ffffff">' + liveP.name + '</span>' +
      '<span style="font-size:10px;color:var(--mx)">' + Math.round(liveP.conf * 100) + '%</span>' +
      '<span style="font-size:8px;color:var(--dm)">[' + (liveP.model || '') + ']</span>' +
      '</div>' +
      '</div>';
  } else {
    livePredHTML =
      '<div style="font-size:8px;color:var(--dm);margin-top:4px">' +
      (state.camActive ? 'Camera active' : 'Waiting…') +
      ((state.staticTrained || state.dynamicTrained) ? ' · live prediction ON' : ' · train model to enable') +
      '</div>';
  }

  var recBorder = state.recording ? 'border-color:var(--r);box-shadow:0 0 18px rgba(251,113,133,.15)' : '';

  return '<div style="position:sticky;top:0;z-index:50;padding-bottom:8px;background:var(--bg)">' +
    '<div class="cd" style="margin-bottom:0;' + recBorder + '">' +
    '<div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap">' +

    '<div style="position:relative;width:180px;min-height:130px;border-radius:8px;overflow:hidden;background:var(--s1);border:1px solid ' + (state.recording ? 'var(--r)' : 'var(--brd)') + ';flex-shrink:0">' +
    '<div id="trainVidContainer" style="width:100%;height:100%;min-height:130px"></div>' +
    (!state.camActive
      ? '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;color:var(--dm);gap:4px;pointer-events:none"><div style="font-size:8px">Loading…</div></div>'
      : '') +
    (state.recording
      ? '<div style="position:absolute;top:5px;right:5px"><span class="bg bg-r" style="font-size:8px"><span class="dot dot-r dot-pulse"></span>REC</span></div>'
      : '') +
    (countdown > 0
      ? '<div class="countdown-overlay"><div class="countdown-num">' + countdown + '</div></div>'
      : '') +
    '<div style="position:absolute;bottom:0;left:0;right:0;display:flex;gap:3px;padding:3px">' +
    '<span id="trainFpsB" style="font-size:7px;background:rgba(0,0,0,.7);color:#ffffff;padding:2px 5px;border-radius:4px;font-weight:700">-- FPS</span>' +
    '<span id="trainHandB" style="font-size:7px;background:rgba(0,0,0,.7);color:var(--mx);padding:2px 5px;border-radius:4px;font-weight:700">No Hand</span>' +
    '</div>' +
    '</div>' +

    '<div style="flex:1;min-width:160px">' +
    '<div style="font-size:8px;color:var(--mx);letter-spacing:.1em;margin-bottom:4px">LIVE FINGER CURL</div>' +
    fingerBars +
    livePredHTML +
    '</div>' +
    '</div>' +
    '</div>' +
    '</div>';
}

function _renderPerGestureAccuracy(perGestAcc, sTrained, dTrained) {
  var allAcc = {};
  var pg;
  if (perGestAcc.static)  { for (pg in perGestAcc.static)  allAcc[pg] = perGestAcc.static[pg]; }
  if (perGestAcc.dynamic) { for (pg in perGestAcc.dynamic) allAcc[pg] = perGestAcc.dynamic[pg]; }

  var accKeys = Object.keys(allAcc);
  if (!(sTrained || dTrained) || accKeys.length === 0) return '';

  accKeys.sort(function(a, b) { return allAcc[a] - allAcc[b]; });

  var rows = '';
  for (var ai = 0; ai < accKeys.length; ai++) {
    var gn = accKeys[ai];
    var ga = allAcc[gn];
    rows += '<div style="padding:6px 8px;background:var(--s1);border-radius:6px;border:1px solid rgba(255,255,255,0.08)">' +
      '<div style="display:flex;justify-content:space-between;margin-bottom:3px">' +
      '<span style="font-size:10px;font-weight:700">' + gn + '</span>' +
      '<span style="font-size:9px;font-weight:700;color:#ffffff">' + Math.round(ga * 100) + '%</span>' +
      '</div>' +
      pBar(Math.round(ga * 100), 100) +
      '</div>';
  }

  return '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">Per-Gesture Accuracy <span style="font-size:8px;color:var(--dm);font-weight:400">— sorted lowest first</span></div>' +
    '<div style="font-size:9px;color:var(--dm);margin-bottom:8px">Below 70% = needs more samples for that gesture</div>' +
    '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:5px">' +
    rows +
    '</div>' +
    '</div>';
}

function _renderSourceSelector(src) {
  var camActive   = src === 'camera';
  var gloveActive = src === 'glove';
  var camStyle = 'flex:1;padding:7px 0;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;border:1px solid ' +
    (camActive ? 'rgba(255,255,255,0.4)' : 'var(--brd)') + ';background:' + (camActive ? 'rgba(255,255,255,0.08)' : 'var(--s1)') + ';color:#ffffff';
  var gloveStyle = 'flex:1;padding:7px 0;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;border:1px solid ' +
    (gloveActive ? 'rgba(255,255,255,0.4)' : 'var(--brd)') + ';background:' + (gloveActive ? 'rgba(255,255,255,0.08)' : 'var(--s1)') + ';color:#ffffff';

  return '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">Data Source</div>' +
    '<div style="display:flex;gap:8px">' +
    '<button style="' + camStyle + '" onclick="window._app.setTrainSource(\'camera\')">Camera</button>' +
    '<button style="' + gloveStyle + '" onclick="window._app.setTrainSource(\'glove\')">Glove / MQTT</button>' +
    '</div>' +
    '<div style="font-size:9px;color:var(--dm);margin-top:6px">' +
    (camActive ? 'Recording from webcam landmarks' : 'Recording from sensor glove via MQTT') +
    '</div>' +
    '</div>';
}

function _renderSectionTabs(section) {
  var tabs = ['alphabet', 'numbers', 'words', 'custom'];
  var tabLabels = { alphabet: 'A–Z', numbers: '0–9', words: 'Words', custom: 'Custom' };
  var html = '<div class="section-tabs">';
  for (var ti = 0; ti < tabs.length; ti++) {
    var s = tabs[ti];
    html += '<button class="tab' + (section === s ? ' active' : '') + '" onclick="window._app.setTrainSection(\'' + s + '\')">' + tabLabels[s] + '</button>';
  }
  return html + '</div>';
}

function _renderSessionHistory(histStats) {
  var totalCaptures = _get(histStats, 'totalCaptures', 0);
  var totalTrains   = _get(histStats, 'totalTrains',   0);
  var newSince      = _get(histStats, 'newSamplesSinceLastTrain', 0);
  var lastTrain     = _get(histStats, 'lastTrainAt', null);

  return '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">Session History</div>' +
    '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px">' +
    '<div style="text-align:center"><div style="font-size:20px;font-weight:700;color:#ffffff">' + totalCaptures + '</div><div style="font-size:8px;color:var(--mx)">Captures</div></div>' +
    '<div style="text-align:center"><div style="font-size:20px;font-weight:700;color:#ffffff">' + totalTrains + '</div><div style="font-size:8px;color:var(--mx)">Trains</div></div>' +
    '<div style="text-align:center"><div style="font-size:20px;font-weight:700;color:#ffffff">' + newSince + '</div><div style="font-size:8px;color:var(--mx)">New Since</div></div>' +
    (lastTrain
      ? '<div style="text-align:center"><div style="font-size:10px;font-weight:600;color:var(--tx)">' + new Date(lastTrain).toLocaleDateString() + '</div><div style="font-size:8px;color:var(--mx)">Last Trained</div></div>'
      : '') +
    '</div>' +
    Btn('View History', 'window._app.loadSessionHistory()', 'o', 'sm') +
    '<div id="historyPanel" style="display:none;margin-top:10px;max-height:200px;overflow-y:auto"></div>' +
    '</div>';
}

function _renderNlpCard(nlpStats) {
  var nlpCorpus   = (nlpStats.corpus_size !== undefined) ? nlpStats.corpus_size : 0;
  var nlpPersonal = nlpStats.personal_model_active ? true : false;

  return '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">Adaptive NLP ' +
    (nlpPersonal
      ? '<span class="bg bg-g" style="font-size:8px">Personal Model Active</span>'
      : '<span class="bg bg-d" style="font-size:8px">Building…</span>') +
    '</div>' +
    '<div style="font-size:9px;color:var(--mx);margin-bottom:8px">NLP learns from your signing history and personalises word suggestions over time.</div>' +
    '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:10px">' +
    '<div style="text-align:center;padding:8px;background:var(--s1);border-radius:8px">' +
    '<div style="font-size:18px;font-weight:700;color:#ffffff">' + nlpCorpus + '</div>' +
    '<div style="font-size:8px;color:var(--mx)">Sentences Learned</div>' +
    '</div>' +
    '<div style="text-align:center;padding:8px;background:var(--s1);border-radius:8px">' +
    '<div style="font-size:18px;font-weight:700;color:#ffffff">' + (nlpPersonal ? 'On' : 'Off') + '</div>' +
    '<div style="font-size:8px;color:var(--mx)">Personal Model</div>' +
    '</div>' +
    '<div style="text-align:center;padding:8px;background:var(--s1);border-radius:8px">' +
    '<div style="font-size:18px;font-weight:700;color:#ffffff">' + (nlpCorpus >= 5 ? 'Ready' : nlpCorpus + '/5') + '</div>' +
    '<div style="font-size:8px;color:var(--mx)">Min 5 to Activate</div>' +
    '</div>' +
    '</div>' +
    '<div style="font-size:9px;color:var(--dm)">Speak sentences using the Recognize tab to train the personal model. Retrains automatically every 3 new sentences.</div>' +
    '</div>';
}

// ── Gesture grid renderers ────────────────────────────────────────────────────

function _renderAlpha(meta, readiness, guided) {
  var html = '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">Alphabet A–Z <span class="flex"></span>' +
    Btn('Guided', 'window._app.startGuidedMode(\'alphabet\')', 'p', 'sm') +
    '</div>' +
    '<div style="font-size:9px;color:var(--dm);margin-bottom:10px">Capture = static · Record = dynamic motion (J, Z)</div>' +
    '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(88px,1fr));gap:5px">';
  for (var i = 0; i < ALPHA.length; i++) {
    var lt = ALPHA[i];
    html += _card(lt, meta, readiness, DYN_LETTERS[lt] ? 'dynamic' : 'static', guided === lt);
  }
  return html + '</div></div>';
}

function _renderNums(meta, readiness) {
  var html = '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">Numbers 0–9</div>' +
    '<div style="font-size:9px;color:var(--dm);margin-bottom:10px">All digits are static hand poses.</div>' +
    '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(88px,1fr));gap:5px">';
  for (var i = 0; i < NUMS.length; i++) {
    html += _card(NUMS[i], meta, readiness, 'static', false);
  }
  return html + '</div></div>';
}

function _renderWords(meta, readiness, gestures, customs) {
  var alphaSet  = {};
  var numSet    = {};
  var customSet = {};
  var words = [];
  var i;
  for (i = 0; i < ALPHA.length;   i++) alphaSet[ALPHA[i]]   = 1;
  for (i = 0; i < NUMS.length;    i++) numSet[NUMS[i]]       = 1;
  for (i = 0; i < customs.length; i++) customSet[customs[i]] = 1;
  for (i = 0; i < gestures.length; i++) {
    var g = gestures[i];
    if (!alphaSet[g] && !numSet[g] && !customSet[g]) words.push(g);
  }

  var html = '<div class="cd" style="margin-bottom:12px"><div class="cd-label">Word Gestures</div>';
  if (words.length === 0) {
    html += '<div style="font-size:10px;color:var(--dm);padding:10px 0">No word gestures yet.</div>';
  } else {
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:6px">';
    for (i = 0; i < words.length; i++) html += _wideCard(words[i], meta, readiness, false);
    html += '</div>';
  }
  return html + '</div>';
}

function _renderCustom(meta, readiness, customs) {
  var html = '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">Custom Gestures</div>' +
    '<div style="font-size:9px;color:var(--dm);margin-bottom:10px">Add your own static pose or dynamic motion gesture.</div>' +
    '<div style="display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap">' +
    '<input class="inp" id="newGInput" placeholder="Gesture name…" style="flex:1;min-width:100px" ' +
    'onkeydown="if(event.key===\'Enter\'){window._app.addCustomGesture(this.value,document.getElementById(\'newGType\').value);this.value=\'\'}">' +
    '<select id="newGType" class="inp" style="width:auto;padding:9px 8px">' +
    '<option value="static">Static (pose)</option>' +
    '<option value="dynamic">Dynamic (motion)</option>' +
    '</select>' +
    Btn('+ Add', 'window._app.addCustomGesture(document.getElementById(\'newGInput\').value,document.getElementById(\'newGType\').value);document.getElementById(\'newGInput\').value=\'\'', 'g', 'sm') +
    '</div>';

  if (customs.length === 0) {
    html += '<div style="font-size:10px;color:var(--dm);text-align:center;padding:10px 0">No custom gestures yet. Add one above!</div>';
  } else {
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px">';
    for (var i = 0; i < customs.length; i++) html += _wideCard(customs[i], meta, readiness, true);
    html += '</div>';
  }
  return html + '</div>';
}

// ── Gesture card components ───────────────────────────────────────────────────

function _card(name, meta, readiness, defaultType, isGuided) {
  var m = meta[name] || { static: 0, dynamic: 0, static_quality: 0, dynamic_quality: 0 };
  var r = readiness[name] || { ready: false, needed: 0, type: defaultType };
  var t = r.type || defaultType;
  var isDyn = t === 'dynamic';
  var src    = window._trainSource || 'camera';
  var camKey = isDyn ? 'dynamic_camera' : 'static_camera';
  var gloveKey = isDyn ? 'dynamic_glove' : 'static_glove';
  var count  = src === 'camera' ? (m[camKey] || 0) : (m[gloveKey] || 0);
  var total  = isDyn ? (m.dynamic || 0) : (m.static || 0);
  var target = isDyn ? 5 : 10;
  var q      = isDyn ? (m.dynamic_quality || 0) : (m.static_quality || 0);
  var ready  = r.ready;

  return '<div style="padding:8px;background:var(--s1);border-radius:8px;border:1px solid var(--brd);text-align:center">' +
    '<div style="font-size:18px;font-weight:800;margin-bottom:2px">' + name + '</div>' +
    '<span class="bg ' + (isDyn ? 'bg-p' : 'bg-g') + '" style="font-size:7px">' + (isDyn ? 'DYN' : 'STA') + '</span>' +
    '<div style="display:flex;align-items:center;gap:4px;margin:4px 0">' +
    pBar(count, target) +
    '<span style="font-size:8px;color:var(--mx);white-space:nowrap">' + count + '/' + target + (total > count ? ' (' + total + ' tot)' : '') + '</span>' +
    '</div>' +
    (count > 0
      ? '<div style="font-size:7px;color:' + qColor(q) + ';margin-bottom:4px">' + qLabel(q) + '</div>'
      : '<div style="height:14px"></div>') +
    '<div style="display:flex;gap:3px;justify-content:center">' +
    (isDyn
      ? '<button style="padding:4px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--s2);color:#ffffff;border:1px solid var(--brd);border-radius:6px;cursor:pointer" onclick="window._app.addSample(\'' + esc(name) + '\',\'dynamic\')">Record</button>'
      : '<button style="padding:4px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--s2);color:#ffffff;border:1px solid var(--brd);border-radius:6px;cursor:pointer" onclick="window._app.addSample(\'' + esc(name) + '\',\'static\')">+1</button>' +
        '<button style="padding:4px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--s2);color:#ffffff;border:1px solid rgba(255,255,255,0.25);border-radius:6px;cursor:pointer" onclick="window._app.addBulkSamples(\'' + esc(name) + '\',5)">×5</button>') +
    (count > 0
      ? '<button style="padding:4px 6px;font-size:9px;font-weight:700;font-family:inherit;background:var(--s2);color:#ffffff;border:1px solid var(--brd);border-radius:6px;cursor:pointer" onclick="window._app.deleteGestureSamples(\'' + esc(name) + '\',\'' + t + '\')">Del</button>'
      : '') +
    '</div>' +
    '</div>';
}

function _wideCard(name, meta, readiness, isCustom) {
  var m = meta[name] || { static: 0, dynamic: 0, static_quality: 0, dynamic_quality: 0 };
  var r = readiness[name] || { ready: false, needed: 0, type: 'static' };
  var isDyn  = r.type === 'dynamic';
  var sc     = m.static  || 0;
  var dc     = m.dynamic || 0;
  var ready  = r.ready;
  var needed = r.needed || 0;

  return '<div style="padding:10px 12px;background:var(--s1);border-radius:8px;border:1px solid var(--brd)">' +
    '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px">' +
    '<div>' +
    '<span style="font-size:13px;font-weight:700">' + name + '</span> ' +
    '<span class="bg ' + (isDyn ? 'bg-p' : 'bg-g') + '" style="font-size:7px">' + (isDyn ? 'DYN' : 'STA') + '</span>' +
    '</div>' +
    (isCustom
      ? '<button style="padding:3px 7px;font-size:9px;font-weight:700;font-family:inherit;background:var(--s2);color:#ffffff;border:1px solid var(--brd);border-radius:5px;cursor:pointer" onclick="window._app.deleteGesture(\'' + esc(name) + '\')">Remove</button>'
      : '') +
    '</div>' +
    '<div style="font-size:8px;color:var(--mx);margin-bottom:5px">' + sc + ' static · ' + dc + ' dynamic</div>' +
    '<div style="display:flex;align-items:center;gap:4px;margin-bottom:6px">' +
    pBar(isDyn ? dc : sc, isDyn ? 5 : 10) +
    '<span style="font-size:8px;color:var(--mx);white-space:nowrap">' + (ready ? 'Ready' : 'Need ' + needed) + '</span>' +
    '</div>' +
    '<div style="display:flex;gap:4px;flex-wrap:wrap">' +
    '<button style="padding:5px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--s2);color:#ffffff;border:1px solid var(--brd);border-radius:6px;cursor:pointer" onclick="window._app.addSample(\'' + esc(name) + '\',\'static\')">+1</button>' +
    '<button style="padding:5px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--s2);color:#ffffff;border:1px solid rgba(255,255,255,0.25);border-radius:6px;cursor:pointer" onclick="window._app.addBulkSamples(\'' + esc(name) + '\',5)">×5</button>' +
    '<button style="padding:5px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--s2);color:#ffffff;border:1px solid var(--brd);border-radius:6px;cursor:pointer" onclick="window._app.addSample(\'' + esc(name) + '\',\'dynamic\')">Record</button>' +
    (sc + dc > 0
      ? '<button style="padding:5px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--s2);color:#ffffff;border:1px solid var(--brd);border-radius:6px;cursor:pointer" onclick="window._app.deleteGestureSamples(\'' + esc(name) + '\',\'all\')">Delete</button>'
      : '') +
    '</div>' +
    '</div>';
}


// ═══ views/SettingsView.js ═══
// views/SettingsView.js
function renderSettingsTab(state) {
  var tts       = state.tts       || { enabled: true, auto: false, rate: 1.0 };
  var confThresh = state.confThresh || 0.65;
  var apiStatus = state.apiStatus || 'none';
  var mqttConnected = state.mqttConnected || false;
  var mqttBroker    = state.mqttBroker    || '';
  var mqttTopic     = state.mqttTopic     || '';

  return (
    _renderGeminiCard(apiStatus) +
    _renderMQTTCard(mqttConnected, mqttBroker, mqttTopic) +
    _renderCameraCard(state.camActive) +
    _renderTTSCard(tts) +
    _renderRecognitionCard(confThresh) +
    _renderBackupCard() +
    _renderAdminCard()
  );
}

function _renderGeminiCard(apiStatus) {
  var isConnected = apiStatus === 'ok';
  var borderColor = isConnected ? 'rgba(255,255,255,.15)' : 'transparent';

  return Card(
    'Gemini AI',
    '<div style="font-size:11px;color:var(--mx);line-height:1.6;margin-bottom:10px">' +
      'Next-word suggestions, grammar correction and sentence completion — powered by Gemini.' +
    '</div>' +
    SettingRow(
      isConnected ? 'Active' : 'Unavailable',
      isConnected
        ? 'Server-side key configured — all calls are proxied securely'
        : 'Set GEMINI_API_KEY env var on the server to enable',
      isConnected ? Badge('ON', 'g') : Badge('OFF', 'd')
    ),
    'border-color:' + borderColor
  );
}

function _renderMQTTCard(connected, broker, topic) {
  var statusDot = connected
    ? '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:rgba(255,255,255,0.8);margin-right:5px"></span>'
    : '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:rgba(255,255,255,0.2);margin-right:5px"></span>';

  return Card(
    'MQTT Publish',
    '<div style="font-size:11px;color:var(--mx);line-height:1.6;margin-bottom:10px">' +
      'Publish recognized gestures to any MQTT subscriber — Raspberry Pi, Arduino, another phone.' +
    '</div>' +
    SettingRow(
      statusDot + (connected ? 'Connected' : 'Disconnected'),
      connected ? broker : 'HiveMQ public broker',
      connected
        ? Btn('Disconnect', 'window._app.disconnectMQTT()', 'r', 'sm')
        : Btn('Connect', 'window._app.connectMQTT()', 'g', 'sm')
    ) +
    (connected
      ? '<div style="font-size:9px;color:var(--dm);margin-top:6px">Topic: <span style="color:#ffffff">' + topic + '</span></div>' +
        '<div style="font-size:10px;color:#ffffff;padding:6px 10px;background:rgba(255,255,255,0.04);border-radius:6px;margin-top:8px">Publishing gestures to broker</div>'
      : '<div style="font-size:9px;color:var(--dm);margin-top:8px">' +
          'Subscribe anywhere: <code style="color:#ffffff">mosquitto_sub -h broker.hivemq.com -t "' + (topic || 'gesture-detection/results/gesture') + '"</code>' +
        '</div>')
  );
}

function _renderCameraCard(camActive) {
  return Card(
    'Camera',
    SettingRow(
      'Live camera',
      'MediaPipe Holistic — hands + face + body',
      Toggle(camActive, camActive ? 'window._app.stopCamera()' : 'window._app.startCamera()')
    ) +
    '<div style="font-size:9px;color:var(--dm);margin-top:6px">41 features: hand×11×2 + face×8 + pose×6 + flags×3</div>'
  );
}

function _renderTTSCard(tts) {
  return Card(
    'Text-to-Speech',
    SettingRow('Enable TTS', '', Toggle(tts.enabled, 'window._app.setTTSEnabled(' + !tts.enabled + ')')) +
    SettingRow('Auto-speak on gesture', '', Toggle(tts.auto, 'window._app.setAutoSpeak(' + !tts.auto + ')')) +
    SettingRow(
      'Speed: ' + tts.rate.toFixed(1) + '×',
      '',
      '<input type="range" min=".5" max="2" step=".1" value="' + tts.rate + '" ' +
      'oninput="window._app.setTTSRate(parseFloat(this.value))" style="width:100px;accent-color:#ffffff">'
    )
  );
}

function _renderRecognitionCard(confThresh) {
  return Card(
    'Recognition',
    SettingRow(
      'Confidence: ' + Math.round(confThresh * 100) + '%',
      'Minimum confidence before a gesture is accepted',
      '<input type="range" min=".3" max=".95" step=".05" value="' + confThresh + '" ' +
      'oninput="window._app.setConfThreshold(parseFloat(this.value))" style="width:100px;accent-color:#ffffff">'
    ) +
    '<div style="font-size:9px;color:var(--dm);margin-top:6px">' +
      'Letter hold 600ms · Word hold 900ms · Same letter cooldown 1.2s' +
    '</div>'
  );
}

function _renderBackupCard() {
  return Card(
    'Data Backup',
    '<div style="font-size:11px;color:var(--mx);line-height:1.6;margin-bottom:12px">' +
      'Export all training samples and settings to a JSON file. Import on any device.' +
    '</div>' +
    '<div class="fr g6 mb8">' +
      Btn('Export', 'window._app.exportDataset()', 'g') +
      Btn('Import', 'document.getElementById(\'importFileInput\').click()', 'o') +
    '</div>' +
    '<input type="file" id="importFileInput" accept=".json" style="display:none" ' +
      'onchange="window._app.importDataset(this.files[0]);this.value=\'\'">'
  );
}

function _renderAdminCard() {
  return Card(
    'Admin',
    SettingRow(
      'Change PIN',
      '',
      '<div class="fr g6">' +
        '<input class="inp" style="width:80px" id="newPin" placeholder="PIN" maxlength="6" type="password">' +
        '<button class="btn btn-o btn-sm" onclick="' +
          'var p=document.getElementById(\'newPin\').value.trim();' +
          'if(!p){alert(\'Enter a PIN first\');}' +
          'else{window._app.setAdminPin(p);document.getElementById(\'newPin\').value=\'\';alert(\'PIN updated\')}' +
        '">Set</button>' +
      '</div>'
    ) +
    SettingRow('Exit Admin', '', Btn('User Mode', "window._app.switchMode('user')", 'o', 'sm'))
  );
}


// ═══ views/UserModeView.js ═══
// views/UserModeView.js — Clean end-user interface
function renderUserMode(state) {
  var {
    camActive, cameraError, running, displayText, spelling,
    suggestions, wordSuggestions, completion, contextState,
    staticTrained, dynamicTrained, inputMode
  } = state;

  var trained = staticTrained || dynamicTrained;

  return (
    _renderUserHeader(state) +
    _renderUserSettings(state, inputMode) +
    _renderUserCamera(camActive, cameraError, trained, running) +
    _renderUserSentence(state, displayText, spelling, suggestions, wordSuggestions, completion, contextState) +
    _renderUserFingers()
  );
}

function _renderUserHeader(state) {
  return '<div style="padding:14px 0 10px;display:flex;align-items:center;justify-content:space-between">' +
    '<div>' +
      '<div style="font-size:11px;color:var(--g);font-weight:600;margin-bottom:3px">gesture detection</div>' +
      '<div style="font-size:16px;font-weight:700">Sign to <span style="color:var(--g)">Communicate</span></div>' +
    '</div>' +
    '<button class="btn btn-o btn-sm" ' +
      'onclick="document.getElementById(\'userSettings\').style.display=' +
        'document.getElementById(\'userSettings\').style.display===\'none\'?\'block\':\'none\'" ' +
      '>Settings</button>' +
  '</div>';
}

function _renderUserSettings(state, inputMode) {
  return '<div id="userSettings" style="display:none" class="cd">' +
    '<div class="srow"><div><div class="srow-label">Speech rate</div></div>' +
      '<input type="range" min=".5" max="2" step=".1" value="' + state.tts.rate + '" ' +
        'oninput="window._app.setTTSRate(parseFloat(this.value))" ' +
        'style="width:100px;accent-color:var(--g)">' +
    '</div>' +
    '<div class="srow"><div><div class="srow-label">Input source</div></div>' +
      '<select onchange="window._app.setInputMode(this.value)" ' +
        'style="background:var(--s1);color:var(--tx);border:1px solid var(--brd);border-radius:6px;padding:4px 8px;font-family:inherit;font-size:11px">' +
        '<option value="camera" ' + (inputMode === 'camera' ? 'selected' : '') + '>Camera</option>' +
        '<option value="glove"  ' + (inputMode === 'glove'  ? 'selected' : '') + '>Glove</option>' +
        '<option value="both"   ' + (inputMode === 'both'   ? 'selected' : '') + '>Both</option>' +
      '</select>' +
    '</div>' +
    '<div style="padding-top:10px;margin-top:4px">' +
      '<button class="btn btn-o btn-sm" onclick="' +
        'var pin=prompt(\'Enter admin PIN:\');' +
        'if(pin&&window._app.checkAdminPin(pin)){window._app.switchMode(\'admin\')}' +
        'else if(pin){alert(\'Wrong PIN\')}' +
      '">Admin</button>' +
    '</div>' +
  '</div>';
}

function _renderUserCamera(camActive, cameraError, trained, running) {
  return '<div class="cd" style="padding:0;overflow:hidden;position:relative;margin-bottom:10px">' +
    '<div class="vid-wrap" style="min-height:280px">' +
      '<div id="vidContainer" style="width:100%;height:100%"></div>' +
      '<div class="vid-badges">' +
        '<span class="bg bg-g" id="fpsB">-- FPS</span>' +
        '<span class="bg bg-d" id="handB">No Hand</span>' +
      '</div>' +
      '<div class="vid-gesture" id="gestDisp" style="display:none">' +
        '<div class="gesture-name" id="gestName"></div>' +
        '<div class="gesture-conf" id="gestConf"></div>' +
      '</div>' +
      '<div class="vid-gesture" id="auxGestDisp" style="display:none;bottom:60px;font-size:0.75em;opacity:0.75">' +
        '<div class="gesture-name" id="auxGestName" style="font-size:18px"></div>' +
        '<div class="gesture-conf" id="auxGestConf"></div>' +
      '</div>' +
      (!camActive ? '<div class="vid-overlay"><div style="font-size:11px;letter-spacing:.1em">TAP START</div></div>' : '') +
    '</div>' +
    '<div style="height:3px;background:var(--s2)"><div id="sysGestProg" style="height:100%;width:0%;background:var(--g);transition:width .1s"></div></div>' +
  '</div>' +

  '<div style="display:flex;gap:8px;justify-content:center;margin-bottom:12px;flex-wrap:wrap">' +
    (camActive
      ? '<button class="btn btn-r" onclick="window._app.stopCamera()">Stop Camera</button>'
      : '<button class="btn btn-o" onclick="window._app.startCamera()">' + (cameraError ? 'Retry Camera' : 'Start Camera') + '</button>') +
    (camActive ? '<button class="btn btn-o" onclick="window._app.switchCamera()">Flip Camera</button>' : '') +
    (cameraError ? '<div style="font-size:10px;color:var(--r);padding:7px 12px;background:var(--rD);border-radius:8px;border:1px solid var(--r)">' + cameraError + '</div>' : '') +
    (trained && !running ? '<button class="btn btn-g" onclick="window._app.startRecognition()">Recognize</button>' : '') +
    (running ? '<button class="btn btn-r" onclick="window._app.stopRecognition()">Stop</button>' : '') +
  '</div>';
}

function _renderUserSentence(state, displayText, spelling, suggestions, wordSuggestions, completion, contextState) {
  return '<div class="cd">' +
    '<div style="background:var(--bg);border:1px solid var(--brd);border-radius:8px;padding:14px 16px;' +
      'min-height:52px;font-size:19px;font-weight:600;line-height:1.5;margin-bottom:12px;' +
      'color:' + (displayText ? 'var(--tx)' : 'var(--dm)') + '">' +
      (displayText || 'Start signing to communicate…') +
      '<span class="cursor"></span>' +
    '</div>' +

    (spelling ? '<div style="font-size:10px;color:var(--p);letter-spacing:.1em;margin-bottom:6px">SPELLING: ' + spelling.toUpperCase() + '_</div>' : '') +

    (wordSuggestions.length
      ? '<div style="margin-bottom:8px">' +
          '<div style="font-size:9px;color:var(--mx);margin-bottom:4px;letter-spacing:.1em">WORD MATCHES</div>' +
          '<div class="suggs">' +
            wordSuggestions.map(function(w, i) {
              return '<span class="sugg ai"><span style="font-size:8px;opacity:.6">' + (i + 1) + '.</span> ' + w + '</span>';
            }).join('') +
          '</div>' +
        '</div>'
      : '') +

    (!spelling && suggestions.length
      ? '<div>' +
          '<div style="font-size:9px;color:var(--mx);margin-bottom:4px;letter-spacing:.1em">' +
            (state.geminiEnabled ? 'AI' : 'LOCAL') + ' PREDICTIONS' +
          '</div>' +
          '<div class="suggs">' +
            suggestions.map(function(w, i) {
              return '<span class="sugg' + (state.geminiEnabled ? ' ai' : '') + '">' +
                '<span style="font-size:8px;opacity:.6">' + (i + 1) + '.</span> ' + w +
              '</span>';
            }).join('') +
          '</div>' +
        '</div>'
      : '') +

    (completion && completion !== state.sentence
      ? '<div class="completion" onclick="window._app.acceptCompletion()">' +
          '<div class="completion-label">AI suggests</div>' +
          '<div class="completion-text">"' + completion + '"</div>' +
        '</div>'
      : '') +

    '<div style="display:flex;gap:8px;align-items:center;justify-content:space-between;margin-top:8px">' +
      '<span style="font-size:9px;color:var(--dm);letter-spacing:.1em">' + contextState + ' ' + (state.geminiEnabled ? '· Gemini AI' : '') + '</span>' +
      '<div style="font-size:9px;color:var(--dm)">Fist=Speak · Palm=Clear · Thumbs down=Undo</div>' +
    '</div>' +
  '</div>';
}

function _renderUserFingers() {
  return '<div class="cd">' + FingerBars(APP_CONFIG.FINGER_NAMES, APP_CONFIG.FINGER_COLORS) + '</div>';
}


// ═══ controllers/RecognitionController.js ═══
// controllers/RecognitionController.js — Gesture Detection v1.0
// Optimisations:
//   1. Rate limiting: predict every Nth frame (not every frame)
//   2. Confidence smoothing: rolling average over last N frames
//   3. Time-based stability: hold gesture for N ms (not N frames)
//   4. Separate cooldowns: letter vs word vs same-letter
//   5. Motion detection: LSTM only activates on movement
//   6. NLP debounce: wait 300ms before fetching suggestions
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


// ═══ controllers/TrainingController.js ═══
// controllers/TrainingController.js — Gesture Detection v1.0
// Camera only, 41-feature Holistic vectors — v6.1 Phase 2A
// Mirror augmentation: saves mirrored copy on every static capture
// LSTM abort: if hand lost mid dynamic recording, abort cleanly
// No optional-chaining — Safari 12 safe
class TrainingController {
  constructor(staticNN, dynamicNN, gestureModel, sensorModel) {
    this.staticNN   = staticNN;
    this.dynamicNN  = dynamicNN;
    this.gm         = gestureModel;
    this.sensor     = sensorModel;
    this.isTraining = false;
    this.progress   = 0;
    this._camera    = null; // set by AppController after camera init

    // LSTM abort tracking
    this._dynLostFrames  = 0;
    this._dynAborted     = false;
  }

  // ── Attach camera reference (needed for mirror augmentation) ──
  setCamera(camera) { this._camera = camera; }

  // ── Countdown (3-2-1) ─────────────────────────────────────────
  async runCountdown(secs) {
    secs = secs || 3;
    for (var i = secs; i > 0; i--) {
      window._countdown = i;
      eventBus.emit(Events.STATE_UPDATED);
      await new Promise(function(r){ setTimeout(r, 1000); });
    }
    window._countdown = 0;
    eventBus.emit(Events.STATE_UPDATED);
  }

  // ── Static single sample capture + mirror augmentation ────────
  async collectStaticSample(name, skipCountdown) {
    if (!skipCountdown) await this.runCountdown(3);

    var useLive = this.sensor && this.sensor.handDetected &&
      (this.sensor.source === 'camera' || this.sensor.source === 'ble' || this.sensor.source === 'sim');

    var sample, mirroredSample;

    if (useLive) {
      sample = this.sensor.getFeatureVector().slice();

      // Mirror augmentation — only possible with camera (landmarks available)
      if (this._camera && this._camera._lastHandsData && this._camera._lastHandsData.length > 0) {
        mirroredSample = this._camera.getMirroredFeatures(this._camera._lastHandsData);
      }
    } else {
      // Simulate — backend generates the 24-feature vector
      var res  = await fetch(API + '/samples/simulate', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ gesture:name, count:1, sample_type:'static', source:'camera' })
      });
      var data = await res.json();
      sample   = data.samples && data.samples[0];
      // For simulated data, generate a mirrored version too
      if (data.mirrored && data.mirrored.length > 0) {
        mirroredSample = data.mirrored[0];
      }
    }

    // Guard: if no sample was produced (hand absent + simulation returned empty), bail out
    if (!sample || !sample.length) {
      eventBus.emit(Events.SAMPLES_COLLECTED, { gesture:name, live:false, count:0 });
      return { live:false, sample:null, mirrored:false };
    }

    // Save original sample
    await this.gm.addStaticSamples(name, [sample], false, window._trainSource||'camera');

    // Save mirrored sample if available (doubles training data for free)
    if (mirroredSample) {
      await this.gm.addStaticSamples(name, [mirroredSample], true, window._trainSource||'camera');
    }

    eventBus.emit(Events.SAMPLES_COLLECTED, { gesture:name, live:useLive, count: mirroredSample ? 2 : 1 });
    return { live:useLive, sample:sample, mirrored: !!mirroredSample };
  }

  // ── Burst (5 at once) ─────────────────────────────────────────
  async collectLiveBurst(name, count) {
    count = count || 5;
    var samples = [];
    var useLive = this.sensor && this.sensor.handDetected;
    if (useLive) {
      for (var i = 0; i < count; i++) {
        samples.push(this.sensor.getFeatureVector().slice());
        await new Promise(function(r){ setTimeout(r, 80); });
      }
    } else {
      var res  = await fetch(API + '/samples/simulate', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ gesture:name, count:count, sample_type:'static', source:'camera' })
      });
      var data = await res.json();
      samples  = data.samples;
    }
    await this.gm.addStaticSamples(name, samples);
    eventBus.emit(Events.SAMPLES_COLLECTED, { gesture:name, live:useLive, count:count });
  }

  async collectStaticSamples(name, count) { return this.collectLiveBurst(name, count || 5); }

  // ── Bulk capture (one countdown, N samples spaced 350ms apart) ───────
  async collectBulkStatic(name, count) {
    count = count || 5;
    await this.runCountdown(3);
    for (var i = 0; i < count; i++) {
      await this.collectStaticSample(name, true);
      if (i < count - 1) await new Promise(function(r){ setTimeout(r, 350); });
    }
  }

  // ── Dynamic recording — with abort on hand loss ───────────────
  startDynamicRecording(name) {
    this._dynLostFrames = 0;
    this._dynAborted    = false;
    this.gm.startRecording(name);
    eventBus.emit(Events.RECORDING_START, { gesture:name });
  }

  // Called every frame by AppController during dynamic recording
  // Returns: 'done' | 'aborted' | 'recording'
  pushRecordingFrame(features, handCount) {
    if (this._dynAborted) return 'aborted';

    // Track hand loss — abort if lost too many consecutive frames
    var ABORT_THRESHOLD = APP_CONFIG.MEDIAPIPE.HAND_LOSS_ABORT_FRAMES || 3;
    if (handCount === 0) {
      this._dynLostFrames++;
      if (this._dynLostFrames >= ABORT_THRESHOLD) {
        this._dynAborted = true;
        this.gm.stopRecording();
        eventBus.emit(Events.RECORDING_DONE, { gesture: this.gm.recordTarget, aborted: true });
        return 'aborted';
      }
      // Don't push zero frames — hold last frame
      return 'recording';
    } else {
      this._dynLostFrames = 0; // reset on hand re-detected
    }

    var done = this.gm.pushFrame(features);
    eventBus.emit(Events.RECORDING_TICK, {
      frame: this.gm.frameBuffer.length,
      total: APP_CONFIG.NN.DYNAMIC_FRAMES,
    });
    if (done) {
      eventBus.emit(Events.RECORDING_DONE, { gesture: this.gm.recordTarget, aborted: false });
      return 'done';
    }
    return 'recording';
  }

  addGesture(name)    { return this.gm.addGesture(name); }
  removeGesture(name) { this.gm.removeGesture(name); }

  // ── Training (Unified Bidirectional LSTM — static + dynamic) ─────────
  async train() {
    this.isTraining = true;
    this.progress   = 0;
    eventBus.emit(Events.TRAIN_STARTED);

    // Load only samples for the active source
    this.gm.trainSource = window._trainSource || 'camera';
    await this.gm.loadFromDB();

    var sd = this.gm.getStaticTrainingData();
    var dd = this.gm.getDynamicTrainingData();

    if (sd.inputs.length === 0 && dd.inputs.length === 0) {
      this.isTraining = false;
      eventBus.emit(Events.TRAIN_COMPLETE, { error:'No samples' });
      return;
    }

    // ── Build unified gesture list (static first, then dynamic-only) ────
    var allNames = sd.names.slice();
    for (var di = 0; di < dd.names.length; di++) {
      if (allNames.indexOf(dd.names[di]) === -1) allNames.push(dd.names[di]);
    }

    if (allNames.length < 2) {
      this.isTraining = false;
      eventBus.emit(Events.TRAIN_COMPLETE, { error:'Need at least 2 gestures with samples' });
      return;
    }

    // Map name → unified index
    var nameToIdx = {};
    for (var ni = 0; ni < allNames.length; ni++) nameToIdx[allNames[ni]] = ni;

    // Which gestures are primarily static vs dynamic
    var gestureTypes = {};
    for (var si2 = 0; si2 < sd.names.length; si2++) gestureTypes[sd.names[si2]] = 'static';
    for (var di2 = 0; di2 < dd.names.length; di2++) gestureTypes[dd.names[di2]] = 'dynamic';

    // ── Build combined inputs + labels ───────────────────────────────────
    // Static inputs:  41-feature vectors — backend repeats to fill sequence
    // Dynamic inputs: DYNAMIC_FRAMES*41-feature flat sequences
    var allInputs = [];
    var allLabels = [];
    for (var si3 = 0; si3 < sd.inputs.length; si3++) {
      allInputs.push(sd.inputs[si3]);
      allLabels.push(nameToIdx[sd.names[sd.labels[si3]]]);
    }
    for (var di3 = 0; di3 < dd.inputs.length; di3++) {
      allInputs.push(dd.inputs[di3]);
      allLabels.push(nameToIdx[dd.names[dd.labels[di3]]]);
    }

    // ── Single init call ─────────────────────────────────────────────────
    await fetch(API + '/nn/init', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        model_type:    'unified',
        output_size:   allNames.length,
        gestures:      allNames,
        gesture_types: gestureTypes,
        feature_version: APP_CONFIG.NN.FEATURE_VERSION,
      })
    });

    // ── Train unified BiLSTM — single call, backend handles SWA + warmup ────
    var EPOCHS = APP_CONFIG.NN.TRAINING_EPOCHS;
    var lr     = (dd.inputs.length > 0) ? 0.001 : 0.008;

    // Progress updates come via WebSocket train_progress events (handled in AppController).
    // A single fetch lets the backend run the full training pipeline uninterrupted.
    var res  = await fetch(API + '/nn/train', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ model_type:'unified', inputs:allInputs, labels:allLabels, epochs:EPOCHS, lr:lr })
    });
    var data = await res.json();

    // Mirror final stats to both frontend NN objects
    this.staticNN.accuracy      = data.accuracy;
    this.staticNN.val_accuracy  = data.val_accuracy || 0;
    this.staticNN.loss          = data.loss;
    this.staticNN.epochs        = data.epochs;
    this.staticNN.trained       = true;
    this.dynamicNN.accuracy     = data.accuracy;
    this.dynamicNN.val_accuracy = data.val_accuracy || 0;
    this.dynamicNN.loss         = data.loss;
    this.dynamicNN.epochs       = data.epochs;
    this.dynamicNN.trained      = true;

    if (!window._perGestAcc) window._perGestAcc = {};
    window._perGestAcc.static  = data.per_gesture_acc || {};
    window._perGestAcc.dynamic = data.per_gesture_acc || {};

    await fetch(API + '/nn/save/unified?source=' + (window._trainSource||'camera'), { method:'POST' });

    this.isTraining = false;
    this.progress   = 100;
    eventBus.emit(Events.TRAIN_COMPLETE, {
      staticAcc:  this.staticNN.accuracy,
      dynamicAcc: this.dynamicNN.accuracy,
    });

    try {
      var pa = await (await fetch(API + '/nn/per_gesture_accuracy')).json();
      window._perGestAcc = pa;
    } catch(e) {}
  }

  // ── Delete / Reset ────────────────────────────────────────────
  async deleteModel() {
    await fetch(API + '/nn/reset_all', { method:'POST' });
    this.staticNN.reset();
    this.dynamicNN.reset();
    this.isTraining = false;
    this.progress   = 0;
    window._perGestAcc = { static:{}, dynamic:{} };
    eventBus.emit(Events.STATE_UPDATED);
  }

  async resetAll() {
    this.staticNN.reset();
    this.dynamicNN.reset();
    await this.gm.resetAll();
    this.isTraining = false;
    this.progress   = 0;
    window._perGestAcc = { static:{}, dynamic:{} };
    eventBus.emit(Events.STATE_UPDATED);
  }

  getStats() {
    return {
      staticAccuracy:  this.staticNN.accuracy,
      staticLoss:      this.staticNN.loss,
      staticEpochs:    this.staticNN.epochs,
      staticTrained:   this.staticNN.trained,
      valAccuracy:     this.staticNN.val_accuracy || 0,
      dynamicAccuracy: this.dynamicNN.accuracy,
      dynamicLoss:     this.dynamicNN.loss,
      dynamicEpochs:   this.dynamicNN.epochs,
      dynamicTrained:  this.dynamicNN.trained,
      isTraining:      this.isTraining,
      progress:        this.progress,
    };
  }
}


// ═══ controllers/AppController.js ═══
// controllers/AppController.js — Gesture Detection v1.0
// Camera only, Holistic, no BLE/VR
class AppController {
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
        this.staticNN.trained  = true; this.staticNN.accuracy  = ud.accuracy||0; this.staticNN.epochs  = ud.epochs||0;
        this.dynamicNN.trained = true; this.dynamicNN.accuracy = ud.accuracy||0; this.dynamicNN.epochs = ud.epochs||0;
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
          self.staticNN.accuracy = d.accuracy; self.staticNN.loss = d.loss;
          self.staticNN.epochs = d.epochs; self.staticNN.trained = true;
          if (d.per_gesture_acc) { if (!window._perGestAcc) window._perGestAcc = {}; window._perGestAcc.static = d.per_gesture_acc; }
        }
        if (d.model === 'dynamic' || d.model === 'unified') {
          self.dynamicNN.accuracy = d.accuracy; self.dynamicNN.loss = d.loss;
          self.dynamicNN.epochs = d.epochs; self.dynamicNN.trained = true;
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
    console.log('[SignLens v6.1] Ready — Phase 2 (two-hand + adaptive NLP)');
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
    try {
      var results = await Promise.all([
        fetch(API + '/samples/meta').then(function(r){ return r.json(); }),
        fetch(API + '/gestures/readiness?source=' + (window._trainSource||'camera')).then(function(r){ return r.json(); }),
        fetch(API + '/history/stats').then(function(r){ return r.json(); }),
      ]);
      window._trainMeta  = results[0];
      window._readiness  = results[1];
      window._histStats  = results[2];
      self._refreshNlpStats();
    } catch(e) {}
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
    await this.trainCtrl.train();
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
    function showStatus(msg, color) {
      if (!statusEl) return;
      statusEl.style.display = 'block';
      statusEl.textContent   = msg;
      statusEl.style.background = color;
      statusEl.style.color   = 'var(--bg)';
      clearTimeout(statusEl._t);
      statusEl._t = setTimeout(function(){ statusEl.style.display='none'; }, 2600);
    }

    if (type === 'static') {
      showStatus('⏳ Get ready — capturing "' + gestureName + '" in 3s…', 'var(--a)');
      var result = await this.trainCtrl.collectStaticSample(gestureName, false);
      showStatus(
        result.live
          ? '✓ Live sample captured for "' + gestureName + '"'
          : '📦 Simulated sample for "' + gestureName + '" (no hand detected)',
        result.live ? 'var(--g)' : 'var(--a)'
      );
      window._lastSampledGesture = gestureName;
      await this._refreshTrainMeta();
      this.view.render();
      setTimeout(function(){ window._lastSampledGesture = null; }, 1600);

    } else if (type === 'dynamic') {
      if (!this.camera.active) {
        showStatus('⚠ Start camera first for dynamic recording!', 'var(--r)');
        return;
      }
      showStatus('⏳ Get ready — recording "' + gestureName + '" in 3s…', 'var(--a)');
      await this.trainCtrl.runCountdown(3);
      showStatus('🎬 Recording "' + gestureName + '"… perform gesture NOW!', 'var(--p)');
      this.trainCtrl.startDynamicRecording(gestureName);
      this.view.render();

      var unsub = eventBus.on(Events.FEATURES_EXTRACTED, function(data) {
        var result = self.trainCtrl.pushRecordingFrame(data.features, data.handCount || 1);
        if (result === 'aborted') {
          unsub();
          showStatus('⚠ Hand lost during recording — please try again', 'var(--r)');
          window._lastSampledGesture = null;
          self.view.render();
          return;
        }
        var done = result === 'done';
        if (done) {
          unsub();
          showStatus('✓ Dynamic sample recorded for "' + gestureName + '"!', 'var(--g)');
          window._lastSampledGesture = gestureName;
          self._refreshTrainMeta().then(function(){ self.view.render(); });
          setTimeout(function(){ window._lastSampledGesture = null; self.view.render(); }, 1600);
        }
      });
    }
  }

  async addBulkSamples(name, count) {
    var self = this;
    var statusEl = document.getElementById('trainStatus');
    function showStatus(msg, color) {
      if (!statusEl) return;
      statusEl.style.display = 'block';
      statusEl.textContent   = msg;
      statusEl.style.background = color;
      statusEl.style.color   = 'var(--bg)';
      clearTimeout(statusEl._t);
      statusEl._t = setTimeout(function(){ statusEl.style.display='none'; }, 3000);
    }
    showStatus('⏳ Get ready — capturing ' + count + '× "' + name + '" in 3s…', 'var(--a)');
    await this.trainCtrl.collectBulkStatic(name, count);
    showStatus('✓ ' + count + ' samples captured for "' + name + '"', 'var(--g)');
    await this._refreshTrainMeta();
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

