// config/app.config.js — Gesture Detection v1.0
// MediaPipe Holistic: hands + face + body = 41 features
// Optimised: rate limiting, confidence smoothing, time-based stability, separate cooldowns
'use strict';

var APP_CONFIG = {
  name: 'Gesture Detection',
  version: '4.8',
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
    // Raised from 0.60 → 0.72 to suppress false positives from imbalanced/noisy training data
    CONFIDENCE_THRESHOLD:  0.72,
    // Stability hold — longer hold reduces jitter from frame-to-frame noise
    STABLE_MS_LETTER:      700,
    STABLE_MS_WORD:        950,
    STABLE_MS_NUMBER:      700,
    // Cooldowns — prevent duplicate confirmations
    COOLDOWN_SAME_LETTER:  1500,
    COOLDOWN_DIFF_LETTER:  500,
    // Word cooldown raised to 3s — prevents same word re-firing while TTS is still speaking it
    COOLDOWN_WORD:         3000,
    // Prediction rate limiting — every 2 frames
    PREDICT_EVERY_N:       2,
    // Confidence smoothing — 7 frames for smoother averaging (was 5)
    CONF_SMOOTH_WINDOW:    7,
    // Motion detection threshold for LSTM activation
    MOTION_THRESHOLD:      0.06,
    DYNAMIC_CONF_THRESH:   0.72,
    DWELL_TIME:            1500,
    SPELL_PAUSE:           1800,
    // NLP debounce
    NLP_DEBOUNCE_MS:       300,
    // Ensemble voting — wider window for more stable averaged probabilities (was 5)
    ENSEMBLE_WINDOW:       7,
    // TTS word-queue buffer
    TTS_BUFFER_MS:         2000,
    // TTS dedup — same word/phrase suppressed if spoken within this window (ms)
    TTS_DEDUP_MS:          5000,
    // Hysteresis — raised from 0.50 to make it harder to drop an active gesture
    HYSTERESIS_EXIT:          0.60,
    // Streak gate — require 5 consecutive same-gesture frames (was 3) — kills single-frame noise
    MIN_STREAK_FRAMES:        5,
    // Rejection zone — hard floor raised from 0.15 to 0.28 — outright discard weak predictions
    MIN_REJECT_CONF:          0.28,
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
