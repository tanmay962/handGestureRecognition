// config/app.config.js — Gesture Detection v1.0
// MediaPipe Holistic: hands + face + body = 41 features
// Optimised: rate limiting, confidence smoothing, time-based stability, separate cooldowns
'use strict';

var APP_CONFIG = {
  name: 'Gesture Detection',
  version: '1.0',
  DEFAULT_GESTURES: ['Hello','Thank You','Yes','No','Help','Please','Sorry','Stop','Go','Water'],

  NN: {
    STATIC_HIDDEN:       [256, 128, 64],
    DYNAMIC_HIDDEN:      [256, 128, 64],
    LEARNING_RATE:       0.008,
    TRAINING_EPOCHS:     500,   // was 300 — more epochs = better convergence
    EPOCH_BATCH:         10,
    SAMPLES_PER_GESTURE: 35,
    STATIC_INPUT:        41,
    DYNAMIC_INPUT:       1845,
    DYNAMIC_FRAMES:      45,
    FEATURE_VERSION:     '1.0',
  },

  RECOGNITION: {
    CONFIDENCE_THRESHOLD:  0.65,
    // Stability hold — longer = fewer false triggers, more deliberate
    STABLE_MS_LETTER:      750,   // was 600
    STABLE_MS_WORD:        1100,  // was 900
    STABLE_MS_NUMBER:      750,   // was 600
    // Cooldowns — space out repeated recognitions
    COOLDOWN_SAME_LETTER:  1600,  // was 1200
    COOLDOWN_DIFF_LETTER:  550,   // was 400
    COOLDOWN_WORD:         2200,  // was 1800 — prevents word bursts
    // Prediction rate limiting
    PREDICT_EVERY_N:       3,
    // Confidence smoothing — larger window = smoother, less jittery
    CONF_SMOOTH_WINDOW:    7,     // was 5
    // Motion detection threshold for LSTM activation
    MOTION_THRESHOLD:      0.08,
    DYNAMIC_CONF_THRESH:   0.75,
    DWELL_TIME:            1500,
    SPELL_PAUSE:           2200,  // was 2000
    // NLP debounce
    NLP_DEBOUNCE_MS:       400,   // was 300
    // Ensemble voting — more history = more stable argmax
    ENSEMBLE_WINDOW:       7,     // was 5
    // TTS word-queue buffer — accumulate words before speaking (ms)
    TTS_BUFFER_MS:         1800,

    // Hysteresis — lower exit threshold keeps active gesture alive during brief dips
    HYSTERESIS_EXIT:          0.45,
    // Streak gate — require this many consecutive same-gesture frames before hold timer starts
    MIN_STREAK_FRAMES:        3,
    // Rejection zone — below this confidence always treat as no gesture
    MIN_REJECT_CONF:          0.15,
    // Dynamic priority margin — dynamic model wins if within this fraction of static conf
    DYNAMIC_PRIORITY_MARGIN:  0.05,
    // Prediction trail — number of recent confirmed gestures to show
    PRED_TRAIL_SIZE:          5,
    // Adaptive cooldown — factor and window applied after user manually interacts
    ADAPTIVE_COOLDOWN_FACTOR: 0.65,
    ADAPTIVE_COOLDOWN_WINDOW: 3000,
  },

  MEDIAPIPE: {
    CDN_BASE:                'https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629',
    MODEL_COMPLEXITY:        1,
    MIN_DETECTION_CONFIDENCE:0.7,
    MIN_TRACKING_CONFIDENCE: 0.5,
    MIRROR_DISPLAY:          true,
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

  SEQUENCE: { WINDOW_SIZE:45, MAX_HISTORY:20, DEFAULT_TIMEOUT:3000 },

  TABS_ADMIN: [
    { id:'detect',    label:'Detect'    },
    { id:'train',     label:'Train'     },
    { id:'sequences', label:'Sequences' },
    { id:'settings',  label:'Settings'  },
  ],

  FINGER_NAMES:  ['Thumb','Index','Middle','Ring','Pinky'],
  FINGER_COLORS: ['#5eead4','#a78bfa','#fbbf24','#fb7185','#60a5fa'],
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
