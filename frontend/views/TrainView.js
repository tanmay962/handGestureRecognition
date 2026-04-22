// views/TrainView.js
// No optional-chaining (?.), no nullish-coalescing (??) — Safari 12 safe.

import {APP_CONFIG} from '../config/app.config.js';
import {Card, Btn, StatBox, Bar, Badge} from './components/index.js';

var ALPHA       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
var NUMS        = '0123456789'.split('');
var DYN_LETTERS = {J: 1, Z: 1};

// ── Tiny helpers ──────────────────────────────────────────────────────────────

function _get(obj, key, fallback) {
  return (obj && obj[key] !== undefined && obj[key] !== null) ? obj[key] : fallback;
}

function qColor(q) {
  return q >= 0.8 ? 'var(--g)' : q >= 0.5 ? 'var(--a)' : 'var(--r)';
}

function qLabel(q) {
  return q >= 0.8 ? '★ Good' : q >= 0.5 ? '◐ Fair' : '▲ Low';
}

function pBar(val, max, color) {
  color = color || 'var(--g)';
  var pct = Math.min(100, Math.round((val / max) * 100));
  return '<div style="height:5px;background:var(--s1);border-radius:3px;overflow:hidden;flex:1;min-width:40px">' +
    '<div style="width:' + pct + '%;height:100%;background:' + color + ';border-radius:3px;transition:width .4s"></div></div>';
}

function esc(s) {
  return String(s).replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

// ── Main export ───────────────────────────────────────────────────────────────

export function renderTrainTab(state) {
  var trainStats  = state.trainStats  || {};
  var meta        = window._trainMeta    || {};
  var readiness   = window._readiness   || {};
  var histStats   = window._histStats   || {};
  var confThresh  = (window._confThresh !== undefined && window._confThresh !== null) ? window._confThresh : 0.65;
  var perGestAcc  = window._perGestAcc  || {static: {}, dynamic: {}};
  var section     = window._trainSection || 'alphabet';
  var guided      = window._guidedGesture || null;
  var liveP       = window._livePrediction || null;
  var countdown   = window._countdown   || 0;
  var customs     = window._customGestures || [];
  var nlpStats    = window._nlpStats    || {};
  var trainSource = window._trainSource  || 'camera';

  var isTraining = _get(trainStats, 'isTraining', false);
  var progress   = _get(trainStats, 'progress',   0);
  var sTrained   = _get(trainStats, 'staticTrained',  false);
  var dTrained   = _get(trainStats, 'dynamicTrained', false);

  var needsRetrain = _get(histStats, 'needsRetrain', false) && (sTrained || dTrained);
  var newSince     = _get(histStats, 'newSamplesSinceLastTrain', 0);

  var readyCount = 0;
  var totalCount = 0;
  var k;
  for (k in readiness) {
    totalCount++;
    if (readiness[k] && readiness[k].ready) readyCount++;
  }

  var html = _renderCameraStrip(state, liveP, countdown);

  html += '<div id="trainStatus" style="display:none;padding:8px 14px;border-radius:8px;margin-bottom:10px;font-size:11px;font-weight:600;text-align:center"></div>';

  if (needsRetrain) {
    html += _renderRetrainAlert(newSince);
  }

  html += _renderModelStats(trainStats);

  if (isTraining) {
    html += _renderTrainingProgress(progress);
  }

  html += _renderControls(isTraining, confThresh, readyCount, totalCount);
  html += _renderSourceSelector(trainSource);
  html += _renderSectionTabs(section);

  if (section === 'alphabet')  html += _renderAlpha(meta, readiness, guided);
  else if (section === 'numbers') html += _renderNums(meta, readiness);
  else if (section === 'words')   html += _renderWords(meta, readiness, state.gestures, customs);
  else                             html += _renderCustom(meta, readiness, customs);

  html += _renderPerGestureAccuracy(perGestAcc, sTrained, dTrained);
  html += _renderSessionHistory(histStats);
  html += _renderNlpCard(nlpStats);

  return html;
}

// ── Section renderers ─────────────────────────────────────────────────────────

function _renderCameraStrip(state, liveP, countdown) {
  var fingerBars = '';
  for (var fi = 0; fi < APP_CONFIG.FINGER_NAMES.length; fi++) {
    fingerBars +=
      '<div style="display:flex;align-items:center;gap:5px;margin-bottom:2px">' +
        '<span style="font-size:8px;width:36px;color:' + APP_CONFIG.FINGER_COLORS[fi] + ';font-weight:600">' + APP_CONFIG.FINGER_NAMES[fi] + '</span>' +
        '<div class="bar-wrap"><div class="bar-fill" id="trainFb' + fi + '" style="width:0%;background:' + APP_CONFIG.FINGER_COLORS[fi] + '"></div></div>' +
        '<span style="font-size:8px;width:26px;text-align:right" id="trainFv' + fi + '">0%</span>' +
      '</div>';
  }

  var livePredHTML;
  if (liveP && liveP.name && liveP.name !== 'Unknown') {
    livePredHTML =
      '<div class="live-pred-box">' +
        '<div style="font-size:8px;color:var(--mx);letter-spacing:.08em;margin-bottom:2px">LIVE PREDICTION</div>' +
        '<div style="display:flex;align-items:center;gap:6px">' +
          '<span style="font-size:18px;font-weight:800;color:var(--g)">' + liveP.name + '</span>' +
          '<span style="font-size:10px;color:var(--mx)">' + Math.round(liveP.conf * 100) + '%</span>' +
          '<span style="font-size:8px;color:var(--dm)">[' + (liveP.model || '') + ']</span>' +
        '</div>' +
      '</div>';
  } else {
    livePredHTML =
      '<div style="font-size:8px;color:var(--dm);margin-top:4px">' +
        (state.camActive ? '✓ Camera active' : '⏳ Waiting…') +
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
            ? '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;color:var(--dm);gap:4px;pointer-events:none"><div style="font-size:22px">📷</div><div style="font-size:8px">Loading…</div></div>'
            : '') +
          (state.recording
            ? '<div style="position:absolute;top:5px;right:5px"><span class="bg bg-r" style="font-size:8px"><span class="dot dot-r dot-pulse"></span>REC</span></div>'
            : '') +
          (countdown > 0
            ? '<div class="countdown-overlay"><div class="countdown-num">' + countdown + '</div></div>'
            : '') +
          '<div style="position:absolute;bottom:0;left:0;right:0;display:flex;gap:3px;padding:3px">' +
            '<span id="trainFpsB" style="font-size:7px;background:rgba(0,0,0,.7);color:var(--g);padding:2px 5px;border-radius:4px;font-weight:700">-- FPS</span>' +
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

function _renderRetrainAlert(newSince) {
  return '<div style="padding:10px 14px;background:var(--aD);border:1px solid var(--a);border-radius:8px;margin-bottom:10px;display:flex;align-items:center;gap:10px;font-size:10px;flex-wrap:wrap">' +
    '<span style="font-size:16px">⚠️</span>' +
    '<div style="flex:1"><strong style="color:var(--a)">' + newSince + ' new samples</strong> since last training — model is outdated.</div>' +
    Btn('Retrain Now', 'window._app.trainModel()', 'a', 'sm') +
  '</div>';
}

function _renderModelStats(trainStats) {
  var sAcc     = _get(trainStats, 'staticAccuracy',  0);
  var sLoss    = _get(trainStats, 'staticLoss',      1);
  var sEp      = _get(trainStats, 'staticEpochs',    0);
  var sTrained = _get(trainStats, 'staticTrained',   false);
  var dAcc     = _get(trainStats, 'dynamicAccuracy', 0);
  var dLoss    = _get(trainStats, 'dynamicLoss',     1);
  var dEp      = _get(trainStats, 'dynamicEpochs',   0);
  var dTrained = _get(trainStats, 'dynamicTrained',  false);

  return '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">' +
    '<div class="cd" style="margin-bottom:0">' +
      '<div class="cd-label">🧠 Static MLP ' + (sTrained ? '<span class="bg bg-g" style="font-size:8px">Trained</span>' : '<span class="bg bg-d" style="font-size:8px">Untrained</span>') + '</div>' +
      '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px">' +
        StatBox((sAcc * 100).toFixed(1) + '%', 'Acc', 'var(--g)') +
        StatBox(sLoss.toFixed(3), 'Loss', 'var(--p)') +
        StatBox(sEp, 'Ep', 'var(--a)') +
      '</div>' +
    '</div>' +
    '<div class="cd" style="margin-bottom:0">' +
      '<div class="cd-label">🔄 LSTM Dyn ' + (dTrained ? '<span class="bg bg-p" style="font-size:8px">Trained</span>' : '<span class="bg bg-d" style="font-size:8px">Untrained</span>') + '</div>' +
      '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px">' +
        StatBox((dAcc * 100).toFixed(1) + '%', 'Acc', 'var(--g)') +
        StatBox(dLoss.toFixed(3), 'Loss', 'var(--p)') +
        StatBox(dEp, 'Ep', 'var(--a)') +
      '</div>' +
    '</div>' +
  '</div>';
}

function _renderTrainingProgress(progress) {
  return '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">⚡ Training… ' + Math.round(progress) + '%</div>' +
    Bar(progress) +
  '</div>';
}

function _renderControls(isTraining, confThresh, readyCount, totalCount) {
  return '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">⚙ Controls</div>' +
    '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">' +
      Btn('📦 Demo Data',   'window._app.generateDemoData()', 'a', 'sm') +
      Btn(isTraining ? '⏳ Training…' : '🚀 Train Models', 'window._app.trainModel()', 'g', '', isTraining) +
      Btn('🗑 Delete Model', 'window._app.deleteModel()', 'r', 'sm', isTraining) +
      Btn('📤 Export',       'window._app.exportDataset()', 'o', 'sm') +
      '<label class="btn btn-o btn-sm" style="cursor:pointer">📥 Import' +
        '<input type="file" accept=".json" style="display:none" onchange="window._app.importDataset(this.files[0])">' +
      '</label>' +
    '</div>' +
    '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">' +
      '<span style="font-size:9px;color:var(--mx);white-space:nowrap">CONFIDENCE</span>' +
      '<input type="range" min="0.3" max="0.95" step="0.05" value="' + confThresh + '" ' +
        'style="flex:1;min-width:80px;accent-color:var(--g)" ' +
        'oninput="window._app.setConfThreshold(parseFloat(this.value));document.getElementById(\'confLbl\').textContent=Math.round(parseFloat(this.value)*100)+\'%\'">' +
      '<span id="confLbl" style="font-size:10px;font-weight:700;color:var(--g);width:32px">' + Math.round(confThresh * 100) + '%</span>' +
    '</div>' +
    '<div style="font-size:9px;color:var(--dm);margin-top:6px">' + readyCount + '/' + totalCount + ' gestures ready · samples in SQLite DB</div>' +
  '</div>';
}

function _renderSourceSelector(src) {
  var camActive   = src === 'camera';
  var gloveActive = src === 'glove';
  var camStyle    = 'flex:1;padding:7px 0;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;border:1px solid ' +
    (camActive ? 'var(--g)' : 'var(--brd)') + ';background:' + (camActive ? 'var(--gD)' : 'var(--s1)') + ';color:' + (camActive ? 'var(--g)' : 'var(--mx)');
  var gloveStyle  = 'flex:1;padding:7px 0;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;border:1px solid ' +
    (gloveActive ? 'var(--p)' : 'var(--brd)') + ';background:' + (gloveActive ? 'var(--pD)' : 'var(--s1)') + ';color:' + (gloveActive ? 'var(--p)' : 'var(--mx)');

  return '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">📡 Data Source</div>' +
    '<div style="display:flex;gap:8px">' +
      '<button style="' + camStyle   + '" onclick="window._app.setTrainSource(\'camera\')">📷 Camera</button>' +
      '<button style="' + gloveStyle + '" onclick="window._app.setTrainSource(\'glove\')">🧤 Glove / MQTT</button>' +
    '</div>' +
    '<div style="font-size:9px;color:var(--dm);margin-top:6px">' +
      (camActive ? 'Recording from webcam landmarks' : 'Recording from sensor glove via MQTT') +
    '</div>' +
  '</div>';
}

function _renderSectionTabs(section) {
  var tabs      = ['alphabet', 'numbers', 'words', 'custom'];
  var tabLabels = {alphabet: '🔤 A–Z', numbers: '🔢 0–9', words: '💬 Words', custom: '✏️ Custom'};
  var html = '<div class="section-tabs">';
  for (var ti = 0; ti < tabs.length; ti++) {
    var s = tabs[ti];
    html += '<button class="tab' + (section === s ? ' active' : '') + '" onclick="window._app.setTrainSection(\'' + s + '\')">' + tabLabels[s] + '</button>';
  }
  return html + '</div>';
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
    var gc = ga < 0.7 ? 'var(--r)' : ga < 0.85 ? 'var(--a)' : 'var(--g)';
    rows += '<div style="padding:6px 8px;background:var(--s1);border-radius:6px;border:1px solid ' + gc + '40">' +
      '<div style="display:flex;justify-content:space-between;margin-bottom:3px">' +
        '<span style="font-size:10px;font-weight:700">' + gn + '</span>' +
        '<span style="font-size:9px;font-weight:700;color:' + gc + '">' + Math.round(ga * 100) + '%</span>' +
      '</div>' +
      pBar(Math.round(ga * 100), 100, gc) +
    '</div>';
  }

  return '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">📊 Per-Gesture Accuracy</div>' +
    '<div style="font-size:9px;color:var(--dm);margin-bottom:8px">Below 70% = needs more samples</div>' +
    '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:5px">' +
      rows +
    '</div>' +
  '</div>';
}

function _renderSessionHistory(histStats) {
  var totalCaptures = _get(histStats, 'totalCaptures', 0);
  var totalTrains   = _get(histStats, 'totalTrains',   0);
  var newSince      = _get(histStats, 'newSamplesSinceLastTrain', 0);
  var lastTrain     = _get(histStats, 'lastTrainAt',   null);

  return '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">🕐 Session History</div>' +
    '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px">' +
      '<div style="text-align:center"><div style="font-size:20px;font-weight:700;color:var(--g)">'  + totalCaptures + '</div><div style="font-size:8px;color:var(--mx)">Captures</div></div>' +
      '<div style="text-align:center"><div style="font-size:20px;font-weight:700;color:var(--p)">'  + totalTrains   + '</div><div style="font-size:8px;color:var(--mx)">Trains</div></div>' +
      '<div style="text-align:center"><div style="font-size:20px;font-weight:700;color:var(--a)">'  + newSince      + '</div><div style="font-size:8px;color:var(--mx)">New Since</div></div>' +
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
    '<div class="cd-label">🧠 Adaptive NLP ' +
      (nlpPersonal
        ? '<span class="bg bg-g" style="font-size:8px">Personal Model Active</span>'
        : '<span class="bg bg-d" style="font-size:8px">Building…</span>') +
    '</div>' +
    '<div style="font-size:9px;color:var(--mx);margin-bottom:8px">NLP learns from your signing history and personalises word suggestions over time.</div>' +
    '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:10px">' +
      '<div style="text-align:center;padding:8px;background:var(--s1);border-radius:8px">' +
        '<div style="font-size:18px;font-weight:700;color:var(--p)">' + nlpCorpus + '</div>' +
        '<div style="font-size:8px;color:var(--mx)">Sentences Learned</div>' +
      '</div>' +
      '<div style="text-align:center;padding:8px;background:var(--s1);border-radius:8px">' +
        '<div style="font-size:18px;font-weight:700;color:' + (nlpPersonal ? 'var(--g)' : 'var(--dm)') + '">' + (nlpPersonal ? '✓' : '✗') + '</div>' +
        '<div style="font-size:8px;color:var(--mx)">Personal Model</div>' +
      '</div>' +
      '<div style="text-align:center;padding:8px;background:var(--s1);border-radius:8px">' +
        '<div style="font-size:18px;font-weight:700;color:var(--a)">' + (nlpCorpus >= 5 ? '✓' : nlpCorpus + '/5') + '</div>' +
        '<div style="font-size:8px;color:var(--mx)">Min 5 to Activate</div>' +
      '</div>' +
    '</div>' +
    '<div style="font-size:9px;color:var(--dm)">Speak sentences using the Recognize tab to train the personal model. Retrains automatically every 3 new sentences.</div>' +
  '</div>';
}

// ── Gesture grid renderers ────────────────────────────────────────────────────

function _renderAlpha(meta, readiness, guided) {
  var html = '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">🔤 Alphabet A–Z <span class="flex"></span>' +
      Btn('🎯 Guided', 'window._app.startGuidedMode(\'alphabet\')', 'p', 'sm') +
    '</div>' +
    '<div style="font-size:9px;color:var(--dm);margin-bottom:10px">📸 = static · 🎬 = dynamic motion (J, Z) · 🗑 = delete samples</div>' +
    '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(88px,1fr));gap:5px">';
  for (var i = 0; i < ALPHA.length; i++) {
    var lt = ALPHA[i];
    html += _card(lt, meta, readiness, DYN_LETTERS[lt] ? 'dynamic' : 'static', guided === lt);
  }
  return html + '</div></div>';
}

function _renderNums(meta, readiness) {
  var html = '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">🔢 Numbers 0–9</div>' +
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
  var words     = [];
  var i;
  for (i = 0; i < ALPHA.length;  i++) alphaSet[ALPHA[i]]    = 1;
  for (i = 0; i < NUMS.length;   i++) numSet[NUMS[i]]        = 1;
  for (i = 0; i < customs.length; i++) customSet[customs[i]] = 1;
  for (i = 0; i < gestures.length; i++) {
    var g = gestures[i];
    if (!alphaSet[g] && !numSet[g] && !customSet[g]) words.push(g);
  }

  var html = '<div class="cd" style="margin-bottom:12px"><div class="cd-label">💬 Word Gestures</div>';
  if (words.length === 0) {
    html += '<div style="font-size:10px;color:var(--dm);padding:10px 0">No word gestures. Generate demo data to add some.</div>';
  } else {
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:6px">';
    for (i = 0; i < words.length; i++) html += _wideCard(words[i], meta, readiness, false);
    html += '</div>';
  }
  return html + '</div>';
}

function _renderCustom(meta, readiness, customs) {
  var html = '<div class="cd" style="margin-bottom:12px">' +
    '<div class="cd-label">✏️ Custom Gestures</div>' +
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
  var m   = meta[name] || {static: 0, dynamic: 0, static_quality: 0, dynamic_quality: 0};
  var r   = readiness[name] || {ready: false, needed: 0, type: defaultType};
  var t   = r.type || defaultType;
  var isDyn   = t === 'dynamic';
  var src     = window._trainSource || 'camera';
  var camKey  = isDyn ? 'dynamic_camera' : 'static_camera';
  var gloveKey= isDyn ? 'dynamic_glove'  : 'static_glove';
  var count   = src === 'camera' ? (m[camKey] || 0) : (m[gloveKey] || 0);
  var total   = isDyn ? (m.dynamic || 0) : (m.static || 0);
  var target  = isDyn ? 5 : 10;
  var q       = isDyn ? (m.dynamic_quality || 0) : (m.static_quality || 0);
  var ready   = r.ready;

  var bg  = isGuided ? 'var(--pD)' : ready ? 'var(--gD)' : 'var(--s1)';
  var brd = isGuided ? 'var(--p)'  : ready ? 'var(--g)'  : 'var(--brd)';

  return '<div style="padding:8px;background:' + bg + ';border-radius:8px;border:1px solid ' + brd + ';text-align:center">' +
    '<div style="font-size:18px;font-weight:800;margin-bottom:2px">' + name + '</div>' +
    '<span class="bg ' + (isDyn ? 'bg-p' : 'bg-g') + '" style="font-size:7px">' + (isDyn ? 'DYN' : 'STA') + '</span>' +
    '<div style="display:flex;align-items:center;gap:4px;margin:4px 0">' +
      pBar(count, target, ready ? 'var(--g)' : 'var(--a)') +
      '<span style="font-size:8px;color:var(--mx);white-space:nowrap">' + count + '/' + target + (total > count ? ' (' + total + ' tot)' : '') + '</span>' +
    '</div>' +
    (count > 0
      ? '<div style="font-size:7px;color:' + qColor(q) + ';margin-bottom:4px">' + qLabel(q) + '</div>'
      : '<div style="height:14px"></div>') +
    '<div style="display:flex;gap:3px;justify-content:center">' +
      (isDyn
        ? '<button style="padding:4px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--p);color:var(--bg);border:none;border-radius:6px;cursor:pointer" onclick="window._app.addSample(\'' + esc(name) + '\',\'dynamic\')">🎬</button>'
        : '<button style="padding:4px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--g);color:var(--bg);border:none;border-radius:6px;cursor:pointer" onclick="window._app.addSample(\'' + esc(name) + '\',\'static\')">📸</button>') +
      (count > 0
        ? '<button style="padding:4px 6px;font-size:9px;font-weight:700;font-family:inherit;background:var(--r);color:var(--bg);border:none;border-radius:6px;cursor:pointer" onclick="window._app.deleteGestureSamples(\'' + esc(name) + '\',\'' + t + '\')">🗑</button>'
        : '') +
    '</div>' +
  '</div>';
}

function _wideCard(name, meta, readiness, isCustom) {
  var m     = meta[name] || {static: 0, dynamic: 0, static_quality: 0, dynamic_quality: 0};
  var r     = readiness[name] || {ready: false, needed: 0, type: 'static'};
  var isDyn = r.type === 'dynamic';
  var sc    = m.static  || 0;
  var dc    = m.dynamic || 0;
  var ready = r.ready;
  var needed= r.needed || 0;

  return '<div style="padding:10px 12px;background:var(--s1);border-radius:8px;border:1px solid ' + (ready ? 'var(--g)' : 'var(--brd)') + '">' +
    '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px">' +
      '<div>' +
        '<span style="font-size:13px;font-weight:700">' + name + '</span> ' +
        '<span class="bg ' + (isDyn ? 'bg-p' : 'bg-g') + '" style="font-size:7px">' + (isDyn ? 'DYN' : 'STA') + '</span>' +
      '</div>' +
      (isCustom
        ? '<button style="padding:3px 7px;font-size:9px;font-weight:700;font-family:inherit;background:var(--r);color:var(--bg);border:none;border-radius:5px;cursor:pointer" onclick="window._app.deleteGesture(\'' + esc(name) + '\')">✕</button>'
        : '') +
    '</div>' +
    '<div style="font-size:8px;color:var(--mx);margin-bottom:5px">📸 ' + sc + ' static · 🎬 ' + dc + ' dynamic</div>' +
    '<div style="display:flex;align-items:center;gap:4px;margin-bottom:6px">' +
      pBar(isDyn ? dc : sc, isDyn ? 5 : 10, ready ? 'var(--g)' : 'var(--a)') +
      '<span style="font-size:8px;color:var(--mx);white-space:nowrap">' + (ready ? '✓ Ready' : 'Need ' + needed) + '</span>' +
    '</div>' +
    '<div style="display:flex;gap:4px;flex-wrap:wrap">' +
      '<button style="padding:5px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--g);color:var(--bg);border:none;border-radius:6px;cursor:pointer" onclick="window._app.addSample(\'' + esc(name) + '\',\'static\')">📸 Static</button>' +
      '<button style="padding:5px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--p);color:var(--bg);border:none;border-radius:6px;cursor:pointer" onclick="window._app.addSample(\'' + esc(name) + '\',\'dynamic\')">🎬 Dynamic</button>' +
      (sc + dc > 0
        ? '<button style="padding:5px 8px;font-size:9px;font-weight:700;font-family:inherit;background:var(--r);color:var(--bg);border:none;border-radius:6px;cursor:pointer" onclick="window._app.deleteGestureSamples(\'' + esc(name) + '\',\'all\')">🗑</button>'
        : '') +
    '</div>' +
  '</div>';
}
