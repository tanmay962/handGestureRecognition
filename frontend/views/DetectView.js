// views/DetectView.js
import { APP_CONFIG, GESTURE_TEMPLATES } from '../config/app.config.js';
import { Badge, DotBadge, Card, Btn, FingerBars, LogEntry } from './components/index.js';

export function renderDetectTab(state) {
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
