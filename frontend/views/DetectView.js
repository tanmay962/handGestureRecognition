// views/DetectView.js
import {APP_CONFIG, GESTURE_TEMPLATES} from '../config/app.config.js';
import {Badge, DotBadge, Card, Btn, FingerBars, LogEntry} from './components/index.js';

export function renderDetectTab(state) {
  var {
    camActive, cameraError, running, displayText, spelling,
    suggestions, wordSuggestions, completion, log, gestures,
    geminiEnabled, staticTrained, dynamicTrained, inputMode, contextState
  } = state;

  var trained = staticTrained || dynamicTrained;

  return (
    _renderCamera(state, camActive, cameraError, running, trained, inputMode, contextState) +
    Card('Finger Curl', FingerBars(APP_CONFIG.FINGER_NAMES, APP_CONFIG.FINGER_COLORS)) +
    _renderSentenceBuilder(state, displayText, spelling, suggestions, wordSuggestions, completion, geminiEnabled) +
    (trained ? _renderQuickTest(gestures) : '') +
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
      '<div class="vid-gesture" id="gestDisp" style="display:none">' +
        '<div class="gesture-name" id="gestName"></div>' +
        '<div class="gesture-conf" id="gestConf"></div>' +
      '</div>' +
      (!camActive ? '<div class="vid-overlay"><div style="font-size:36px">📷</div><div style="font-size:10px;letter-spacing:.12em">START CAMERA</div></div>' : '') +
    '</div>' +

    // system gesture progress bar
    '<div style="height:3px;background:var(--s2);position:relative;margin-bottom:10px">' +
      '<div id="sysGestProg" style="height:100%;width:0%;background:var(--g);transition:width .1s"></div>' +
    '</div>' +

    '<div class="fr fr-center mb8" style="gap:6px;flex-wrap:wrap">' +
      (camActive
        ? Btn('■ Stop Camera', 'window._app.stopCamera()', 'r', 'sm')
        : Btn(cameraError ? '⚠ Retry Camera' : '📷 Start Camera', 'window._app.startCamera()', 'o', 'sm')) +
      (camActive ? Btn('⇄ Flip', 'window._app.switchCamera()', 'o', 'sm') : '') +
      (!running
        ? Btn('▶ Recognize', 'window._app.startRecognition()', camActive ? 'g' : 'o', '', !trained)
        : Btn('■ Stop', 'window._app.stopRecognition()', 'r')) +
      '<select onchange="window._app.setInputMode(this.value)" style="background:var(--s2);color:var(--tx);border:1px solid var(--brd);border-radius:6px;padding:6px 9px;font-family:inherit;font-size:10px;font-weight:600">' +
        '<option value="camera"' + (inputMode === 'camera' ? ' selected' : '') + '>📷 Camera</option>' +
        '<option value="glove"'  + (inputMode === 'glove'  ? ' selected' : '') + '>🧤 Glove</option>' +
        '<option value="both"'   + (inputMode === 'both'   ? ' selected' : '') + '>⚡ Both</option>' +
      '</select>' +
    '</div>' +
    (cameraError ? '<div style="font-size:10px;color:var(--r);padding:6px 10px;background:var(--rD);border-radius:6px;border:1px solid var(--r)">⚠ ' + cameraError + '</div>' : '');

  var statusBadges = (running || camActive ? DotBadge('Active', 'g', running || camActive) : DotBadge('Idle', 'dm', false)) +
    (contextState !== 'IDLE' ? ' ' + Badge(contextState, 'p') : '');

  return Card(
    '<span>Live Recognition</span><span class="flex"></span>' + statusBadges,
    cameraContent,
    'position:relative;overflow:hidden'
  );
}

function _renderSentenceBuilder(state, displayText, spelling, suggestions, wordSuggestions, completion, geminiEnabled) {
  var titleRight = (geminiEnabled ? Badge('Gemini AI', 'a') : Badge('Local', 'd')) +
    ' ' + Btn('🔊', 'window._app.speakSentence()', 'o', 'sm', !state.sentence);

  var body =
    '<div class="sent-box' + (displayText ? '' : ' empty') + '">' +
      (displayText || 'Signs appear here…') +
      '<span class="cursor"></span>' +
    '</div>' +

    (spelling ? '<div style="font-size:10px;color:var(--p);margin-bottom:6px">SPELLING: <strong>' + spelling.toUpperCase() + '_</strong></div>' : '') +

    (wordSuggestions.length
      ? '<div class="mb8">' +
          '<div style="font-size:9px;color:var(--a);letter-spacing:.1em;margin-bottom:4px">WORD MATCHES</div>' +
          '<div class="suggs">' +
            wordSuggestions.map(function(w, i) {
              return '<span class="sugg ai" onclick="window._app.addSuggestionWord(\'' + w + '\')">' +
                '<span style="font-size:8px;opacity:.5">' + (i + 1) + '</span> ' + w +
              '</span>';
            }).join('') +
          '</div>' +
        '</div>'
      : '') +

    (completion && completion !== state.sentence
      ? '<div class="completion" onclick="window._app.acceptCompletion()">' +
          '<div class="completion-label">✨ Gemini suggests</div>' +
          '<div class="completion-text">"' + completion + '"</div>' +
        '</div>'
      : '') +

    '<div style="font-size:9px;color:var(--mx);letter-spacing:.1em;margin-bottom:6px">' +
      (geminiEnabled ? 'AI' : 'LOCAL') + ' NEXT WORD' +
    '</div>' +
    '<div class="suggs">' +
      suggestions.map(function(w, i) {
        return '<span class="sugg' + (geminiEnabled ? ' ai' : '') + '" onclick="window._app.addSuggestionWord(\'' + w + '\')">' +
          '<span style="font-size:8px;opacity:.5">' + (i + 1) + '</span> ' + w +
        '</span>';
      }).join('') +
    '</div>' +

    '<div class="fr fr-end g6 mt8">' +
      (geminiEnabled && state.sentence ? Btn('✨ Fix Grammar', 'window._app.fixGrammar()', 'ghost', 'sm') : '') +
      Btn('↩ Undo', 'window._app.undoWord()', 'ghost', 'sm') +
      Btn('✕ Clear', 'window._app.clearSentence()', 'ghost', 'sm') +
    '</div>' +

    '<div style="font-size:8px;color:var(--dm);margin-top:8px">' +
      '✊ Hold fist = Speak · 🖐 Open palm = Clear · 👎 Thumbs down = Backspace · Hold 1–5 fingers = Select suggestion' +
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
    'Recognition Log',
    '<div style="max-height:180px;overflow-y:auto">' +
      log.slice(0, 15).map(function(entry, i) {
        return LogEntry(entry, 1 - i * 0.05);
      }).join('') +
    '</div>'
  );
}
