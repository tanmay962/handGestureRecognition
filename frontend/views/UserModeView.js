// views/UserModeView.js — Clean end-user interface
import {APP_CONFIG} from '../config/app.config.js';
import {FingerBars} from './components/index.js';

export function renderUserMode(state) {
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
