// views/UserModeView.js — Clean end-user interface
import {APP_CONFIG} from '../config/app.config.js';
import {FingerBars} from './components/index.js';

export function renderUserMode(state){
  const {camActive,cameraError,running,displayText,spelling,suggestions,wordSuggestions,completion,contextState,staticTrained,dynamicTrained}=state;
  const trained=staticTrained||dynamicTrained;

  return`
  <div style="padding:12px 0 6px;display:flex;align-items:center;justify-content:space-between">
    <div>
      <div style="font-size:10px;letter-spacing:.2em;color:var(--g);font-weight:700">✋ SIGNLENS</div>
      <div style="font-size:16px;font-weight:700">Gesture <span style="color:var(--g)">Communication</span></div>
    </div>
    <button class="btn btn-o btn-sm" onclick="document.getElementById('userSettings').style.display=document.getElementById('userSettings').style.display==='none'?'block':'none'" style="font-size:16px;padding:8px">⚙</button>
  </div>

  <!-- Mini settings (hidden by default) -->
  <div id="userSettings" style="display:none" class="cd">
    <div class="srow"><div><div class="srow-label">Speech Rate</div></div>
      <input type="range" min=".5" max="2" step=".1" value="${state.tts.rate}" oninput="window._app.setTTSRate(parseFloat(this.value))" style="width:100px;accent-color:var(--g)">
    </div>
    <div class="srow"><div><div class="srow-label">Input</div></div>
      <select onchange="window._app.setInputMode(this.value)" style="background:var(--s1);color:var(--tx);border:1px solid var(--brd);border-radius:6px;padding:4px 8px;font-family:inherit;font-size:11px">
        <option value="camera" ${state.inputMode==='camera'?'selected':''}>Camera</option>
        <option value="glove" ${state.inputMode==='glove'?'selected':''}>Glove</option>
        <option value="both" ${state.inputMode==='both'?'selected':''}>Both</option>
      </select>
    </div>
    <div style="border-top:1px solid var(--brd);padding-top:10px;margin-top:8px">
      <button class="btn btn-o btn-sm" onclick="const pin=prompt('Enter Admin PIN:');if(pin&&window._app.checkAdminPin(pin)){window._app.switchMode('admin')}else if(pin){alert('Wrong PIN')}">🔒 Admin Mode</button>
    </div>
  </div>

  <!-- Camera Feed (large) -->
  <div class="cd" style="padding:0;overflow:hidden;position:relative">
    <div class="vid-wrap" style="min-height:300px">
      <div id="vidContainer" style="width:100%;height:100%"></div>
      <div class="vid-badges">
        <span class="bg bg-g" id="fpsB">-- FPS</span>
        <span class="bg bg-p" id="handB">No Hand</span>
      </div>
      <div class="vid-gesture" id="gestDisp" style="display:none">
        <div class="gesture-name" id="gestName"></div>
        <div class="gesture-conf" id="gestConf"></div>
      </div>
      ${!camActive?'<div class="vid-overlay"><div style="font-size:48px">✋</div><div style="font-size:12px;letter-spacing:.1em">TAP START</div></div>':''}
    </div>
    <!-- System gesture progress bar -->
    <div style="height:3px;background:var(--s1);position:relative"><div id="sysGestProg" style="height:100%;width:0%;background:var(--g);transition:width .1s"></div></div>
  </div>

  <!-- Start/Stop -->
  <div style="display:flex;gap:8px;justify-content:center;margin-bottom:12px">
    ${camActive?`<button class="btn btn-r" onclick="window._app.stopCamera()">■ Stop Camera</button>`
      :`<button class="btn btn-o" onclick="window._app.startCamera()">${cameraError?'⚠ Retry Camera':'📷 Start Camera'}</button>`}
    ${cameraError?`<div style="font-size:10px;color:var(--r);margin-top:6px;padding:8px 12px;background:rgba(251,113,133,.1);border-radius:8px;border:1px solid var(--r);text-align:center">⚠ ${cameraError}</div>`:''}
    ${trained&&!running?`<button class="btn btn-g" onclick="window._app.startRecognition()">▶ Recognize</button>`:''}
    ${running?`<button class="btn btn-r" onclick="window._app.stopRecognition()">■ Stop</button>`:''}
  </div>

  <!-- Sentence Display (large, prominent) -->
  <div class="cd">
    <div style="background:var(--s1);border:1px solid var(--brd);border-radius:10px;padding:16px 20px;min-height:56px;font-size:20px;font-weight:600;line-height:1.5;margin-bottom:12px;color:${displayText?'var(--tx)':'var(--dm)'}">
      ${displayText||'Start signing to communicate...'}<span class="cursor"></span>
    </div>

    ${spelling?`<div style="font-size:10px;color:var(--p);letter-spacing:.12em;margin-bottom:6px">SPELLING: ${spelling.toUpperCase()}_</div>`:''}

    <!-- Word suggestions from spelling -->
    ${wordSuggestions.length?`<div style="margin-bottom:8px"><div style="font-size:9px;color:var(--mx);margin-bottom:4px;letter-spacing:.1em">WORD MATCHES (hold 1-${Math.min(5,wordSuggestions.length)} fingers to select)</div>
      <div class="suggs">${wordSuggestions.map((w,i)=>`<span class="sugg ai"><span style="font-size:8px;opacity:.6">${i+1}.</span> ${w}</span>`).join('')}</div></div>`:''}

    <!-- Sentence-level suggestions -->
    ${!spelling&&suggestions.length?`<div><div style="font-size:9px;color:var(--mx);margin-bottom:4px;letter-spacing:.1em">${state.geminiEnabled?'AI':'LOCAL'} PREDICTIONS (hold fingers to select)</div>
      <div class="suggs">${suggestions.map((w,i)=>`<span class="sugg${state.geminiEnabled?' ai':''}"><span style="font-size:8px;opacity:.6">${i+1}.</span> ${w}</span>`).join('')}</div></div>`:''}

    <!-- Completion -->
    ${completion&&completion!==state.sentence?`<div class="completion" onclick="window._app.acceptCompletion()"><div class="completion-label">✨ AI suggests</div><div class="completion-text">"${completion}"</div></div>`:''}

    <!-- Context state indicator -->
    <div style="display:flex;gap:8px;align-items:center;justify-content:space-between;margin-top:8px">
      <span style="font-size:9px;color:var(--dm);letter-spacing:.1em">${contextState} ${state.geminiEnabled?'· GEMINI AI':''}</span>
      <div style="font-size:9px;color:var(--dm)">✊=Speak  🖐=Clear  👎=Undo</div>
    </div>
  </div>

  <!-- Finger bars (compact) -->
  <div class="cd">
    ${FingerBars(APP_CONFIG.FINGER_NAMES,APP_CONFIG.FINGER_COLORS)}
  </div>`;
}
