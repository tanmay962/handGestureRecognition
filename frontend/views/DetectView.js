// views/DetectView.js — Gesture Detection v1.0
import {APP_CONFIG,GESTURE_TEMPLATES} from '../config/app.config.js';
import {Badge,DotBadge,Card,Btn,FingerBars,LogEntry} from './components/index.js';

export function renderDetectTab(state){
  const{camActive,cameraError,running,displayText,spelling,suggestions,wordSuggestions,completion,log,gestures,geminiEnabled,staticTrained,dynamicTrained,inputMode,contextState}=state;
  const trained=staticTrained||dynamicTrained;
  return`
  ${Card(`<span>Live Recognition</span><span class="flex"></span>${DotBadge(running||camActive?'Active':'Idle',running||camActive?'g':'a',running||camActive)}${contextState!=='IDLE'?Badge(contextState,'p'):''}`,
    `<div class="vid-wrap" style="min-height:260px">
      <div id="vidContainer" style="width:100%;height:100%"></div>
      <div class="vid-badges"><span class="bg bg-g" id="fpsB">-- FPS</span><span class="bg bg-p" id="handB">No Hand</span></div>
      <div class="vid-gesture" id="gestDisp" style="display:none"><div class="gesture-name" id="gestName"></div><div class="gesture-conf" id="gestConf"></div></div>
      ${!camActive?'<div class="vid-overlay"><div style="font-size:40px">📷</div><div style="font-size:11px;letter-spacing:.1em">START CAMERA</div></div>':''}
    </div>
    <div style="height:3px;background:var(--s1);position:relative;margin-bottom:8px"><div id="sysGestProg" style="height:100%;width:0%;background:var(--g);transition:width .1s"></div></div>
    <div class="fr fr-center mb8" style="gap:6px;flex-wrap:wrap">
      ${camActive?Btn('■ Stop Camera','window._app.stopCamera()','r','sm'):Btn(cameraError?'⚠ Retry Camera':'📷 Start Camera','window._app.startCamera()','o','sm')}
      ${cameraError?`<div style="font-size:10px;color:var(--r);margin-top:6px;padding:6px 10px;background:var(--rD);border-radius:6px;border:1px solid var(--r)">⚠ ${cameraError}</div>`:''}
      ${!running?Btn('▶ Recognize','window._app.startRecognition()',camActive?'o':'g','',!trained):Btn('■ Stop','window._app.stopRecognition()','r')}
      <select onchange="window._app.setInputMode(this.value)" style="background:var(--s1);color:var(--tx);border:1px solid var(--brd);border-radius:6px;padding:6px 10px;font-family:inherit;font-size:10px;font-weight:600">
        <option value="camera"${inputMode==='camera'?' selected':''}>📷 Camera</option>
        <option value="glove"${inputMode==='glove'?' selected':''}>🧤 Glove</option>
        <option value="both"${inputMode==='both'?' selected':''}>⚡ Both</option>
      </select>
    </div>`,'position:relative;overflow:hidden')}

  ${Card('👆 Finger Detection',FingerBars(APP_CONFIG.FINGER_NAMES,APP_CONFIG.FINGER_COLORS))}

  ${Card(`<span>Sentence Builder</span><span class="flex"></span>${geminiEnabled?Badge('Gemini AI','a'):Badge('Local','d')}${Btn('🔊','window._app.speakSentence()','o','sm',!state.sentence)}`,
    `<div class="sent-box${displayText?'':' empty'}">${displayText||'Signs appear here...'}<span class="cursor"></span></div>
    ${spelling?`<div style="font-size:10px;color:var(--p);margin-bottom:6px">SPELLING: <strong>${spelling.toUpperCase()}_</strong></div>`:''}
    ${wordSuggestions.length?`<div class="mb8"><div style="font-size:9px;color:var(--a);letter-spacing:.1em;margin-bottom:4px">WORD MATCHES</div><div class="suggs">${wordSuggestions.map((w,i)=>`<span class="sugg ai" onclick="window._app.addSuggestionWord('${w}')"><span style="font-size:8px;opacity:.5">${i+1}</span> ${w}</span>`).join('')}</div></div>`:''}
    ${completion&&completion!==state.sentence?`<div class="completion" onclick="window._app.acceptCompletion()"><div class="completion-label">✨ Gemini suggests</div><div class="completion-text">"${completion}"</div></div>`:''}
    <div style="font-size:9px;color:var(--mx);letter-spacing:.1em;margin-bottom:6px">${geminiEnabled?'AI':'LOCAL'} NEXT WORD</div>
    <div class="suggs">${suggestions.map((w,i)=>`<span class="sugg${geminiEnabled?' ai':''}" onclick="window._app.addSuggestionWord('${w}')"><span style="font-size:8px;opacity:.5">${i+1}</span> ${w}</span>`).join('')}</div>
    <div class="fr fr-end g6 mt8">
      ${geminiEnabled&&state.sentence?Btn('✨ Fix Grammar','window._app.fixGrammar()','ghost','sm'):''}
      ${Btn('↩ Undo','window._app.undoWord()','ghost','sm')}
      ${Btn('✕ Clear','window._app.clearSentence()','ghost','sm')}
    </div>
    <div style="font-size:8px;color:var(--dm);margin-top:8px">✊ Hold fist=Speak · 🖐 Open palm=Clear · 👎 Thumbs down=Backspace · Hold 1-5 fingers=Select suggestion</div>`)}

  ${trained?Card('Quick Test',`<div class="fr g5" style="flex-wrap:wrap">${state.gestures.slice(0,20).map(g=>Btn(g,`window._app.quickTest('${g}')`,'o','sm')).join('')}</div>`):''}

  ${log.length?Card('Recognition Log',`<div style="max-height:180px;overflow-y:auto">${log.slice(0,15).map((e,i)=>LogEntry(e,1-i*.05)).join('')}</div>`):''}`;
}
