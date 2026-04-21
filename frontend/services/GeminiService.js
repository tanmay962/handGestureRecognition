// services/GeminiService.js
import {APP_CONFIG} from '../config/app.config.js';
import {eventBus,Events} from '../utils/EventBus.js';

export class GeminiService{
  constructor(){this.apiKey='';this.enabled=false}
  setApiKey(k){this.apiKey=k.trim();this.enabled=this.apiKey.length>10}
  async testConnection(k){
    this.setApiKey(k);
    try{const r=await this._call('Say OK',{maxOutputTokens:5});if(r!==null){eventBus.emit(Events.GEMINI_CONNECTED);return true}}catch{}
    eventBus.emit(Events.GEMINI_ERROR);return false;
  }
  async getSuggestions(sentence,lastWord,context){
    if(!this.enabled)return null;
    const p=`You are a predictive text assistant for sign language. User builds sentences word by word.\nContext: "${context}"\nSentence: "${sentence||'(empty)'}"\n${lastWord?`Last word: "${lastWord}"`:''}
Suggest exactly 5 next words. Respond ONLY with JSON array like ["w1","w2","w3","w4","w5"]`;
    const t=await this._call(p,{temperature:.4,maxOutputTokens:50});return this._parseArr(t);
  }
  async completeSentence(s){if(!this.enabled||!s)return null;return this._call(`Complete naturally: "${s}"\nReturn ONLY completed sentence, no quotes, under 10 words.`,{temperature:.3,maxOutputTokens:30})}
  async correctGrammar(s){if(!this.enabled||!s)return null;return this._call(`Fix grammar: "${s}"\nReturn ONLY corrected sentence.`,{temperature:.1,maxOutputTokens:50})}
  async _call(prompt,cfg={}){
    if(!this.apiKey)return null;
    try{const r=await fetch(`${APP_CONFIG.GEMINI.ENDPOINT}/${APP_CONFIG.GEMINI.MODEL}:generateContent?key=${this.apiKey}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({contents:[{parts:[{text:prompt}]}],generationConfig:{temperature:cfg.temperature||.4,maxOutputTokens:cfg.maxOutputTokens||50,topP:.8}})});
      if(!r.ok)return null;const d=await r.json();return (d&&d.candidates&&d.candidates[0]&&d.candidates[0].content&&d.candidates[0].content.parts&&d.candidates[0].content.parts[0]?d.candidates[0].content.parts[0].text.trim():null);
    }catch(e){console.warn('[Gemini]',e);return null}
  }
  _parseArr(t){if(!t)return null;try{const m=t.replace(/```json|```/g,'').trim().match(/\[[\s\S]*?\]/);if(m){const p=JSON.parse(m[0]);if(Array.isArray(p)&&p.length)return p.slice(0,5).map(w=>String(w).toLowerCase())}}catch{}return null}
}
