// views/SequenceView.js
import {Card,Btn,Badge} from './components/index.js';

export function renderSequenceTab(state){
  const{gestures,combos}=state;const seq=window._comboSeq||[];
  return`${Card('🔗 Gesture Combos',`<div style="font-size:12px;color:var(--mx);line-height:1.7"><strong style="color:var(--g)">Combos</strong> detect ordered gesture sequences. <strong style="color:var(--p)">Dynamic model</strong> handles motion-based recognition.</div>`)}
  ${Card('Create Combo',`<div class="fr g5 mb8" style="flex-wrap:wrap">${gestures.slice(0,20).map(g=>`<button class="btn btn-o btn-sm" onclick="window._comboSeq=(window._comboSeq||[]);window._comboSeq.push('${g}');window._app.switchTab('sequences')">${g}</button>`).join('')}</div>
    ${seq.length?`<div class="fr g5 mb8" style="padding:8px 12px;background:var(--pD);border-radius:8px;border:1px solid rgba(167,139,250,.2);flex-wrap:wrap"><span style="font-size:10px;color:var(--p);font-weight:700">SEQ:</span>${seq.map((g,i)=>`${Badge(g,'p')}${i<seq.length-1?'<span style="color:var(--dm)">→</span>':''}`).join('')}<button class="btn btn-ghost btn-sm" onclick="window._comboSeq=[];window._app.switchTab('sequences')">✕</button></div>`:''}
    <div class="fr g6"><input class="inp f1" id="comboAct" placeholder="Output phrase..."><button class="btn btn-g" onclick="if((window._comboSeq||[]).length>=2&&document.getElementById('comboAct').value.trim()){window._app.addCombo(window._comboSeq,document.getElementById('comboAct').value.trim());window._comboSeq=[];document.getElementById('comboAct').value='';window._app.switchTab('sequences')}"${seq.length<2?' disabled':''}>Save</button></div>`)}
  ${Card('Registered Combos',combos.map(c=>`<div class="fr mb8" style="padding:10px 14px;background:var(--s1);border-radius:8px;border:1px solid var(--brd)"><div class="f1"><div class="fr g5 mb8" style="flex-wrap:wrap">${c.sequence.map((g,i)=>`${Badge(g,'p')}${i<c.sequence.length-1?'<span style="color:var(--dm)">→</span>':''}`).join('')}</div><div style="font-size:12px;font-weight:600;color:var(--g)">"${c.action}"</div></div>${Badge((c.timeout/1000).toFixed(1)+'s','a')}</div>`).join(''))}`;
}
