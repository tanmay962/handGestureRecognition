// controllers/SequenceController.js
import {DEFAULT_COMBOS,APP_CONFIG} from '../config/app.config.js';
import {eventBus,Events} from '../utils/EventBus.js';

export class SequenceController{
  constructor(){this.combos=new Map();this.history=[];this._loadDefaults()}
  _loadDefaults(){for(const c of DEFAULT_COMBOS){this.combos.set(c.seq.join('→'),{sequence:c.seq,action:c.action,timeout:c.timeout})}}
  addCombo(seq,action,timeout=APP_CONFIG.SEQUENCE.DEFAULT_TIMEOUT){const n=seq.join('→');this.combos.set(n,{sequence:seq,action,timeout});eventBus.emit(Events.COMBO_DETECTED,{combo:n})}
  pushGesture(name){
    this.history.push({name,time:Date.now()});if(this.history.length>APP_CONFIG.SEQUENCE.MAX_HISTORY)this.history.shift();
    const now=Date.now();
    for(const[cn,c]of this.combos){const recent=this.history.filter(g=>now-g.time<c.timeout).map(g=>g.name);
      let mi=0;for(const g of recent){if(g===c.sequence[mi]){mi++;if(mi===c.sequence.length){this.history=[];return{combo:cn,action:c.action}}}}
    }
    return null;
  }
  getAllCombos(){return[...this.combos.entries()].map(([n,c])=>({name:n,sequence:c.sequence,action:c.action,timeout:c.timeout}))}
}
