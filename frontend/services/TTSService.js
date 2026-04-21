// services/TTSService.js
export class TTSService{
  constructor(){this.enabled=true;this.autoSpeak=true;this.rate=1}
  speak(t){if(!this.enabled||!t||!window.speechSynthesis)return;window.speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(t);u.rate=this.rate;window.speechSynthesis.speak(u)}
  speakIfAuto(t){if(this.autoSpeak)this.speak(t)}
  stop(){if(window.speechSynthesis)window.speechSynthesis.cancel()}
  setRate(r){this.rate=Math.max(.5,Math.min(2,r))}
}
