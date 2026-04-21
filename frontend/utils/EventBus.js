// utils/EventBus.js — Pure pub-sub, no browser side effects at import time
// WebSocket bridge lives in WSClient.js (imported separately, deferred)

class EventBus {
  constructor() { this._l = new Map(); }
  on(e, cb) {
    if (!this._l.has(e)) this._l.set(e, []);
    this._l.get(e).push(cb);
    return () => this.off(e, cb);
  }
  off(e, cb) {
    const l = this._l.get(e);
    if (l) this._l.set(e, l.filter(c => c !== cb));
  }
  emit(e, d = null) {
    (this._l.get(e) || []).forEach(cb => {
      try { cb(d); } catch (err) { console.error(`[Event:${e}]`, err); }
    });
  }
  once(e, cb) { const w = d => { cb(d); this.off(e, w); }; this.on(e, w); }
}

export const eventBus = new EventBus();

export const Events = {
  TAB_CHANGED: 'app:tab', STATE_UPDATED: 'app:state', MODE_CHANGED: 'app:mode',
  CAMERA_STARTED: 'cam:start', CAMERA_STOPPED: 'cam:stop',
  HAND_DETECTED: 'cam:hand', HAND_LOST: 'cam:lost', FEATURES_EXTRACTED: 'cam:features',
  BLE_CONNECTED: 'ble:conn', BLE_DISCONNECTED: 'ble:disc', BLE_DATA: 'ble:data',
  MQTT_CONNECTED: 'mqtt:conn', MQTT_DISCONNECTED: 'mqtt:disc', MQTT_DATA: 'mqtt:data',
  RECOG_STARTED: 'rec:start', RECOG_STOPPED: 'rec:stop',
  GESTURE_RECOGNIZED: 'rec:gesture', GESTURE_PREDICTING: 'rec:predict', COMBO_DETECTED: 'rec:combo',
  TRAIN_STARTED: 'tr:start', TRAIN_PROGRESS: 'tr:prog',
  TRAIN_COMPLETE: 'tr:done', SAMPLES_COLLECTED: 'tr:samples',
  WORD_ADDED: 'nlp:word', SENTENCE_UPDATED: 'nlp:sent',
  SUGGESTIONS_UPDATED: 'nlp:sugg', COMPLETION_READY: 'nlp:comp',
  SPELLING_UPDATED: 'spell:update', WORD_PREDICTED: 'spell:predict',
  GEMINI_CONNECTED: 'gem:conn', GEMINI_ERROR: 'gem:err', GEMINI_LOADING: 'gem:load',
  TTS_SPEAK: 'tts:speak', VR_POSE: 'vr:pose', VR_INIT: 'vr:init',
  SUGGESTION_SELECTED: 'ui:suggsel', SYSTEM_GESTURE: 'ui:sysgest',
  RECORDING_START: 'rec:recstart', RECORDING_TICK: 'rec:rectick', RECORDING_DONE: 'rec:recdone',
};
