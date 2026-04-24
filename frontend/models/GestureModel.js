// models/GestureModel.js — v6.1 Phase 2 + Source tracking
// Each sample tagged with source: 'camera' | 'glove'
// Training loads only samples matching active source
import {APP_CONFIG, GESTURE_TEMPLATES} from '../config/app.config.js';

// (API already defined in bundle header)

export class GestureModel {
  constructor(defaultGestures) {
    this.gestures       = [...defaultGestures];
    this.staticSamples  = {};
    this.dynamicSamples = {};
    this.sampleCounts   = {};
    this.templates      = GESTURE_TEMPLATES;
    this.recording      = false;
    this.recordTarget   = null;
    this.frameBuffer    = [];
    this.trainSource    = 'camera'; // 'camera' | 'glove' — set by AppController
  }

  async loadFromDB() {
    try {
      var url = API + '/samples/meta';
      var loadUrl = API + '/samples/load?source=' + this.trainSource;
      var metaUrl = API + '/samples/meta';

      var results = await Promise.all([
        fetch(metaUrl).then(function(r){ return r.json(); }),
        fetch(loadUrl).then(function(r){ return r.json(); }),
      ]);
      var meta = results[0]; var samples = results[1];

      this.sampleCounts = {};
      for (var g in meta) {
        this.sampleCounts[g] = {
          static:         meta[g].static         || 0,
          dynamic:        meta[g].dynamic         || 0,
          static_camera:  meta[g].static_camera   || 0,
          static_glove:   meta[g].static_glove    || 0,
          dynamic_camera: meta[g].dynamic_camera  || 0,
          dynamic_glove:  meta[g].dynamic_glove   || 0,
        };
        if (this.gestures.indexOf(g) < 0) this.gestures.push(g);
      }
      for (var g in samples) {
        if (samples[g].static)  this.staticSamples[g]  = samples[g].static;
        if (samples[g].dynamic) this.dynamicSamples[g] = samples[g].dynamic;
      }
      console.log('[GestureModel] loaded from Python DB:', Object.keys(meta).length, 'gestures, source:', this.trainSource);
    } catch(e) { console.warn('[GestureModel] load failed:', e); }
  }

  addGesture(name) {
    var t = name.trim();
    if (t && this.gestures.indexOf(t) < 0) { this.gestures.push(t); return true; }
    return false;
  }
  removeGesture(name) {
    this.gestures = this.gestures.filter(function(g){ return g !== name; });
    delete this.staticSamples[name];
    delete this.dynamicSamples[name];
    delete this.sampleCounts[name];
  }

  getType(name) {
    var t = this.templates[name];
    return (t ? t.type : null) || 'static';
  }

  async simulateStaticAsync(name, count) {
    var res = await fetch(API + '/samples/simulate', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({gesture: name, count: count || 1, sample_type: 'static'})
    });
    var data = await res.json();
    return data.samples;
  }

  // source param: 'camera' | 'glove' — defaults to this.trainSource
  async addStaticSamples(name, vectors, isMirror, source) {
    source = source || this.trainSource;
    if (!this.staticSamples[name]) this.staticSamples[name] = [];
    this.staticSamples[name].push.apply(this.staticSamples[name], vectors);
    if (!this.sampleCounts[name]) this.sampleCounts[name] = {static:0,dynamic:0,static_camera:0,static_glove:0,dynamic_camera:0,dynamic_glove:0};
    this.sampleCounts[name].static++;
    if (source === 'camera') this.sampleCounts[name].static_camera++;
    else if (source === 'glove') this.sampleCounts[name].static_glove++;

    var saveRes = await fetch(API + '/samples/save', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({gesture: name, samples: vectors, sample_type: 'static', source: source})
    });
    if (!saveRes.ok) throw new Error('Server returned ' + saveRes.status);
  }

  async addDynamicSamples(name, trajectories, source) {
    source = source || this.trainSource;
    if (!this.dynamicSamples[name]) this.dynamicSamples[name] = [];
    this.dynamicSamples[name].push.apply(this.dynamicSamples[name], trajectories);
    if (!this.sampleCounts[name]) this.sampleCounts[name] = {static:0,dynamic:0,static_camera:0,static_glove:0,dynamic_camera:0,dynamic_glove:0};
    this.sampleCounts[name].dynamic++;
    if (source === 'camera') this.sampleCounts[name].dynamic_camera++;
    else if (source === 'glove') this.sampleCounts[name].dynamic_glove++;

    await fetch(API + '/samples/save', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({gesture: name, samples: trajectories, sample_type: 'dynamic', source: source})
    });
  }

  pushFrame(features) {
    if (!this.recording) return false;
    this.frameBuffer.push([...features]);
    if (this.frameBuffer.length >= APP_CONFIG.NN.DYNAMIC_FRAMES) {
      var trajectory = this.frameBuffer.flat ? this.frameBuffer.flat() :
        [].concat.apply([], this.frameBuffer);
      this.addDynamicSamples(this.recordTarget, [trajectory], this.trainSource);
      this.frameBuffer = []; this.recording = false;
      return true;
    }
    return false;
  }

  startRecording(gestureName) { this.recording = true; this.recordTarget = gestureName; this.frameBuffer = []; }
  stopRecording()              { this.recording = false; this.frameBuffer = []; }

  getStaticTrainingData() {
    var inputs = [], labels = [], names = [], idx = 0;
    for (var i = 0; i < this.gestures.length; i++) {
      var name = this.gestures[i];
      if (!this.staticSamples[name] || !this.staticSamples[name].length) continue;
      names.push(name);
      for (var j = 0; j < this.staticSamples[name].length; j++) {
        inputs.push(this.staticSamples[name][j]); labels.push(idx);
      }
      idx++;
    }
    return {inputs: inputs, labels: labels, names: names};
  }

  getDynamicTrainingData() {
    var inputs = [], labels = [], names = [], idx = 0;
    for (var i = 0; i < this.gestures.length; i++) {
      var name = this.gestures[i];
      if (!this.dynamicSamples[name] || !this.dynamicSamples[name].length) continue;
      names.push(name);
      for (var j = 0; j < this.dynamicSamples[name].length; j++) {
        inputs.push(this.dynamicSamples[name][j]); labels.push(idx);
      }
      idx++;
    }
    return {inputs: inputs, labels: labels, names: names};
  }

  getSampleCount(name) { return this.sampleCounts[name] || {static:0,dynamic:0,static_camera:0,static_glove:0,dynamic_camera:0,dynamic_glove:0}; }

  async resetAll() {
    await fetch(API + '/samples/clear', {method: 'DELETE'});
    this.staticSamples = {}; this.dynamicSamples = {}; this.sampleCounts = {};
    this.gestures = [...APP_CONFIG.DEFAULT_GESTURES];
  }
}
