// controllers/TrainingController.js — Gesture Detection v1.0
// Camera only, 41-feature Holistic vectors — v6.1 Phase 2A
// Mirror augmentation: saves mirrored copy on every static capture
// LSTM abort: if hand lost mid dynamic recording, abort cleanly
// No optional-chaining — Safari 12 safe
import {APP_CONFIG} from '../config/app.config.js';
import {eventBus, Events} from '../utils/EventBus.js';

var API = '/api';

export class TrainingController {
  constructor(staticNN, dynamicNN, gestureModel, sensorModel) {
    this.staticNN   = staticNN;
    this.dynamicNN  = dynamicNN;
    this.gm         = gestureModel;
    this.sensor     = sensorModel;
    this.isTraining = false;
    this.progress   = 0;
    this._camera    = null; // set by AppController after camera init

    // LSTM abort tracking
    this._dynLostFrames  = 0;
    this._dynAborted     = false;
  }

  // ── Attach camera reference (needed for mirror augmentation) ──
  setCamera(camera) { this._camera = camera; }

  // ── Countdown (3-2-1) ─────────────────────────────────────────
  async runCountdown(secs) {
    secs = secs || 3;
    for (var i = secs; i > 0; i--) {
      window._countdown = i;
      eventBus.emit(Events.STATE_UPDATED);
      await new Promise(function(r){ setTimeout(r, 1000); });
    }
    window._countdown = 0;
    eventBus.emit(Events.STATE_UPDATED);
  }

  // ── Static single sample capture + mirror augmentation ────────
  async collectStaticSample(name, skipCountdown) {
    if (!skipCountdown) await this.runCountdown(3);

    var useLive = this.sensor && this.sensor.handDetected &&
      (this.sensor.source === 'camera' || this.sensor.source === 'ble' || this.sensor.source === 'sim');

    var sample, mirroredSample;

    try {
      if (useLive) {
        sample = this.sensor.getFeatureVector().slice();

        // Mirror augmentation — only possible with camera (landmarks available)
        if (this._camera && this._camera._lastHandsData && this._camera._lastHandsData.length > 0) {
          mirroredSample = this._camera.getMirroredFeatures(this._camera._lastHandsData);
        }
      } else {
        // Simulate — backend generates the feature vector
        var res  = await fetch(API + '/samples/simulate', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ gesture:name, count:1, sample_type:'static', source:'camera' })
        });
        if (!res.ok) throw new Error('simulate failed: ' + res.status);
        var data = await res.json();
        sample   = data.samples && data.samples[0];
        // For simulated data, generate a mirrored version too
        if (data.mirrored && data.mirrored.length > 0) {
          mirroredSample = data.mirrored[0];
        }
      }
    } catch(e) {
      console.error('[TrainCtrl] capture error:', e);
      eventBus.emit(Events.SAMPLES_COLLECTED, { gesture:name, live:false, count:0, error: String(e) });
      return { live:false, sample:null, mirrored:false, error: String(e) };
    }

    // Guard: if no sample was produced, bail out
    if (!sample || !sample.length) {
      eventBus.emit(Events.SAMPLES_COLLECTED, { gesture:name, live:false, count:0 });
      return { live:false, sample:null, mirrored:false };
    }

    try {
      // Save original sample
      await this.gm.addStaticSamples(name, [sample], false, window._trainSource||'camera');

      // Save mirrored sample if available (doubles training data for free)
      if (mirroredSample) {
        await this.gm.addStaticSamples(name, [mirroredSample], true, window._trainSource||'camera');
      }
    } catch(e) {
      console.error('[TrainCtrl] save error:', e);
      eventBus.emit(Events.SAMPLES_COLLECTED, { gesture:name, live:false, count:0, error: 'Save failed — is the server running?' });
      return { live:false, sample:null, mirrored:false, error: 'Save failed — is the server running?' };
    }

    eventBus.emit(Events.SAMPLES_COLLECTED, { gesture:name, live:useLive, count: mirroredSample ? 2 : 1 });
    return { live:useLive, sample:sample, mirrored: !!mirroredSample };
  }

  // ── Burst (5 at once) ─────────────────────────────────────────
  async collectLiveBurst(name, count) {
    count = count || 5;
    var samples = [];
    var useLive = this.sensor && this.sensor.handDetected;
    if (useLive) {
      for (var i = 0; i < count; i++) {
        samples.push(this.sensor.getFeatureVector().slice());
        await new Promise(function(r){ setTimeout(r, 80); });
      }
    } else {
      var res  = await fetch(API + '/samples/simulate', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ gesture:name, count:count, sample_type:'static', source:'camera' })
      });
      var data = await res.json();
      samples  = data.samples;
    }
    await this.gm.addStaticSamples(name, samples);
    eventBus.emit(Events.SAMPLES_COLLECTED, { gesture:name, live:useLive, count:count });
  }

  async collectStaticSamples(name, count) { return this.collectLiveBurst(name, count || 5); }

  // ── Bulk capture (one countdown, N samples spaced 350ms apart) ───────
  async collectBulkStatic(name, count) {
    count = count || 5;
    await this.runCountdown(3);
    for (var i = 0; i < count; i++) {
      await this.collectStaticSample(name, true);
      if (i < count - 1) await new Promise(function(r){ setTimeout(r, 350); });
    }
  }

  // ── Dynamic recording — with abort on hand loss ───────────────
  startDynamicRecording(name) {
    this._dynLostFrames = 0;
    this._dynAborted    = false;
    this.gm.startRecording(name);
    eventBus.emit(Events.RECORDING_START, { gesture:name });
  }

  // Called every frame by AppController during dynamic recording
  // Returns: 'done' | 'aborted' | 'recording'
  pushRecordingFrame(features, handCount) {
    if (this._dynAborted) return 'aborted';

    // Track hand loss — abort if lost too many consecutive frames
    var ABORT_THRESHOLD = APP_CONFIG.MEDIAPIPE.HAND_LOSS_ABORT_FRAMES || 3;
    if (handCount === 0) {
      this._dynLostFrames++;
      if (this._dynLostFrames >= ABORT_THRESHOLD) {
        this._dynAborted = true;
        this.gm.stopRecording();
        eventBus.emit(Events.RECORDING_DONE, { gesture: this.gm.recordTarget, aborted: true });
        return 'aborted';
      }
      // Don't push zero frames — hold last frame
      return 'recording';
    } else {
      this._dynLostFrames = 0; // reset on hand re-detected
    }

    var done = this.gm.pushFrame(features);
    eventBus.emit(Events.RECORDING_TICK, {
      frame: this.gm.frameBuffer.length,
      total: APP_CONFIG.NN.DYNAMIC_FRAMES,
    });
    if (done) {
      eventBus.emit(Events.RECORDING_DONE, { gesture: this.gm.recordTarget, aborted: false });
      return 'done';
    }
    return 'recording';
  }

  addGesture(name)    { return this.gm.addGesture(name); }
  removeGesture(name) { this.gm.removeGesture(name); }

  // ── Training (Unified Bidirectional LSTM — static + dynamic) ─────────
  async train() {
    this.isTraining = true;
    this.progress   = 0;
    eventBus.emit(Events.TRAIN_STARTED);

    // Load only samples for the active source
    this.gm.trainSource = window._trainSource || 'camera';
    await this.gm.loadFromDB();

    var sd = this.gm.getStaticTrainingData();
    var dd = this.gm.getDynamicTrainingData();

    if (sd.inputs.length === 0 && dd.inputs.length === 0) {
      this.isTraining = false;
      eventBus.emit(Events.TRAIN_COMPLETE, { error:'No samples' });
      return;
    }

    // ── Build unified gesture list (static first, then dynamic-only) ────
    var allNames = sd.names.slice();
    for (var di = 0; di < dd.names.length; di++) {
      if (allNames.indexOf(dd.names[di]) === -1) allNames.push(dd.names[di]);
    }

    if (allNames.length < 2) {
      this.isTraining = false;
      eventBus.emit(Events.TRAIN_COMPLETE, { error:'Need at least 2 gestures with samples' });
      return;
    }

    // Map name → unified index
    var nameToIdx = {};
    for (var ni = 0; ni < allNames.length; ni++) nameToIdx[allNames[ni]] = ni;

    // Which gestures are primarily static vs dynamic
    var gestureTypes = {};
    for (var si2 = 0; si2 < sd.names.length; si2++) gestureTypes[sd.names[si2]] = 'static';
    for (var di2 = 0; di2 < dd.names.length; di2++) gestureTypes[dd.names[di2]] = 'dynamic';

    // ── Build combined inputs + labels ───────────────────────────────────
    // Static inputs:  41-feature vectors — backend repeats to fill sequence
    // Dynamic inputs: DYNAMIC_FRAMES*41-feature flat sequences
    var allInputs = [];
    var allLabels = [];
    for (var si3 = 0; si3 < sd.inputs.length; si3++) {
      allInputs.push(sd.inputs[si3]);
      allLabels.push(nameToIdx[sd.names[sd.labels[si3]]]);
    }
    for (var di3 = 0; di3 < dd.inputs.length; di3++) {
      allInputs.push(dd.inputs[di3]);
      allLabels.push(nameToIdx[dd.names[dd.labels[di3]]]);
    }

    // ── Single init call ─────────────────────────────────────────────────
    await fetch(API + '/nn/init', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        model_type:    'unified',
        output_size:   allNames.length,
        gestures:      allNames,
        gesture_types: gestureTypes,
        feature_version: APP_CONFIG.NN.FEATURE_VERSION,
      })
    });

    // ── Train unified BiLSTM — single call, backend handles SWA + warmup ────
    var EPOCHS = APP_CONFIG.NN.TRAINING_EPOCHS;
    var lr     = (dd.inputs.length > 0) ? 0.001 : 0.008;

    // Progress updates come via WebSocket train_progress events (handled in AppController).
    // A single fetch lets the backend run the full training pipeline uninterrupted.
    var res  = await fetch(API + '/nn/train', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ model_type:'unified', inputs:allInputs, labels:allLabels, epochs:EPOCHS, lr:lr })
    });
    var data = await res.json();

    // Guard: backend returned an error (e.g. 401 admin PIN, 500 internal error)
    // NOTE: do NOT use !data.accuracy — that is falsy for 0.0 and would hide a real (if low) result
    if (!res.ok || data.error || data.accuracy == null) {
      this.isTraining = false;
      eventBus.emit(Events.TRAIN_COMPLETE, { error: data.error || data.detail || 'Training failed (server error)' });
      return;
    }

    // Mirror final stats to both frontend NN objects
    this.staticNN.accuracy      = data.accuracy;
    this.staticNN.val_accuracy  = data.val_accuracy || 0;
    this.staticNN.loss          = data.loss;
    this.staticNN.epochs        = data.epochs;
    this.staticNN.trained       = true;
    this.dynamicNN.accuracy     = data.accuracy;
    this.dynamicNN.val_accuracy = data.val_accuracy || 0;
    this.dynamicNN.loss         = data.loss;
    this.dynamicNN.epochs       = data.epochs;
    this.dynamicNN.trained      = true;

    if (!window._perGestAcc) window._perGestAcc = {};
    window._perGestAcc.static  = data.per_gesture_acc || {};
    window._perGestAcc.dynamic = data.per_gesture_acc || {};

    await fetch(API + '/nn/save/unified?source=' + (window._trainSource||'camera'), { method:'POST' });

    this.isTraining = false;
    this.progress   = 100;
    eventBus.emit(Events.TRAIN_COMPLETE, {
      staticAcc:  this.staticNN.accuracy,
      dynamicAcc: this.dynamicNN.accuracy,
    });

    try {
      var pa = await (await fetch(API + '/nn/per_gesture_accuracy')).json();
      window._perGestAcc = pa;
    } catch(e) {}
  }

  // ── Delete / Reset ────────────────────────────────────────────
  async deleteModel() {
    await fetch(API + '/nn/reset_all', { method:'POST' });
    this.staticNN.reset();
    this.dynamicNN.reset();
    this.isTraining = false;
    this.progress   = 0;
    window._perGestAcc = { static:{}, dynamic:{} };
    eventBus.emit(Events.STATE_UPDATED);
  }

  async resetAll() {
    this.staticNN.reset();
    this.dynamicNN.reset();
    await this.gm.resetAll();
    this.isTraining = false;
    this.progress   = 0;
    window._perGestAcc = { static:{}, dynamic:{} };
    eventBus.emit(Events.STATE_UPDATED);
  }

  getStats() {
    return {
      staticAccuracy:  this.staticNN.accuracy,
      staticLoss:      this.staticNN.loss,
      staticEpochs:    this.staticNN.epochs,
      staticTrained:   this.staticNN.trained,
      valAccuracy:     this.staticNN.val_accuracy || 0,
      dynamicAccuracy: this.dynamicNN.accuracy,
      dynamicLoss:     this.dynamicNN.loss,
      dynamicEpochs:   this.dynamicNN.epochs,
      dynamicTrained:  this.dynamicNN.trained,
      isTraining:      this.isTraining,
      progress:        this.progress,
    };
  }
}
