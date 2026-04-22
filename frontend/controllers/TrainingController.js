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

    if (useLive) {
      sample = this.sensor.getFeatureVector().slice();

      // Mirror augmentation — only possible with camera (landmarks available)
      if (this._camera && this._camera._lastHandsData && this._camera._lastHandsData.length > 0) {
        mirroredSample = this._camera.getMirroredFeatures(this._camera._lastHandsData);
      }
    } else {
      // Simulate — backend generates the 24-feature vector
      var res  = await fetch(API + '/samples/simulate', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ gesture:name, count:1, sample_type:'static', source:'camera' })
      });
      var data = await res.json();
      sample   = data.samples[0];
      // For simulated data, generate a mirrored version too
      if (data.mirrored && data.mirrored.length > 0) {
        mirroredSample = data.mirrored[0];
      }
    }

    // Save original sample
    await this.gm.addStaticSamples(name, [sample], false, window._trainSource||'camera');

    // Save mirrored sample if available (doubles training data for free)
    if (mirroredSample) {
      await this.gm.addStaticSamples(name, [mirroredSample], true, window._trainSource||'camera');
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

  async generateAllDemo() {
    await this.gm.generateAllDemo();
    eventBus.emit(Events.SAMPLES_COLLECTED, { gesture:'all' });
  }

  // ── Training (MLP static + LSTM dynamic) ─────────────────────
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

    // Validate feature size — warn if mismatch with current config
    var expectedSize = APP_CONFIG.NN.STATIC_INPUT;
    if (sd.inputs.length > 0 && sd.inputs[0].length !== expectedSize) {
      console.warn('[Train] Feature size mismatch — old samples have', sd.inputs[0].length,
                   'features, current config expects', expectedSize,
                   '— clearing old model and retraining with available data');
    }

    // — Static MLP —
    if (sd.inputs.length > 0 && sd.names.length >= 2) {
      for (var si = 0; si < sd.names.length; si++) this.staticNN.addGesture(sd.names[si], si);
      this.staticNN.initialize(sd.inputs[0].length, APP_CONFIG.NN.STATIC_HIDDEN, sd.names.length);

      await fetch(API + '/nn/init', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          model_type:   'static',
          input_size:   sd.inputs[0].length,
          hidden_sizes: APP_CONFIG.NN.STATIC_HIDDEN,
          output_size:  sd.names.length,
          gestures:     sd.names,
          feature_version: APP_CONFIG.NN.FEATURE_VERSION,
        })
      });

      var STATIC_EPOCHS = APP_CONFIG.NN.TRAINING_EPOCHS;
      var BATCH = APP_CONFIG.NN.EPOCH_BATCH;
      for (var ep = 0; ep < STATIC_EPOCHS; ep += BATCH) {
        await new Promise(function(r){ setTimeout(r, 4); });
        var ep2  = Math.min(BATCH, STATIC_EPOCHS - ep);
        var sRes = await fetch(API + '/nn/train', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ model_type:'static', inputs:sd.inputs, labels:sd.labels, epochs:ep2, lr:0.008 })
        });
        var sData = await sRes.json();
        this.staticNN.accuracy     = sData.accuracy;
        this.staticNN.val_accuracy = sData.val_accuracy || 0;
        this.staticNN.loss         = sData.loss;
        this.staticNN.epochs       = sData.epochs;
        this.staticNN.trained      = true;
        if (!window._perGestAcc) window._perGestAcc = {};
        window._perGestAcc.static = sData.per_gesture_acc || {};
        this.progress = Math.min(85, ((ep + ep2) / STATIC_EPOCHS) * 85);
        eventBus.emit(Events.TRAIN_PROGRESS, { progress:this.progress, model:'static', accuracy:sData.accuracy });
      }
      await fetch(API + '/nn/save/static?source=' + (window._trainSource||'camera'), { method:'POST' });
    }

    // — Dynamic LSTM —
    if (dd.inputs.length > 0 && dd.names.length >= 2) {
      await fetch(API + '/nn/init', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          model_type:   'dynamic',
          output_size:  dd.names.length,
          gestures:     dd.names,
          feature_version: APP_CONFIG.NN.FEATURE_VERSION,
        })
      });

      var DYN_EPOCHS = 300, DYN_BATCH = 15;
      for (var dep = 0; dep < DYN_EPOCHS; dep += DYN_BATCH) {
        await new Promise(function(r){ setTimeout(r, 4); });
        var dep2 = Math.min(DYN_BATCH, DYN_EPOCHS - dep);
        var dRes = await fetch(API + '/nn/train', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ model_type:'dynamic', inputs:dd.inputs, labels:dd.labels, epochs:dep2, lr:0.001 })
        });
        var dData = await dRes.json();
        this.dynamicNN.accuracy     = dData.accuracy;
        this.dynamicNN.val_accuracy = dData.val_accuracy || 0;
        this.dynamicNN.loss         = dData.loss;
        this.dynamicNN.epochs       = dData.epochs;
        this.dynamicNN.trained      = true;
        if (!window._perGestAcc) window._perGestAcc = {};
        window._perGestAcc.dynamic = dData.per_gesture_acc || {};
        this.progress = 85 + ((dep + dep2) / DYN_EPOCHS) * 15;
        eventBus.emit(Events.TRAIN_PROGRESS, { progress:this.progress, model:'dynamic', accuracy:dData.accuracy });
      }
      await fetch(API + '/nn/save/dynamic', { method:'POST' });
    }

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
      dynamicAccuracy: this.dynamicNN.accuracy,
      dynamicLoss:     this.dynamicNN.loss,
      dynamicEpochs:   this.dynamicNN.epochs,
      dynamicTrained:  this.dynamicNN.trained,
      isTraining:      this.isTraining,
      progress:        this.progress,
    };
  }
}
