// services/CameraService.js — Gesture Detection v1.0
// MediaPipe Holistic: hands + face + body = 41 features
// [hand1x11] + [hand2x11] + [face x8] + [pose x6] + [flags x3] + [pad x2] = 41
'use strict';

var CameraService = (function() {
  function CameraService() {
    this.holistic     = null;
    this.active       = false;
    this.fps          = 0;
    this._fc          = 0;
    this._ft          = Date.now();
    this._stream      = null;
    this._rafId       = null;
    this.handCount    = 0;
    this.faceDetected = false;
    this.poseDetected = false;
    this.lastFeatures = null;
    this._lastHandsData = [];
    this._lastFaceLM    = null;
    this._lastPoseLM    = null;

    // Camera facing mode: 'user' (front) or 'environment' (back)
    this.facingMode = 'user';
    // Current live prediction for canvas overlay
    this._currentPrediction = null;
    // Status text shown on canvas when no prediction is active
    this._statusText = null;

    // Persistent DOM elements — never destroyed by re-renders
    this.videoEl = document.createElement('video');
    this.videoEl.setAttribute('autoplay', '');
    this.videoEl.setAttribute('playsinline', '');
    this.videoEl.muted = true;
    this.videoEl.style.cssText = 'width:100%;display:block;transform:scaleX(-1)';

    this.canvasEl = document.createElement('canvas');
    this.canvasEl.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;transform:scaleX(-1)';
  }

  CameraService.prototype.initialize = function() {
    if (typeof Holistic === 'undefined') {
      console.warn('[Camera] MediaPipe Holistic not loaded');
      return null;
    }
    if (this.holistic) return Promise.resolve(true);
    var c = APP_CONFIG.MEDIAPIPE;
    var self = this;
    this._holisticReady = false;
    this.holistic = new Holistic({
      locateFile: function(f) { return c.CDN_BASE + '/' + f; }
    });
    this.holistic.setOptions({
      modelComplexity:          c.MODEL_COMPLEXITY,
      smoothLandmarks:          c.SMOOTH_LANDMARKS,
      minDetectionConfidence:   c.MIN_DETECTION_CONFIDENCE,
      minTrackingConfidence:    c.MIN_TRACKING_CONFIDENCE,
    });
    this.holistic.onResults(function(r) { self._onResults(r); });
    // Wait for all model files to finish downloading before allowing send()
    return this.holistic.initialize().then(function() {
      self._holisticReady = true;
      console.log('[Camera] Holistic v1.0 initialized and ready');
      return true;
    });
  };

  CameraService.prototype.start = function() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera API unavailable. Open the app at http://localhost:8000 (not 0.0.0.0) — browsers block camera access on non-localhost addresses.');
    }
    var initPromise = this.initialize();
    if (!initPromise) throw new Error('MediaPipe Holistic not available');
    var self = this;
    return Promise.all([
      initPromise,
      navigator.mediaDevices.getUserMedia({
        video: { facingMode: self.facingMode, width: { ideal: 640 }, height: { ideal: 480 } }
      }).then(function(stream) {
        self._stream = stream;
        self.videoEl.srcObject = stream;
        self._applyMirror();
        return self.videoEl.play();
      })
    ]).then(function() {
      self.active = true;
      self._startLoop();
      eventBus.emit(Events.CAMERA_STARTED);
      console.log('[Camera] started facingMode=' + self.facingMode);
    });
  };

  // ── Mirror transform (front camera = mirrored, back = normal) ────
  CameraService.prototype._applyMirror = function() {
    var t = this.facingMode === 'user' ? 'scaleX(-1)' : 'none';
    this.videoEl.style.transform  = t;
    this.canvasEl.style.transform = t;
  };

  // ── Switch between front / back camera ───────────────────────
  CameraService.prototype.switchCamera = function() {
    var self = this;
    this.facingMode = this.facingMode === 'user' ? 'environment' : 'user';
    this._applyMirror();
    this._currentPrediction = null;
    if (!this.active || !this._stream) return Promise.resolve();
    // Stop existing tracks then open new stream
    this._stream.getTracks().forEach(function(t) { t.stop(); });
    this._stream = null;
    // Try 'exact' first (needed on some Android devices), fall back on failure
    return navigator.mediaDevices.getUserMedia({
      video: { facingMode: { exact: self.facingMode }, width: { ideal: 640 }, height: { ideal: 480 } }
    }).catch(function() {
      return navigator.mediaDevices.getUserMedia({
        video: { facingMode: self.facingMode, width: { ideal: 640 }, height: { ideal: 480 } }
      });
    }).then(function(stream) {
      self._stream = stream;
      self.videoEl.srcObject = stream;
      return self.videoEl.play();
    }).catch(function(err) {
      // Roll back if new camera unavailable
      self.facingMode = self.facingMode === 'user' ? 'environment' : 'user';
      self._applyMirror();
      console.warn('[Camera] switchCamera failed, reverting:', err);
    });
  };

  // ── Set live prediction for canvas overlay ───────────────────
  CameraService.prototype.setPrediction = function(name, conf, model) {
    this._currentPrediction = name ? { name: name, conf: conf || 0, model: model || '' } : null;
  };

  // ── Set status text shown when no prediction is active ───────
  CameraService.prototype.setStatus = function(text) {
    this._statusText = text || null;
  };

  CameraService.prototype.stop = function() {
    this.active = false;
    if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
    if (this._stream) {
      this._stream.getTracks().forEach(function(t) { t.stop(); });
      this._stream = null;
    }
    this.videoEl.srcObject = null;
    this.handCount = 0;
    eventBus.emit(Events.CAMERA_STOPPED);
  };

  CameraService.prototype.mountInto = function(container) {
    if (!container) return;
    if (this.videoEl.parentNode !== container)  container.appendChild(this.videoEl);
    if (this.canvasEl.parentNode !== container) container.appendChild(this.canvasEl);
    this.videoEl.style.display  = this.active ? 'block' : 'none';
    this.canvasEl.style.display = this.active ? 'block' : 'none';
  };

  CameraService.prototype._startLoop = function() {
    var self = this;
    function loop() {
      if (!self.active) return;
      self._rafId = requestAnimationFrame(loop);
      if (self.videoEl.readyState >= 2 && self.holistic && self._holisticReady) {
        self.holistic.send({ image: self.videoEl }).catch(function() {});
      }
      // FPS counter
      self._fc++;
      var now = Date.now();
      if (now - self._ft >= 1000) {
        self.fps = self._fc;
        self._fc = 0;
        self._ft = now;
        var el = document.getElementById('trainFpsB') || document.getElementById('recogFpsB');
        if (el) el.textContent = self.fps + ' FPS';
      }
    }
    loop();
  };

  // ── Holistic results handler ─────────────────────────────────
  CameraService.prototype._onResults = function(results) {
    // Draw skeleton on canvas
    this._drawResults(results);

    // Collect hands data
    var handsData = [];
    if (results.leftHandLandmarks) {
      handsData.push({ label: 'Left',  lm: results.leftHandLandmarks });
    }
    if (results.rightHandLandmarks) {
      handsData.push({ label: 'Right', lm: results.rightHandLandmarks });
    }

    this._lastHandsData = handsData;
    this._lastFaceLM    = results.faceLandmarks    || null;
    this._lastPoseLM    = results.poseLandmarks    || null;
    this.handCount      = handsData.length;
    this.faceDetected   = !!results.faceLandmarks;
    this.poseDetected   = !!results.poseLandmarks;

    // Build 41-feature vector
    var features = this._buildFeatureVector(handsData, results.faceLandmarks, results.poseLandmarks);
    this.lastFeatures = features;

    // Update hand badge
    var hb = document.getElementById('trainHandB') || document.getElementById('recogHandB');
    if (hb) {
      if (this.handCount === 0) hb.textContent = 'No Hand';
      else if (this.handCount === 1) hb.textContent = '1 Hand';
      else hb.textContent = '2 Hands';
    }

    eventBus.emit(Events.FEATURES_EXTRACTED, {
      features:  features,
      handCount: this.handCount,
      faceDetected: this.faceDetected,
      poseDetected: this.poseDetected,
    });
  };

  // ── 41-feature vector builder ────────────────────────────────
  CameraService.prototype._buildFeatureVector = function(handsData, faceLM, poseLM) {
    var ZEROS_11 = [0,0,0,0,0, 0,0,0,0,0,0];
    var dom = null, aux = null, domPresent = 0, auxPresent = 0;

    for (var i = 0; i < handsData.length; i++) {
      var label = this._correctHandedness(handsData[i].label);
      var feat  = this._extractOneHand(handsData[i].lm);
      if (label === APP_CONFIG.MEDIAPIPE.DOMINANT_HAND) {
        dom = feat; domPresent = 1;
      } else {
        aux = feat; auxPresent = 1;
      }
    }

    // Face features (8)
    var faceFeat = this._extractFaceFeatures(faceLM, dom, aux);
    // Pose features (6)
    var poseFeat = this._extractPoseFeatures(poseLM, dom, aux);
    // Presence flags (3)
    var flags = [domPresent, auxPresent, this.faceDetected ? 1 : 0];

    var vec = (dom || ZEROS_11).concat(aux || ZEROS_11)
              .concat(faceFeat)
              .concat(poseFeat)
              .concat(flags);

    // Ensure exactly 41 features
    while (vec.length < 41) vec.push(0);
    return vec.slice(0, 41);
  };

  // ── Single hand: 11 features ─────────────────────────────────
  CameraService.prototype._extractOneHand = function(lm) {
    var tips  = [4, 8, 12, 16, 20];
    var pips  = [3, 6, 10, 14, 18];
    var mcps  = [2, 5,  9, 13, 17];
    var curls = [];
    for (var i = 0; i < 5; i++) {
      var t2m = dist3D(lm[tips[i]], lm[mcps[i]]);
      var p2m = dist3D(lm[pips[i]], lm[mcps[i]]);
      curls.push(clamp(1 - (p2m > 0.001 ? t2m / (p2m * 2.5) : 0), 0, 1));
    }
    // Hand direction: wrist(0) → mid-finger MCP(9)
    var hd = norm3D({ x: lm[9].x-lm[0].x, y: lm[9].y-lm[0].y, z: lm[9].z-lm[0].z });
    // Side direction: pinky MCP(17) → index MCP(5)
    var sd = norm3D({ x: lm[17].x-lm[5].x, y: lm[17].y-lm[5].y, z: lm[17].z-lm[5].z });
    return curls.concat([hd.x, hd.y, hd.z, sd.x*0.1, sd.y*0.1, sd.z*0.1]);
  };

  // ── Face features: 8 values ──────────────────────────────────
  CameraService.prototype._extractFaceFeatures = function(faceLM, dom, aux) {
    if (!faceLM || faceLM.length < 468) return [0,0,0, 0,0, 0,0, 0];
    // Nose tip = landmark 1
    var nose = faceLM[1];
    // Eye distance: left eye outer (33) to right eye outer (263)
    var eyeDist = dist3D(faceLM[33], faceLM[263]);
    // Face tilt: angle of line from left to right eye
    var dx = faceLM[263].x - faceLM[33].x;
    var dy = faceLM[263].y - faceLM[33].y;
    var tilt = Math.atan2(dy, dx);

    // Hand-to-nose offsets (only if hands detected)
    var h1nx = 0, h1ny = 0, h2nx = 0, h2ny = 0;
    if (dom && dom.length >= 2) {
      // wrist position approximated from hand features — use nose as reference
      // Use feature[5,6] (hand dir) as proxy for relative position
      h1nx = clamp(dom[5] - nose.x, -1, 1);
      h1ny = clamp(dom[6] - nose.y, -1, 1);
    }
    if (aux && aux.length >= 2) {
      h2nx = clamp(aux[5] - nose.x, -1, 1);
      h2ny = clamp(aux[6] - nose.y, -1, 1);
    }

    return [
      nose.x, nose.y,           // [0,1] nose position
      clamp(eyeDist * 5, 0, 1), // [2]   face scale
      h1nx, h1ny,               // [3,4] hand1 to nose
      h2nx, h2ny,               // [5,6] hand2 to nose
      clamp(tilt / Math.PI, -1, 1), // [7] face tilt
    ];
  };

  // ── Pose features: 6 values ──────────────────────────────────
  CameraService.prototype._extractPoseFeatures = function(poseLM, dom, aux) {
    if (!poseLM || poseLM.length < 33) return [0,0, 0,0, 0,0];
    // Shoulders: left=11, right=12
    var lSh = poseLM[11], rSh = poseLM[12];
    var shoulderMidX = (lSh.x + rSh.x) / 2;
    var shoulderMidY = (lSh.y + rSh.y) / 2;

    // Hand height relative to shoulder (using wrist landmarks 15=left, 16=right)
    var lWr = poseLM[15], rWr = poseLM[16];
    var h1ShDy = dom ? clamp(shoulderMidY - rWr.y, -1, 1) : 0;
    var h2ShDy = aux ? clamp(shoulderMidY - lWr.y, -1, 1) : 0;

    // Elbow angle: right elbow = landmark 14, right shoulder = 12, right wrist = 16
    var rEl = poseLM[14];
    var v1x = rSh.x - rEl.x, v1y = rSh.y - rEl.y;
    var v2x = rWr.x - rEl.x, v2y = rWr.y - rEl.y;
    var mag = Math.sqrt(v1x*v1x+v1y*v1y) * Math.sqrt(v2x*v2x+v2y*v2y);
    var elbowAngle = mag > 0.001 ? Math.acos(clamp((v1x*v2x+v1y*v2y)/mag,-1,1)) / Math.PI : 0;

    var bodyVisible = (poseLM[11].visibility > 0.5 && poseLM[12].visibility > 0.5) ? 1 : 0;

    return [
      clamp(shoulderMidX, 0, 1),  // [0]
      clamp(shoulderMidY, 0, 1),  // [1]
      h1ShDy,                     // [2]
      h2ShDy,                     // [3]
      elbowAngle,                 // [4]
      bodyVisible,                // [5]
    ];
  };

  // ── Mirror augmentation ──────────────────────────────────────
  CameraService.prototype.getMirroredFeatures = function(handsData) {
    if (!handsData || handsData.length === 0) return null;
    var mirrored = handsData.map(function(h) {
      return {
        label: h.label === 'Left' ? 'Right' : 'Left',
        lm: h.lm.map(function(p) {
          return { x: 1 - p.x, y: p.y, z: -p.z };
        }),
      };
    });
    return this._buildFeatureVector(mirrored, this._lastFaceLM, this._lastPoseLM);
  };

  CameraService.prototype._correctHandedness = function(label) {
    // Front camera is mirrored → flip handedness so dominant hand matches visual
    // Back camera is NOT mirrored → keep MediaPipe's labels as-is
    if (this.facingMode !== 'user') return label;
    return label === 'Left' ? 'Right' : 'Left';
  };

  // ── Skeleton drawing ─────────────────────────────────────────
  CameraService.prototype._drawResults = function(results) {
    var canvas = this.canvasEl;
    var video  = this.videoEl;
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw hand connections
    var handColor = '#5eead4';
    var connections = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],
      [0,13],[13,14],[14,15],[15,16],
      [0,17],[17,18],[18,19],[19,20],
      [5,9],[9,13],[13,17],
    ];

    var self = this;
    function drawHand(lm) {
      ctx.strokeStyle = handColor;
      ctx.lineWidth   = 2;
      connections.forEach(function(c) {
        var a = lm[c[0]], b = lm[c[1]];
        ctx.beginPath();
        ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
        ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
        ctx.stroke();
      });
      lm.forEach(function(p) {
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, Math.PI*2);
        ctx.fillStyle = handColor;
        ctx.fill();
      });
    }

    if (results.leftHandLandmarks)  drawHand(results.leftHandLandmarks);
    if (results.rightHandLandmarks) drawHand(results.rightHandLandmarks);

    // Draw pose skeleton (just shoulders + arms)
    if (results.poseLandmarks) {
      var pose = results.poseLandmarks;
      var poseConns = [[11,12],[11,13],[13,15],[12,14],[14,16]];
      ctx.strokeStyle = '#a78bfa';
      ctx.lineWidth   = 2;
      poseConns.forEach(function(c) {
        var a = pose[c[0]], b = pose[c[1]];
        if (a.visibility > 0.5 && b.visibility > 0.5) {
          ctx.beginPath();
          ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
          ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
          ctx.stroke();
        }
      });
    }

    // ── Status text overlay (when no prediction is active) ───
    if (!this._currentPrediction && this._statusText) {
      var cw2 = canvas.width, ch2 = canvas.height;
      var sFontSize = Math.min(18, Math.max(11, cw2 * 0.033));
      var bgGrad2 = ctx.createLinearGradient(0, ch2 * 0.78, 0, ch2);
      bgGrad2.addColorStop(0, 'rgba(6,8,13,0)');
      bgGrad2.addColorStop(1, 'rgba(6,8,13,0.75)');
      ctx.fillStyle = bgGrad2;
      ctx.fillRect(0, ch2 * 0.78, cw2, ch2 * 0.22);
      ctx.save();
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.font        = 'bold ' + sFontSize + 'px "IBM Plex Mono",monospace';
      ctx.shadowColor = 'rgba(0,0,0,0.9)';
      ctx.shadowBlur  = 10;
      ctx.fillStyle   = 'rgba(180,185,200,0.85)';
      ctx.fillText(this._statusText, cw2 / 2, ch2 - 12);
      ctx.restore();
    }

    // ── Prediction overlay drawn directly on canvas ───────────
    if (this._currentPrediction) {
      var pred = this._currentPrediction;
      var cw = canvas.width, ch = canvas.height;
      var nameFontSize = Math.min(60, Math.max(28, cw * 0.11));
      var confFontSize = Math.round(nameFontSize * 0.28);

      // Fade gradient at bottom so text is readable over any background
      var bgGrad = ctx.createLinearGradient(0, ch * 0.58, 0, ch);
      bgGrad.addColorStop(0, 'rgba(6,8,13,0)');
      bgGrad.addColorStop(1, 'rgba(6,8,13,0.88)');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, ch * 0.58, cw, ch * 0.42);

      ctx.save();
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'alphabetic';

      // Gesture name — teal→purple gradient with glow
      ctx.font        = 'bold ' + nameFontSize + 'px "IBM Plex Mono",monospace';
      ctx.shadowColor = 'rgba(0,0,0,0.95)';
      ctx.shadowBlur  = 18;
      var nameGrad = ctx.createLinearGradient(cw * 0.2, 0, cw * 0.8, 0);
      nameGrad.addColorStop(0, '#5eead4');
      nameGrad.addColorStop(1, '#a78bfa');
      ctx.fillStyle = nameGrad;
      ctx.fillText(pred.name, cw / 2, ch - 30);

      // Confidence + model tag
      ctx.font        = 'bold ' + confFontSize + 'px "IBM Plex Mono",monospace';
      ctx.fillStyle   = 'rgba(220,224,236,0.80)';
      ctx.shadowBlur  = 6;
      var confLabel   = (pred.conf * 100).toFixed(1) + '%';
      if (pred.model) confLabel += '  [' + pred.model + ']';
      ctx.fillText(confLabel, cw / 2, ch - 8);

      ctx.restore();
    }
  };

  return CameraService;
})();
