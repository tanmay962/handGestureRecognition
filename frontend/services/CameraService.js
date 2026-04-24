// services/CameraService.js — Gesture Detection v1.0
// MediaPipe Holistic: hands + face + body = 41 features
// [hand1x11] + [hand2x11] + [face x10] + [pose x6] + [flags x3] = 41
// Face (10): nose_x, nose_y, eye_scale, dom_wrist_dx, dom_wrist_dy,
//            aux_wrist_dx, aux_wrist_dy, tilt, mouth_open, eye_open
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

    // Recording trail — filled by AppController event handlers
    this._isRecording    = false;
    this._recordingTrail = []; // [{x, y}] canvas-normalised positions

    // Camera facing mode: 'user' (front) or 'environment' (back)
    this.facingMode = 'user';
    // Current live prediction for canvas overlay (dom hand)
    this._currentPrediction = null;
    // Independent aux-hand prediction for canvas overlay
    this._auxPrediction     = null;
    // Status text shown on canvas when no prediction is active
    this._statusText = null;

    // Persistent DOM elements — never destroyed by re-renders
    this.videoEl = document.createElement('video');
    this.videoEl.setAttribute('autoplay', '');
    this.videoEl.setAttribute('playsinline', '');
    this.videoEl.muted = true;
    this.videoEl.style.cssText = 'width:100%;display:block';

    this.canvasEl = document.createElement('canvas');
    this.canvasEl.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%';
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

  // ── Mirror transform — no flip so video shows natural orientation ──
  CameraService.prototype._applyMirror = function() {
    this.videoEl.style.transform  = 'none';
    this.canvasEl.style.transform = 'none';
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
  CameraService.prototype.setPrediction = function(name, conf, model, auxName, auxConf) {
    this._currentPrediction = name ? { name: name, conf: conf || 0, model: model || '' } : null;
    this._auxPrediction = (auxName && auxConf > 0.3) ? { name: auxName, conf: auxConf || 0 } : null;
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

    // Record wrist trail during dynamic gesture recording
    if (this._isRecording && handsData.length > 0) {
      var wrist = handsData[0].lm[0]; // landmark 0 = wrist
      this._recordingTrail.push({ x: wrist.x, y: wrist.y });
      if (this._recordingTrail.length > 60) this._recordingTrail.shift();
    }

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

    // Face features (10): nose, scale, wrist→nose offsets, tilt, mouth open, eye open
    var faceFeat = this._extractFaceFeatures(faceLM, poseLM, dom, aux);
    // Pose features (6)
    var poseFeat = this._extractPoseFeatures(poseLM, dom, aux);
    // Presence flags (3)
    var flags = [domPresent, auxPresent, this.faceDetected ? 1 : 0];

    // 11 + 11 + 10 + 6 + 3 = 41 exactly
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

  // ── Face features: 10 values ─────────────────────────────────
  // [0,1] nose position  [2] face scale  [3,4] dom wrist→nose  [5,6] aux wrist→nose
  // [7] face tilt  [8] mouth openness  [9] eye openness
  CameraService.prototype._extractFaceFeatures = function(faceLM, poseLM, dom, aux) {
    if (!faceLM || faceLM.length < 468) return [0,0,0, 0,0, 0,0, 0, 0,0];
    // Nose tip = landmark 1
    var nose = faceLM[1];
    // Eye distance: left eye outer (33) to right eye outer (263)
    var eyeDist = dist3D(faceLM[33], faceLM[263]);
    var eyeScale = eyeDist > 0.001 ? eyeDist : 0.1;
    // Face tilt: angle of line from left to right eye
    var dx = faceLM[263].x - faceLM[33].x;
    var dy = faceLM[263].y - faceLM[33].y;
    var tilt = Math.atan2(dy, dx);

    // Wrist-to-nose offsets using actual pose wrist landmarks
    // Pose: 16 = right wrist (dominant), 15 = left wrist (aux)
    var h1nx = 0, h1ny = 0, h2nx = 0, h2ny = 0;
    if (poseLM && poseLM.length >= 17) {
      var rWrist = poseLM[16];
      var lWrist = poseLM[15];
      if (dom && rWrist.visibility > 0.1) {
        h1nx = clamp(rWrist.x - nose.x, -1, 1);
        h1ny = clamp(rWrist.y - nose.y, -1, 1);
      }
      if (aux && lWrist.visibility > 0.1) {
        h2nx = clamp(lWrist.x - nose.x, -1, 1);
        h2ny = clamp(lWrist.y - nose.y, -1, 1);
      }
    }

    // Mouth openness: upper lip center (13) to lower lip center (14)
    var mouthOpen = clamp(dist3D(faceLM[13], faceLM[14]) / eyeScale, 0, 1);

    // Eye openness: average of left eye (159→145) and right eye (386→374) vertical span
    var leftEyeH  = dist3D(faceLM[159], faceLM[145]);
    var rightEyeH = dist3D(faceLM[386], faceLM[374]);
    var eyeOpen   = clamp((leftEyeH + rightEyeH) / 2 / eyeScale, 0, 1);

    return [
      nose.x, nose.y,                // [0,1] nose position
      clamp(eyeDist * 5, 0, 1),      // [2]   face scale
      h1nx, h1ny,                    // [3,4] dom wrist → nose
      h2nx, h2ny,                    // [5,6] aux wrist → nose
      clamp(tilt / Math.PI, -1, 1),  // [7]   face tilt
      mouthOpen,                     // [8]   mouth openness
      eyeOpen,                       // [9]   eye openness
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
    // MediaPipe identifies hands anatomically (Right = right hand, Left = left hand)
    // regardless of camera orientation — use labels as-is so the dominant (Right) hand
    // always maps to feature slots 0-10 and the finger curl bars update correctly.
    return label;
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
    var handColor = 'rgba(255,255,255,0.75)';
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

    // Draw hand movement trail during dynamic recording
    if (self._isRecording && self._recordingTrail.length > 1) {
      var trail = self._recordingTrail;
      for (var ti = 1; ti < trail.length; ti++) {
        var alpha = ti / trail.length;
        ctx.beginPath();
        ctx.moveTo(trail[ti - 1].x * canvas.width, trail[ti - 1].y * canvas.height);
        ctx.lineTo(trail[ti].x * canvas.width, trail[ti].y * canvas.height);
        ctx.strokeStyle = 'rgba(255,255,255,' + (alpha * 0.7) + ')';
        ctx.lineWidth   = 3 * alpha;
        ctx.stroke();
      }
      // Draw a dot at the latest position
      var last = trail[trail.length - 1];
      ctx.beginPath();
      ctx.arc(last.x * canvas.width, last.y * canvas.height, 5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.fill();
    }

    // Draw pose skeleton (just shoulders + arms)
    if (results.poseLandmarks) {
      var pose = results.poseLandmarks;
      var poseConns = [[11,12],[11,13],[13,15],[12,14],[14,16]];
      ctx.strokeStyle = 'rgba(255,255,255,0.45)';
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

    // Draw face tracking: oval contour, eyes, nose, mouth
    if (results.faceLandmarks && results.faceLandmarks.length >= 468) {
      var fl = results.faceLandmarks;
      var fw = canvas.width, fh = canvas.height;

      function drawPolyline(indices, close) {
        if (indices.length < 2) return;
        ctx.beginPath();
        ctx.moveTo(fl[indices[0]].x * fw, fl[indices[0]].y * fh);
        for (var pi = 1; pi < indices.length; pi++) {
          ctx.lineTo(fl[indices[pi]].x * fw, fl[indices[pi]].y * fh);
        }
        if (close) ctx.closePath();
        ctx.stroke();
      }

      // Face oval
      ctx.strokeStyle = 'rgba(255,255,255,0.20)';
      ctx.lineWidth   = 1;
      drawPolyline([10,338,297,332,284,251,389,356,454,323,361,288,397,365,
                    379,378,400,377,152,148,176,149,150,136,172,58,132,93,
                    234,127,162,21,54,103,67,109,10], false);

      // Eyes
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.lineWidth   = 1.2;
      drawPolyline([33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33], true);
      drawPolyline([362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362], true);

      // Outer lips
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.lineWidth   = 1.2;
      drawPolyline([61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61], true);

      // Nose bridge + tip
      ctx.strokeStyle = 'rgba(255,255,255,0.35)';
      ctx.lineWidth   = 1;
      drawPolyline([168,6,197,195,5,4,1], false);
      drawPolyline([98,240,64,235,236,3,237,238,241,125,327], false);

      // Nose tip dot
      ctx.beginPath();
      ctx.arc(fl[1].x * fw, fl[1].y * fh, 3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fill();
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
    if (this._currentPrediction || this._auxPrediction) {
      var pred = this._currentPrediction;
      var aux  = this._auxPrediction;
      var cw = canvas.width, ch = canvas.height;
      var nameFontSize = Math.min(60, Math.max(28, cw * 0.11));
      var confFontSize = Math.round(nameFontSize * 0.28);
      var auxFontSize  = Math.round(nameFontSize * 0.55);

      var alpha = pred ? Math.min(1, 0.35 + pred.conf * 0.65) : 0.5;

      // Fade gradient — taller when aux label is also showing, normal otherwise
      var gradStart = aux ? 0.50 : 0.58;
      var bgGrad = ctx.createLinearGradient(0, ch * gradStart, 0, ch);
      bgGrad.addColorStop(0, 'rgba(6,8,13,0)');
      bgGrad.addColorStop(1, 'rgba(6,8,13,' + (0.88 * alpha) + ')');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, ch * gradStart, cw, ch * (1 - gradStart));

      ctx.save();
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'alphabetic';

      if (pred) {
        ctx.globalAlpha = alpha;
        // Dom gesture name
        ctx.font        = 'bold ' + nameFontSize + 'px "IBM Plex Mono",monospace';
        ctx.shadowColor = 'rgba(0,0,0,0.95)';
        ctx.shadowBlur  = 18;
        ctx.fillStyle   = '#ffffff';
        ctx.fillText(pred.name, cw / 2, ch - 30);
        // Confidence + model tag
        ctx.font      = 'bold ' + confFontSize + 'px "IBM Plex Mono",monospace';
        ctx.fillStyle = 'rgba(220,224,236,0.80)';
        ctx.shadowBlur = 6;
        var confLabel = (pred.conf * 100).toFixed(1) + '%';
        if (pred.model) confLabel += '  [' + pred.model + ']';
        ctx.fillText(confLabel, cw / 2, ch - 8);
      }

      // Aux-hand prediction — smaller badge above dom prediction
      if (aux) {
        var auxAlpha = Math.min(0.85, 0.3 + aux.conf * 0.55);
        ctx.globalAlpha = auxAlpha;
        ctx.font        = 'bold ' + auxFontSize + 'px "IBM Plex Mono",monospace';
        ctx.shadowColor = 'rgba(0,0,0,0.9)';
        ctx.shadowBlur  = 10;
        ctx.fillStyle   = 'rgba(180,220,255,0.9)'; // tinted blue to distinguish from dom
        var auxY = pred ? ch - 30 - nameFontSize - 10 : ch - 30;
        ctx.fillText(aux.name, cw / 2, auxY);
        ctx.font      = 'bold ' + confFontSize + 'px "IBM Plex Mono",monospace';
        ctx.fillStyle = 'rgba(160,200,240,0.70)';
        ctx.fillText((aux.conf * 100).toFixed(1) + '% [aux]', cw / 2, auxY + confFontSize + 4);
      }

      ctx.restore();
    }
  };

  return CameraService;
})();
