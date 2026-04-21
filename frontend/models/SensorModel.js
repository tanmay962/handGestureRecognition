// models/SensorModel.js — Gesture Detection v1.0
// Camera-only sensor state. Stores full 41-feature Holistic vector.
'use strict';

var SensorModel = (function() {
  function SensorModel() {
    this.handDetected    = false;
    this.handCount       = 0;
    this.faceDetected    = false;
    this.poseDetected    = false;
    this.source          = 'camera';
    this._fullVector     = null; // 41-feature vector from Holistic
    // For finger bar display (first 5 features are curls)
    this.flex            = [0, 0, 0, 0, 0];
    this.calibration     = null;
  }

  // Called by RecognitionController on every FEATURES_EXTRACTED event
  SensorModel.prototype.setFromFeatures = function(features, meta) {
    meta = meta || {};
    if (!features || features.length < 41) return;
    this._fullVector  = features.slice(0, 41);
    this.flex         = features.slice(0, 5); // curls for finger bar display
    this.handDetected = features[36] > 0.5 || features[37] > 0.5; // dom or aux present
    this.handCount    = (features[36] > 0.5 ? 1 : 0) + (features[37] > 0.5 ? 1 : 0);
    this.faceDetected = features[38] > 0.5;
    this.poseDetected = meta.poseDetected || false;
    this.source       = 'camera';
  };

  SensorModel.prototype.getFeatureVector = function() {
    if (this._fullVector && this._fullVector.length === 41) {
      return this._fullVector.slice();
    }
    // No data yet — return zeros
    return new Array(41).fill(0);
  };

  // Count extended fingers (features 0-4 are curls, low curl = extended)
  SensorModel.prototype.countExtendedFingers = function() {
    var count = 0;
    for (var i = 0; i < 5; i++) {
      if (this.flex[i] < 0.3) count++;
    }
    return count;
  };

  SensorModel.prototype.reset = function() {
    this.handDetected = false;
    this.handCount    = 0;
    this.faceDetected = false;
    this._fullVector  = null;
    this.flex         = [0, 0, 0, 0, 0];
  };

  return SensorModel;
})();
