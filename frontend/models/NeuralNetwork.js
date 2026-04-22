// models/NeuralNetwork.js — State container for trained model metadata
// Training and inference happen in Python; this holds the result for the UI.

export class NeuralNetwork {
  constructor(id) {
    this.id       = id || 'default';
    this.trained  = false;
    this.gestures = new Map();
    this.accuracy = 0;
    this.val_accuracy = 0;
    this.loss     = 1;
    this.epochs   = 0;
  }

  // Called by TrainingController before training to record expected shape
  initialize(inputSize, hiddenSizes, outputSize) {
    this._inputSize   = inputSize;
    this._hiddenSizes = hiddenSizes;
    this._outputSize  = outputSize;
  }

  addGesture(name, id) { this.gestures.set(id, name); }
  getName(id)          { return this.gestures.get(id) || 'Unknown'; }
  getInputSize()       { return this._inputSize || 0; }

  reset() {
    this.trained  = false;
    this.accuracy = 0;
    this.loss     = 1;
    this.epochs   = 0;
    this.gestures = new Map();
    fetch('/api/nn/reset/' + this.id, { method: 'POST' }).catch(function(){});
  }

  toJSON() {
    return {
      id:       this.id,
      gestures: [...this.gestures.entries()],
      accuracy: this.accuracy,
      loss:     this.loss,
      epochs:   this.epochs,
    };
  }

  fromJSON(d) {
    this.gestures = new Map(d.gestures);
    this.accuracy = d.accuracy;
    this.loss     = d.loss;
    this.epochs   = d.epochs;
    this.trained  = true;
    this.id       = d.id || this.id;
  }
}
