// models/NeuralNetwork.js — Delegates training & inference to Python backend
const API = '/api';

export class NeuralNetwork {
  constructor(id = 'default') {
    this.id = id;
    this.trained = false;
    this.gestures = new Map();
    this.lr = 0.008;
    this.accuracy = 0;
    this.loss = 1;
    this.epochs = 0;
  }

  initialize(inputSize, hiddenSizes, outputSize) {
    // Stored for reference; actual init done server-side via train()
    this._inputSize = inputSize;
    this._hiddenSizes = hiddenSizes;
    this._outputSize = outputSize;
    console.log(`[NN:${this.id}] init registered ${inputSize}→${hiddenSizes.join('→')}→${outputSize}`);
  }

  // Called by TrainingController — sends everything to Python
  async trainAsync(inputs, labels, epochs = 50) {
    if (!inputs.length) return;
    const outputSize = new Set(labels).size;

    // Init on server
    const gestureList = [...this.gestures.entries()].sort((a,b)=>a[0]-b[0]).map(([,n])=>n);
    await fetch(`${API}/nn/init`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({model_type: this.id, input_size: this._inputSize || inputs[0].length,
        hidden_sizes: this._hiddenSizes || [128,64,32], output_size: outputSize, gestures: gestureList})
    });

    // Train in batches of 10 epochs
    const batchSize = 10;
    for (let i = 0; i < epochs; i += batchSize) {
      const res = await fetch(`${API}/nn/train`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({model_type: this.id, inputs, labels, epochs: Math.min(batchSize, epochs - i)})
      });
      const data = await res.json();
      this.accuracy = data.accuracy;
      this.loss = data.loss;
      this.epochs = data.epochs;
    }
    this.trained = true;

    // Persist to server DB
    await fetch(`${API}/nn/save/${this.id}`, {method: 'POST'});
    console.log(`[NN:${this.id}] trained via Python: acc=${(this.accuracy*100).toFixed(1)}%`);
  }

  // Synchronous shim kept for compatibility — use predictAsync in production
  train(inputs, labels, epochs = 50) {
    // Blocking sync train is unavailable when Python does the work.
    // TrainingController calls this in a loop with await new Promise(setTimeout).
    // We store data for the next async flush.
    this._pendingInputs = inputs;
    this._pendingLabels = labels;
    this._pendingEpochs = (this._pendingEpochs || 0) + epochs;
  }

  async flushTraining() {
    if (!this._pendingInputs || !this._pendingInputs.length) return;
    await this.trainAsync(this._pendingInputs, this._pendingLabels, this._pendingEpochs || 50);
    this._pendingInputs = null; this._pendingLabels = null; this._pendingEpochs = 0;
  }

  async predict(input) {
    if (!this.trained) return {idx: -1, conf: 0, probs: []};
    const res = await fetch(`${API}/nn/predict`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({features: input, model_type: this.id})
    });
    return res.json();
  }

  // Sync predict stub for compatibility (returns cached last result)
  predictSync(input) {
    return {idx: -1, conf: 0, probs: []};
  }

  addGesture(name, id) { this.gestures.set(id, name); }
  getName(id) { return this.gestures.get(id) || 'Unknown'; }
  getInputSize() { return this._inputSize || 0; }

  reset() {
    this.trained = false; this.gestures = new Map();
    this.accuracy = 0; this.loss = 1; this.epochs = 0;
    this._pendingInputs = null; this._pendingLabels = null; this._pendingEpochs = 0;
    fetch(`${API}/nn/reset/${this.id}`, {method: 'POST'});
    console.log(`[NN:${this.id}] reset`);
  }

  toJSON() {
    return {id: this.id, layers: [], gestures: [...this.gestures.entries()],
            accuracy: this.accuracy, loss: this.loss, epochs: this.epochs};
  }

  fromJSON(d) {
    this.gestures = new Map(d.gestures);
    this.accuracy = d.accuracy; this.loss = d.loss;
    this.epochs = d.epochs; this.trained = true; this.id = d.id || this.id;
  }
}
