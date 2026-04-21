// services/MQTTService.js — Browser MQTT client (camera mode publish)
// Uses mqtt.js over WSS to publish recognized gestures to HiveMQ
'use strict';

var MQTTService = (function() {
  function MQTTService() {
    this._client    = null;
    this.connected  = false;
    this.enabled    = false;
    this.broker     = APP_CONFIG.MQTT.DEFAULT_BROKER;   // wss://broker.hivemq.com:8884/mqtt
    this.topicGesture  = APP_CONFIG.MQTT.TOPIC_RESULTS;   // gesture-detection/results/gesture
    this.topicStatus   = APP_CONFIG.MQTT.TOPIC_STATUS;    // gesture-detection/status
    this._clientId  = 'gesture-browser-' + Math.random().toString(36).slice(2, 9);
    this._onStatusChange = null;
  }

  MQTTService.prototype.connect = function(onStatus) {
    if (typeof mqtt === 'undefined') {
      console.warn('[MQTT] mqtt.js not loaded');
      return;
    }
    if (this._client) return;
    this._onStatusChange = onStatus || null;
    var self = this;

    this._client = mqtt.connect(this.broker, {
      clientId: this._clientId,
      clean:    true,
      reconnectPeriod: 5000,
    });

    this._client.on('connect', function() {
      self.connected = true;
      self.enabled   = true;
      console.log('[MQTT] Browser client connected to ' + self.broker);
      self._publish(self.topicStatus, JSON.stringify({ status: 'online', source: 'camera', ts: Date.now() }));
      if (self._onStatusChange) self._onStatusChange('connected');
    });

    this._client.on('reconnect', function() {
      console.log('[MQTT] Reconnecting…');
      if (self._onStatusChange) self._onStatusChange('reconnecting');
    });

    this._client.on('disconnect', function() {
      self.connected = false;
      if (self._onStatusChange) self._onStatusChange('disconnected');
    });

    this._client.on('error', function(err) {
      console.warn('[MQTT] Error:', err.message || err);
      self.connected = false;
      if (self._onStatusChange) self._onStatusChange('error');
    });

    this._client.on('offline', function() {
      self.connected = false;
      if (self._onStatusChange) self._onStatusChange('offline');
    });
  };

  MQTTService.prototype.disconnect = function() {
    if (!this._client) return;
    this._client.end(true);
    this._client   = null;
    this.connected = false;
    this.enabled   = false;
    if (this._onStatusChange) this._onStatusChange('disconnected');
  };

  MQTTService.prototype.publishGesture = function(gesture, conf, model) {
    if (!this.connected || !this.enabled) return;
    var payload = JSON.stringify({
      gesture:    gesture,
      confidence: Math.round(conf * 100) / 100,
      model:      model || 'static',
      source:     'camera',
      ts:         Date.now(),
    });
    this._publish(this.topicGesture, payload);
  };

  MQTTService.prototype._publish = function(topic, payload) {
    if (!this._client || !this.connected) return;
    this._client.publish(topic, payload, { qos: 0, retain: false });
  };

  return MQTTService;
})();
