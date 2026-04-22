// views/SettingsView.js
'use strict';

function renderSettingsTab(state) {
  var tts       = state.tts       || { enabled: true, auto: false, rate: 1.0 };
  var confThresh = state.confThresh || 0.65;
  var apiKey    = state.apiKey    || '';
  var apiStatus = state.apiStatus || 'none';
  var mqttConnected = state.mqttConnected || false;
  var mqttBroker    = state.mqttBroker    || '';
  var mqttTopic     = state.mqttTopic     || '';

  return (
    _renderGeminiCard(apiKey, apiStatus) +
    _renderMQTTCard(mqttConnected, mqttBroker, mqttTopic) +
    _renderCameraCard(state.camActive) +
    _renderTTSCard(tts) +
    _renderRecognitionCard(confThresh) +
    _renderBackupCard() +
    _renderAdminCard()
  );
}

function _renderGeminiCard(apiKey, apiStatus) {
  var isConnected = apiStatus === 'ok';
  var borderColor = isConnected ? 'rgba(78,203,158,.2)' : 'transparent';
  var gemFn = "(async function(){await window._app.connectGemini(document.getElementById('apiKeyInput').value)})()";

  return Card(
    '✨ Gemini AI',
    '<div style="font-size:11px;color:var(--mx);line-height:1.6;margin-bottom:12px">' +
      'Next-word suggestions, grammar correction, sentence completion. ' +
      'Get a free key at <span style="color:var(--g)">aistudio.google.com</span>.' +
    '</div>' +
    '<div class="fr g6 mb8">' +
      '<input class="inp f1" id="apiKeyInput" type="password" placeholder="Paste API key…" value="' + apiKey + '">' +
      Btn(isConnected ? '✓ Connected' : 'Connect', gemFn, isConnected ? 'g' : 'a') +
    '</div>' +
    (isConnected ? '<div style="font-size:10px;color:var(--g)">✓ Gemini is active</div>' : ''),
    'border-color:' + borderColor
  );
}

function _renderMQTTCard(connected, broker, topic) {
  var statusDot = connected
    ? '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--g);margin-right:5px"></span>'
    : '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--dm);margin-right:5px"></span>';

  return Card(
    '📡 MQTT Publish',
    '<div style="font-size:11px;color:var(--mx);line-height:1.6;margin-bottom:10px">' +
      'Publish recognized gestures to any MQTT subscriber — Raspberry Pi, Arduino, another phone.' +
    '</div>' +
    SettingRow(
      statusDot + (connected ? 'Connected' : 'Disconnected'),
      connected ? broker : 'HiveMQ public broker',
      connected
        ? Btn('Disconnect', 'window._app.disconnectMQTT()', 'r', 'sm')
        : Btn('Connect', 'window._app.connectMQTT()', 'g', 'sm')
    ) +
    (connected
      ? '<div style="font-size:9px;color:var(--dm);margin-top:6px">Topic: <span style="color:var(--p)">' + topic + '</span></div>' +
        '<div style="font-size:10px;color:var(--g);padding:6px 10px;background:var(--gD);border-radius:6px;margin-top:8px">✓ Publishing gestures to broker</div>'
      : '<div style="font-size:9px;color:var(--dm);margin-top:8px">' +
          'Subscribe anywhere: <code style="color:var(--a)">mosquitto_sub -h broker.hivemq.com -t "' + (topic || 'gesture-detection/results/gesture') + '"</code>' +
        '</div>')
  );
}

function _renderCameraCard(camActive) {
  return Card(
    '📷 Camera',
    SettingRow(
      'Live camera',
      'MediaPipe Holistic — hands + face + body',
      Toggle(camActive, camActive ? 'window._app.stopCamera()' : 'window._app.startCamera()')
    ) +
    '<div style="font-size:9px;color:var(--dm);margin-top:6px">41 features: hand×11×2 + face×8 + pose×6 + flags×3</div>'
  );
}

function _renderTTSCard(tts) {
  return Card(
    '🔊 Text-to-Speech',
    SettingRow('Enable TTS', '', Toggle(tts.enabled, 'window._app.setTTSEnabled(' + !tts.enabled + ')')) +
    SettingRow('Auto-speak on gesture', '', Toggle(tts.auto, 'window._app.setAutoSpeak(' + !tts.auto + ')')) +
    SettingRow(
      'Speed: ' + tts.rate.toFixed(1) + '×',
      '',
      '<input type="range" min=".5" max="2" step=".1" value="' + tts.rate + '" ' +
      'oninput="window._app.setTTSRate(parseFloat(this.value))" style="width:100px;accent-color:var(--g)">'
    )
  );
}

function _renderRecognitionCard(confThresh) {
  return Card(
    '🎯 Recognition',
    SettingRow(
      'Confidence: ' + Math.round(confThresh * 100) + '%',
      'Minimum confidence before a gesture is accepted',
      '<input type="range" min=".3" max=".95" step=".05" value="' + confThresh + '" ' +
      'oninput="window._app.setConfThreshold(parseFloat(this.value))" style="width:100px;accent-color:var(--g)">'
    ) +
    '<div style="font-size:9px;color:var(--dm);margin-top:6px">' +
      'Letter hold 600ms · Word hold 900ms · Same letter cooldown 1.2s' +
    '</div>'
  );
}

function _renderBackupCard() {
  return Card(
    '💾 Data Backup',
    '<div style="font-size:11px;color:var(--mx);line-height:1.6;margin-bottom:12px">' +
      'Export all training samples and settings to a JSON file. Import on any device.' +
    '</div>' +
    '<div class="fr g6 mb8">' +
      Btn('⬇ Export', 'window._app.exportDataset()', 'g') +
      Btn('⬆ Import', 'document.getElementById(\'importFileInput\').click()', 'o') +
    '</div>' +
    '<input type="file" id="importFileInput" accept=".json" style="display:none" ' +
      'onchange="window._app.importDataset(this.files[0]);this.value=\'\'">'
  );
}

function _renderAdminCard() {
  return Card(
    '🔒 Admin',
    SettingRow(
      'Change PIN',
      '',
      '<div class="fr g6">' +
        '<input class="inp" style="width:80px" id="newPin" placeholder="PIN" maxlength="6" type="password">' +
        '<button class="btn btn-o btn-sm" onclick="' +
          'var p=document.getElementById(\'newPin\').value.trim();' +
          'if(!p){alert(\'Enter a PIN first\');}' +
          'else{window._app.setAdminPin(p);document.getElementById(\'newPin\').value=\'\';alert(\'PIN updated\')}' +
        '">Set</button>' +
      '</div>'
    ) +
    SettingRow('Exit Admin', '', Btn('← User Mode', "window._app.switchMode('user')", 'o', 'sm'))
  );
}
