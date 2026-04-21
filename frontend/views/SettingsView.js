// views/SettingsView.js — Gesture Detection v1.0
// Gemini AI, Camera, TTS, Recognition confidence, Admin
'use strict';

function renderSettingsTab(state) {
  var camActive      = state.camActive;
  var tts            = state.tts        || { enabled: true, auto: false, rate: 1.0 };
  var confThresh     = state.confThresh || 0.65;
  var apiKey         = state.apiKey     || '';
  var apiStatus      = state.apiStatus  || 'none';
  var mqttConnected  = state.mqttConnected  || false;
  var mqttEnabled    = state.mqttEnabled    || false;
  var mqttBroker     = state.mqttBroker     || '';
  var mqttTopic      = state.mqttTopic      || '';

  var gemBorder = apiStatus === 'ok' ? 'var(--g)' : apiStatus === 'error' ? 'var(--r)' : 'var(--brd)';
  var gemBtn    = apiStatus === 'ok' ? 'g' : 'a';
  var gemLabel  = apiStatus === 'ok' ? '✓ Active' : 'Connect';
  var gemStatus = apiStatus === 'ok'
    ? '<div style="padding:6px 10px;background:var(--gD);border-radius:6px;font-size:10px;color:var(--g)">✓ Gemini connected</div>'
    : '';

  var gemFn = "(async function(){await window._app.connectGemini(document.getElementById('apiKeyInput').value)})()";

  var out = '';

  out += Card('✨ Google Gemini AI',
    '<div style="font-size:11px;color:var(--mx);line-height:1.6;margin-bottom:12px">' +
      'AI suggestions, grammar correction, sentence completion. Free at aistudio.google.com' +
    '</div>' +
    '<div class="fr g6 mb8">' +
      '<input class="inp f1" id="apiKeyInput" type="password" placeholder="Paste Gemini API key..." value="' + apiKey + '" style="border-color:' + gemBorder + '">' +
      Btn(gemLabel, gemFn, gemBtn) +
    '</div>' + gemStatus,
    'border-color:rgba(251,191,36,.2)'
  );

  var mqttDot    = mqttConnected ? '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--g);margin-right:5px"></span>' : '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--dm);margin-right:5px"></span>';
  var mqttStatus = mqttConnected ? '<div style="font-size:10px;color:var(--g);padding:6px 10px;background:var(--gD);border-radius:6px;margin-top:8px">✓ Publishing gestures to broker</div>' : '';
  out += Card('📡 MQTT Publish (Camera Mode)',
    '<div style="font-size:11px;color:var(--mx);line-height:1.6;margin-bottom:10px">' +
      'Publish recognized gestures in real-time so other devices (Raspberry Pi, Arduino, phone) can subscribe and react.' +
    '</div>' +
    SettingRow(mqttDot + (mqttConnected ? 'Connected' : 'Disconnected'), mqttConnected ? mqttBroker : 'HiveMQ public broker',
      mqttConnected
        ? Btn('Disconnect', 'window._app.disconnectMQTT()', 'r', 'sm')
        : Btn('Connect', 'window._app.connectMQTT()', 'g', 'sm')
    ) +
    (mqttConnected
      ? '<div style="font-size:9px;color:var(--dm);margin-top:6px">Topic: <span style="color:var(--p)">' + mqttTopic + '</span></div>'
      : '') +
    mqttStatus +
    '<div style="font-size:9px;color:var(--dm);margin-top:8px">Subscribe on any device:<br>' +
    '<code style="color:var(--a)">mosquitto_sub -h broker.hivemq.com -t "' + (mqttTopic || 'gesture-detection/results/gesture') + '"</code></div>'
  );

  out += Card('📷 Camera',
    SettingRow('Live Camera', 'MediaPipe Holistic — hands + face + body',
      Toggle(camActive, camActive ? 'window._app.stopCamera()' : 'window._app.startCamera()')
    ) +
    '<div style="font-size:9px;color:var(--dm);margin-top:6px">41 features: hand×11×2 + face×8 + pose×6 + flags×3</div>'
  );

  out += Card('🔊 Text-to-Speech',
    SettingRow('Enable TTS', '', Toggle(tts.enabled, 'window._app.setTTSEnabled(' + String(!tts.enabled) + ')')) +
    SettingRow('Auto-Speak', '', Toggle(tts.auto, 'window._app.setAutoSpeak(' + String(!tts.auto) + ')')) +
    SettingRow('Speed: ' + tts.rate.toFixed(1) + 'x', '',
      '<input type="range" min=".5" max="2" step=".1" value="' + tts.rate + '" ' +
      'oninput="window._app.setTTSRate(parseFloat(this.value))" style="width:100px;accent-color:var(--g)">'
    )
  );

  out += Card('🎯 Recognition',
    SettingRow('Confidence: ' + Math.round(confThresh * 100) + '%', '',
      '<input type="range" min=".3" max=".95" step=".05" value="' + confThresh + '" ' +
      'oninput="window._app.setConfThreshold(parseFloat(this.value))" style="width:100px;accent-color:var(--g)">'
    ) +
    '<div style="font-size:9px;color:var(--dm);margin-top:6px">' +
    'Letter hold 600ms · Word hold 900ms · LSTM threshold 75%<br>' +
    'Letter cooldown 400ms · Same letter 1200ms · Word cooldown 1800ms' +
    '</div>'
  );

  out += Card('💾 Model Backup',
    '<div style="font-size:11px;color:var(--mx);line-height:1.6;margin-bottom:12px">' +
      'Save all your training data to a file. Reload it anytime — on this device or a new deployment.' +
    '</div>' +
    '<div class="fr g6 mb8">' +
      Btn('⬇ Export', 'window._app.exportDataset()', 'g') +
      Btn('⬆ Import', 'document.getElementById(\'importFileInput\').click()', 'a') +
    '</div>' +
    '<input type="file" id="importFileInput" accept=".json" style="display:none" ' +
      'onchange="window._app.importDataset(this.files[0]);this.value=\'\'">' +
    '<div style="font-size:9px;color:var(--dm);margin-top:4px">' +
      'Export saves: training samples + gesture list · Import merges into current dataset' +
    '</div>'
  );

  out += Card('🔒 Admin',
    SettingRow('Change PIN', '',
      '<input class="inp" style="width:80px" id="newPin" placeholder="PIN" maxlength="6">' +
      '<button class="btn btn-o btn-sm" style="margin-left:8px" ' +
      'onclick="window._app.setAdminPin(document.getElementById(\'newPin\').value);' +
      'document.getElementById(\'newPin\').value=\'\';alert(\'PIN updated\')">Set</button>'
    ) +
    SettingRow('Exit Admin', '', Btn('← User Mode', 'window._app.switchMode("user")', 'o', 'sm'))
  );

  return out;
}
