// views/AppView.js — Gesture Detection v1.0
// Clean header, 4 tabs only: Detect | Train | Sequences | Settings
'use strict';

var AppView = (function() {
  function AppView(root, ctrl) {
    this.root = root;
    this.ctrl = ctrl;
  }

  AppView.prototype.render = function() {
    var state = this.ctrl.getState();
    var html  = '';

    if (state.mode === 'user') {
      html = renderUserMode(state);
    } else {
      // Header
      html += '<div class="hdr">' +
        '<div>' +
          '<div class="hdr-brand">✋ Gesture Detection ' +
            Badge('v1.0', 'p') + ' ' +
            (state.geminiEnabled ? Badge('Gemini', 'a') + ' ' : '') +
          '</div>' +
          '<div class="hdr-title">Hand Gesture Recognition <em>System</em></div>' +
        '</div>' +
        '<div class="hdr-badges">' +
          Badge(state.camActive ? '📷 Live' : '📷 Off',   state.camActive      ? 'g' : 'd') + ' ' +
          Badge(state.staticTrained  ? 'MLP ✓'  : 'MLP ✗',  state.staticTrained  ? 'g' : 'd') + ' ' +
          Badge(state.dynamicTrained ? 'LSTM ✓' : 'LSTM ✗', state.dynamicTrained ? 'p' : 'd') +
        '</div>' +
      '</div>';

      // Tabs
      html += '<div class="tabs">';
      for (var i = 0; i < APP_CONFIG.TABS_ADMIN.length; i++) {
        var t = APP_CONFIG.TABS_ADMIN[i];
        var active = state.tab === t.id ? ' active' : '';
        html += '<button class="tab' + active + '" data-tab="' + t.id + '" onclick="window._app.switchTab(this.dataset.tab)">' + t.label + '</button>';
      }
      html += '</div>';

      // Tab content
      switch (state.tab) {
        case 'detect':    html += renderDetectTab(state);    break;
        case 'train':     html += renderTrainTab(state);     break;
        case 'sequences': html += renderSequenceTab(state);  break;
        case 'settings':  html += renderSettingsTab(state);  break;
        default:          html += renderDetectTab(state);
      }

      html += '<div class="footer">GESTURE DETECTION v1.0 · MLP + LSTM · HOLISTIC · GEMINI · PWA</div>';
    }

    this.root.innerHTML = html;
    this.ctrl._mountCamera();
  };

  return AppView;
})();
