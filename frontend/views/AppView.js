// views/AppView.js
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
      html += _renderHeader(state);
      html += _renderTabs(state);
      html += _renderTabContent(state);
      html += '<div class="footer">gesture-detection · MLP + LSTM · holistic · gemini</div>';
    }

    this.root.innerHTML = html;
    this.ctrl._mountCamera();
  };

  return AppView;
})();

function _renderHeader(state) {
  var camBadge    = state.camActive ? '<span class="bg bg-g"><span class="dot dot-g dot-pulse"></span>Live</span>' : '';
  var modelBadge  = (state.staticTrained || state.dynamicTrained)
    ? '<span class="bg bg-g">Model ready</span>'
    : '<span class="bg bg-d">No model</span>';
  var geminiBadge = state.geminiEnabled ? '<span class="bg bg-a">Gemini</span>' : '';

  return '<div class="hdr">' +
    '<div>' +
      '<div class="hdr-brand">✋ Gesture Detection</div>' +
      '<div class="hdr-title">Hand Recognition <em>System</em></div>' +
    '</div>' +
    '<div class="hdr-badges">' +
      camBadge +
      modelBadge +
      geminiBadge +
    '</div>' +
  '</div>';
}

function _renderTabs(state) {
  var tabs = APP_CONFIG.TABS_ADMIN;
  var html = '<div class="tabs">';
  for (var i = 0; i < tabs.length; i++) {
    var t = tabs[i];
    var active = state.tab === t.id ? ' active' : '';
    html += '<button class="tab' + active + '" onclick="window._app.switchTab(\'' + t.id + '\')">' + t.label + '</button>';
  }
  html += '</div>';
  return html;
}

function _renderTabContent(state) {
  switch (state.tab) {
    case 'detect':    return renderDetectTab(state);
    case 'train':     return renderTrainTab(state);
    case 'sequences': return renderSequenceTab(state);
    case 'settings':  return renderSettingsTab(state);
    default:          return renderDetectTab(state);
  }
}
