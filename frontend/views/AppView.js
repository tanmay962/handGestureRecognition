// views/AppView.js
'use strict';

var AppView = (function () {
  function AppView(root, ctrl) {
    this.root = root;
    this.ctrl = ctrl;
  }

  AppView.prototype.render = function () {
    var state = this.ctrl.getState();
    var html = '';

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
  var trained = state.staticTrained || state.dynamicTrained;
  var modelBadge = trained
    ? '<span class="bg bg-g">model ready</span>'
    : '';
  var camBadge = state.camActive
    ? '<span class="bg bg-g"><span class="dot dot-g dot-pulse"></span> live</span>'
    : '';
  var geminiBadge = state.geminiEnabled ? '<span class="bg bg-a">gemini</span>' : '';

  return '<div class="hdr">' +
    '<div>' +
    '<div class="hdr-brand">gesture detection</div>' +
    '<div class="hdr-title">Sign Language <em>Recognition</em></div>' +
    '</div>' +
    '<div class="hdr-badges">' + camBadge + modelBadge + geminiBadge +
    '<button class="btn btn-o btn-sm" onclick="window._app.switchMode(\'user\')" title="Back to user view" ' +
      'style="font-size:11px;padding:4px 9px;border-color:var(--brd);color:var(--mx)">User View</button>' +
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
    case 'detect': return renderDetectTab(state);
    case 'train': return renderTrainTab(state);
    case 'sequences': return renderSequenceTab(state);
    case 'settings': return renderSettingsTab(state);
    default: return renderDetectTab(state);
  }
}
