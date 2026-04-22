// views/SequenceView.js
import {Card, Btn, Badge} from './components/index.js';

export function renderSequenceTab(state) {
  var gestures = state.gestures;
  var combos   = state.combos;
  var seq      = window._comboSeq || [];

  return (
    _renderIntro() +
    _renderBuilder(gestures, seq) +
    _renderSavedCombos(combos)
  );
}

function _renderIntro() {
  return Card(
    'Gesture Combos',
    '<div style="font-size:12px;color:var(--mx);line-height:1.7">' +
      '<strong style="color:var(--g)">Combos</strong> fire when you sign a specific sequence of gestures. ' +
      'The <strong style="color:var(--p)">dynamic model</strong> handles motion-based gestures like J and Z.' +
    '</div>'
  );
}

function _renderBuilder(gestures, seq) {
  var gestureButtons = gestures.slice(0, 20).map(function(g) {
    return '<button class="btn btn-o btn-sm" onclick="' +
      'window._comboSeq=(window._comboSeq||[]);window._comboSeq.push(\'' + g + '\');window._app.switchTab(\'sequences\')" >' +
      g +
    '</button>';
  }).join('');

  var seqPreview = seq.length
    ? '<div class="fr g5 mb8" style="padding:8px 12px;background:var(--pD);border-radius:8px;border:1px solid rgba(155,143,212,.2);flex-wrap:wrap">' +
        '<span style="font-size:10px;color:var(--p);font-weight:700">SEQ:</span>' +
        seq.map(function(g, i) {
          return Badge(g, 'p') + (i < seq.length - 1 ? '<span style="color:var(--dm)">→</span>' : '');
        }).join('') +
        '<button class="btn btn-ghost btn-sm" onclick="window._comboSeq=[];window._app.switchTab(\'sequences\')">clear</button>' +
      '</div>'
    : '<div style="font-size:10px;color:var(--dm);margin-bottom:8px">Tap gestures above to build a sequence (min 2).</div>';

  var saveBtn = '<button class="btn btn-g" ' +
    'onclick="if((window._comboSeq||[]).length>=2&&document.getElementById(\'comboAct\').value.trim()){' +
      'window._app.addCombo(window._comboSeq,document.getElementById(\'comboAct\').value.trim());' +
      'window._comboSeq=[];document.getElementById(\'comboAct\').value=\'\';window._app.switchTab(\'sequences\')' +
    '}" ' + (seq.length < 2 ? 'disabled' : '') + '>Save combo</button>';

  return Card(
    'Create Combo',
    '<div class="fr g5 mb8" style="flex-wrap:wrap">' + gestureButtons + '</div>' +
    seqPreview +
    '<div class="fr g6">' +
      '<input class="inp f1" id="comboAct" placeholder="Output phrase…">' +
      saveBtn +
    '</div>'
  );
}

function _renderSavedCombos(combos) {
  if (!combos || combos.length === 0) {
    return Card('Saved Combos', '<div style="font-size:10px;color:var(--dm)">No combos yet. Build one above.</div>');
  }

  var rows = combos.map(function(c) {
    var arrows = c.sequence.map(function(g, i) {
      return Badge(g, 'p') + (i < c.sequence.length - 1 ? '<span style="color:var(--dm)">→</span>' : '');
    }).join('');

    return '<div class="fr mb8" style="padding:10px 14px;background:var(--s2);border-radius:8px;border:1px solid var(--brd)">' +
      '<div class="f1">' +
        '<div class="fr g5 mb8" style="flex-wrap:wrap">' + arrows + '</div>' +
        '<div style="font-size:12px;font-weight:600;color:var(--g)">"' + c.action + '"</div>' +
      '</div>' +
      Badge((c.timeout / 1000).toFixed(1) + 's', 'a') +
    '</div>';
  }).join('');

  return Card('Saved Combos', rows);
}
