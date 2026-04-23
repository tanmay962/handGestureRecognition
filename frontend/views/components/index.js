// views/components/index.js
// Reusable UI primitives. Small functions that just return HTML strings.
// Kept simple — no state, no lifecycle, just string → string.
'use strict';

export const Badge = (text, variant) =>
  `<span class="bg bg-${variant || 'g'}">${text}</span>`;

export const DotBadge = (text, variant, pulse) => {
  variant = variant || 'g';
  return `<span class="bg bg-${variant}">` +
    `<span class="dot dot-${variant}${pulse ? ' dot-pulse' : ''}"></span>` +
    `${text}</span>`;
};

// variant: 'g' | 'p' | 'r' | 'a' | 'o' | 'ghost' (default: outline 'o')
export const Btn = (label, onclick, variant, size, disabled) =>
  `<button class="btn${variant ? ' btn-' + variant : ' btn-o'}${size ? ' btn-' + size : ''}" ` +
  `onclick="${onclick}"${disabled ? ' disabled' : ''}>${label}</button>`;

// title is optional — pass '' or null to get a plain container
export const Card = (title, body, style) =>
  `<div class="cd"${style ? ` style="${style}"` : ''}>` +
    (title ? `<div class="cd-label">${title}</div>` : '') +
    body +
  `</div>`;

export const Toggle = (on, onclick) =>
  `<div class="toggle" onclick="${onclick}">` +
    `<div class="knob" style="left:${on ? '21px' : '3px'}"></div>` +
  `</div>`;

export const Bar = (pct, color) =>
  `<div class="bar-wrap"><div class="bar-fill" style="width:${pct}%;background:rgba(255,255,255,0.5)"></div></div>`;

export const StatBox = (value, label, color) =>
  `<div class="stat">` +
    `<div class="stat-v" style="color:#ffffff">${value}</div>` +
    `<div class="stat-l">${label}</div>` +
  `</div>`;

export const SettingRow = (label, desc, control) =>
  `<div class="srow">` +
    `<div><div class="srow-label">${label}</div>${desc ? `<div class="srow-desc">${desc}</div>` : ''}</div>` +
    control +
  `</div>`;

export const LogEntry = (entry, opacity) => {
  opacity = opacity !== undefined ? opacity : 1;
  return `<div class="log-entry" style="opacity:${opacity}">` +
    `<div class="fr g5">` +
      Badge(entry.gesture, 'd') +
      (entry.combo ? Badge(entry.combo, 'd') : '') +
      (entry.model ? `<span style="font-size:8px;color:var(--dm)">[${entry.model}]</span>` : '') +
    `</div>` +
    `<div class="fr g5">` +
      `<span style="font-size:12px;font-weight:700;color:#ffffff">${(entry.conf * 100).toFixed(1)}%</span>` +
      `<span style="font-size:9px;color:var(--dm)">${entry.time.toLocaleTimeString()}</span>` +
    `</div>` +
  `</div>`;
};

// Ten vertical bars showing finger curl 0–100% for both hands.
// IDs fb0–fb9 and fv0–fv9 updated directly by AppController on each frame.
// Bars 0–4 = dominant hand, 5–9 = auxiliary hand; divided by a thin separator.
export const FingerBars = (names, colors) => {
  var half = Math.floor(names.length / 2);
  var cols = names.map(function(name, i) {
    var color = (colors && colors[i]) ? colors[i] : 'rgba(255,255,255,0.5)';
    return `<div class="fcol">` +
      `<div class="fbar-w"><div class="fbar-f" id="fb${i}" style="height:0%;background:${color}"></div></div>` +
      `<div class="fname" style="color:${color};font-size:9px">${name}</div>` +
      `<div class="fval" id="fv${i}" style="font-size:8px">0%</div>` +
    `</div>`;
  });
  // Insert visual divider between the two hands
  if (names.length > 5) {
    cols.splice(half, 0, '<div class="fgrid-div"></div>');
  }
  return `<div style="font-size:8px;color:var(--mx);margin-bottom:4px">` +
    `<span style="margin-right:4px">Dom hand</span>` +
    `<span style="opacity:.5">·</span>` +
    `<span style="margin-left:4px">Aux hand</span>` +
  `</div>` +
  `<div class="fgrid">${cols.join('')}</div>`;
};
