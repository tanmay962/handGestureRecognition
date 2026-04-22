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
  `<div class="bar-wrap"><div class="bar-fill" style="width:${pct}%;background:${color || 'var(--g)'}"></div></div>`;

export const StatBox = (value, label, color) =>
  `<div class="stat">` +
    `<div class="stat-v" style="color:${color || 'var(--g)'}">${value}</div>` +
    `<div class="stat-l">${label}</div>` +
  `</div>`;

export const SettingRow = (label, desc, control) =>
  `<div class="srow">` +
    `<div><div class="srow-label">${label}</div>${desc ? `<div class="srow-desc">${desc}</div>` : ''}</div>` +
    control +
  `</div>`;

export const LogEntry = (entry, opacity) => {
  opacity = opacity !== undefined ? opacity : 1;
  var confColor = entry.conf > 0.9 ? 'var(--g)' : entry.conf > 0.7 ? 'var(--a)' : 'var(--r)';
  return `<div class="log-entry" style="opacity:${opacity}">` +
    `<div class="fr g5">` +
      Badge(entry.gesture, 'g') +
      (entry.combo ? Badge(entry.combo, 'p') : '') +
      (entry.model ? `<span style="font-size:8px;color:var(--dm)">[${entry.model}]</span>` : '') +
    `</div>` +
    `<div class="fr g5">` +
      `<span style="font-size:12px;font-weight:700;color:${confColor}">${(entry.conf * 100).toFixed(1)}%</span>` +
      `<span style="font-size:9px;color:var(--dm)">${entry.time.toLocaleTimeString()}</span>` +
    `</div>` +
  `</div>`;
};

// Five vertical bars showing finger curl 0–100%.
// IDs fb0–fb4 and fv0–fv4 are updated directly by AppController on each frame.
export const FingerBars = (names, colors) =>
  `<div class="fgrid">` +
    names.map(function(name, i) {
      return `<div class="fcol">` +
        `<div class="fbar-w"><div class="fbar-f" id="fb${i}" style="height:0%;background:${colors[i]}"></div></div>` +
        `<div class="fname" style="color:${colors[i]}">${name}</div>` +
        `<div class="fval" id="fv${i}">0%</div>` +
      `</div>`;
    }).join('') +
  `</div>`;
