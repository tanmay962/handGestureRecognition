// utils/MathUtils.js — JS stubs; heavy math runs in Python backend
// Used only by CameraService.js for feature extraction (browser-side only)

export const randomMatrix = (r, c) => {
  const s = Math.sqrt(2 / r);
  return Array.from({length: r}, () => Array.from({length: c}, () => (Math.random() - .5) * 2 * s));
};
export const relu = x => Math.max(0, x);
export const reluDeriv = x => x > 0 ? 1 : 0;
export const softmax = a => { const m = Math.max(...a); const e = a.map(v => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };
export const argmax = a => a.indexOf(Math.max(...a));
export const noise = (s = .1) => (Math.random() - .5) * s;
export const clamp = (v, mn, mx) => Math.max(mn, Math.min(mx, v));
export const dist3D = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
export const norm3D = v => { const l = Math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) || .001; return {x: v.x / l, y: v.y / l, z: v.z / l}; };
