// utils/MathUtils.js — browser-side math helpers used by CameraService
// Heavy ML math (softmax, matrix ops) runs in Python backend

export const noise  = (s = .1) => (Math.random() - .5) * s;
export const clamp  = (v, mn, mx) => Math.max(mn, Math.min(mx, v));
export const dist3D = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
export const norm3D = v => { const l = Math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) || .001; return {x: v.x / l, y: v.y / l, z: v.z / l}; };
