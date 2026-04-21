// utils/DOMHelper.js
export const $ = id => document.getElementById(id);
export const clearEl = el => { while (el && el.firstChild) el.removeChild(el.firstChild); };
