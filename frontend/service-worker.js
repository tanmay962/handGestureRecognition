// service-worker.js — Gesture Detection v1.0 PWA
const CACHE = 'gd-v3.1';
const STATIC = [
  '/',
  '/index.html',
  '/bundle.js?v=3.1',
  '/assets/css/styles.css',
  '/manifest.json',
  '/assets/icons/icon-192.png',
  '/assets/icons/icon-512.png',
];

self.addEventListener('install', function(e) {
  e.waitUntil(
    caches.open(CACHE).then(function(c) { return c.addAll(STATIC); })
  );
  self.skipWaiting();
});

self.addEventListener('activate', function(e) {
  e.waitUntil(
    caches.keys().then(function(keys) {
      return Promise.all(keys.filter(function(k){ return k !== CACHE; }).map(function(k){ return caches.delete(k); }));
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', function(e) {
  var url = e.request.url;
  // API calls — always network
  if (url.includes('/api/') || url.includes('/ws')) {
    e.respondWith(fetch(e.request).catch(function() {
      return new Response(JSON.stringify({error:'offline'}), {headers:{'Content-Type':'application/json'}});
    }));
    return;
  }
  // MediaPipe CDN — always network, never cache
  // Caching tflite/wasm model files conflicts with MediaPipe's internal WASM virtual FS
  if (url.includes('mediapipe') || url.includes('jsdelivr')) {
    e.respondWith(fetch(e.request));
    return;
  }
  // Google Fonts — cache after first load
  if (url.includes('fonts.googleapis') || url.includes('fonts.gstatic')) {
    e.respondWith(
      caches.match(e.request).then(function(cached) {
        if (cached) return cached;
        return fetch(e.request).then(function(resp) {
          var clone = resp.clone();
          caches.open(CACHE).then(function(c) { c.put(e.request, clone); });
          return resp;
        });
      })
    );
    return;
  }
  // Static assets — cache first
  e.respondWith(
    caches.match(e.request).then(function(cached) {
      return cached || fetch(e.request);
    })
  );
});
