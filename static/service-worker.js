const CACHE_NAME = 'ai-agent-cache-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/static/index.html',
  '/static/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS_TO_CACHE))
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
    ))
  );
});

// Basic fetch handler: serve cached assets, try network for API calls
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  // For API calls (server or external), prefer network then fallback to cache
  if (url.pathname.startsWith('/chat') || url.hostname.includes('api.apiverve.com')) {
    event.respondWith(
      fetch(event.request).catch(() => caches.match(event.request))
    );
    return;
  }

  // For other requests, respond with cache-first strategy
  event.respondWith(
    caches.match(event.request).then((resp) => resp || fetch(event.request))
  );
});
