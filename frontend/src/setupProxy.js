/* eslint-disable no-console */
const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  console.log('🔧 setupProxy.js chargé - Configuration du proxy...');
  console.log('⚠️ [DEBUG] setupProxy.js EXÉCUTÉ - app:', app ? 'OK' : 'NULL');

  // 🔌 Proxy Socket.IO avec support WebSocket
  console.log('✅ Configuring /socket.io proxy...');
  app.use(
    '/socket.io',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',
      changeOrigin: true,
      ws: true, // Support WebSocket
      secure: false,
      logLevel: 'info',
      // IMPORTANT : remettre /socket.io dans le chemin
      pathRewrite: function (path) {
        return path;
      },
      onProxyReq: (proxyReq, req) => {
        console.log(`[SOCKET.IO] ${req.method} ${req.url} -> ${proxyReq.path}`);
      },
      onProxyReqWs: (proxyReq, req) => {
        console.log(`[SOCKET.IO WS] Upgrade: ${req.url}`);
      },
      onError: (err, _req, _res) => {
        console.error('[SOCKET.IO ERROR]:', err.message);
      },
    })
  );

  // 📁 Proxy Uploads (images, PDFs, etc.)
  console.log('✅ Configuring /uploads proxy...');
  app.use(
    '/uploads',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',
      changeOrigin: true,
      secure: false,
      logLevel: 'warn',
      pathRewrite: function (path) {
        return '/uploads' + path;
      },
    })
  );

  // 📡 Proxy API v1 explicite (prioritaire)
  console.log('✅ Configuring /api/v1 proxy...');
  app.use(
    '/api/v1',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',
      changeOrigin: true,
      secure: false,
      logLevel: 'debug',
      timeout: 120000,
      proxyTimeout: 120000,
      pathRewrite: function (path) {
        const rewritten = path.startsWith('/api/v1')
          ? path
          : `/api/v1${path.startsWith('/') ? '' : '/'}${path}`;
        console.log(`[API V1] pathRewrite: ${path} -> ${rewritten}`);
        return rewritten;
      },
      onProxyRes: (proxyRes, req) => {
        console.log(`[API V1] ${req.method} ${req.url} -> ${proxyRes.statusCode}`);
      },
    })
  );

  // 📡 Proxy API REST (inclut /api/v1/*) sans réécriture
  console.log('✅ Configuring /api proxy...');
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',
      changeOrigin: true,
      secure: false,
      logLevel: 'debug',
      timeout: 120000,
      proxyTimeout: 120000,
      // Ne pas préfixer à nouveau par /api pour éviter /api/api/...
      pathRewrite: function (path) {
        return path; // conserve /api/... tel quel
      },
      onProxyRes: (proxyRes, req) => {
        console.log(`[API] ${req.method} ${req.url} -> ${proxyRes.statusCode}`);
      },
    })
  );

  console.log('✅ Tous les proxies configurés !');
  console.log('📋 Routes: /socket.io, /uploads, /api');
};
