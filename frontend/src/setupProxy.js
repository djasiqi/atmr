/* eslint-disable no-console */
const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  console.log('🔧 setupProxy.js chargé - Configuration du proxy...');

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
        return '/socket.io' + path;
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

  // 📡 Proxy API REST
  console.log('✅ Configuring /api proxy...');
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',
      changeOrigin: true,
      secure: false,
      logLevel: 'warn',
      pathRewrite: function (path) {
        return '/api' + path;
      },
    })
  );

  console.log('✅ Tous les proxies configurés !');
  console.log('📋 Routes: /socket.io, /uploads, /api');
};
