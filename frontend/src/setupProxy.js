const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = function (app) {
  app.use(
    "/api",
    createProxyMiddleware({
      // Important: cible sur /api pour conserver le préfixe côté backend RESTX
      target: "http://127.0.0.1:5000",
      changeOrigin: true,
      logLevel: "debug",
      secure: false,
      // Le mount "/api" d'Express est retiré du path; on le réinjecte pour le backend
      pathRewrite: (path) => `/api${path}`,
      // Pas de pathRewrite: le mount "/api" côté CRA est retiré automatiquement,
      // donc la cible incluant "/api" reconstruit bien /api/... côté backend
    })
  );

  // ⚡ Proxy Socket.IO (indispensable pour éviter le timeout)
  app.use(
    "/socket.io",
    createProxyMiddleware({
      target: "http://127.0.0.1:5000",

      changeOrigin: true,
      ws: true, // Active le proxy WebSocket
      logLevel: "debug",
      secure: false,
    })
  );

  // 📄 Proxy pour les fichiers uploads (PDFs, images, etc.)
  app.use(
    "/uploads",
    createProxyMiddleware({
      target: "http://127.0.0.1:5000",
      changeOrigin: true,
      logLevel: "debug",
      secure: false,
    })
  );
};
