const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api', // Cible toutes les requêtes qui commencent par /api
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000', // L'adresse de votre serveur backend
      changeOrigin: true,
    })
  );
};