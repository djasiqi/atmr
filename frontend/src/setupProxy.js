// frontend/src/setupProxy.js
const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',
      changeOrigin: true,
      // ❗ NE PAS supprimer /api :
      // ✘ pathRewrite: { '^/api': '' }  <-- à supprimer s’il existe
      logLevel: 'silent',

      // (Optionnel) facilite les cookies en dev HTTP
      onProxyRes(proxyRes) {
        if (proxyRes.headers['set-cookie']) {
          proxyRes.headers['set-cookie'] = proxyRes.headers['set-cookie'].map(c =>
            c.replace(/; *secure/gi, '')
          );
        }
      },
    })
  );

  // ❗ N’ajoute PAS de règle séparée pour '/auth'
  // Tout doit passer par '/api'
};
