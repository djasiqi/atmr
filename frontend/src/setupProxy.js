 const { createProxyMiddleware } = require('http-proxy-middleware');

 module.exports = function (app) {
   app.use(
     '/api',
     createProxyMiddleware({
       target: 'http://127.0.0.1:5000',
       changeOrigin: true,
       logLevel: 'debug',
      secure: false,
     })

     
   );


  // ⚡ Proxy Socket.IO (indispensable pour éviter le timeout)
  app.use(
    '/socket.io',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',

      changeOrigin: true,
      ws: true,         // Active le proxy WebSocket
      logLevel: 'debug',
      secure: false,
    })
  );
 };

