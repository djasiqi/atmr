/* eslint-disable no-console */
const { createProxyMiddleware } = require('http-proxy-middleware');

// ✅ Résoudre l'URL du backend depuis les variables d'environnement
// IMPORTANT: Utiliser 127.0.0.1 (IPv4) et jamais localhost — évite 504/ERR_CONNECTION_RESET (IPv6 + Docker Windows)
const getBackendUrl = () => {
  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL;
  if (apiBaseUrl) {
    try {
      const url = new URL(apiBaseUrl);
      // Forcer 127.0.0.1 si c'est localhost (évite IPv6)
      const host = url.hostname === 'localhost' ? '127.0.0.1' : url.hostname;
      const port = url.port || '5000';
      return `${url.protocol}//${host}:${port}`;
    } catch (e) {
      console.error('Invalid REACT_APP_API_BASE_URL:', apiBaseUrl, e);
    }
  }
  
  // Fallback pour développement — 127.0.0.1 obligatoire (pas localhost)
  const isProduction = process.env.NODE_ENV === 'production';
  if (isProduction) {
    console.warn(
      '⚠️ REACT_APP_API_BASE_URL non défini en production. ' +
      'Utilisation du fallback http://127.0.0.1:5000 (non recommandé).'
    );
  }
  
  return 'http://127.0.0.1:5000';
};

const BACKEND_URL = getBackendUrl();

// Par défaut, /api/demo pointe vers le MÊME backend que /api/app (5000).
// Ainsi, provision admin et consume magic link utilisent la même base.
// Pour multi-backend (docker-compose.multi-backend.yml), définir REACT_APP_DEMO_API_BASE_URL=http://127.0.0.1:5100
const getDemoBackendUrl = () => {
  const demoApiBaseUrl = process.env.REACT_APP_DEMO_API_BASE_URL || process.env.REACT_APP_DEMO_API_URL;
  if (demoApiBaseUrl) {
    try {
      const url = new URL(demoApiBaseUrl);
      const host = url.hostname === 'localhost' ? '127.0.0.1' : url.hostname;
      const port = url.port || '5100';
      return `${url.protocol}//${host}:${port}`;
    } catch (e) {
      console.error('Invalid REACT_APP_DEMO_API_BASE_URL:', demoApiBaseUrl, e);
    }
  }
  // Fallback: même backend que /api/app pour éviter "Token invalide" (provision sur 5000, consume sur 5100 = DB différente)
  return getBackendUrl();
};

const DEMO_BACKEND_URL = getDemoBackendUrl();

module.exports = function (app) {
  console.log('🔧 setupProxy.js chargé - Configuration du proxy...');
  console.log('⚠️ [DEBUG] setupProxy.js EXÉCUTÉ - app:', app ? 'OK' : 'NULL');
  console.log(`📡 Backend URL: ${BACKEND_URL}`);
  console.log(`📡 Demo Backend URL: ${DEMO_BACKEND_URL}`);

  // 🔌 Proxy Socket.IO avec support WebSocket
  // ✅ IMPORTANT: Ce middleware doit être AVANT les autres pour capturer les requêtes Socket.IO
  // ✅ IMPORTANT: Exclure /ws pour que webpack-dev-server puisse l'utiliser pour le HMR
  console.log('✅ Configuring /socket.io proxy...');
  const socketIoProxy = createProxyMiddleware({
    target: BACKEND_URL,
    changeOrigin: true, // ✅ Changer l'origine pour éviter les problèmes CORS
    ws: false, // ✅ DÉSACTIVER le support WebSocket pour ce proxy - webpack-dev-server utilise /ws pour le HMR
    // Note: Socket.IO utilisera polling puis upgrade vers WebSocket via /socket.io, pas via /ws
    secure: false,
    logLevel: 'info',
    // ✅ Exclure /ws pour que webpack-dev-server puisse l'utiliser pour le HMR
    filter: function (pathname, req) {
      const isWsPath = pathname === '/ws' || pathname.startsWith('/ws/');
      // Ne pas capturer /ws - webpack-dev-server en a besoin pour le HMR
      if (isWsPath) {
        console.log(`[SOCKET.IO PROXY] Filtre: /ws exclu (pathname: ${pathname}, url: ${req.url})`);
        return false; // Ne pas proxifier
      }
      // Ne capturer que les requêtes qui commencent par /socket.io
      const shouldProxy = pathname.startsWith('/socket.io');
      if (shouldProxy) {
        console.log(`[SOCKET.IO PROXY] Filtre: /socket.io accepté (pathname: ${pathname})`);
      }
      return shouldProxy;
    },
    // ✅ Transmettre les cookies httpOnly
    // En développement, ne pas réécrire le domaine (laisser les cookies tels quels)
    cookieDomainRewrite: false, // ✅ Ne pas réécrire le domaine en dev
    cookiePathRewrite: false, // ✅ Ne pas réécrire le path en dev
    // ✅ IMPORTANT : http-proxy-middleware supprime automatiquement le préfixe du middleware
    // Donc si le client fait /socket.io/?EIO=4, le pathRewrite reçoit /?EIO=4
    // Il faut donc réajouter /socket.io au début
    pathRewrite: function (path, req) {
      // ✅ IMPORTANT: Ne jamais transformer /ws en /socket.io/ws
      // webpack-dev-server utilise /ws pour le HMR et ne doit pas être proxifié
      if (path === '/ws' || path.startsWith('/ws/')) {
        console.warn(`[SOCKET.IO PROXY] ⚠️ Tentative de proxifier /ws - cela ne devrait pas arriver (filtre devrait l'exclure)`);
        return path; // Retourner tel quel (ne devrait jamais arriver grâce au filtre)
      }

      // ✅ FIX CRITIQUE: Éviter le double préfixe /socket.io/socket.io/
      // http-proxy-middleware supprime automatiquement le préfixe du middleware (/socket.io)
      // MAIS pour certaines requêtes (WebSocket), le préfixe peut ne pas être supprimé
      // Il faut donc vérifier à la fois path et req.url pour éviter le doublon
      
      let rewrittenPath;
      
      // ✅ Vérifier si le path commence déjà par /socket.io (double préfixe détecté)
      if (path.startsWith('/socket.io')) {
        // Le path contient déjà /socket.io - NE PAS ajouter de préfixe
        // Retourner le path tel quel pour éviter /socket.io/socket.io/
        rewrittenPath = path;
      } 
      // ✅ Vérifier si req.url contient /socket.io mais pas le path (cas WebSocket)
      else if (req.url && req.url.includes('/socket.io') && !path.startsWith('/socket.io')) {
        // Extraire la partie après /socket.io de req.url pour éviter le doublon
        const socketIoIndex = req.url.indexOf('/socket.io');
        rewrittenPath = req.url.substring(socketIoIndex);
      }
      // ✅ Cas normal: le path reçu est SANS le préfixe /socket.io (supprimé par le middleware)
      // Il faut le réajouter pour que Flask-SocketIO reçoive /socket.io/...
      else {
        rewrittenPath = `/socket.io${path}`;
      }
      
      // ✅ Vérification de sécurité: détecter et corriger tout double préfixe résiduel
      if (rewrittenPath.includes('/socket.io/socket.io')) {
        console.error(`[SOCKET.IO PROXY] ⚠️ Double préfixe détecté! Corrigeant: ${rewrittenPath}`);
        rewrittenPath = rewrittenPath.replace('/socket.io/socket.io', '/socket.io');
      }
      
      console.log(`[SOCKET.IO PROXY] pathRewrite: ${path} -> ${rewrittenPath} (original: ${req.url})`);
      return rewrittenPath;
    },
    onProxyReq: (proxyReq, req) => {
      // ✅ Log de test pour vérifier que le callback est appelé
      console.log('[SOCKET.IO PROXY] onProxyReq appelé pour:', req.url);
      // ✅ S'assurer que les cookies sont transmis
      if (req.headers.cookie) {
        proxyReq.setHeader('Cookie', req.headers.cookie);
        console.log(`[SOCKET.IO] Cookies transmis: ${req.headers.cookie.substring(0, 50)}...`);
      } else {
        console.warn(`[SOCKET.IO] ⚠️ Aucun cookie dans la requête`);
        // ✅ Logger tous les headers pour debug
        console.log(`[SOCKET.IO] Headers reçus:`, Object.keys(req.headers));
      }
      // ✅ S'assurer que les headers d'authentification sont transmis
      if (req.headers.authorization) {
        proxyReq.setHeader('Authorization', req.headers.authorization);
        console.log(`[SOCKET.IO] Authorization header transmis`);
      }
      // ✅ Transmettre les headers WebSocket si présents (pour les upgrades WebSocket)
      if (req.headers.upgrade === 'websocket') {
        proxyReq.setHeader('Upgrade', 'websocket');
        console.log(`[SOCKET.IO] Upgrade header transmis: websocket`);
      }
      if (req.headers.connection) {
        proxyReq.setHeader('Connection', req.headers.connection);
        console.log(`[SOCKET.IO] Connection header transmis: ${req.headers.connection}`);
      }
      if (req.headers['sec-websocket-key']) {
        proxyReq.setHeader('Sec-WebSocket-Key', req.headers['sec-websocket-key']);
      }
      if (req.headers['sec-websocket-version']) {
        proxyReq.setHeader('Sec-WebSocket-Version', req.headers['sec-websocket-version']);
      }
      if (req.headers['sec-websocket-protocol']) {
        proxyReq.setHeader('Sec-WebSocket-Protocol', req.headers['sec-websocket-protocol']);
      }
      if (req.headers['sec-websocket-extensions']) {
        proxyReq.setHeader('Sec-WebSocket-Extensions', req.headers['sec-websocket-extensions']);
      }
      console.log(`[SOCKET.IO] ${req.method} ${req.url} -> ${proxyReq.path}`);
    },
    onProxyRes: (proxyRes, req, res) => {
      // ✅ Log de test pour vérifier que le callback est appelé
      console.log('[SOCKET.IO PROXY] onProxyRes appelé pour:', req.url, 'status:', proxyRes.statusCode);
      // ✅ Logger les cookies reçus du backend
      const setCookieHeaders = proxyRes.headers['set-cookie'];
      if (setCookieHeaders) {
        console.log(`[SOCKET.IO] Cookies reçus du backend: ${setCookieHeaders.length} cookie(s)`);
        // ✅ Transmettre les cookies au client
        res.setHeader('Set-Cookie', setCookieHeaders);
      }
      // ✅ Logger le status code pour debug
      console.log(`[SOCKET.IO] Response: ${proxyRes.statusCode} ${req.url}`);
    },
    onProxyReqWs: (proxyReq, req) => {
      // ✅ Transmettre les cookies lors de l'upgrade WebSocket
      if (req.headers.cookie) {
        proxyReq.setHeader('Cookie', req.headers.cookie);
        console.log(`[SOCKET.IO WS] Cookies transmis lors de l'upgrade`);
      } else {
        console.warn(`[SOCKET.IO WS] ⚠️ Aucun cookie lors de l'upgrade WebSocket`);
      }
      // ✅ S'assurer que les headers d'authentification sont transmis
      if (req.headers.authorization) {
        proxyReq.setHeader('Authorization', req.headers.authorization);
        console.log(`[SOCKET.IO WS] Authorization header transmis`);
      }
      console.log(`[SOCKET.IO WS] Upgrade: ${req.url}`);
    },
    onError: (err, req, res) => {
      console.error('[SOCKET.IO ERROR]:', err.message);
      console.error('[SOCKET.IO ERROR] Request:', req.method, req.url);
      // ✅ Retourner une erreur structurée au client
      if (res && !res.headersSent) {
        res.status(503).json({
          error: 'Socket.IO proxy error',
          message: err.message,
        });
      }
    },
  });
  
  // ✅ IMPORTANT: Ne pas créer de proxy /ws car webpack-dev-server l'utilise pour le HMR
  // Les requêtes Socket.IO de l'application utilisent déjà /socket.io
  // Le proxy /ws causait des conflits avec webpack-dev-server
  
  app.use('/socket.io', socketIoProxy);

  // ✅ Sprint 3 local: routes unifiées disponibles aussi depuis localhost:3000
  app.use(
    '/api/gateway',
    createProxyMiddleware({
      target: `${BACKEND_URL}/api/gateway`,
      changeOrigin: true,
      secure: false,
      logLevel: 'warn',
    })
  );

  app.use(
    '/api/app',
    createProxyMiddleware({
      target: `${BACKEND_URL}/api/v1`,
      changeOrigin: true,
      secure: false,
      logLevel: 'warn',
    })
  );

  app.use(
    '/api/demo',
    createProxyMiddleware({
      target: `${DEMO_BACKEND_URL}/api/v1`,
      changeOrigin: true,
      secure: false,
      logLevel: 'warn',
    })
  );

  // 📁 Proxy Uploads (images, PDFs, etc.)
  console.log('✅ Configuring /uploads proxy...');
  app.use(
    '/uploads',
    createProxyMiddleware({
      target: BACKEND_URL,
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
      target: BACKEND_URL,
      changeOrigin: true,
      secure: false,
      logLevel: 'debug',
      timeout: 120000, // 120s timeout
      proxyTimeout: 120000, // 120s proxy timeout
      pathRewrite: function (path) {
        const rewritten = path.startsWith('/api/v1')
          ? path
          : `/api/v1${path.startsWith('/') ? '' : '/'}${path}`;
        console.log(`[API V1] pathRewrite: ${path} -> ${rewritten}`);
        return rewritten;
      },
      onProxyReq: (proxyReq, req) => {
        console.log(`[API V1] Proxying ${req.method} ${req.url} -> ${BACKEND_URL}${proxyReq.path}`);
      },
      onProxyRes: (proxyRes, req) => {
        console.log(`[API V1] ${req.method} ${req.url} -> ${proxyRes.statusCode}`);
      },
      onError: (err, req, res) => {
        // ✅ Gestion d'erreur améliorée pour 504 Gateway Timeout
        console.error(`[API V1] Proxy error for ${req.method} ${req.url}:`, err.message);
        if (res && !res.headersSent) {
          const isRefused = err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT';
          res.status(isRefused ? 503 : 504).json({
            error: isRefused ? 'Service temporairement indisponible' : 'Gateway Timeout',
            message: `Backend (${BACKEND_URL}) non accessible. ` +
              'Vérifiez: 1) docker compose up -d  2) docker compose ps (api doit être healthy)  3) curl http://127.0.0.1:5000/health',
            code: err.code,
          });
        }
      },
    })
  );

  // 📡 Proxy API REST (inclut /api/v1/*) sans réécriture
  console.log('✅ Configuring /api proxy...');
  app.use(
    '/api',
    createProxyMiddleware({
      target: BACKEND_URL,
      changeOrigin: true,
      secure: false,
      ws: false, // ✅ Désactiver le support WebSocket pour ce proxy (webpack-dev-server utilise /ws pour le HMR)
      logLevel: 'debug',
      timeout: 120000, // 120s timeout
      proxyTimeout: 120000, // 120s proxy timeout
      // ✅ IMPORTANT: Exclure /ws pour que webpack-dev-server puisse l'utiliser pour le HMR
      filter: function (pathname, _req) {
        // Ne pas capturer /ws - webpack-dev-server en a besoin pour le HMR
        if (pathname === '/ws' || pathname.startsWith('/ws/')) {
          return false;
        }
        // Capturer toutes les autres requêtes qui commencent par /api
        return pathname.startsWith('/api');
      },
      // ✅ IMPORTANT: http-proxy-middleware supprime automatiquement le préfixe /api avant pathRewrite
      // Donc si le client fait /api/shadow-mode/status, pathRewrite reçoit /shadow-mode/status
      // Il faut réajouter /api pour que le backend reçoive /api/shadow-mode/status
      // MAIS ne pas le faire pour /api/v1/* qui est déjà géré par le proxy /api/v1
      pathRewrite: function (path) {
        // ✅ IMPORTANT: Ne jamais transformer /ws en /api/ws
        // webpack-dev-server utilise /ws pour le HMR et ne doit pas être proxifié
        if (path === '/ws' || path.startsWith('/ws/')) {
          console.warn(`[API PROXY] ⚠️ Tentative de proxifier /ws - cela ne devrait pas arriver (filtre devrait l'exclure)`);
          return path; // Retourner tel quel (ne devrait jamais arriver grâce au filtre)
        }
        // Si le path commence déjà par /api, le retourner tel quel (ne devrait pas arriver)
        if (path.startsWith('/api')) {
          return path;
        }
        // Si le path commence par /v1, c'est géré par le proxy /api/v1, ne pas le modifier ici
        if (path.startsWith('/v1')) {
          return path;
        }
        // Pour toutes les autres routes (/shadow-mode, etc.), réajouter /api
        return `/api${path}`;
      },
      onProxyRes: (proxyRes, req) => {
        console.log(`[API] ${req.method} ${req.url} -> ${proxyRes.statusCode}`);
      },
      onError: (err, req, res) => {
        console.error(`[API] Proxy error for ${req.method} ${req.url}:`, err.message);
        if (res && !res.headersSent) {
          const isRefused = err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT';
          res.status(isRefused ? 503 : 504).json({
            error: isRefused ? 'Service temporairement indisponible' : 'Gateway Timeout',
            message: `Backend (${BACKEND_URL}) non accessible. Vérifiez: docker compose up -d`,
            code: err.code,
          });
        }
      },
    })
  );

  console.log('✅ Tous les proxies configurés !');
  console.log(`📡 Backend cible: ${BACKEND_URL} (si 504: vérifier "docker compose ps" et "curl ${BACKEND_URL}/health")`);
  console.log(`📡 Demo backend cible: ${DEMO_BACKEND_URL}`);
  console.log('📋 Routes: /socket.io, /uploads, /api, /api/gateway, /api/app, /api/demo');
};
