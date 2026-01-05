/* eslint-disable no-console */
const { createProxyMiddleware } = require('http-proxy-middleware');

// ✅ Résoudre l'URL du backend depuis les variables d'environnement
const getBackendUrl = () => {
  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL;
  if (apiBaseUrl) {
    try {
      const url = new URL(apiBaseUrl);
      return url.origin;
    } catch (e) {
      console.error('Invalid REACT_APP_API_BASE_URL:', apiBaseUrl, e);
    }
  }
  
  // Fallback pour développement uniquement
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

module.exports = function (app) {
  console.log('🔧 setupProxy.js chargé - Configuration du proxy...');
  console.log('⚠️ [DEBUG] setupProxy.js EXÉCUTÉ - app:', app ? 'OK' : 'NULL');
  console.log(`📡 Backend URL: ${BACKEND_URL}`);

  // ✅ CRITIQUE: Handler Express pour /ws qui répond directement sans proxifier
  // webpack-dev-server utilise /ws pour le HMR et ne doit JAMAIS être proxifié
  // Ce handler doit être AVANT tous les proxies pour éviter qu'ils interceptent /ws
  // IMPORTANT: On utilise app.get() au lieu de app.use() pour être plus spécifique
  // et on ne fait rien (pas de next()) pour laisser webpack-dev-server gérer la connexion
  app.get('/ws', (req, _res, _next) => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:/ws GET handler',message:'/ws GET request intercepted',data:{url:req.url,method:req.method,is_upgrade:req.headers.upgrade === 'websocket',headers_upgrade:req.headers.upgrade},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'}),credentials:'omit'}).catch(()=>{});
    // #endregion
    console.log('[PROXY] /ws GET request - laissé à webpack-dev-server pour HMR');
    // Ne pas appeler next() - laisser webpack-dev-server gérer directement
    // webpack-dev-server intercepte les connexions WebSocket avant les middlewares Express
    // En ne faisant rien, on empêche les proxies d'intercepter la connexion
  });
  
  // Handler pour les requêtes WebSocket upgrade (méthode GET avec header Upgrade: websocket)
  app.use('/ws', (req, res, next) => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:/ws middleware',message:'/ws request intercepted',data:{url:req.url,method:req.method,is_upgrade:req.headers.upgrade === 'websocket',headers_upgrade:req.headers.upgrade},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'}),credentials:'omit'}).catch(()=>{});
    // #endregion
    // Si c'est une requête WebSocket upgrade, on ne fait rien et on laisse webpack-dev-server la gérer
    if (req.headers.upgrade === 'websocket') {
      console.log('[PROXY] /ws WebSocket upgrade - laissé à webpack-dev-server pour HMR');
      // Ne pas appeler next() pour les WebSocket - laisser webpack-dev-server les gérer directement
      // webpack-dev-server intercepte les connexions WebSocket avant les middlewares Express
      return; // Sortir sans appeler next() pour empêcher les proxies d'intercepter
    }
    // Pour les requêtes HTTP normales, on peut simplement passer
    console.log('[PROXY] /ws HTTP request - laissé à webpack-dev-server pour HMR');
    next(); // Passer au middleware suivant (webpack-dev-server)
  });

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
      // #region agent log
      const isWsPath = pathname === '/ws' || pathname.startsWith('/ws/');
      const isUpgrade = req.headers.upgrade === 'websocket';
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:socketIoProxy.filter',message:'Filter check',data:{pathname,url:req.url,is_ws_path:isWsPath,is_upgrade:isUpgrade,starts_with_socketio:pathname.startsWith('/socket.io')},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'}),credentials:'omit'}).catch(()=>{});
      // #endregion
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
      
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:pathRewrite',message:'Path rewrite before transformation',data:{original_path:path,original_url:req.url,path_starts_with_slash:path.startsWith('/'),path_already_has_socketio:path.startsWith('/socket.io'),url_has_socketio:req.url.includes('/socket.io'),is_ws_path:path === '/ws' || path.startsWith('/ws/')},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'}),credentials:'omit'}).catch(()=>{});
      // #endregion
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
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:pathRewrite',message:'Path rewrite after transformation',data:{rewritten_path:rewrittenPath,original_path:path,has_double_prefix:rewrittenPath.includes('/socket.io/socket.io'),is_valid_socketio_path:rewrittenPath.startsWith('/socket.io'),was_already_socketio:path.startsWith('/socket.io'),url_had_socketio:req.url && req.url.includes('/socket.io')},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'}),credentials:'omit'}).catch(()=>{});
      // #endregion
      return rewrittenPath;
    },
    onProxyReq: (proxyReq, req) => {
      // ✅ Log de test pour vérifier que le callback est appelé
      console.log('[SOCKET.IO PROXY] onProxyReq appelé pour:', req.url);
      // #region agent log
      const cookieHeader = req.headers.cookie || '';
      const hasAccessTokenCookie = cookieHeader.includes('access_token');
      const isWebSocketRequest = req.url?.includes('transport=websocket') || req.headers.upgrade === 'websocket';
      const requestHeaders = {
        upgrade: req.headers.upgrade,
        connection: req.headers.connection,
        'sec-websocket-key': req.headers['sec-websocket-key'],
        'sec-websocket-version': req.headers['sec-websocket-version'],
        'sec-websocket-protocol': req.headers['sec-websocket-protocol'],
        'sec-websocket-extensions': req.headers['sec-websocket-extensions'],
      };
      try {
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:onProxyReq',message:'Proxy request before cookie transmission',data:{url:req.url,method:req.method,is_websocket_request:isWebSocketRequest,request_headers:requestHeaders,has_cookie_header:!!req.headers.cookie,has_access_token:hasAccessTokenCookie,cookie_keys:cookieHeader.split(';').map(c=>c.split('=')[0].trim()).filter(Boolean),target_path:proxyReq.path,proxy_headers:{upgrade:proxyReq.getHeader('upgrade'),connection:proxyReq.getHeader('connection')}},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'}),credentials:'omit'}).catch((e)=>console.error('[LOG ERROR]',e));
      } catch(e) {
        console.error('[LOG ERROR] onProxyReq:', e);
      }
      // #endregion
      // ✅ S'assurer que les cookies sont transmis
      if (req.headers.cookie) {
        proxyReq.setHeader('Cookie', req.headers.cookie);
        console.log(`[SOCKET.IO] Cookies transmis: ${req.headers.cookie.substring(0, 50)}...`);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:onProxyReq',message:'Cookies transmitted to backend',data:{cookie_header_set:true,has_access_token:hasAccessTokenCookie},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'A'}),credentials:'omit'}).catch(()=>{});
        // #endregion
      } else {
        console.warn(`[SOCKET.IO] ⚠️ Aucun cookie dans la requête`);
        // ✅ Logger tous les headers pour debug
        console.log(`[SOCKET.IO] Headers reçus:`, Object.keys(req.headers));
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:onProxyReq',message:'No cookies in request',data:{headers:Object.keys(req.headers)},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'A'}),credentials:'omit'}).catch(()=>{});
        // #endregion
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
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:onProxyReq',message:'Proxy request path rewrite',data:{original_url:req.url,rewritten_path:proxyReq.path,path_matches_socketio:proxyReq.path.startsWith('/socket.io')},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'}),credentials:'omit'}).catch(()=>{});
      // #endregion
    },
    onProxyRes: (proxyRes, req, res) => {
      // ✅ Log de test pour vérifier que le callback est appelé
      console.log('[SOCKET.IO PROXY] onProxyRes appelé pour:', req.url, 'status:', proxyRes.statusCode);
      // #region agent log
      const isWebSocketUpgrade = req.headers.upgrade === 'websocket' || req.url?.includes('transport=websocket');
      const responseHeaders = {
        upgrade: proxyRes.headers.upgrade,
        connection: proxyRes.headers.connection,
        'sec-websocket-accept': proxyRes.headers['sec-websocket-accept'],
        'sec-websocket-protocol': proxyRes.headers['sec-websocket-protocol'],
        'set-cookie': proxyRes.headers['set-cookie'] ? 'present' : 'missing',
        'access-control-allow-origin': proxyRes.headers['access-control-allow-origin'],
        'access-control-allow-credentials': proxyRes.headers['access-control-allow-credentials'],
      };
      try {
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:onProxyRes',message:'Proxy response received',data:{status_code:proxyRes.statusCode,url:req.url,is_websocket_upgrade:isWebSocketUpgrade,response_headers:responseHeaders,is_error:proxyRes.statusCode >= 400},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'}),credentials:'omit'}).catch((e)=>console.error('[LOG ERROR]',e));
      } catch(e) {
        console.error('[LOG ERROR] onProxyRes:', e);
      }
      // #endregion
      // ✅ Logger les cookies reçus du backend
      const setCookieHeaders = proxyRes.headers['set-cookie'];
      if (setCookieHeaders) {
        console.log(`[SOCKET.IO] Cookies reçus du backend: ${setCookieHeaders.length} cookie(s)`);
        // ✅ Transmettre les cookies au client
        res.setHeader('Set-Cookie', setCookieHeaders);
      }
      // ✅ Logger le status code pour debug
      console.log(`[SOCKET.IO] Response: ${proxyRes.statusCode} ${req.url}`);
      // #region agent log
      if (proxyRes.statusCode >= 400) {
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:onProxyRes',message:'Error response from backend',data:{status_code:proxyRes.statusCode,url:req.url,error_type:'backend_error'},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'}),credentials:'omit'}).catch(()=>{});
      }
      // #endregion
    },
    onProxyReqWs: (proxyReq, req) => {
      // #region agent log
      const upgradeHeaders = {
        upgrade: req.headers.upgrade,
        connection: req.headers.connection,
        'sec-websocket-key': req.headers['sec-websocket-key'],
        'sec-websocket-version': req.headers['sec-websocket-version'],
        'sec-websocket-protocol': req.headers['sec-websocket-protocol'],
        'sec-websocket-extensions': req.headers['sec-websocket-extensions'],
        cookie: req.headers.cookie ? 'present' : 'missing',
        authorization: req.headers.authorization ? 'present' : 'missing',
      };
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:onProxyReqWs',message:'WebSocket upgrade request',data:{url:req.url,method:req.method,headers:upgradeHeaders,proxy_path:proxyReq.path,has_cookie:!!req.headers.cookie,has_authorization:!!req.headers.authorization},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'}),credentials:'omit'}).catch(()=>{});
      // #endregion
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
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:/api/v1 onProxyReq',message:'Proxy request to backend',data:{url:req.url,method:req.method,path:proxyReq.path,target:BACKEND_URL,has_cookie:!!req.headers.cookie,has_authorization:!!req.headers.authorization},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'}),credentials:'omit'}).catch(()=>{});
        // #endregion
        console.log(`[API V1] Proxying ${req.method} ${req.url} -> ${BACKEND_URL}${proxyReq.path}`);
      },
      onProxyRes: (proxyRes, req) => {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'setupProxy.js:/api/v1 onProxyRes',message:'Proxy response from backend',data:{url:req.url,method:req.method,status_code:proxyRes.statusCode,is_error:proxyRes.statusCode >= 400},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'}),credentials:'omit'}).catch(()=>{});
        // #endregion
        console.log(`[API V1] ${req.method} ${req.url} -> ${proxyRes.statusCode}`);
      },
      onError: (err, req, res) => {
        // ✅ Gestion d'erreur améliorée pour 504 Gateway Timeout
        console.error(`[API V1] Proxy error for ${req.method} ${req.url}:`, err.message);
        if (err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT') {
          res.status(503).json({
            error: 'Service temporairement indisponible',
            message: 'Le serveur backend n\'est pas accessible. Vérifiez qu\'il est démarré sur le port 5000.',
            code: err.code,
          });
        } else {
          res.status(504).json({
            error: 'Gateway Timeout',
            message: 'Le serveur backend n\'a pas répondu dans les délais impartis.',
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
        // ✅ Gestion d'erreur améliorée pour 504 Gateway Timeout
        console.error(`[API] Proxy error for ${req.method} ${req.url}:`, err.message);
        if (err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT') {
          res.status(503).json({
            error: 'Service temporairement indisponible',
            message: 'Le serveur backend n\'est pas accessible. Vérifiez qu\'il est démarré sur le port 5000.',
            code: err.code,
          });
        } else {
          res.status(504).json({
            error: 'Gateway Timeout',
            message: 'Le serveur backend n\'a pas répondu dans les délais impartis.',
            code: err.code,
          });
        }
      },
    })
  );

  console.log('✅ Tous les proxies configurés !');
  console.log('📋 Routes: /socket.io, /uploads, /api');
};
