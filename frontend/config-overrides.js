// config-overrides.js

const SML_EXCLUDE = /node_modules[\\/](svg-engine)/; // adapte le nom du paquet si besoin
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const fs = require('fs');
const path = require('path');

module.exports = {
  webpack: function override(config, _env) {
  const isSmlRule = (rule) => {
    const inUse =
      Array.isArray(rule?.use) &&
      rule.use.some((u) => {
        const loader = typeof u === 'string' ? u : u?.loader || '';
        return loader.includes('source-map-loader');
      });
    const inLoader = typeof rule?.loader === 'string' && rule.loader.includes('source-map-loader');
    return rule?.enforce === 'pre' && (inUse || inLoader);
  };

  const appendExclude = (rule, pattern) => {
    if (!rule.exclude) rule.exclude = [pattern];
    else if (Array.isArray(rule.exclude)) rule.exclude.push(pattern);
    else rule.exclude = [rule.exclude, pattern]; // convertit RegExp/Fn/String -> Array
  };

  const visit = (rule) => {
    if (!rule) return;
    if (isSmlRule(rule)) appendExclude(rule, SML_EXCLUDE);
    if (Array.isArray(rule.oneOf)) rule.oneOf.forEach(visit);
    if (Array.isArray(rule.rules)) rule.rules.forEach(visit);
  };

  (config.module?.rules || []).forEach(visit);

  // Forcer MiniCssExtractPlugin à ignorer l'ordre des imports CSS (corrige les conflits de build CI)
  config.plugins = (config.plugins || []).map((plugin) => {
    if (plugin instanceof MiniCssExtractPlugin) {
      plugin.options = {
        ...plugin.options,
        ignoreOrder: true,
      };
    }
    return plugin;
  });

  // Ignore les warnings courants (source maps, baseline-browser-mapping, etc.)
  config.ignoreWarnings = [
    ...(config.ignoreWarnings || []),
    /Failed to parse source map/,
    /baseline-browser-mapping/,
    /The data in this module is over/,
    /to ensure accurate Baseline data/,
  ];

  // ✅ PERF: Optimisations de bundle
  if (_env === 'production') {
    // Split chunks plus aggressif
    config.optimization = {
      ...config.optimization,
      splitChunks: {
        chunks: 'all',
        cacheGroups: {
          // Vendor libs séparées
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name(module) {
              // Chunks nommés par package pour meilleur caching
              const packageName = module.context.match(/[\\/]node_modules[\\/](.*?)([\\/]|$)/)?.[1];
              return `vendor.${packageName?.replace('@', '')}`;
            },
            priority: 10,
          },
          // Leaflet séparé (150 KB)
          leaflet: {
            test: /[\\/]node_modules[\\/](react-)?leaflet/,
            name: 'vendor.leaflet',
            priority: 20,
          },
          // Recharts séparé (380 KB)
          recharts: {
            test: /[\\/]node_modules[\\/]recharts/,
            name: 'vendor.recharts',
            priority: 20,
          },
          // Socket.IO séparé
          socketio: {
            test: /[\\/]node_modules[\\/]socket\.io-client/,
            name: 'vendor.socketio',
            priority: 20,
          },
          // Libs communes
          common: {
            minChunks: 2,
            priority: 5,
            reuseExistingChunk: true,
          },
        },
      },
      // Minimize plus agressif
      minimizer: config.optimization.minimizer?.map((plugin) => {
        if (plugin.constructor.name === 'TerserPlugin') {
          plugin.options.terserOptions = {
            ...plugin.options.terserOptions,
            compress: {
              ...plugin.options.terserOptions?.compress,
              drop_console: true, // Supprimer console.log en prod
              drop_debugger: true,
              pure_funcs: ['console.log', 'console.info', 'console.debug'],
            },
          };
        }
        return plugin;
      }),
    };

    // PWA : précache shell/offline uniquement (budget Lot 0 ≤ 1,5 Mo manifeste)
    const WorkboxWebpackPlugin = require('workbox-webpack-plugin');
    config.plugins.push(
      new WorkboxWebpackPlugin.GenerateSW({
        clientsClaim: true,
        skipWaiting: true,
        maximumFileSizeToCacheInBytes: 512 * 1024,
        navigateFallback: '/index.html',
        navigateFallbackDenylist: [
          /^\/api/,
          /^\/socket\.io/,
          /^\/uploads/,
          /\/[^/?]+\.[^/]+$/,
        ],
        // Exclure chunks métier massifs (maps, factures, analytics, dispatch)
        exclude: [
          /\.map$/,
          /maps/i,
          /Invoice/i,
          /Analytics/i,
          /Dispatch/i,
          /pdf\.worker/i,
          /recharts/i,
          /Vendor/i,
        ],
        additionalManifestEntries: [
          { url: '/offline.html', revision: '20260408-offline-v1' },
        ],
        // Pas de runtimeCaching d'API auth
      })
    );
  }

  return config;
  },

  // ✅ Configuration personnalisée pour webpack-dev-server v5
  devServer: function(configFunction) {
    return function(proxy, allowedHost) {
      // Obtenir la configuration par défaut
      const config = configFunction(proxy, allowedHost);
      
      // ✅ Supprimer les propriétés obsolètes de webpack-dev-server v5
      delete config.onAfterSetupMiddleware;
      delete config.onBeforeSetupMiddleware;
      
      // ✅ Convertir https en server si nécessaire
      if (config.https !== undefined) {
        // Si https est true ou un objet de configuration, convertir en server
        if (config.https === true) {
          config.server = 'https';
        } else if (typeof config.https === 'object' && config.https !== null) {
          config.server = {
            type: 'https',
            options: config.https,
          };
        }
        // Sinon (false, undefined), on ne fait rien, juste supprimer https
        delete config.https;
      }
      
      // ✅ Migrer les middlewares vers setupMiddlewares
      if (!config.setupMiddlewares) {
        config.setupMiddlewares = (middlewares, devServer) => {
          if (!devServer) {
            throw new Error('webpack-dev-server is not defined');
          }
          
          // Charger setupProxy.js s'il existe
          const appPath = path.resolve(fs.realpathSync(process.cwd()));
          const appSrc = path.resolve(appPath, 'src');
          const proxySetupPath = path.resolve(appSrc, 'setupProxy.js');
          if (fs.existsSync(proxySetupPath)) {
            require(proxySetupPath)(devServer.app);
          }
          
          return middlewares;
        };
      }

      // ✅ Ne pas afficher l'overlay pour le rejet Socket.IO uniquement (python-socketio #590)
      // Condition stricte : message "Connection rejected" ET stack/message indique socket.io ou engine.io
      config.client = config.client || {};
      const baseOverlay = typeof config.client.overlay === 'object' ? config.client.overlay : {};
      config.client.overlay = {
        ...baseOverlay,
        runtimeErrors: (error) => {
          const msg = (error && error.message) ? String(error.message) : String(error);
          const stack = (error && error.stack) ? String(error.stack) : '';
          const fromSocket = stack.includes('socket.io') || stack.includes('engine.io') || msg.includes('socket.io') || msg.includes('engine.io');
          if (msg.includes('Connection rejected by server') && fromSocket) return false;
          if (typeof baseOverlay.runtimeErrors === 'function') {
            return baseOverlay.runtimeErrors(error);
          }
          return true;
        },
      };
      
      return config;
    };
  },
};
