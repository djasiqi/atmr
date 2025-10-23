// config-overrides.js

const SML_EXCLUDE = /node_modules[\\/](svg-engine)/; // adapte le nom du paquet si besoin

module.exports = function override(config, _env) {
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

  // Ignore "Failed to parse source map" (optionnel)
  config.ignoreWarnings = [...(config.ignoreWarnings || []), /Failed to parse source map/];

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
  }

  return config;
};
