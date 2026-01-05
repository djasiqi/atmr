// craco.config.js
// Configuration minimale pour @craco/craco

const SML_EXCLUDE = /node_modules[\\/](svg-engine)/;
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = {
  eslint: {
    enable: false, // ✅ Désactiver ESLint pour éviter l'erreur eslint-loader
  },
  devServer: {
    // ✅ Configuration minimale pour le serveur de développement
    hot: false,
    liveReload: true,
    webSocketServer: false,
  },
  webpack: {
    configure: (config) => {
      // ✅ Fix: Supprimer les propriétés obsolètes de devServer
      if (config.devServer) {
        delete config.devServer.onAfterSetupMiddleware;
        delete config.devServer.onBeforeSetupMiddleware;
      }

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
        else rule.exclude = [rule.exclude, pattern];
      };

      const visit = (rule) => {
        if (!rule) return;
        if (isSmlRule(rule)) appendExclude(rule, SML_EXCLUDE);
        if (Array.isArray(rule.oneOf)) rule.oneOf.forEach(visit);
        if (Array.isArray(rule.rules)) rule.rules.forEach(visit);
      };

      (config.module?.rules || []).forEach(visit);

      // Forcer MiniCssExtractPlugin à ignorer l'ordre des imports CSS
      config.plugins = (config.plugins || []).map((plugin) => {
        if (plugin instanceof MiniCssExtractPlugin) {
          plugin.options = {
            ...plugin.options,
            ignoreOrder: true,
          };
        }
        return plugin;
      });

      // Ignore "Failed to parse source map"
      config.ignoreWarnings = [...(config.ignoreWarnings || []), /Failed to parse source map/];
      
      // ✅ Ignorer les erreurs ModuleHotAcceptDependency
      config.ignoreWarnings = [
        ...(config.ignoreWarnings || []),
        /No template for dependency: ModuleHotAcceptDependency/,
        /ModuleHotAcceptDependency/,
      ];
      
      // ✅ Désactiver complètement HMR
      if (config.plugins) {
        config.plugins = config.plugins.filter(
          (plugin) => !(plugin.constructor.name === 'HotModuleReplacementPlugin')
        );
      }
      
      if (config.devServer) {
        config.devServer.hot = false;
        config.devServer.liveReload = true;
      }

      return config;
    },
  },
};

