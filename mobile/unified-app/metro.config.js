const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Requis pour expo-sqlite (web / wa-sqlite.wasm) — voir docs Expo SQLite « Web setup ».
if (!config.resolver.assetExts.includes('wasm')) {
  config.resolver.assetExts.push('wasm');
}

// SharedArrayBuffer pour wa-sqlite (dev server Expo).
const previousEnhanceMiddleware = config.server?.enhanceMiddleware;
config.server = {
  ...config.server,
  enhanceMiddleware: (middleware, server) => {
    const base =
      typeof previousEnhanceMiddleware === 'function'
        ? previousEnhanceMiddleware(middleware, server)
        : middleware;
    return (req, res, next) => {
      res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      return base(req, res, next);
    };
  },
};

module.exports = config;
