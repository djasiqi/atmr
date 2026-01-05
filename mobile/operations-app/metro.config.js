// metro.config.js
const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Configuration pour exclure/react-native-maps et expo-secure-store sur le web
const originalResolveRequest = config.resolver.resolveRequest;
config.resolver.resolveRequest = (context, moduleName, platform) => {
  // Sur le web, remplacer react-native-maps par un stub vide
  if (platform === 'web' && moduleName === 'react-native-maps') {
    return {
      type: 'sourceFile',
      filePath: path.resolve(__dirname, 'metro-stubs/react-native-maps.js'),
    };
  }
  // Sur le web, remplacer expo-secure-store par un stub utilisant localStorage
  if (platform === 'web' && moduleName === 'expo-secure-store') {
    return {
      type: 'sourceFile',
      filePath: path.resolve(__dirname, 'metro-stubs/expo-secure-store.js'),
    };
  }
  // Comportement par défaut pour les autres modules
  if (originalResolveRequest) {
    return originalResolveRequest(context, moduleName, platform);
  }
  return context.resolveRequest(context, moduleName, platform);
};

module.exports = config;

