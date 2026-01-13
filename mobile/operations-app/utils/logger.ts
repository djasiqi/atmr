/**
 * Logger conditionnel qui n'affiche les logs qu'en développement
 */

import Constants from 'expo-constants';

const isDevelopment = __DEV__ || Constants.expoConfig?.extra?.environment !== 'production';

export const logger = {
  log: (...args: any[]) => {
    if (isDevelopment) {
      console.log(...args);
    }
  },
  
  info: (...args: any[]) => {
    if (isDevelopment) {
      console.info(...args);
    }
  },
  
  warn: (...args: any[]) => {
    // Warnings toujours affichés
    console.warn(...args);
  },
  
  error: (...args: any[]) => {
    // Erreurs toujours affichées
    console.error(...args);
  },
  
  debug: (...args: any[]) => {
    if (isDevelopment) {
      console.debug(...args);
    }
  },
};

// Export aussi pour compatibilité
export const isDev = isDevelopment;
