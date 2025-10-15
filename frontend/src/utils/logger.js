/**
 * Logger structuré avec intégration Sentry
 * Remplace les console.log basiques par des logs tracés
 */
import * as Sentry from '@sentry/react';

const isDevelopment = process.env.NODE_ENV === 'development';

export const logger = {
  /**
   * Log d'information (non-critique)
   */
  info: (message, data = {}) => {
    if (isDevelopment) {
      console.info(`ℹ️ ${message}`, data);
    }
    Sentry.addBreadcrumb({
      category: 'info',
      message,
      level: 'info',
      data,
    });
  },

  /**
   * Log d'erreur (critique - sera envoyé à Sentry)
   */
  error: (message, error, additionalData = {}) => {
    console.error(`❌ ${message}`, error, additionalData);
    
    if (error instanceof Error) {
      Sentry.captureException(error, {
        tags: { context: message },
        extra: additionalData,
      });
    } else {
      Sentry.captureMessage(message, {
        level: 'error',
        extra: { error, ...additionalData },
      });
    }
  },

  /**
   * Log d'avertissement
   */
  warn: (message, data = {}) => {
    if (isDevelopment) {
      console.warn(`⚠️ ${message}`, data);
    }
    Sentry.addBreadcrumb({
      category: 'warning',
      message,
      level: 'warning',
      data,
    });
  },

  /**
   * Log de debug (développement uniquement)
   */
  debug: (message, data = {}) => {
    if (isDevelopment) {
      console.debug(`🐛 ${message}`, data);
    }
  },

  /**
   * Log d'événement utilisateur (pour analytics)
   */
  event: (eventName, properties = {}) => {
    if (isDevelopment) {
      console.log(`📊 Event: ${eventName}`, properties);
    }
    Sentry.addBreadcrumb({
      category: 'user',
      message: eventName,
      level: 'info',
      data: properties,
    });
  },
};

export default logger;

