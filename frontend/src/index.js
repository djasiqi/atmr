import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/globals.css';
import reportWebVitals from './reportWebVitals';
import * as Sentry from '@sentry/react';
import { onCLS, onINP, onFCP, onLCP, onTTFB } from 'web-vitals';
import { SpeedInsights } from '@vercel/speed-insights/react';

// ===== SENTRY CONFIGURATION =====
const SENTRY_DSN = process.env.REACT_APP_SENTRY_DSN;
const ENVIRONMENT =
  process.env.REACT_APP_SENTRY_ENVIRONMENT || process.env.NODE_ENV || 'development';

// ✅ Initialiser Sentry uniquement en production (évite les erreurs de connexion en dev)
if (SENTRY_DSN && ENVIRONMENT !== 'development') {
  Sentry.init({
    dsn: SENTRY_DSN,
    environment: ENVIRONMENT,
    integrations: [
      Sentry.browserTracingIntegration({
        // Tracer les performances des pages
        tracePropagationTargets: ['localhost', /^\//],
      }),
      Sentry.replayIntegration({
        // Enregistrer les sessions avec erreurs
        maskAllText: true,
        blockAllMedia: true,
      }),
    ],
    // Performance Monitoring
    tracesSampleRate: 0.1, // 10% en production
    // Session Replay
    replaysSessionSampleRate: 0.1, // 10% des sessions
    replaysOnErrorSampleRate: 1.0, // 100% des sessions avec erreur
  });
  console.log('✅ Sentry initialisé en mode', ENVIRONMENT);
} else {
  // Silence en dev: pas de log Sentry non pertinent.
}
// Note: Sentry est désactivé en développement pour éviter les erreurs de connexion

// ===== WEB VITALS MONITORING =====
function sendWebVitalToSentry({ name, delta, value, id }) {
  // ✅ Envoyer les métriques Web Vitals à Sentry uniquement en production
  if (SENTRY_DSN && ENVIRONMENT !== 'development') {
    Sentry.captureMessage(`Web Vital: ${name}`, {
      level: 'info',
      tags: {
        web_vital: name,
        metric_id: id,
      },
      contexts: {
        'Web Vitals': {
          name,
          value: Math.round(value),
          delta: Math.round(delta),
          rating: value < 100 ? 'good' : value < 300 ? 'needs-improvement' : 'poor',
        },
      },
    });
  }
}

// ✅ Mesurer les Web Vitals uniquement en production
if (ENVIRONMENT !== 'development') {
  onCLS(sendWebVitalToSentry);
  onINP(sendWebVitalToSentry);
  onFCP(sendWebVitalToSentry);
  onLCP(sendWebVitalToSentry);
  onTTFB(sendWebVitalToSentry);
}

// ===== GESTION ERREURS EXTENSIONS NAVIGATEUR + SOCKET =====
// ⚡ Ignorer l'erreur "listener indicated asynchronous response"
// qui est généralement causée par des extensions de navigateur
// ⚡ Intercepter "Connection rejected by server" (socket.io / python-socketio #590)
//    pour éviter le crash quand le serveur refuse la connexion sans data.message
window.addEventListener(
  'error',
  (event) => {
    const message = event.message || (event.error?.message ?? '') || '';
    const stack = event.error?.stack ?? '';
    const chunkFailedRegex = /Loading chunk [\w-]+ failed/i;

    if (
      message.includes('listener indicated an asynchronous response') ||
      message.includes('message channel closed')
    ) {
      event.preventDefault();
      return false;
    }

    if (chunkFailedRegex.test(message)) {
      if (!window.__RELOADING_FROM_CHUNK_ERROR__) {
        window.__RELOADING_FROM_CHUNK_ERROR__ = true;
        window.location.reload();
      }
      event.preventDefault();
      return false;
    }

    // Socket.IO : rejet sans data.message (python-socketio #590) — on masque seulement si ça vient bien du transport
    const fromSocketTransport =
      stack.includes('socket.io') || stack.includes('engine.io') || message.includes('socket.io') || message.includes('engine.io');
    const isSocketRejection =
      message.includes('Connection rejected by server') && fromSocketTransport;
    const isOnpacketUndefined =
      message.includes('message') && message.includes('undefined') && stack.includes('onpacket') && fromSocketTransport;
    if (isSocketRejection || isOnpacketUndefined) {
      event.preventDefault();
      event.stopPropagation();
      if (process.env.NODE_ENV === 'development') {
        window.__SOCKET_REJECTION_SUPPRESS_COUNT = (window.__SOCKET_REJECTION_SUPPRESS_COUNT || 0) + 1;
        console.warn(
          '[App] Connexion Socket refusée (interceptée). Vérifiez auth/backend.',
          'Count:', window.__SOCKET_REJECTION_SUPPRESS_COUNT,
          event.error
        );
      } else {
        console.warn('[App] Connexion Socket refusée par le serveur.', event.error);
      }
      window.dispatchEvent(
        new CustomEvent('socket_connection_rejected', { detail: { message, originalError: event.error } })
      );
      return true;
    }
  },
  true
);

// Gérer aussi les erreurs de promesse non catchées (extensions + Socket rejet)
window.addEventListener('unhandledrejection', (event) => {
  const reasonMessage = event.reason?.message || String(event.reason || '');
  const chunkFailedRegex = /Loading chunk [\w-]+ failed/i;

  if (
    reasonMessage.includes('listener indicated an asynchronous response') ||
    reasonMessage.includes('message channel closed')
  ) {
    // Ignorer silencieusement cette erreur (elle vient des extensions)
    event.preventDefault();
    return;
  }

  // Socket.IO : rejet sans data.message (python-socketio #590) — uniquement si stack/raison indique socket/engine
  const reasonStack = (event.reason?.stack || '').toString();
  const fromSocket = reasonStack.includes('socket.io') || reasonStack.includes('engine.io')
    || reasonMessage.includes('socket.io') || reasonMessage.includes('engine.io');
  if (reasonMessage.includes('Connection rejected by server') && fromSocket) {
    event.preventDefault();
    if (process.env.NODE_ENV === 'development') {
      window.__SOCKET_REJECTION_SUPPRESS_COUNT = (window.__SOCKET_REJECTION_SUPPRESS_COUNT || 0) + 1;
      console.warn('[App] Connexion Socket refusée (promise, interceptée). Count:', window.__SOCKET_REJECTION_SUPPRESS_COUNT, event.reason);
    } else {
      console.warn('[App] Connexion Socket refusée par le serveur.', event.reason);
    }
    window.dispatchEvent(
      new CustomEvent('socket_connection_rejected', {
        detail: { message: reasonMessage, originalError: event.reason },
      })
    );
    return;
  }

  if (chunkFailedRegex.test(reasonMessage)) {
    if (!window.__RELOADING_FROM_CHUNK_ERROR__) {
      window.__RELOADING_FROM_CHUNK_ERROR__ = true;
      window.location.reload();
    }
    event.preventDefault();
  }
});

// ===== ERROR BOUNDARY (Sentry) =====
const SentryErrorBoundary = Sentry.ErrorBoundary;

const root = ReactDOM.createRoot(document.getElementById('root'));
const isDev = ENVIRONMENT === 'development';
root.render(
  <React.StrictMode>
    <SentryErrorBoundary
      fallback={({ error, resetError }) => (
        <div style={{ padding: '2rem', textAlign: 'center' }}>
          <h1>⚠️ Une erreur est survenue</h1>
          <p style={{ color: '#666' }}>L'équipe technique a été notifiée.</p>
          <button
            onClick={resetError}
            style={{
              marginTop: '1rem',
              padding: '0.5rem 1rem',
              background: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Réessayer
          </button>
          {process.env.NODE_ENV === 'development' && (
            <details style={{ marginTop: '1rem', textAlign: 'left' }}>
              <summary>Détails de l'erreur (dev only)</summary>
              <pre style={{ background: '#f5f5f5', padding: '1rem', overflow: 'auto' }}>
                {error.toString()}
              </pre>
            </details>
          )}
        </div>
      )}
    >
      <App />
      {!isDev && <SpeedInsights />}
    </SentryErrorBoundary>
  </React.StrictMode>
);

// Si vous souhaitez mesurer les performances de votre application, passez une fonction
// pour logger les résultats (par exemple : reportWebVitals(console.log))
// ou envoyez-les à un endpoint d'analytics. En savoir plus : https://bit.ly/CRA-vitals
reportWebVitals();
