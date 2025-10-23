import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/globals.css';
import 'leaflet/dist/leaflet.css';
import reportWebVitals from './reportWebVitals';
import * as Sentry from '@sentry/react';
import { onCLS, onFID, onFCP, onLCP, onTTFB } from 'web-vitals';

// ===== SENTRY CONFIGURATION =====
const SENTRY_DSN = process.env.REACT_APP_SENTRY_DSN;
const ENVIRONMENT =
  process.env.REACT_APP_SENTRY_ENVIRONMENT || process.env.NODE_ENV || 'development';

// Initialiser Sentry uniquement si DSN est configuré
if (SENTRY_DSN) {
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
    tracesSampleRate: ENVIRONMENT === 'production' ? 0.1 : 1.0, // 10% en prod, 100% en dev
    // Session Replay
    replaysSessionSampleRate: 0.1, // 10% des sessions
    replaysOnErrorSampleRate: 1.0, // 100% des sessions avec erreur
    // Ne pas envoyer en développement local (optionnel)
    beforeSend(event) {
      if (ENVIRONMENT === 'development' && window.location.hostname === 'localhost') {
        // En dev local, on affiche juste dans la console sans envoyer
        return null;
      }
      return event;
    },
  });
}
// Note: Sentry est désactivé si REACT_APP_SENTRY_DSN n'est pas configuré

// ===== WEB VITALS MONITORING =====
function sendWebVitalToSentry({ name, delta, value, id }) {
  // Envoyer les métriques Web Vitals à Sentry
  if (SENTRY_DSN) {
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

// Mesurer les Web Vitals
onCLS(sendWebVitalToSentry);
onFID(sendWebVitalToSentry);
onFCP(sendWebVitalToSentry);
onLCP(sendWebVitalToSentry);
onTTFB(sendWebVitalToSentry);

// ===== ERROR BOUNDARY (Sentry) =====
const SentryErrorBoundary = Sentry.ErrorBoundary;

const root = ReactDOM.createRoot(document.getElementById('root'));
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
    </SentryErrorBoundary>
  </React.StrictMode>
);

// Si vous souhaitez mesurer les performances de votre application, passez une fonction
// pour logger les résultats (par exemple : reportWebVitals(console.log))
// ou envoyez-les à un endpoint d'analytics. En savoir plus : https://bit.ly/CRA-vitals
reportWebVitals();
