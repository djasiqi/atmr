import React from 'react';
import './ErrorFallback.css';

/**
 * Composant de fallback affiché quand une erreur est capturée par ErrorBoundary
 */
const ErrorFallback = ({ error, errorInfo, onReset }) => {
  const isDev = process.env.NODE_ENV === 'development';

  return (
    <div className="error-fallback-container">
      <div className="error-fallback-content">
        <div className="error-icon">⚠️</div>
        <h1>Oups ! Une erreur s'est produite</h1>
        <p className="error-message">
          Nous sommes désolés, quelque chose s'est mal passé. L'équipe technique a été notifiée.
        </p>

        {isDev && error && (
          <details className="error-details">
            <summary>Détails de l'erreur (dev mode)</summary>
            <div className="error-stack">
              <p>
                <strong>Message :</strong> {error.toString()}
              </p>
              {errorInfo && errorInfo.componentStack && <pre>{errorInfo.componentStack}</pre>}
            </div>
          </details>
        )}

        <div className="error-actions">
          <button onClick={onReset} className="btn-reset">
            Réessayer
          </button>
          <button onClick={() => (window.location.href = '/')} className="btn-home">
            Retour à l'accueil
          </button>
        </div>

        <div className="error-help">
          <p>
            Si le problème persiste, veuillez contacter le support technique ou recharger la page.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ErrorFallback;
