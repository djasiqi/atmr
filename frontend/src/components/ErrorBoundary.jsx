import React from 'react';
import ErrorFallback from './ErrorFallback';

/**
 * Error Boundary pour capturer les erreurs React et éviter le crash de l'application
 *
 * Usage:
 * <ErrorBoundary>
 *   <YourComponent />
 * </ErrorBoundary>
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error) {
    // Mettre à jour l'état pour afficher l'UI de fallback au prochain rendu
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error, errorInfo) {
    // Vous pouvez aussi logger l'erreur vers un service externe (Sentry, LogRocket, etc.)
    console.error('Error caught by ErrorBoundary:', error, errorInfo);

    this.setState({
      errorInfo,
    });

    // Optionnel : envoyer l'erreur à un service de monitoring
    // if (process.env.NODE_ENV === 'production') {
    //   Sentry.captureException(error, { extra: errorInfo });
    // }
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      return (
        <ErrorFallback
          error={this.state.error}
          errorInfo={this.state.errorInfo}
          onReset={this.handleReset}
        />
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
