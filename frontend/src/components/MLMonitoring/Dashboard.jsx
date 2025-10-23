import React, { useState, useEffect } from 'react';
import './Dashboard.css';

/**
 * Dashboard de monitoring ML temps réel
 *
 * Affiche:
 * - Métriques en temps réel (MAE, R², accuracy)
 * - Graphiques performance 7 derniers jours
 * - Feature flags status
 * - Anomalies détectées
 */
const MLDashboard = () => {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Charger les données
  const fetchSummary = async () => {
    try {
      const response = await fetch('/api/ml-monitoring/summary');
      if (!response.ok) throw new Error('Failed to fetch summary');

      const data = await response.json();
      setSummary(data);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching ML summary:', err);
    } finally {
      setLoading(false);
    }
  };

  // Effect initial + auto-refresh
  useEffect(() => {
    fetchSummary();

    if (autoRefresh) {
      const interval = setInterval(fetchSummary, 30000); // 30s
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  if (loading) {
    return (
      <div className="ml-dashboard loading">
        <div className="spinner"></div>
        <p>Chargement du dashboard...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="ml-dashboard error">
        <h3>❌ Erreur</h3>
        <p>{error}</p>
        <button onClick={fetchSummary}>Réessayer</button>
      </div>
    );
  }

  const { metrics_24h, feature_flags, anomalies_count, total_predictions } = summary || {};

  return (
    <div className="ml-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <h1>📊 Monitoring ML - Prédiction de Retards</h1>
        <div className="header-controls">
          <label>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh (30s)
          </label>
          <button onClick={fetchSummary} className="refresh-btn">
            🔄 Actualiser
          </button>
        </div>
      </div>

      {/* Feature Flags Status */}
      <div className="feature-flags-section">
        <h2>🚦 Feature Flags</h2>
        <div className="flags-grid">
          <div className="flag-card">
            <span className="flag-label">ML Activé</span>
            <span className={`flag-value ${feature_flags?.ml_enabled ? 'enabled' : 'disabled'}`}>
              {feature_flags?.ml_enabled ? '✅ Oui' : '❌ Non'}
            </span>
          </div>
          <div className="flag-card">
            <span className="flag-label">Trafic ML</span>
            <span className="flag-value">{feature_flags?.ml_traffic_percentage || 0}%</span>
          </div>
          <div className="flag-card">
            <span className="flag-label">Taux Succès</span>
            <span
              className={`flag-value ${
                (feature_flags?.ml_success_rate || 0) > 0.95 ? 'good' : 'warning'
              }`}
            >
              {((feature_flags?.ml_success_rate || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flag-card">
            <span className="flag-label">Total Prédictions</span>
            <span className="flag-value">{total_predictions || 0}</span>
          </div>
        </div>
      </div>

      {/* Métriques 24h */}
      <div className="metrics-section">
        <h2>📈 Métriques 24 Heures</h2>
        <div className="metrics-grid">
          <MetricCard
            title="MAE (Mean Absolute Error)"
            value={metrics_24h?.mae}
            unit="min"
            target={3.0}
            format={(v) => v?.toFixed(2)}
            isGood={(v) => v < 3.0}
          />
          <MetricCard
            title="R² Score"
            value={metrics_24h?.r2}
            unit=""
            target={0.65}
            format={(v) => v?.toFixed(4)}
            isGood={(v) => v > 0.65}
          />
          <MetricCard
            title="Accuracy Rate"
            value={metrics_24h?.accuracy_rate}
            unit="%"
            target={0.8}
            format={(v) => (v * 100)?.toFixed(1)}
            isGood={(v) => v > 0.8}
          />
          <MetricCard
            title="Temps Prédiction Moyen"
            value={metrics_24h?.avg_prediction_time_ms}
            unit="ms"
            target={150}
            format={(v) => v?.toFixed(1)}
            isGood={(v) => v < 150}
          />
        </div>
        <div className="metrics-info">
          <p>
            Basé sur <strong>{metrics_24h?.count || 0}</strong> prédictions avec résultats réels
          </p>
        </div>
      </div>

      {/* Anomalies */}
      {anomalies_count > 0 && (
        <div className="anomalies-section alert">
          <h2>⚠️ Anomalies Détectées</h2>
          <p>
            <strong>{anomalies_count}</strong> prédictions avec erreur {'>'} 5 min dans les
            dernières 24h
          </p>
          <button onClick={() => (window.location.href = '#anomalies')}>Voir détails</button>
        </div>
      )}

      {/* Footer */}
      <div className="dashboard-footer">
        <p>Dernière mise à jour : {new Date(summary?.timestamp).toLocaleString('fr-FR')}</p>
      </div>
    </div>
  );
};

/**
 * Composant pour afficher une métrique individuelle
 */
const MetricCard = ({ title, value, unit, target, format, isGood }) => {
  if (value === null || value === undefined) {
    return (
      <div className="metric-card no-data">
        <h3>{title}</h3>
        <div className="metric-value">-</div>
        <div className="metric-subtitle">Aucune donnée</div>
      </div>
    );
  }

  const formattedValue = format ? format(value) : value;
  const status = isGood(value) ? 'good' : 'warning';

  return (
    <div className={`metric-card ${status}`}>
      <h3>{title}</h3>
      <div className="metric-value">
        {formattedValue}
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
      <div className="metric-subtitle">
        Cible: {target}
        {unit}
      </div>
      <div className={`metric-status ${status}`}>{status === 'good' ? '✅' : '⚠️'}</div>
    </div>
  );
};

export default MLDashboard;
