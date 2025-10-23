// frontend/src/pages/company/Dispatch/Dashboard/RLMetricsDashboard.jsx
import React, { useEffect, useState, useCallback } from 'react';
import {
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import apiClient from '../../../../utils/apiClient';
import './RLMetricsDashboard.css';

const RLMetricsDashboard = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [period, setPeriod] = useState(30); // Jours

  const loadMetrics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const { data } = await apiClient.get('/company_dispatch/rl/metrics', {
        params: { days: period },
      });
      setMetrics(data);
    } catch (err) {
      console.error('[RLMetricsDashboard] Error loading metrics:', err);
      setError(err.message || 'Erreur chargement métriques');
    } finally {
      setLoading(false);
    }
  }, [period]);

  useEffect(() => {
    loadMetrics();

    // Auto-refresh toutes les 60 secondes
    const interval = setInterval(loadMetrics, 60000);
    return () => clearInterval(interval);
  }, [loadMetrics]);

  if (loading && !metrics) {
    return (
      <div className="rl-metrics-dashboard">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Chargement des métriques RL...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rl-metrics-dashboard">
        <div className="error-container">
          <h3>❌ Erreur</h3>
          <p>{error}</p>
          <button onClick={loadMetrics} className="btn-retry">
            🔄 Réessayer
          </button>
        </div>
      </div>
    );
  }

  if (!metrics || metrics.total_suggestions === 0) {
    return (
      <div className="rl-metrics-dashboard">
        <div className="empty-state">
          <h3>📊 Métriques Système RL</h3>
          <p>Aucune suggestion générée pour la période sélectionnée.</p>
          <p className="hint">
            Les métriques apparaîtront dès que des suggestions RL seront générées.
          </p>
        </div>
      </div>
    );
  }

  // Préparer données pour PieChart
  const sourceData = [
    { name: 'DQN Model', value: metrics.by_source.dqn_model, color: '#4CAF50' },
    { name: 'Heuristique', value: metrics.by_source.basic_heuristic, color: '#FF9800' },
  ];

  // Couleurs pour les KPI cards
  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return 'kpi-excellent';
    if (conf >= 0.6) return 'kpi-good';
    return 'kpi-warning';
  };

  const getAccuracyColor = (acc) => {
    if (!acc) return 'kpi-neutral';
    if (acc >= 0.85) return 'kpi-excellent';
    if (acc >= 0.7) return 'kpi-good';
    return 'kpi-warning';
  };

  // Déterminer alertes
  const alerts = [];
  if (metrics.fallback_rate > 0.2) {
    alerts.push({
      level: 'danger',
      icon: '🚨',
      message: `Taux fallback heuristique élevé (${(metrics.fallback_rate * 100).toFixed(0)}%)`,
      action: 'Vérifier fonctionnement modèle DQN',
    });
  }
  if (metrics.avg_gain_accuracy && metrics.avg_gain_accuracy < 0.7) {
    alerts.push({
      level: 'warning',
      icon: '⚠️',
      message: `Précision gain faible (${(metrics.avg_gain_accuracy * 100).toFixed(0)}%)`,
      action: 'Ré-entraînement du modèle recommandé',
    });
  }
  if (metrics.application_rate < 0.3) {
    alerts.push({
      level: 'info',
      icon: '💡',
      message: `Taux application bas (${(metrics.application_rate * 100).toFixed(0)}%)`,
      action: 'Les suggestions sont-elles pertinentes ?',
    });
  }
  if (metrics.avg_confidence >= 0.9) {
    alerts.push({
      level: 'success',
      icon: '✅',
      message: `Confiance excellente (${(metrics.avg_confidence * 100).toFixed(0)}%)`,
      action: 'Le modèle performe très bien !',
    });
  }

  return (
    <div className="rl-metrics-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-content">
          <h2>📊 Métriques Système RL</h2>
          <p className="subtitle">Performance des suggestions RL en temps réel</p>
        </div>

        <div className="header-controls">
          {/* Sélecteur période */}
          <div className="period-selector">
            <label>Période :</label>
            <select value={period} onChange={(e) => setPeriod(Number(e.target.value))}>
              <option value={7}>7 jours</option>
              <option value={30}>30 jours</option>
              <option value={90}>90 jours</option>
            </select>
          </div>

          {/* Bouton refresh */}
          <button onClick={loadMetrics} className="btn-refresh" disabled={loading}>
            {loading ? '⏳' : '🔄'} Actualiser
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-value">{metrics.total_suggestions}</div>
          <div className="kpi-label">Suggestions générées</div>
          <div className="kpi-period">{period} jours</div>
        </div>

        <div className={`kpi-card ${getConfidenceColor(metrics.avg_confidence)}`}>
          <div className="kpi-value">{(metrics.avg_confidence * 100).toFixed(0)}%</div>
          <div className="kpi-label">Confiance moyenne</div>
          <div className="kpi-trend">
            {metrics.avg_confidence >= 0.8
              ? '📈 Excellent'
              : metrics.avg_confidence >= 0.6
              ? '✅ Bon'
              : '⚠️ À améliorer'}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-value">{(metrics.application_rate * 100).toFixed(0)}%</div>
          <div className="kpi-label">Taux application</div>
          <div className="kpi-detail">
            {metrics.applied_count} / {metrics.total_suggestions} appliquées
          </div>
        </div>

        <div className={`kpi-card ${getAccuracyColor(metrics.avg_gain_accuracy)}`}>
          <div className="kpi-value">
            {metrics.avg_gain_accuracy ? `${(metrics.avg_gain_accuracy * 100).toFixed(0)}%` : 'N/A'}
          </div>
          <div className="kpi-label">Précision gain</div>
          <div className="kpi-detail">
            {metrics.avg_gain_accuracy ? `Estimé vs réel` : 'Pas encore de données'}
          </div>
        </div>
      </div>

      {/* Alertes */}
      {alerts.length > 0 && (
        <div className="alerts-section">
          {alerts.map((alert, idx) => (
            <div key={idx} className={`alert alert-${alert.level}`}>
              <span className="alert-icon">{alert.icon}</span>
              <div className="alert-content">
                <div className="alert-message">{alert.message}</div>
                <div className="alert-action">{alert.action}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Graphiques */}
      <div className="charts-grid">
        {/* Graphique 1: Évolution confiance */}
        <div className="chart-card">
          <h3>📈 Évolution confiance moyenne</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metrics.confidence_history}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis domain={[0, 1]} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
              <Tooltip
                formatter={(value) => `${(value * 100).toFixed(1)}%`}
                labelStyle={{ color: '#333' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="avg_confidence"
                stroke="#4CAF50"
                strokeWidth={2}
                name="Confiance moyenne"
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Graphique 2: Répartition sources */}
        <div className="chart-card">
          <h3>🔀 Répartition sources</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={sourceData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {sourceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>

          {/* Légende personnalisée */}
          <div className="source-legend">
            <div className="legend-item">
              <span className="legend-dot" style={{ backgroundColor: '#4CAF50' }}></span>
              <span>DQN Model: {metrics.by_source.dqn_model}</span>
            </div>
            <div className="legend-item">
              <span className="legend-dot" style={{ backgroundColor: '#FF9800' }}></span>
              <span>Heuristique: {metrics.by_source.basic_heuristic}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Statistiques détaillées */}
      <div className="stats-grid">
        <div className="stat-card">
          <h4>📋 Suggestions</h4>
          <div className="stat-row">
            <span>Total générées</span>
            <strong>{metrics.total_suggestions}</strong>
          </div>
          <div className="stat-row">
            <span>Appliquées</span>
            <strong className="text-success">{metrics.applied_count}</strong>
          </div>
          <div className="stat-row">
            <span>Rejetées</span>
            <strong className="text-warning">{metrics.rejected_count}</strong>
          </div>
          <div className="stat-row">
            <span>En attente</span>
            <strong className="text-info">{metrics.pending_count}</strong>
          </div>
        </div>

        <div className="stat-card">
          <h4>⏱️ Gains temporels</h4>
          <div className="stat-row">
            <span>Gain estimé total</span>
            <strong>{metrics.total_expected_gain_minutes} min</strong>
          </div>
          <div className="stat-row">
            <span>Gain réel total</span>
            <strong className="text-success">{metrics.total_actual_gain_minutes} min</strong>
          </div>
          <div className="stat-row">
            <span>Écart</span>
            <strong>
              {metrics.total_expected_gain_minutes - metrics.total_actual_gain_minutes} min
            </strong>
          </div>
        </div>

        <div className="stat-card">
          <h4>🎯 Performance modèle</h4>
          <div className="stat-row">
            <span>Confiance moyenne</span>
            <strong>{(metrics.avg_confidence * 100).toFixed(1)}%</strong>
          </div>
          <div className="stat-row">
            <span>Précision gain</span>
            <strong>
              {metrics.avg_gain_accuracy
                ? `${(metrics.avg_gain_accuracy * 100).toFixed(1)}%`
                : 'N/A'}
            </strong>
          </div>
          <div className="stat-row">
            <span>Taux fallback</span>
            <strong className={metrics.fallback_rate > 0.2 ? 'text-warning' : 'text-success'}>
              {(metrics.fallback_rate * 100).toFixed(1)}%
            </strong>
          </div>
        </div>
      </div>

      {/* Top suggestions */}
      {metrics.top_suggestions && metrics.top_suggestions.length > 0 && (
        <div className="top-suggestions-section">
          <h3>🏆 Top 5 suggestions (gain réel)</h3>
          <div className="suggestions-table">
            <table>
              <thead>
                <tr>
                  <th>Booking</th>
                  <th>Confiance</th>
                  <th>Gain estimé</th>
                  <th>Gain réel</th>
                  <th>Précision</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {metrics.top_suggestions.slice(0, 5).map((sugg, idx) => (
                  <tr key={idx}>
                    <td>#{sugg.booking_id}</td>
                    <td>{(sugg.confidence * 100).toFixed(0)}%</td>
                    <td>{sugg.expected_gain || 0} min</td>
                    <td className="text-success">{sugg.actual_gain || 0} min</td>
                    <td>
                      {sugg.gain_accuracy ? `${(sugg.gain_accuracy * 100).toFixed(0)}%` : 'N/A'}
                    </td>
                    <td>
                      <span
                        className={`badge badge-${
                          sugg.source === 'dqn_model' ? 'success' : 'warning'
                        }`}
                      >
                        {sugg.source === 'dqn_model' ? 'DQN' : 'Heuristique'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="dashboard-footer">
        <p className="footer-info">
          ℹ️ Dernière mise à jour : {new Date(metrics.timestamp).toLocaleString('fr-FR')}
        </p>
        <p className="footer-hint">
          💡 Les métriques se rafraîchissent automatiquement toutes les minutes
        </p>
      </div>
    </div>
  );
};

export default RLMetricsDashboard;
