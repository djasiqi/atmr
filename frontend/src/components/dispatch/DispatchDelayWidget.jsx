// frontend/src/components/dispatch/DispatchDelayWidget.jsx
/**
 * Widget compact pour afficher les retards en temps réel
 * Peut être intégré dans n'importe quelle page (Dashboard, Sidebar, etc.)
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getLiveDelays } from '../../services/dispatchMonitoringService';
import './DispatchDelayWidget.css';

const DispatchDelayWidget = ({ 
  companyPublicId,
  refreshInterval = 60000, // 1 minute par défaut
  showDetails = false,
  compact = false 
}) => {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const today = new Date().toISOString().split('T')[0];
        const data = await getLiveDelays(today);
        setSummary(data.summary);
        setLastUpdate(new Date());
        setLoading(false);
      } catch (error) {
        console.error('[DispatchDelayWidget] Error:', error);
        setLoading(false);
      }
    };

    fetchSummary();
    const interval = setInterval(fetchSummary, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  const handleClick = () => {
    navigate(`/dashboard/company/${companyPublicId}/dispatch/monitor`);
  };

  if (loading && !summary) {
    return (
      <div className={`dispatch-widget ${compact ? 'compact' : ''} loading`}>
        <div className="widget-spinner" />
      </div>
    );
  }

  if (!summary) {
    return null;
  }

  const hasDelays = summary.late > 0;
  const status = hasDelays ? 'warning' : 'success';

  // Mode compact
  if (compact) {
    return (
      <div 
        className={`dispatch-widget compact ${status}`}
        onClick={handleClick}
        title="Cliquer pour voir les détails"
      >
        <div className="widget-icon">
          {hasDelays ? '⚠️' : '✅'}
        </div>
        <div className="widget-content">
          <span className="widget-label">Dispatch</span>
          <span className="widget-value">
            {hasDelays ? `${summary.late} retard(s)` : 'À l\'heure'}
          </span>
        </div>
      </div>
    );
  }

  // Mode normal
  return (
    <div className={`dispatch-widget ${status}`}>
      <div className="widget-header">
        <h3>📡 Monitoring Dispatch</h3>
        {lastUpdate && (
          <span className="widget-timestamp">
            Màj: {lastUpdate.toLocaleTimeString('fr-FR', { 
              hour: '2-digit', 
              minute: '2-digit' 
            })}
          </span>
        )}
      </div>

      <div className="widget-stats">
        <div className="stat-item">
          <span className="stat-value">{summary.total}</span>
          <span className="stat-label">Total</span>
        </div>

        <div className="stat-item success">
          <span className="stat-value">{summary.on_time}</span>
          <span className="stat-label">À l'heure</span>
        </div>

        {hasDelays && (
          <div className="stat-item warning">
            <span className="stat-value">{summary.late}</span>
            <span className="stat-label">En retard</span>
          </div>
        )}

        {summary.early > 0 && (
          <div className="stat-item info">
            <span className="stat-value">{summary.early}</span>
            <span className="stat-label">En avance</span>
          </div>
        )}
      </div>

      {showDetails && summary.average_delay !== undefined && (
        <div className="widget-details">
          <p className="avg-delay">
            Retard moyen: <strong>{summary.average_delay.toFixed(1)} min</strong>
          </p>
        </div>
      )}

      <div className="widget-footer">
        <button 
          className="view-details-btn"
          onClick={handleClick}
        >
          Voir détails →
        </button>
      </div>
    </div>
  );
};

export default DispatchDelayWidget;

