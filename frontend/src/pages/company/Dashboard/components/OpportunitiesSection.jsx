// src/pages/company/Dashboard/components/OpportunitiesSection.jsx
/**
 * ✅ 3.4.2: Section affichant les opportunités d'optimisation depuis le dashboard temps réel
 */

import React from 'react';
import { Link } from 'react-router-dom';
import styles from '../CompanyDashboard.module.css';

const OpportunitiesSection = ({ opportunities, companyPublicId, loading }) => {
  if (loading) {
    return (
      <section className={styles.compactSection}>
        <h2>🔍 Opportunités d'optimisation</h2>
        <p style={{ color: '#666' }}>Chargement...</p>
      </section>
    );
  }

  if (!opportunities || opportunities.length === 0) {
    return (
      <section className={styles.compactSection}>
        <h2>🔍 Opportunités d'optimisation</h2>
        <p style={{ color: '#666' }}>Aucune opportunité détectée pour le moment.</p>
      </section>
    );
  }

  // Filtrer les opportunités critiques et high
  const criticalOpportunities = opportunities.filter(
    (opp) => opp.severity === 'critical' || opp.severity === 'high'
  );

  if (criticalOpportunities.length === 0) {
    return (
      <section className={styles.compactSection}>
        <h2>🔍 Opportunités d'optimisation</h2>
        <p style={{ color: '#666' }}>Aucune opportunité critique détectée.</p>
      </section>
    );
  }

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical':
        return '#f44336';
      case 'high':
        return '#ff9800';
      case 'medium':
        return '#ffc107';
      default:
        return '#9e9e9e';
    }
  };

  const getSeverityLabel = (severity) => {
    switch (severity) {
      case 'critical':
        return 'Critique';
      case 'high':
        return 'Élevée';
      case 'medium':
        return 'Moyenne';
      default:
        return 'Faible';
    }
  };

  return (
    <section className={styles.compactSection}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '12px',
        }}
      >
        <h2>🔍 Opportunités d'optimisation</h2>
        {companyPublicId && (
          <Link
            to={`/dashboard/company/${companyPublicId}/dispatch`}
            style={{ fontSize: '14px', color: '#1976d2' }}
          >
            Voir toutes →
          </Link>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {criticalOpportunities.slice(0, 5).map((opp) => (
          <div
            key={opp.assignment_id || opp.booking_id}
            style={{
              padding: '12px',
              border: `2px solid ${getSeverityColor(opp.severity)}`,
              borderRadius: '8px',
              backgroundColor: `${getSeverityColor(opp.severity)}15`,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span
                style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: getSeverityColor(opp.severity),
                  textTransform: 'uppercase',
                }}
              >
                {getSeverityLabel(opp.severity)}
              </span>
              {opp.current_delay_minutes !== undefined && (
                <span style={{ fontSize: '14px', fontWeight: 'bold' }}>
                  {opp.current_delay_minutes > 0 ? '+' : ''}
                  {opp.current_delay_minutes} min
                </span>
              )}
            </div>
            {opp.suggestions && opp.suggestions.length > 0 && (
              <div style={{ marginTop: '8px' }}>
                {opp.suggestions.slice(0, 2).map((suggestion, idx) => (
                  <div
                    key={idx}
                    style={{
                      fontSize: '13px',
                      color: '#555',
                      marginTop: idx > 0 ? '4px' : 0,
                      paddingLeft: '8px',
                      borderLeft: `3px solid ${getSeverityColor(opp.severity)}`,
                    }}
                  >
                    {suggestion.message || suggestion.action}
                  </div>
                ))}
              </div>
            )}
            {companyPublicId && (
              <div style={{ marginTop: '8px' }}>
                <Link
                  to={`/dashboard/company/${companyPublicId}/dispatch`}
                  style={{ fontSize: '12px', color: '#1976d2' }}
                >
                  Voir détails →
                </Link>
              </div>
            )}
          </div>
        ))}
      </div>
      {criticalOpportunities.length > 5 && (
        <p style={{ marginTop: '12px', fontSize: '13px', color: '#666' }}>
          +{criticalOpportunities.length - 5} autre(s) opportunité(s)
        </p>
      )}
    </section>
  );
};

export default OpportunitiesSection;

