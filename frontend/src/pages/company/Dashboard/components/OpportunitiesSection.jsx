// src/pages/company/Dashboard/components/OpportunitiesSection.jsx
import React from 'react';
import { Link } from 'react-router-dom';
import {
  FiAlertTriangle,
  FiArrowRight,
  FiActivity,
  FiUserPlus,
  FiRefreshCw,
  FiZap,
  FiCheck,
} from 'react-icons/fi';
import styles from './OpportunitiesSection.module.css';

const SEVERITY_CONFIG = {
  critical: { label: 'Critique', cssClass: 'critical' },
  high: { label: 'Élevée', cssClass: 'high' },
  medium: { label: 'Moyenne', cssClass: 'medium' },
  low: { label: 'Faible', cssClass: 'low' },
};

const getActionConfig = (opp) => {
  if (opp.current_delay_minutes >= 15) {
    return { label: 'Réassigner', Icon: FiRefreshCw, variant: 'critical' };
  }
  if (opp.severity === 'critical' || opp.severity === 'high') {
    return { label: 'Assigner', Icon: FiUserPlus, variant: 'high' };
  }
  return { label: 'Optimiser', Icon: FiZap, variant: 'default' };
};

const OpportunitiesSection = ({ opportunities, companyPublicId, loading, onAction }) => {
  if (loading) {
    return (
      <section className={styles.container}>
        <div className={styles.compactStatus}>
          <FiActivity className={styles.compactIcon} size={14} />
          <span className={styles.compactText}>Analyse en cours...</span>
        </div>
      </section>
    );
  }

  if (!opportunities || opportunities.length === 0) {
    return (
      <section className={styles.container}>
        <div className={styles.compactStatus}>
          <FiCheck className={styles.compactIconOk} size={14} />
          <span className={styles.compactTextOk}>Aucun problème détecté</span>
        </div>
      </section>
    );
  }

  const criticalOpportunities = opportunities.filter(
    (opp) => opp.severity === 'critical' || opp.severity === 'high'
  );

  const displayOpportunities = criticalOpportunities.length > 0
    ? criticalOpportunities
    : opportunities;

  if (displayOpportunities.length === 0) {
    return (
      <section className={styles.container}>
        <div className={styles.compactStatus}>
          <FiCheck className={styles.compactIconOk} size={14} />
          <span className={styles.compactTextOk}>Aucune alerte critique</span>
        </div>
      </section>
    );
  }

  return (
    <section className={styles.container}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionTitleGroup}>
          <FiAlertTriangle className={styles.sectionIcon} size={18} />
          <h2 className={styles.sectionTitle}>Intelligence Dispatch</h2>
          <span className={styles.alertCount}>{displayOpportunities.length}</span>
        </div>
        {companyPublicId && (
          <Link
            to={`/dashboard/company/${companyPublicId}/dispatch`}
            className={styles.viewAllLink}
          >
            Voir tout
            <FiArrowRight size={14} />
          </Link>
        )}
      </div>

      <div className={styles.opportunitiesList}>
        {displayOpportunities.slice(0, 5).map((opp) => {
          const config = SEVERITY_CONFIG[opp.severity] || SEVERITY_CONFIG.low;
          const action = getActionConfig(opp);

          return (
            <div
              key={opp.assignment_id || opp.booking_id}
              className={`${styles.opportunityCard} ${styles[config.cssClass] || ''}`}
            >
              <div className={styles.opportunityHeader}>
                <span className={`${styles.severityLabel} ${styles[`severity_${config.cssClass}`] || ''}`}>
                  {config.label}
                </span>
                <div className={styles.headerRight}>
                  {opp.current_delay_minutes !== undefined && (
                    <span className={styles.delayValue}>
                      {opp.current_delay_minutes > 0 ? '+' : ''}
                      {opp.current_delay_minutes} min
                    </span>
                  )}
                  {onAction && (
                    <button
                      className={`${styles.actionCTA} ${styles[`cta_${action.variant}`] || ''}`}
                      onClick={() => onAction(opp)}
                      type="button"
                    >
                      <action.Icon size={12} />
                      {action.label}
                    </button>
                  )}
                </div>
              </div>

              {opp.suggestions && opp.suggestions.length > 0 && (
                <div className={styles.suggestionsList}>
                  {opp.suggestions.slice(0, 2).map((suggestion, idx) => (
                    <div
                      key={idx}
                      className={`${styles.suggestionItem} ${styles[`border_${config.cssClass}`] || ''}`}
                    >
                      {suggestion.message || suggestion.action}
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {displayOpportunities.length > 5 && (
        <p className={styles.moreOpportunities}>
          +{displayOpportunities.length - 5} autre(s) opportunité(s) détectée(s)
        </p>
      )}
    </section>
  );
};

export default OpportunitiesSection;
