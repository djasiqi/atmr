// src/pages/company/Dashboard/components/DispatchModeStatusBar.jsx
// Indicateur compact du mode de dispatch — opérationnel, pas marketing
import React, { useMemo } from 'react';
import { FiSettings, FiCpu, FiZap, FiAlertTriangle, FiCheckCircle } from 'react-icons/fi';
import styles from './DispatchModeStatusBar.module.css';

const MODE_CONFIG = {
  manual: {
    label: 'Manuel',
    Icon: FiSettings,
    accent: 'neutral',
    description: 'Assignations manuelles',
  },
  semi_auto: {
    label: 'Semi-Automatique',
    Icon: FiCpu,
    accent: 'brand',
    description: 'Suggestions IA actives',
  },
  fully_auto: {
    label: 'Automatique',
    Icon: FiZap,
    accent: 'auto',
    description: 'IA pilote les assignations',
  },
  autonomous: {
    label: 'Automatique',
    Icon: FiZap,
    accent: 'auto',
    description: 'IA pilote les assignations',
  },
};

const DispatchModeStatusBar = ({ mode, opportunities = [] }) => {
  const isManual = mode === 'manual';

  const aiSummary = useMemo(() => {
    if (!mode || isManual || !opportunities?.length) return null;

    const critical = opportunities.filter(
      (o) => o.severity === 'critical' || o.severity === 'high'
    ).length;
    const total = opportunities.length;

    return { total, critical };
  }, [mode, opportunities, isManual]);

  if (!mode) return null;

  const config = MODE_CONFIG[mode] || MODE_CONFIG.manual;

  return (
    <div className={`${styles.bar} ${styles[`bar_${config.accent}`] || ''}`}>
      <div className={styles.modeInfo}>
        <config.Icon size={14} className={styles.modeIcon} />
        <span className={styles.modeLabel}>{config.label}</span>
        <span className={styles.modeSep}>—</span>
        <span className={styles.modeDesc}>{config.description}</span>
      </div>

      {aiSummary && (
        <div className={styles.aiStatus}>
          {aiSummary.critical > 0 ? (
            <span className={styles.aiAlert}>
              <FiAlertTriangle size={12} />
              {aiSummary.critical} action{aiSummary.critical > 1 ? 's' : ''} requise{aiSummary.critical > 1 ? 's' : ''}
            </span>
          ) : aiSummary.total > 0 ? (
            <span className={styles.aiSuggestions}>
              <FiCpu size={12} />
              {aiSummary.total} suggestion{aiSummary.total > 1 ? 's' : ''}
            </span>
          ) : (
            <span className={styles.aiOk}>
              <FiCheckCircle size={12} />
              Aucune intervention
            </span>
          )}
        </div>
      )}

      {isManual && (
        <span className={styles.manualHint}>Suggestions désactivées</span>
      )}
    </div>
  );
};

export default DispatchModeStatusBar;
