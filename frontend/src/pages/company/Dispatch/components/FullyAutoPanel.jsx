import React, { useState, useEffect } from 'react';
import DispatchTableSkeleton from '../../../../components/SkeletonLoaders/DispatchTableSkeleton';
import EmptyState from '../../../../components/EmptyState';
import ModeBanner from './ModeBanner';

/**
 * Composant pour le mode automatique de dispatch
 */
const FullyAutoPanel = ({
  dispatches = [],
  delays = [],
  optimizerStatus,
  loading,
  error,
  onStartOptimizer,
  onStopOptimizer,
  autoRefresh = true,
  setAutoRefresh,
  styles = {},
}) => {
  // État pour l'intervalle de vérification (en minutes)
  const [checkInterval, setCheckInterval] = useState(5);

  // État pour le journal d'activité
  const [activityLog, setActivityLog] = useState([
    {
      timestamp: new Date().toLocaleTimeString('fr-FR'),
      icon: '🚀',
      message: 'Système de dispatch automatique initialisé',
    },
  ]);
  const formatTime = (timeString) => {
    if (!timeString) return '—';
    const date = new Date(timeString);
    if (isNaN(date.getTime())) return '—';
    return date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
  };

  // Ajouter un événement au journal d'activité
  const addLogEntry = (icon, message) => {
    const newEntry = {
      timestamp: new Date().toLocaleTimeString('fr-FR'),
      icon,
      message,
    };
    setActivityLog((prev) => [newEntry, ...prev].slice(0, 50)); // Garder max 50 entrées
  };

  // Ecouter les changements de statut de l'optimiseur
  useEffect(() => {
    if (optimizerStatus?.running) {
      addLogEntry('🟢', 'Optimiseur demarre et actif');
    } else if (optimizerStatus?.running === false) {
      addLogEntry('🔴', 'Optimiseur arrete');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [optimizerStatus?.running]);

  // Ecouter les nouvelles alertes
  useEffect(() => {
    if (delays && delays.length > 0) {
      const criticalDelays = delays.filter((d) => d.severity === 'critical');
      if (criticalDelays.length > 0) {
        addLogEntry('🔴', `${criticalDelays.length} retard(s) critique(s) detecte(s)`);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [delays?.length]);

  // Handler pour demarrer l'optimiseur avec l'intervalle configure
  const handleStartOptimizer = () => {
    addLogEntry('▶', `Demarrage de l'optimiseur (intervalle: ${checkInterval} min)`);
    onStartOptimizer();
  };

  // Handler pour arreter l'optimiseur
  const handleStopOptimizer = () => {
    const icon = String.fromCodePoint(0x25a0);
    addLogEntry(icon, 'Arret de l optimiseur demande');
    onStopOptimizer();
  };

  // Handler pour effacer le journal
  const clearActivityLog = () => {
    setActivityLog([
      {
        timestamp: new Date().toLocaleTimeString('fr-FR'),
        icon: String.fromCodePoint(0x00d7),
        message: 'Journal d activite efface',
      },
    ]);
  };

  if (loading) {
    return <DispatchTableSkeleton rows={8} />;
  }

  if (error) {
    return <div className={styles.error}>Erreur: {error}</div>;
  }

  return (
    <>
      {/* Panel Header et Contrôles */}
      <div className={styles.fullyAutoPanel}>
        <div className={styles.panelHeader}>
          <h3>🤖 Mode Automatique - Surveillance en temps réel</h3>
          <p>Le dispatch fonctionne automatiquement. Surveillez les performances et les alertes.</p>
        </div>

        {/* Section Optimiseur - COMPLÈTE */}
        <div className={styles.optimizerSection}>
          <h4>🤖 Optimiseur en temps réel</h4>

          <div className={styles.optimizerStatus}>
            <span className={styles.statusIndicator}>
              {optimizerStatus?.running ? '🟢 Actif' : '🔴 Inactif'}
            </span>
            {optimizerStatus?.last_check && (
              <span className={styles.lastCheck}>
                Dernière vérification:{' '}
                {new Date(optimizerStatus.last_check).toLocaleTimeString('fr-FR')}
              </span>
            )}
          </div>

          <div className={styles.optimizerActions}>
            {optimizerStatus?.running ? (
              <button onClick={handleStopOptimizer} className={styles.stopButton}>
                ⏸️ Arrêter l'optimiseur
              </button>
            ) : (
              <button onClick={handleStartOptimizer} className={styles.startButton}>
                ▶️ Démarrer l'optimiseur
              </button>
            )}
          </div>

          {/* Réglage de l'intervalle */}
          <div className={styles.optimizerSettings}>
            <label htmlFor="checkInterval">
              Intervalle de vérification (minutes):
              <input
                type="number"
                id="checkInterval"
                min="1"
                max="60"
                value={checkInterval}
                onChange={(e) => setCheckInterval(Number(e.target.value))}
                className={styles.intervalInput}
              />
            </label>
            <span className={styles.settingsHelp}>
              L'optimiseur vérifiera les assignations toutes les {checkInterval} minute(s)
            </span>

            {/* Auto-refresh toggle */}
            {setAutoRefresh && (
              <label htmlFor="autoRefresh" className={styles.autoRefreshToggle}>
                <input
                  type="checkbox"
                  id="autoRefresh"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className={styles.checkbox}
                />
                <span>Actualisation automatique (30s)</span>
              </label>
            )}
          </div>
        </div>

        {/* Journal d'activité */}
        <div className={styles.activityLog}>
          <div className={styles.logHeader}>
            <h4>📋 Journal d'activité</h4>
            <button onClick={clearActivityLog} className={styles.clearLogButton}>
              🗑️ Effacer
            </button>
          </div>
          <div className={styles.logEntries}>
            {activityLog.map((entry, idx) => (
              <div key={idx} className={styles.logEntry}>
                <span className={styles.logTime}>{entry.timestamp}</span>
                <span className={styles.logIcon}>{entry.icon}</span>
                <span className={styles.logMessage}>{entry.message}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Section Alertes actives */}
        {delays.length > 0 ? (
          <div className={styles.alertsSection}>
            <h4>⚠️ Alertes actives ({delays.length})</h4>
            <div className={styles.alertsList}>
              {delays.map((delay, index) => (
                <div key={index} className={styles.alertItem}>
                  <div className={styles.alertHeader}>
                    <span className={styles.alertTime}>{formatTime(delay.scheduled_time)}</span>
                    <span
                      className={`${styles.alertSeverity} ${
                        styles[`severity${delay.severity || 'low'}`]
                      }`}
                    >
                      {delay.severity === 'critical'
                        ? '🔴 Critique'
                        : delay.severity === 'high'
                        ? '🟠 Élevé'
                        : delay.severity === 'medium'
                        ? '🟡 Moyen'
                        : '🟢 Faible'}
                    </span>
                  </div>
                  <div className={styles.alertContent}>
                    <strong>{delay.customer_name || 'Client inconnu'}</strong> - Retard:{' '}
                    {delay.delay_minutes || 0} min
                  </div>
                  {delay.suggestions && delay.suggestions.length > 0 && (
                    <div className={styles.autoSuggestions}>
                      {delay.suggestions.map((suggestion, sIndex) => (
                        <div key={sIndex} className={styles.autoSuggestion}>
                          {suggestion.auto_applicable ? '✅' : '⚠️'} {suggestion.message}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className={styles.noAlertsMessage}>
            <span className={styles.successIcon}>✅</span>
            <p>Aucune alerte active - Tout fonctionne normalement</p>
          </div>
        )}

        {/* Section Assignations récentes */}
        <div className={styles.recentAssignments}>
          <h4>✅ Assignations automatiques récentes</h4>
          {dispatches.length > 0 ? (
            <div className={styles.assignmentsList}>
              {dispatches.slice(0, 10).map((dispatch) => (
                <div key={dispatch.id} className={styles.assignmentCard}>
                  <span className={styles.assignmentTime}>
                    {formatTime(dispatch.scheduled_time)}
                  </span>
                  <span className={styles.assignmentDetails}>
                    <strong>{dispatch.customer_name || 'Client inconnu'}</strong>
                    {dispatch.driver_name && (
                      <span className={styles.driverName}> → Chauffeur {dispatch.driver_name}</span>
                    )}
                  </span>
                  <span className={styles.assignmentStatus}>{dispatch.status || 'assigned'}</span>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState
              icon="📭"
              title="Aucune assignation récente"
              message="Les assignations automatiques apparaîtront ici dès qu'elles seront effectuées."
            />
          )}
        </div>
      </div>

      {/* Bannière Mode Fully-Auto */}
      <ModeBanner
        icon="🤖"
        title="Mode Totalement Automatique Activé"
        description="Le système gère automatiquement toutes les assignations selon les règles configurées. Vous pouvez surveiller l'activité en temps réel."
        variant="fullyAuto"
        styles={styles}
      />
    </>
  );
};

export default FullyAutoPanel;
