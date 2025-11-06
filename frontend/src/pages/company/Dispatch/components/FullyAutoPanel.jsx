import React, { useState, useEffect, useMemo } from 'react';
import DispatchTableSkeleton from '../../../../components/SkeletonLoaders/DispatchTableSkeleton';
import EmptyState from '../../../../components/EmptyState';
import ModeBanner from './ModeBanner';
import {
  startAgent,
  getAgentStatus,
  startRealTimeOptimizer,
  getOptimizerStatus,
  resetAssignments,
} from '../../../../services/dispatchMonitoringService';

/**
 * Composant ULTRA PERFORMANT pour le mode automatique de dispatch
 * Interface de commande pour l'Agent Dispatch Intelligent
 */
const FullyAutoPanel = ({
  dispatches = [],
  delays = [],
  optimizerStatus,
  loading,
  error,
  onStartOptimizer,
  onStopOptimizer: _onStopOptimizer, // Non utilisé en mode fully_auto (arrêt désactivé)
  autoRefresh: _autoRefresh = true, // Non utilisé dans la version compacte
  setAutoRefresh: _setAutoRefresh, // Non utilisé dans la version compacte
  onDispatchNow, // Handler pour retour urgent +15min
  styles = {},
}) => {
  // États
  const [checkInterval] = useState(5); // Valeur fixe, pas de modification dans la version compacte
  const [agentStatus, setAgentStatus] = useState(null);
  const [_agentLoading] = useState(false); // Non utilisé dans la version compacte
  const [logFilter, setLogFilter] = useState('all'); // 'all' | 'assign' | 'reassign' | 'alert' | 'osrm' | 'tick'
  const [logSearch, setLogSearch] = useState('');
  const [expandedLogEntry, setExpandedLogEntry] = useState(null);
  const [resetting, setResetting] = useState(false);

  // Journal d'activité enrichi
  const [activityLog, setActivityLog] = useState([
    {
      timestamp: new Date().toISOString(),
      icon: '🚀',
      message: 'Système de dispatch automatique initialisé',
      type: 'system',
      details: null,
    },
  ]);

  const formatTime = (timeString) => {
    if (!timeString) return '⏱️ À définir';
    const date = new Date(timeString);
    if (isNaN(date.getTime())) return '⏱️ À définir';
    // Vérifier si c'est une heure à définir (00:00:00)
    const hours = date.getHours();
    const minutes = date.getMinutes();
    if (hours === 0 && minutes === 0) {
      return '⏱️ À définir';
    }
    return date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
  };

  // Vérifier si une course est un retour avec heure à définir
  const isReturnToSchedule = (dispatch) => {
    if (!dispatch.scheduled_time) return true;
    const date = new Date(dispatch.scheduled_time);
    if (isNaN(date.getTime())) return true;
    // Heure à définir = 00:00:00
    return date.getHours() === 0 && date.getMinutes() === 0;
  };

  // Ajouter un événement au journal avec détails
  const addLogEntry = (icon, message, type = 'info', details = null) => {
    const newEntry = {
      timestamp: new Date().toISOString(),
      icon,
      message,
      type,
      details, // Payload JSON, métriques, etc.
    };
    setActivityLog((prev) => [newEntry, ...prev].slice(0, 100)); // Augmenté à 100
  };

  // Filtrer et rechercher dans le journal
  const filteredLogs = useMemo(() => {
    let filtered = activityLog;

    // Filtre par type
    if (logFilter !== 'all') {
      filtered = filtered.filter((entry) => entry.type === logFilter);
    }

    // Recherche textuelle
    if (logSearch.trim()) {
      const searchLower = logSearch.toLowerCase();
      filtered = filtered.filter(
        (entry) =>
          entry.message.toLowerCase().includes(searchLower) ||
          (entry.details && JSON.stringify(entry.details).toLowerCase().includes(searchLower))
      );
    }

    return filtered;
  }, [activityLog, logFilter, logSearch]);

  // Calculer métriques depuis dispatches
  const metrics = useMemo(() => {
    const total = dispatches.length;
    const assigned = dispatches.filter((d) => d.driver_name || d.driver_id).length;
    const unassigned = total - assigned;
    const withDriverName = dispatches.filter((d) => d.driver_name).length;

    return {
      total,
      assigned,
      unassigned,
      assignmentRate: total > 0 ? Math.round((assigned / total) * 100) : 0,
      withDriverName,
    };
  }, [dispatches]);

  const sortedDispatches = useMemo(() => {
    const getCategory = (dispatch) => {
      const status = (dispatch.status || '').toLowerCase();
      if (status === 'completed') {
        return 2;
      }
      return isReturnToSchedule(dispatch) ? 1 : 0;
    };

    const getTimestamp = (dispatch) => {
      const timeString = dispatch.scheduled_time;
      if (!timeString) return null;
      const date = new Date(timeString);
      return isNaN(date.getTime()) ? null : date.getTime();
    };

    return dispatches
      .map((dispatch, index) => ({ dispatch, index }))
      .sort((a, b) => {
        const categoryA = getCategory(a.dispatch);
        const categoryB = getCategory(b.dispatch);
        if (categoryA !== categoryB) {
          return categoryA - categoryB;
        }

        if (categoryA === 0 || categoryA === 2) {
          const timeA = getTimestamp(a.dispatch);
          const timeB = getTimestamp(b.dispatch);
          if (timeA !== timeB) {
            if (timeA === null) return 1;
            if (timeB === null) return -1;
            return timeA - timeB;
          }
        }

        return a.index - b.index;
      })
      .map((item) => item.dispatch);
  }, [dispatches]);

  // ✅ DÉMARRAGE AUTOMATIQUE en mode fully_auto
  useEffect(() => {
    let hasStartedAgent = false;
    let hasStartedOptimizer = false;

    const autoStartServices = async () => {
      try {
        // 1. Démarrer l'Agent Dispatch si pas déjà actif
        if (!hasStartedAgent) {
          try {
            const agentStatus = await getAgentStatus();
            if (!agentStatus?.running) {
              console.log("[FullyAuto] 🤖 Démarrage automatique de l'Agent Dispatch...");
              await startAgent();
              hasStartedAgent = true;
              addLogEntry(
                '🤖',
                'Agent Dispatch démarré automatiquement (mode fully_auto)',
                'system'
              );
            } else {
              console.log('[FullyAuto] ✅ Agent Dispatch déjà actif');
              hasStartedAgent = true;
            }
          } catch (error) {
            console.error('[FullyAuto] Erreur démarrage agent:', error);
            addLogEntry(
              '❌',
              `Erreur démarrage agent: ${error.message || 'Erreur inconnue'}`,
              'alert'
            );
          }
        }

        // 2. Démarrer l'Optimiseur si pas déjà actif
        if (!hasStartedOptimizer) {
          try {
            // ✅ Charger le statut de l'optimiseur si pas encore disponible
            let currentOptimizerStatus = optimizerStatus;
            if (!currentOptimizerStatus) {
              console.log("[FullyAuto] 🔍 Chargement du statut de l'optimiseur...");
              currentOptimizerStatus = await getOptimizerStatus();
            }

            if (!currentOptimizerStatus || !currentOptimizerStatus.running) {
              console.log("[FullyAuto] 🔄 Démarrage automatique de l'Optimiseur...");
              // ✅ Toujours utiliser onStartOptimizer si disponible (il recharge le statut automatiquement)
              // Sinon, appeler directement le service et recharger le statut manuellement
              if (onStartOptimizer) {
                await onStartOptimizer();
                // onStartOptimizer devrait déjà recharger le statut via loadOptimizerStatus
              } else {
                await startRealTimeOptimizer(checkInterval * 60); // Convertir minutes en secondes
                // ✅ Recharger le statut après démarrage direct
                try {
                  const updatedStatus = await getOptimizerStatus();
                  if (updatedStatus) {
                    console.log('[FullyAuto] ✅ Optimiseur démarré, statut:', updatedStatus);
                    // Note: Le statut sera mis à jour via la prop optimizerStatus au prochain rendu
                  }
                } catch (statusError) {
                  console.warn('[FullyAuto] Impossible de recharger le statut:', statusError);
                }
              }
              hasStartedOptimizer = true;
              addLogEntry('🔄', 'Optimiseur démarré automatiquement (mode fully_auto)', 'system');
            } else if (currentOptimizerStatus.running) {
              console.log('[FullyAuto] ✅ Optimiseur déjà actif');
              hasStartedOptimizer = true;
            }
          } catch (error) {
            console.error('[FullyAuto] Erreur démarrage optimiseur:', error);
            addLogEntry(
              '❌',
              `Erreur démarrage optimiseur: ${error.message || 'Erreur inconnue'}`,
              'alert'
            );
          }
        }
      } catch (error) {
        console.error('[FullyAuto] Erreur démarrage automatique:', error);
      }
    };

    // Démarrer automatiquement au montage
    autoStartServices();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Une seule fois au montage - on ignore les dépendances pour éviter les redémarrages en boucle

  // Charger le statut de l'agent et surveiller les changements
  useEffect(() => {
    let previousActionsToday = 0;
    let previousActionsLastHour = 0;
    let previousOsrmState = null;

    const fetchAgentStatus = async () => {
      try {
        const status = await getAgentStatus();
        if (status) {
          // Détecter nouvelles actions
          if (status.actions_today > previousActionsToday) {
            const newActions = status.actions_today - previousActionsToday;
            addLogEntry(
              '✅',
              `Agent: ${newActions} nouvelle(s) assignation(s) aujourd'hui (total: ${status.actions_today})`,
              'assign',
              { actions_today: status.actions_today, new_actions: newActions }
            );
          }
          if (status.actions_last_hour > previousActionsLastHour) {
            const newActions = status.actions_last_hour - previousActionsLastHour;
            addLogEntry(
              '⚡',
              `Agent: ${newActions} assignation(s) dans la dernière heure`,
              'assign',
              { actions_last_hour: status.actions_last_hour }
            );
          }

          // Détecter changements OSRM
          const currentOsrmState = status.osrm_health?.state || null;
          if (previousOsrmState !== null && previousOsrmState !== currentOsrmState) {
            if (currentOsrmState === 'OPEN') {
              addLogEntry(
                '⚠️',
                `OSRM Circuit Breaker OPEN - Latence: ${
                  status.osrm_health?.latency_ms || 'N/A'
                }ms - Mode dégradé activé`,
                'osrm',
                status.osrm_health
              );
            } else if (currentOsrmState === 'CLOSED') {
              addLogEntry(
                '✅',
                'OSRM Circuit Breaker CLOSED - Fonctionnement normal',
                'osrm',
                status.osrm_health
              );
            }
          }

          previousActionsToday = status.actions_today;
          previousActionsLastHour = status.actions_last_hour;
          previousOsrmState = currentOsrmState;
          setAgentStatus(status);
        }
      } catch (error) {
        console.debug('[Agent] Status fetch error:', error);
      }
    };

    fetchAgentStatus();
    const interval = setInterval(fetchAgentStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Note: handleStartAgent et handleStopAgent supprimés - En mode fully_auto, les services démarrent automatiquement
  // et ne peuvent pas être arrêtés manuellement. L'arrêt se fait uniquement en changeant le mode de dispatch via l'API

  const clearActivityLog = () => {
    setActivityLog([
      {
        timestamp: new Date().toISOString(),
        icon: '🗑️',
        message: "Journal d'activité effacé",
        type: 'system',
        details: null,
      },
    ]);
  };

  const handleResetAssignments = async () => {
    if (
      !window.confirm(
        "⚠️ Êtes-vous sûr de vouloir réinitialiser toutes les assignations ?\n\nCette action va :\n- Supprimer toutes les assignations\n- Remettre les courses au statut ACCEPTED\n- Nettoyer les références aux chauffeurs\n\nL'agent pourra ensuite repartir de zéro."
      )
    ) {
      return;
    }

    setResetting(true);
    try {
      // Récupérer la date actuelle depuis les dispatches (première course)
      const currentDate =
        dispatches.length > 0 && dispatches[0].scheduled_time
          ? new Date(dispatches[0].scheduled_time).toISOString().split('T')[0]
          : null;

      const result = await resetAssignments(currentDate);
      addLogEntry(
        '🔄',
        `Réinitialisation effectuée: ${result.assignments_deleted} assignations supprimées, ${result.bookings_reset} courses réinitialisées`,
        'system',
        result
      );

      // Recharger la page après 1 seconde pour voir les changements
      setTimeout(() => {
        window.location.reload();
      }, 1000);
    } catch (error) {
      console.error('[FullyAuto] Erreur réinitialisation:', error);
      addLogEntry(
        '❌',
        `Erreur lors de la réinitialisation: ${error.message || 'Erreur inconnue'}`,
        'alert',
        { error: error.message }
      );
    } finally {
      setResetting(false);
    }
  };

  if (loading) {
    return <DispatchTableSkeleton rows={8} />;
  }

  if (error) {
    return <div className={styles.error}>Erreur: {error}</div>;
  }

  return (
    <>
      <div className={styles.fullyAutoPanel}>
        {/* ========== HEADER AVEC MÉTRIQUES GLOBALES ========== */}
        <div className={styles.panelHeader}>
          <div className={styles.headerTop}>
            <div>
              <h3>🤖 Mode Automatique - Centre de Commande</h3>
              <p>Surveillance en temps réel de l'Agent Dispatch Intelligent</p>
            </div>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <button
                onClick={handleResetAssignments}
                disabled={resetting}
                style={{
                  padding: '0.5rem 1rem',
                  backgroundColor: '#ff6b6b',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: resetting ? 'not-allowed' : 'pointer',
                  fontSize: '0.875rem',
                  fontWeight: '500',
                  opacity: resetting ? 0.6 : 1,
                }}
                title="Réinitialiser toutes les assignations et remettre les courses au statut ACCEPTED"
              >
                {resetting ? '⏳ Réinitialisation...' : '🔄 Réinitialiser les assignations'}
              </button>
              <div className={styles.globalMetrics}>
                <div className={styles.metricCard}>
                  <span className={styles.metricValue}>{metrics.total}</span>
                  <span className={styles.metricLabel}>Courses totales</span>
                </div>
                <div className={styles.metricCard}>
                  <span className={styles.metricValue}>{metrics.assigned}</span>
                  <span className={styles.metricLabel}>Assignées</span>
                </div>
                <div className={styles.metricCard}>
                  <span className={styles.metricValue}>{metrics.assignmentRate}%</span>
                  <span className={styles.metricLabel}>Taux d'assignation</span>
                </div>
                <div className={styles.metricCard}>
                  <span className={styles.metricValue}>
                    {agentStatus?.osrm_health?.state === 'CLOSED' ? '🟢' : '⚠️'}
                  </span>
                  <span className={styles.metricLabel}>OSRM</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ========== SECTIONS AGENT & OPTIMISEUR (2 colonnes compactes) ========== */}
        <div className={styles.servicesGrid}>
          {/* Colonne 1: Agent Dispatch */}
          <div className={styles.serviceCard}>
            <div className={styles.serviceHeader}>
              <h4>🤖 Agent Dispatch</h4>
              <div className={styles.serviceStatus}>
                <span
                  className={`${styles.statusDot} ${
                    agentStatus?.running ? styles.active : styles.inactive
                  }`}
                ></span>
                <span>{agentStatus?.running ? 'Actif' : 'Inactif'}</span>
              </div>
            </div>
            <div className={styles.serviceMetrics}>
              <div className={styles.metricCompact}>
                <span className={styles.metricValue}>{agentStatus?.actions_today || 0}</span>
                <span className={styles.metricLabel}>Actions aujourd'hui</span>
              </div>
              <div className={styles.metricCompact}>
                <span className={styles.metricValue}>
                  {agentStatus?.osrm_health?.state === 'CLOSED' ? '🟢' : '⚠️'}
                </span>
                <span className={styles.metricLabel}>OSRM</span>
              </div>
            </div>
            {agentStatus?.running && (
              <div className={styles.serviceStatusInfo}>
                <small>
                  Dernier tick: {agentStatus?.last_tick ? formatTime(agentStatus.last_tick) : '—'}
                </small>
              </div>
            )}
          </div>

          {/* Colonne 2: Optimiseur */}
          <div className={styles.serviceCard}>
            <div className={styles.serviceHeader}>
              <h4>🔄 Optimiseur</h4>
              <div className={styles.serviceStatus}>
                <span
                  className={`${styles.statusDot} ${
                    optimizerStatus?.running ? styles.active : styles.inactive
                  }`}
                ></span>
                <span>{optimizerStatus?.running ? 'Actif' : 'Inactif'}</span>
              </div>
            </div>
            <div className={styles.serviceMetrics}>
              <div className={styles.metricCompact}>
                <span className={styles.metricValue}>
                  {optimizerStatus?.opportunities_count || 0}
                </span>
                <span className={styles.metricLabel}>Opportunités</span>
              </div>
              <div className={styles.metricCompact}>
                <span className={styles.metricValue}>{checkInterval}</span>
                <span className={styles.metricLabel}>Intervalle (min)</span>
              </div>
            </div>
            {optimizerStatus?.running && (
              <div className={styles.serviceStatusInfo}>
                <small>
                  Dernière vérif:{' '}
                  {optimizerStatus?.last_check ? formatTime(optimizerStatus.last_check) : '—'}
                </small>
              </div>
            )}
          </div>
        </div>

        {/* ========== SECTION 3: TABLEAU DES ASSIGNATIONS ENRICHIES ========== */}
        <div className={styles.assignmentsTableSection}>
          <h4>📋 Assignations automatiques ({dispatches.length})</h4>
          {dispatches.length > 0 ? (
            <div className={styles.tableContainer}>
              <table className={styles.assignmentsTable}>
                <thead>
                  <tr>
                    <th>Heure</th>
                    <th>Client</th>
                    <th>Chauffeur</th>
                    <th>Pickup → Dropoff</th>
                    <th>Statut</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedDispatches.map((dispatch) => (
                    <tr key={dispatch.id} className={styles.tableRow}>
                      <td className={styles.timeCell}>
                        {formatTime(dispatch.scheduled_time) || '⏱️ À définir'}
                      </td>
                      <td className={styles.clientCell}>
                        <strong>{dispatch.customer_name || 'Client inconnu'}</strong>
                        {dispatch.booking_id && (
                          <span className={styles.bookingId}>#{dispatch.booking_id}</span>
                        )}
                      </td>
                      <td className={styles.driverCell}>
                        {(() => {
                          // Extraire le nom du chauffeur depuis différentes sources possibles
                          // 1. Depuis dispatch.driver (objet driver direct)
                          // 2. Depuis dispatch.assignment.driver (driver dans l'assignment)
                          // 3. Depuis dispatch.driver_name (ancien format)
                          const driverObj = dispatch.driver || dispatch.assignment?.driver || null;
                          const driverName =
                            dispatch.driver_name ||
                            driverObj?.full_name ||
                            driverObj?.name ||
                            (driverObj?.user?.first_name && driverObj?.user?.last_name
                              ? `${driverObj.user.first_name} ${driverObj.user.last_name}`.trim()
                              : driverObj?.user?.full_name) ||
                            driverObj?.user?.username ||
                            null;
                          const driverId =
                            dispatch.driver_id ||
                            dispatch.assignment?.driver_id ||
                            driverObj?.id ||
                            null;

                          // Si le statut est "assigned" mais qu'on n'a pas de nom, afficher au moins l'ID
                          if (dispatch.status === 'assigned' && driverId && !driverName) {
                            return (
                              <span className={styles.driverName}>👤 Chauffeur #{driverId}</span>
                            );
                          }

                          if (driverName || driverId) {
                            return (
                              <span className={styles.driverName}>
                                👤 {driverName || `Chauffeur #${driverId}`}
                              </span>
                            );
                          }
                          return <span className={styles.unassigned}>❌ Non assigné</span>;
                        })()}
                      </td>
                      <td className={styles.locationCell}>
                        <div className={styles.pickup}>
                          <span className={styles.locationIcon}>📍</span>
                          <span className={styles.locationText}>
                            {dispatch.pickup_address || dispatch.pickup_location || 'N/A'}
                          </span>
                        </div>
                        <div className={styles.arrow}>→</div>
                        <div className={styles.dropoff}>
                          <span className={styles.locationIcon}>🎯</span>
                          <span className={styles.locationText}>
                            {dispatch.dropoff_address || dispatch.dropoff_location || 'N/A'}
                          </span>
                        </div>
                      </td>
                      <td className={styles.statusCell}>
                        <span
                          className={`${styles.statusBadge} ${styles[`status${dispatch.status}`]}`}
                        >
                          {dispatch.status || 'accepted'}
                        </span>
                      </td>
                      <td className={styles.actionsCell}>
                        <div className={styles.actionButtons}>
                          {/* Bouton "Retour urgent +15min" pour les retours avec heure à définir */}
                          {isReturnToSchedule(dispatch) && onDispatchNow && (
                            <button
                              className={styles.urgentButton}
                              onClick={() => onDispatchNow(dispatch.id)}
                              title="Retour urgent (+15 min)"
                            >
                              ⚡ Urgent +15min
                            </button>
                          )}
                          <button
                            className={styles.viewDetailsButton}
                            onClick={() => console.log('View details', dispatch.id)}
                            title="Voir les détails"
                          >
                            👁️
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              icon="📭"
              title="Aucune assignation"
              message="Les assignations automatiques apparaîtront ici dès qu'elles seront effectuées par l'agent."
            />
          )}
        </div>

        {/* ========== SECTION 4: JOURNAL D'ACTIVITÉ AMÉLIORÉ ========== */}
        <div className={styles.activityLog}>
          <div className={styles.logHeader}>
            <h4>📋 Journal d'activité ({filteredLogs.length})</h4>
            <div className={styles.logControls}>
              <select
                value={logFilter}
                onChange={(e) => setLogFilter(e.target.value)}
                className={styles.logFilter}
              >
                <option value="all">Tous les types</option>
                <option value="assign">Assignations</option>
                <option value="reassign">Réassignations</option>
                <option value="alert">Alertes</option>
                <option value="osrm">OSRM</option>
                <option value="system">Système</option>
              </select>
              <input
                type="text"
                placeholder="Rechercher..."
                value={logSearch}
                onChange={(e) => setLogSearch(e.target.value)}
                className={styles.logSearch}
              />
              <button onClick={clearActivityLog} className={styles.clearLogButton}>
                🗑️ Effacer
              </button>
            </div>
          </div>
          <div className={styles.logEntries}>
            {filteredLogs.length > 0 ? (
              filteredLogs.map((entry, idx) => (
                <div
                  key={idx}
                  className={`${styles.logEntry} ${styles[`logType${entry.type}`]}`}
                  onClick={() => setExpandedLogEntry(expandedLogEntry === idx ? null : idx)}
                >
                  <span className={styles.logTime}>
                    {new Date(entry.timestamp).toLocaleTimeString('fr-FR')}
                  </span>
                  <span className={styles.logIcon}>{entry.icon}</span>
                  <span className={styles.logMessage}>{entry.message}</span>
                  {entry.details && (
                    <span className={styles.logDetailsToggle}>
                      {expandedLogEntry === idx ? '▼' : '▶'}
                    </span>
                  )}
                  {expandedLogEntry === idx && entry.details && (
                    <div className={styles.logDetails}>
                      <pre>{JSON.stringify(entry.details, null, 2)}</pre>
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className={styles.noLogs}>Aucune entrée correspondant aux filtres</div>
            )}
          </div>
        </div>

        {/* ========== SECTION 5: ALERTES ========== */}
        {delays.length > 0 && (
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
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <ModeBanner
        icon="🤖"
        title="Mode Totalement Automatique Activé"
        description="L'Agent Dispatch Intelligent gère automatiquement toutes les assignations. Surveillez l'activité en temps réel ci-dessus."
        variant="fullyAuto"
        styles={styles}
      />
    </>
  );
};

export default FullyAutoPanel;
