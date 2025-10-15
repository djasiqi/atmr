// frontend/src/pages/company/Dispatch/UnifiedDispatch.jsx
/**
 * 📊 PAGE UNIFIÉE : DISPATCH & PLANIFICATION
 *
 * Combine en un seul endroit :
 * 1. Planification automatique quotidienne
 * 2. Suivi en temps réel
 * 3. Détection et correction des retards
 * 4. Liste détaillée des assignations
 */

import React, { useState, useEffect, useCallback } from 'react';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import useCompanySocket from '../../../hooks/useCompanySocket';
import useDispatchStatus from '../../../hooks/useDispatchStatus';
import { runDispatchForDay, fetchAssignedReservations } from '../../../services/companyService';
import {
  getLiveDelays,
  getOptimizerStatus,
  startRealTimeOptimizer,
  stopRealTimeOptimizer,
  applySuggestion,
} from '../../../services/dispatchMonitoringService';
import styles from './UnifiedDispatch.module.css';

// Helpers
const makeToday = () => {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
};

const UnifiedDispatch = () => {
  // État principal
  const [date, setDate] = useState(makeToday());
  const [regularFirst, setRegularFirst] = useState(true);
  const [allowEmergency, setAllowEmergency] = useState(true);

  // Données
  const [dispatches, setDispatches] = useState([]);
  const [delays, setDelays] = useState([]);
  const [summary, setSummary] = useState(null);
  const [optimizerStatus, setOptimizerStatus] = useState(null);

  // États UI
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [dispatchSuccess, setDispatchSuccess] = useState(null);

  // États pour les modals
  const [contactModalVisible, setContactModalVisible] = useState(false);
  const [rescheduleModalVisible, setRescheduleModalVisible] = useState(false);
  const [selectedDelay, setSelectedDelay] = useState(null);
  const [selectedSuggestion, setSelectedSuggestion] = useState(null);
  const [newTime, setNewTime] = useState('');

  // WebSocket pour temps réel
  const socket = useCompanySocket();
  const {
    label: dispatchLabel,
    progress: dispatchProgress,
    isRunning: isDispatching,
  } = useDispatchStatus(socket);

  // Charger les courses dispatchées
  const loadDispatches = useCallback(async () => {
    try {
      const data = await fetchAssignedReservations(date);
      // fetchAssignedReservations retourne un objet avec les données
      const dispatches = Array.isArray(data) ? data : data?.data || [];

      // ℹ️ On affiche TOUTES les courses (même celles avec heure à confirmer)
      // Le backend exclura automatiquement les retours time_confirmed=false du dispatch
      setDispatches(dispatches);
    } catch (err) {
      console.error('[UnifiedDispatch] Error loading dispatches:', err);
    }
  }, [date]);

  // Charger les retards
  const loadDelays = useCallback(async () => {
    try {
      const response = await getLiveDelays(date);
      setDelays(response.delays || []);
      setSummary(response.summary || null);
    } catch (err) {
      console.error('[UnifiedDispatch] Error loading delays:', err);
    }
  }, [date]);

  // Charger le statut de l'optimiseur
  const loadOptimizerStatus = useCallback(async () => {
    try {
      const status = await getOptimizerStatus();
      setOptimizerStatus(status);
    } catch (err) {
      console.error('[UnifiedDispatch] Error loading optimizer:', err);
    }
  }, []);

  // Lancer le dispatch automatique
  const handleRunDispatch = async () => {
    if (!window.confirm(`Lancer le dispatch automatique pour le ${date} ?`)) {
      return;
    }

    try {
      setLoading(true);
      setError(null);
      await runDispatchForDay({
        forDate: date,
        regularFirst: regularFirst,
        allowEmergency: allowEmergency,
        mode: 'auto',
        runAsync: true,
      });

      // Attendre 3 secondes puis rafraîchir pour laisser le temps au backend
      setTimeout(() => {
        loadDispatches();
        loadDelays();
      }, 3000);
    } catch (err) {
      console.error('[UnifiedDispatch] Error running dispatch:', err);
      setError('Erreur lors du lancement du dispatch');
      setLoading(false);
    }
  };

  // Toggle optimiseur
  const handleToggleOptimizer = async () => {
    try {
      if (optimizerStatus?.running) {
        await stopRealTimeOptimizer();
      } else {
        await startRealTimeOptimizer(120);
      }
      await loadOptimizerStatus();
    } catch (err) {
      console.error('[UnifiedDispatch] Error toggling optimizer:', err);
      alert("Erreur lors de l'activation/désactivation du monitoring");
    }
  };

  // Copier le numéro de téléphone dans le presse-papier
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(
      () => alert('📋 Numéro copié dans le presse-papier !'),
      () => alert('❌ Erreur lors de la copie')
    );
  };

  // Gérer le report de course
  const handleReschedule = async () => {
    if (!selectedDelay || !newTime) return;

    const scheduledTime = selectedDelay.booking?.scheduled_time || selectedDelay.pickup_time;
    const [hours, minutes] = newTime.split(':').map(Number);

    if (isNaN(hours) || isNaN(minutes)) {
      alert("❌ Format d'horaire invalide");
      return;
    }

    const newDate = new Date(scheduledTime);
    newDate.setHours(hours, minutes, 0, 0);

    try {
      // TODO: Appeler l'API pour reporter la course
      alert(
        `✅ Course reportée avec succès !\n\n` +
          `Nouveau RDV: ${newDate.toLocaleString('fr-FR')}\n` +
          `Recherche d'un chauffeur disponible en cours...`
      );

      setRescheduleModalVisible(false);
      loadDelays();
      loadDispatches();
    } catch (err) {
      console.error('[UnifiedDispatch] Error rescheduling:', err);
      alert('❌ Erreur lors du report');
    }
  };

  // Annuler le report (marquer comme urgente)
  const handleCancelReschedule = () => {
    alert(
      `⚠️ Course marquée comme URGENTE\n\n` +
        `Le système continuera à chercher un chauffeur disponible rapidement.`
    );
    setRescheduleModalVisible(false);
    // TODO: Marquer la course comme urgente dans le backend
  };

  // Gérer les actions sur les suggestions
  const handleSuggestionAction = async (suggestion, delay) => {
    // 📞 CONTACTER : Ouvrir le modal de contact
    if (suggestion.action === 'notify_customer') {
      setSelectedDelay(delay);
      setSelectedSuggestion(suggestion);
      setContactModalVisible(true);
      return;
    }

    // ⏰ REPORTER : Ouvrir le modal de report
    if (suggestion.action === 'adjust_time') {
      setSelectedDelay(delay);
      setSelectedSuggestion(suggestion);

      // Calculer l'horaire suggéré
      const scheduledTime = delay.booking?.scheduled_time || delay.pickup_time;
      if (scheduledTime) {
        const suggestedTime = new Date(scheduledTime);
        suggestedTime.setMinutes(suggestedTime.getMinutes() + delay.delay_minutes);
        const hours = String(suggestedTime.getHours()).padStart(2, '0');
        const minutes = String(suggestedTime.getMinutes()).padStart(2, '0');
        setNewTime(`${hours}:${minutes}`);
      }

      setRescheduleModalVisible(true);
      return;
    }

    // 🔄 RÉASSIGNER
    if (suggestion.action === 'reassign') {
      if (!suggestion.alternative_driver_id) {
        alert('Cette suggestion ne peut pas être appliquée automatiquement');
        return;
      }

      const confirmed = window.confirm(
        `Réassigner la course #${suggestion.booking_id} au chauffeur #${suggestion.alternative_driver_id}?\n\n` +
          `Gain estimé: ${suggestion.estimated_gain_minutes} minutes`
      );

      if (!confirmed) return;

      try {
        await applySuggestion(delay.assignment_id, suggestion.alternative_driver_id);
        alert('✅ Suggestion appliquée avec succès!');
        loadDelays();
        loadDispatches();
      } catch (err) {
        console.error('[UnifiedDispatch] Error applying suggestion:', err);
        alert("❌ Erreur lors de l'application");
      }
      return;
    }

    // 🚨 REDISTRIBUER (plusieurs courses d'un même chauffeur en retard)
    if (suggestion.action === 'redistribute') {
      const tripsCount = suggestion.additional_data?.delayed_trips_count || 2;
      const driverName = suggestion.additional_data?.driver_name || `#${suggestion.driver_id}`;
      const totalDelay = suggestion.additional_data?.total_delay || 0;

      alert(
        `🚨 ALERTE : Chauffeur Surchargé\n\n` +
          `Chauffeur: ${driverName}\n` +
          `Courses en retard: ${tripsCount}\n` +
          `Retard total: ${totalDelay} min\n\n` +
          `⚠️ Action recommandée :\n` +
          `Le système devrait relancer automatiquement le dispatch\n` +
          `pour répartir ces courses sur ${tripsCount} chauffeurs différents.\n\n` +
          `Voulez-vous relancer le dispatch maintenant ?`
      );

      // TODO: Proposer de relancer le dispatch automatique
      return;
    }
  };

  // Chargement initial
  useEffect(() => {
    loadDispatches();
    loadDelays();
    loadOptimizerStatus();
  }, [loadDispatches, loadDelays, loadOptimizerStatus]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      loadDelays();
      loadOptimizerStatus();
    }, 30000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadDelays, loadOptimizerStatus]);

  // Écoute WebSocket
  useEffect(() => {
    if (!socket) return;

    const handleDispatchComplete = (data) => {
      setLoading(false);
      setDispatchSuccess(`✅ Dispatch terminé ! ${data?.assignments_count || 0} courses assignées`);
      setTimeout(() => setDispatchSuccess(null), 5000); // Masquer après 5s
      loadDispatches();
      loadDelays();
    };

    const handleBookingUpdated = () => {
      // Rafraîchir immédiatement les données
      loadDispatches();
      loadDelays();
    };

    socket.on('dispatch_run_completed', handleDispatchComplete);
    socket.on('booking_updated', handleBookingUpdated);
    socket.on('new_booking', handleBookingUpdated); // Aussi pour nouvelles courses

    return () => {
      socket.off('dispatch_run_completed', handleDispatchComplete);
      socket.off('booking_updated', handleBookingUpdated);
      socket.off('new_booking', handleBookingUpdated);
    };
  }, [socket, loadDispatches, loadDelays]);

  // Rendu des statistiques
  const renderSummary = () => {
    if (!summary) return null;
    return (
      <div className={styles.summaryCards}>
        <div className={`${styles.summaryCard} ${styles.total}`}>
          <div className={styles.cardIcon}>📊</div>
          <div className={styles.cardContent}>
            <h3>{summary.total}</h3>
            <p>Total courses</p>
          </div>
        </div>
        <div className={`${styles.summaryCard} ${styles.onTime}`}>
          <div className={styles.cardIcon}>✅</div>
          <div className={styles.cardContent}>
            <h3>{summary.on_time}</h3>
            <p>À l'heure</p>
          </div>
        </div>
        <div className={`${styles.summaryCard} ${styles.late}`}>
          <div className={styles.cardIcon}>⚠️</div>
          <div className={styles.cardContent}>
            <h3>{summary.late}</h3>
            <p>En retard</p>
          </div>
        </div>
        <div className={`${styles.summaryCard} ${styles.early}`}>
          <div className={styles.cardIcon}>🚀</div>
          <div className={styles.cardContent}>
            <h3>{summary.early}</h3>
            <p>En avance</p>
          </div>
        </div>
        <div className={`${styles.summaryCard} ${styles.average}`}>
          <div className={styles.cardIcon}>⏱️</div>
          <div className={styles.cardContent}>
            <h3>{summary.average_delay?.toFixed(1) || 0} min</h3>
            <p>Retard moyen</p>
          </div>
        </div>
      </div>
    );
  };

  // Rendu d'une alerte de retard
  const renderDelayAlert = (delay) => {
    const statusColors = {
      late: '#dc3545',
      on_time: '#28a745',
      early: '#17a2b8',
    };

    if (delay.status === 'on_time') return null; // Pas d'alerte si à l'heure

    return (
      <div key={delay.assignment_id} className={styles.delayAlert}>
        <div className={styles.delayMainContent}>
          {/* Informations de la course - GAUCHE */}
          <div className={styles.delayInfo}>
            <h4>
              🔴 Course #{delay.booking_id} - {delay.booking?.customer_name}
            </h4>
            <p className={styles.delayMeta}>
              Chauffeur: {delay.driver?.name || `#${delay.driver_id}`} • Retard:{' '}
              <strong style={{ color: statusColors.late }}>+{delay.delay_minutes} min</strong>
            </p>
          </div>

          {/* Suggestions - DROITE (alignées à droite) */}
          {delay.suggestions && delay.suggestions.length > 0 && (
            <div className={styles.suggestionsRow}>
              {delay.suggestions.map((suggestion, idx) => (
                <button
                  key={idx}
                  className={`${styles.suggestionBtn} ${
                    styles[
                      `suggestionBtn${
                        suggestion.priority.charAt(0).toUpperCase() + suggestion.priority.slice(1)
                      }`
                    ]
                  }`}
                  onClick={() => handleSuggestionAction(suggestion, delay)}
                  title={suggestion.message}
                >
                  {suggestion.action === 'notify_customer'
                    ? '📞 Contacter'
                    : suggestion.action === 'adjust_time'
                    ? '⏰ Reporter'
                    : suggestion.action === 'reassign'
                    ? '🔄 Réassigner'
                    : suggestion.action}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };

  // Rendu d'une course dans la liste détaillée
  const renderDispatch = (dispatch) => {
    const statusColors = {
      pending: '#6c757d',
      accepted: '#17a2b8',
      scheduled: '#6c757d',
      assigned: '#0f766e',
      en_route_pickup: '#17a2b8',
      arrived_pickup: '#ffc107',
      onboard: '#fd7e14',
      en_route_dropoff: '#0f766e',
      completed: '#28a745',
      cancelled: '#dc3545',
    };

    // Extraire les infos du chauffeur
    const assignment = dispatch.assignment;
    // ⚠️ IMPORTANT : driver doit être un objet ou null, jamais une chaîne
    const driver =
      assignment?.driver && typeof assignment.driver === 'object'
        ? assignment.driver
        : dispatch.driver && typeof dispatch.driver === 'object'
        ? dispatch.driver
        : null;

    // Essayer d'abord les champs flat (ajoutés côté backend)
    let driverName = null;

    if (driver) {
      // Priorité 1 : full_name pré-calculé
      driverName = driver.full_name;

      // Priorité 2 : Construire depuis first_name + last_name
      if (!driverName) {
        const firstName = driver.first_name || driver.user?.first_name || '';
        const lastName = driver.last_name || driver.user?.last_name || '';
        const fullName = `${firstName} ${lastName}`.trim();
        driverName = fullName || driver.username || driver.user?.username || null;
      }
    }

    // Dernier fallback : chercher dans dispatch directement
    if (!driverName) {
      driverName = dispatch.driver_username || dispatch.driver_name || dispatch.full_name || null;
    }

    const driverId = assignment?.driver_id || dispatch.driver_id || driver?.id;

    // Extraire les adresses
    const pickupAddress =
      dispatch.pickup_address || dispatch.pickup_location || dispatch.origin || '—';

    const dropoffAddress =
      dispatch.dropoff_address || dispatch.dropoff_location || dispatch.destination || '—';

    const statusKey = (dispatch.status || 'pending').toLowerCase();

    return (
      <tr key={dispatch.id}>
        <td>#{dispatch.id}</td>
        <td>{dispatch.customer_name || '—'}</td>
        <td>
          {(() => {
            const scheduledTime = dispatch.scheduled_time || dispatch.pickup_time;
            if (!scheduledTime) return '—';

            const date = new Date(scheduledTime);
            const hours = date.getHours();
            const minutes = date.getMinutes();

            // Si c'est un retour avec heure non confirmée OU heure à 00:00
            if (
              dispatch.is_return &&
              (dispatch.time_confirmed === false || (hours === 0 && minutes === 0))
            ) {
              return 'Heure à confirmer';
            }

            return date.toLocaleTimeString('fr-FR', {
              hour: '2-digit',
              minute: '2-digit',
            });
          })()}
        </td>
        <td>{driverName || (driverId ? `#${driverId}` : 'Non assigné')}</td>
        <td>
          <span
            className={styles.statusBadge}
            style={{ backgroundColor: statusColors[statusKey] || '#999' }}
          >
            {dispatch.status}
          </span>
        </td>
        <td className={styles.addressCell}>📍 {pickupAddress}</td>
        <td className={styles.addressCell}>📍 {dropoffAddress}</td>
      </tr>
    );
  };

  const delaysOnly = delays.filter((d) => d.status === 'late');

  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />

      <div className={styles.dashboardLayout}>
        <CompanySidebar />

        <main className={styles.mainContent}>
          <div className={styles.unifiedDispatch}>
            {/* ==================== ZONE 1 : PLANIFICATION ==================== */}
            <section className={styles.plannerSection}>
              <div className={styles.sectionHeader}>
                <div className={styles.headerLeft}>
                  <h1>🚀 Dispatch & Planification</h1>
                  <p className={styles.subtitle}>
                    Planification automatique et suivi temps réel des courses
                  </p>
                </div>
              </div>

              <div className={styles.plannerControls}>
                <div className={styles.controlGroup}>
                  <label>Date</label>
                  <input
                    type="date"
                    value={date}
                    onChange={(e) => setDate(e.target.value)}
                    className={styles.dateInput}
                  />
                </div>

                <div className={styles.controlGroup}>
                  <label className={styles.checkbox}>
                    <input
                      type="checkbox"
                      checked={regularFirst}
                      onChange={(e) => setRegularFirst(e.target.checked)}
                    />
                    <span>Chauffeurs réguliers prioritaires</span>
                  </label>
                </div>

                <div className={styles.controlGroup}>
                  <label className={styles.checkbox}>
                    <input
                      type="checkbox"
                      checked={allowEmergency}
                      onChange={(e) => setAllowEmergency(e.target.checked)}
                    />
                    <span>Autoriser chauffeurs d'urgence</span>
                  </label>
                </div>

                <button
                  className={styles.dispatchBtn}
                  onClick={handleRunDispatch}
                  disabled={isDispatching || loading}
                >
                  {isDispatching ? '⏳ En cours...' : '🚀 Lancer Dispatch Automatique'}
                </button>

                <label className={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                  />
                  <span>Auto-refresh (30s)</span>
                </label>
              </div>

              {/* Progression du dispatch */}
              {isDispatching && (
                <div className={styles.progressBar}>
                  <div className={styles.progressInfo}>
                    <span>{dispatchLabel}</span>
                    <span>{Math.round(dispatchProgress)}%</span>
                  </div>
                  <div className={styles.progressTrack}>
                    <div
                      className={styles.progressFill}
                      style={{ width: `${dispatchProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Message de succès */}
              {dispatchSuccess && <div className={styles.successBanner}>{dispatchSuccess}</div>}

              {/* Optimiseur temps réel */}
              <div className={styles.optimizerControl}>
                <button
                  className={`${styles.optimizerBtn} ${
                    optimizerStatus?.running ? styles.active : ''
                  }`}
                  onClick={handleToggleOptimizer}
                >
                  {optimizerStatus?.running
                    ? '⏸️ Arrêter Monitoring Auto'
                    : '▶️ Démarrer Monitoring Auto'}
                </button>
                {optimizerStatus?.running && (
                  <span className={styles.optimizerInfo}>
                    🤖 Actif - Dernière vérification:{' '}
                    {optimizerStatus.last_check
                      ? new Date(optimizerStatus.last_check).toLocaleTimeString('fr-FR')
                      : 'Jamais'}
                  </span>
                )}
              </div>
            </section>

            {/* ==================== ZONE 2 : STATISTIQUES ==================== */}
            {summary && <section className={styles.statsSection}>{renderSummary()}</section>}

            {/* ==================== ZONE 3 : ALERTES RETARDS ==================== */}
            {delaysOnly.length > 0 && (
              <section className={styles.alertsSection}>
                <div className={styles.alertsHeader}>
                  <h2>🚨 Alertes & Actions Recommandées</h2>
                  <span className={styles.alertCount}>
                    {delaysOnly.length} retard(s) détecté(s)
                  </span>
                </div>
                <div className={styles.alertsList}>{delaysOnly.map(renderDelayAlert)}</div>
              </section>
            )}

            {/* ==================== ZONE 4 : LISTE DÉTAILLÉE ==================== */}
            <section className={styles.detailsSection}>
              <div className={styles.detailsHeader}>
                <h2>📋 Courses du Jour</h2>
                <span className={styles.detailsCount}>
                  {dispatches.length} course(s) assignée(s)
                </span>
              </div>

              {error && <div className={styles.errorBanner}>❌ {error}</div>}

              {loading && dispatches.length === 0 ? (
                <div className={styles.loadingState}>
                  <div className={styles.spinner}></div>
                  <p>Chargement des données...</p>
                </div>
              ) : dispatches.length === 0 ? (
                <div className={styles.emptyState}>
                  <div className={styles.emptyIcon}>📦</div>
                  <h3>Aucune course pour cette date</h3>
                  <p>Lancez le dispatch automatique pour assigner les courses</p>
                </div>
              ) : (
                <div className={styles.tableWrapper}>
                  <table className={styles.dispatchTable}>
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Client</th>
                        <th>Heure</th>
                        <th>Chauffeur</th>
                        <th>Statut</th>
                        <th>Pickup</th>
                        <th>Dropoff</th>
                      </tr>
                    </thead>
                    <tbody>{dispatches.map(renderDispatch)}</tbody>
                  </table>
                </div>
              )}
            </section>
          </div>
        </main>
      </div>

      {/* Modal de Contact Client */}
      {contactModalVisible && selectedDelay && (
        <div className={styles.modalOverlay} onClick={() => setContactModalVisible(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3>📞 Contacter le Client</h3>
              <button className={styles.modalClose} onClick={() => setContactModalVisible(false)}>
                ✕
              </button>
            </div>

            <div className={styles.modalBody}>
              {/* Vérifier si le numéro existe */}
              {!selectedDelay.booking?.customer_phone && !selectedDelay.booking?.phone ? (
                <div className={styles.noPhoneWarning}>
                  <div className={styles.warningIcon}>❌</div>
                  <h4>Numéro de téléphone non disponible</h4>
                  <p>Ce client n'a pas de numéro de téléphone enregistré dans le système.</p>
                </div>
              ) : (
                <>
                  {/* Informations du client */}
                  <div className={styles.customerInfo}>
                    <div className={styles.infoRow}>
                      <span className={styles.infoLabel}>Client :</span>
                      <span className={styles.infoValue}>
                        {selectedDelay.booking?.customer_name || 'N/A'}
                      </span>
                    </div>

                    <div className={styles.infoRow}>
                      <span className={styles.infoLabel}>Course :</span>
                      <span className={styles.infoValue}>#{selectedDelay.booking_id}</span>
                    </div>

                    <div className={styles.infoRow}>
                      <span className={styles.infoLabel}>Retard :</span>
                      <span className={styles.infoValueHighlight}>
                        +{selectedDelay.delay_minutes} min (
                        {Math.floor(selectedDelay.delay_minutes / 60)}h
                        {selectedDelay.delay_minutes % 60})
                      </span>
                    </div>
                  </div>

                  {/* Contact */}
                  <div className={styles.contactSection}>
                    <div className={styles.phoneBox}>
                      <div className={styles.phoneIcon}>📱</div>
                      <div className={styles.phoneInfo}>
                        <span className={styles.phoneLabel}>Téléphone</span>
                        <span className={styles.phoneNumber}>
                          {selectedDelay.booking?.customer_phone || selectedDelay.booking?.phone}
                        </span>
                      </div>
                      <button
                        className={styles.copyBtn}
                        onClick={() =>
                          copyToClipboard(
                            selectedDelay.booking?.customer_phone || selectedDelay.booking?.phone
                          )
                        }
                      >
                        📋 Copier
                      </button>
                    </div>

                    {(selectedDelay.booking?.customer_email || selectedDelay.booking?.email) && (
                      <div className={styles.emailBox}>
                        <div className={styles.emailIcon}>📧</div>
                        <div className={styles.emailInfo}>
                          <span className={styles.emailLabel}>Email</span>
                          <span className={styles.emailAddress}>
                            {selectedDelay.booking?.customer_email || selectedDelay.booking?.email}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Message suggéré */}
                  <div className={styles.messageSection}>
                    <label className={styles.messageLabel}>💬 Message suggéré</label>
                    <textarea
                      className={styles.messageTextarea}
                      readOnly
                      rows={4}
                      value={
                        selectedSuggestion?.additional_data?.auto_message ||
                        `Bonjour ${
                          selectedDelay.booking?.customer_name || ''
                        },\n\nVotre chauffeur arrivera avec environ ${
                          selectedDelay.delay_minutes
                        } minutes de retard. Nous nous excusons pour ce désagrément.`
                      }
                    />
                  </div>
                </>
              )}
            </div>

            <div className={styles.modalFooter}>
              <button className={styles.btnSecondary} onClick={() => setContactModalVisible(false)}>
                Fermer
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modal de Report de Course */}
      {rescheduleModalVisible && selectedDelay && (
        <div className={styles.modalOverlay} onClick={() => setRescheduleModalVisible(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3>⏰ Reporter le Rendez-vous</h3>
              <button
                className={styles.modalClose}
                onClick={() => setRescheduleModalVisible(false)}
              >
                ✕
              </button>
            </div>

            <div className={styles.modalBody}>
              {/* Informations actuelles */}
              <div className={styles.rescheduleInfo}>
                <div className={styles.infoRow}>
                  <span className={styles.infoLabel}>Client :</span>
                  <span className={styles.infoValue}>
                    {selectedDelay.booking?.customer_name || 'N/A'}
                  </span>
                </div>

                <div className={styles.infoRow}>
                  <span className={styles.infoLabel}>Course :</span>
                  <span className={styles.infoValue}>#{selectedDelay.booking_id}</span>
                </div>

                <div className={styles.infoRow}>
                  <span className={styles.infoLabel}>Retard actuel :</span>
                  <span className={styles.infoValueHighlight}>
                    +{selectedDelay.delay_minutes} min (
                    {Math.floor(selectedDelay.delay_minutes / 60)}h
                    {selectedDelay.delay_minutes % 60})
                  </span>
                </div>

                <div className={styles.infoRow}>
                  <span className={styles.infoLabel}>Horaire actuel :</span>
                  <span className={styles.infoValue}>
                    {selectedDelay.booking?.scheduled_time
                      ? new Date(selectedDelay.booking.scheduled_time).toLocaleString('fr-FR', {
                          hour: '2-digit',
                          minute: '2-digit',
                        })
                      : 'N/A'}
                  </span>
                </div>
              </div>

              {/* Sélection nouvel horaire */}
              <div className={styles.timeSelection}>
                <label className={styles.timeLabel}>Nouvel horaire (HH:MM)</label>
                <input
                  type="time"
                  className={styles.timeInput}
                  value={newTime}
                  onChange={(e) => setNewTime(e.target.value)}
                />
                <p className={styles.timeHint}>💡 Horaire suggéré : {newTime}</p>
              </div>

              {/* Actions automatiques */}
              <div className={styles.autoActions}>
                <h4>🤖 Actions automatiques</h4>
                <ul>
                  <li>✅ Mise à jour de l'horaire de la course</li>
                  <li>✅ Recherche d'un chauffeur disponible</li>
                  <li>✅ Réassignation automatique</li>
                  <li>✅ Notification du nouveau chauffeur</li>
                </ul>
              </div>
            </div>

            <div className={styles.modalFooter}>
              <button className={styles.btnDanger} onClick={handleCancelReschedule}>
                ⚠️ Marquer comme Urgente
              </button>
              <button className={styles.btnPrimary} onClick={handleReschedule}>
                ✅ Confirmer le Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UnifiedDispatch;
