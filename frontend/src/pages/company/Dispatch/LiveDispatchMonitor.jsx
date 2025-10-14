// frontend/src/pages/company/Dispatch/LiveDispatchMonitor.jsx
/**
 * Composant de monitoring temps réel du dispatch
 * Affiche les retards en direct avec suggestions intelligentes
 */

import React, { useState, useEffect, useCallback } from "react";
import {
  getLiveDelays,
  getOptimizerStatus,
  startRealTimeOptimizer,
  stopRealTimeOptimizer,
  applySuggestion,
} from "../../../services/dispatchMonitoringService";
import CompanyHeader from "../../../components/layout/Header/CompanyHeader";
import CompanySidebar from "../../../components/layout/Sidebar/CompanySidebar/CompanySidebar";
import styles from "./LiveDispatchMonitor.module.css";

const LiveDispatchMonitor = () => {
  const [date, setDate] = useState(new Date().toISOString().split("T")[0]);
  const [delays, setDelays] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [optimizerStatus, setOptimizerStatus] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval] = useState(30000); // 30 secondes

  // Charger les données
  const fetchDelays = useCallback(async () => {
    try {
      setLoading(true);
      const response = await getLiveDelays(date);
      setDelays(response.delays || []);
      setSummary(response.summary || null);
      setError(null);
    } catch (err) {
      console.error("[LiveDispatchMonitor] Error fetching delays:", err);
      setError("Erreur lors du chargement des retards");
    } finally {
      setLoading(false);
    }
  }, [date]);

  // Charger le statut de l'optimiseur
  const fetchOptimizerStatus = useCallback(async () => {
    try {
      const status = await getOptimizerStatus();
      setOptimizerStatus(status);
    } catch (err) {
      console.error(
        "[LiveDispatchMonitor] Error fetching optimizer status:",
        err
      );
    }
  }, []);

  // Démarrer/Arrêter le monitoring automatique
  const toggleOptimizer = async () => {
    try {
      if (optimizerStatus?.running) {
        await stopRealTimeOptimizer();
      } else {
        await startRealTimeOptimizer(120); // 2 minutes
      }
      await fetchOptimizerStatus();
    } catch (err) {
      console.error("[LiveDispatchMonitor] Error toggling optimizer:", err);
      alert("Erreur lors de l'activation/désactivation du monitoring");
    }
  };

  // Appliquer une suggestion
  const handleApplySuggestion = async (suggestion) => {
    if (!suggestion.alternative_driver_id) {
      alert("Cette suggestion ne peut pas être appliquée automatiquement");
      return;
    }

    const confirmed = window.confirm(
      `Voulez-vous vraiment réassigner la course #${suggestion.booking_id} au chauffeur #${suggestion.alternative_driver_id}?\n\n` +
        `Gain estimé: ${suggestion.estimated_gain_minutes} minutes`
    );

    if (!confirmed) return;

    try {
      await applySuggestion(
        delays.find((d) => d.booking_id === suggestion.booking_id)
          ?.assignment_id,
        suggestion.alternative_driver_id
      );
      alert("Suggestion appliquée avec succès!");
      fetchDelays(); // Rafraîchir
    } catch (err) {
      console.error("[LiveDispatchMonitor] Error applying suggestion:", err);
      alert("Erreur lors de l'application de la suggestion");
    }
  };

  // Auto-refresh
  useEffect(() => {
    fetchDelays();
    fetchOptimizerStatus();
  }, [fetchDelays, fetchOptimizerStatus]);

  useEffect(() => {
    if (!autoRefresh) return;

    const intervalId = setInterval(() => {
      fetchDelays();
      fetchOptimizerStatus();
    }, refreshInterval);

    return () => clearInterval(intervalId);
  }, [autoRefresh, refreshInterval, fetchDelays, fetchOptimizerStatus]);

  // Rendu des statistiques
  const renderSummary = () => {
    if (!summary) return null;

    return (
      <div className={styles.summaryCards}>
        <div className={`${styles.summaryCard} ${styles.total}`}>
          <div className={styles.cardIcon}>📊</div>
          <div className={styles.cardContent}>
            <h3>{summary.total}</h3>
            <p>Total assignations</p>
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
            <h3>{summary.average_delay.toFixed(1)} min</h3>
            <p>Retard moyen</p>
          </div>
        </div>
      </div>
    );
  };

  // Rendu d'une suggestion
  const renderSuggestion = (suggestion, delayInfo) => {
    return (
      <div
        key={`${delayInfo.assignment_id}-${suggestion.action}`}
        className={styles.suggestionItem}
      >
        <div className={styles.suggestionHeader}>
          <span
            className={`${styles.priorityBadge} ${styles[suggestion.priority]}`}
          >
            {suggestion.priority.toUpperCase()}
          </span>
          <span className={styles.actionBadge}>{suggestion.action}</span>
        </div>
        <p className={styles.suggestionMessage}>{suggestion.message}</p>
        {suggestion.estimated_gain_minutes && (
          <p className={styles.suggestionGain}>
            💡 Gain estimé:{" "}
            <strong>{suggestion.estimated_gain_minutes} minutes</strong>
          </p>
        )}
        {suggestion.action === "reassign" && (
          <button
            className={styles.applySuggestionBtn}
            onClick={() => handleApplySuggestion(suggestion)}
          >
            Appliquer cette suggestion
          </button>
        )}
      </div>
    );
  };

  // Rendu d'une ligne de retard
  const renderDelay = (delay) => {
    const statusColors = {
      late: "#dc3545",
      on_time: "#28a745",
      early: "#17a2b8",
    };

    const statusIcons = {
      late: "🔴",
      on_time: "🟢",
      early: "🔵",
    };

    const statusLabels = {
      late: "En retard",
      on_time: "À l'heure",
      early: "En avance",
    };

    return (
      <div key={delay.assignment_id} className={styles.delayCard}>
        <div className={styles.delayHeader}>
          <div className={styles.delayInfo}>
            <h4>
              {statusIcons[delay.status]} Course #{delay.booking_id}
            </h4>
            <p className={styles.customerName}>
              {delay.booking?.customer_name}
            </p>
          </div>
          <div
            className={styles.delayBadge}
            style={{ backgroundColor: statusColors[delay.status] }}
          >
            {statusLabels[delay.status]}
            {delay.delay_minutes !== 0 && (
              <span className={styles.delayTime}>
                {delay.delay_minutes > 0 ? "+" : ""}
                {delay.delay_minutes} min
              </span>
            )}
          </div>
        </div>

        <div className={styles.delayDetails}>
          <div className={styles.detailRow}>
            <span className={styles.label}>Chauffeur:</span>
            <span className={styles.value}>
              {delay.driver?.name || `#${delay.driver_id}`}
            </span>
          </div>
          <div className={styles.detailRow}>
            <span className={styles.label}>Horaire prévu:</span>
            <span className={styles.value}>
              {delay.scheduled_time
                ? new Date(delay.scheduled_time).toLocaleTimeString("fr-FR", {
                    hour: "2-digit",
                    minute: "2-digit",
                  })
                : "N/A"}
            </span>
          </div>
          <div className={styles.detailRow}>
            <span className={styles.label}>ETA actuel:</span>
            <span className={styles.value}>
              {delay.current_eta
                ? new Date(delay.current_eta).toLocaleTimeString("fr-FR", {
                    hour: "2-digit",
                    minute: "2-digit",
                  })
                : "N/A"}
            </span>
          </div>
          {delay.booking && (
            <>
              <div className={styles.detailRow}>
                <span className={styles.label}>Pickup:</span>
                <span className={styles.value}>
                  {delay.booking.pickup_address}
                </span>
              </div>
              <div className={styles.detailRow}>
                <span className={styles.label}>Dropoff:</span>
                <span className={styles.value}>
                  {delay.booking.dropoff_address}
                </span>
              </div>
            </>
          )}
        </div>

        {/* Suggestions */}
        {delay.suggestions && delay.suggestions.length > 0 && (
          <div className={styles.suggestionsSection}>
            <h5>💡 Suggestions d'optimisation</h5>
            {delay.suggestions.map((suggestion) =>
              renderSuggestion(suggestion, delay)
            )}
          </div>
        )}

        {/* Impact cascade */}
        {delay.impacts_next_bookings &&
          delay.impacts_next_bookings.length > 0 && (
            <div className={styles.cascadeImpact}>
              <h5>⚠️ Impact sur les courses suivantes</h5>
              <ul>
                {delay.impacts_next_bookings.map((impact) => (
                  <li key={impact.booking_id}>
                    Course #{impact.booking_id} ({impact.customer_name}) -{" "}
                    <strong>
                      Retard potentiel: +{impact.potential_delay_minutes} min
                    </strong>
                  </li>
                ))}
              </ul>
            </div>
          )}
      </div>
    );
  };

  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />

      <div className={styles.dashboardLayout}>
        <CompanySidebar />

        <main className={styles.mainContent}>
          <div className={styles.monitorContainer}>
            <div className={styles.monitorHeader}>
              <div className={styles.headerTop}>
                <div className={styles.headerLeft}>
                  <h1>📡 Monitoring Temps Réel</h1>
                  <p className={styles.subtitle}>
                    Surveillance automatique des retards et optimisations
                  </p>
                </div>

                <div className={styles.headerControls}>
                  <input
                    type="date"
                    value={date}
                    onChange={(e) => setDate(e.target.value)}
                    className={styles.datePicker}
                  />

                  <label className={styles.autoRefreshToggle}>
                    <input
                      type="checkbox"
                      checked={autoRefresh}
                      onChange={(e) => setAutoRefresh(e.target.checked)}
                    />
                    <span>Auto-refresh (30s)</span>
                  </label>

                  <button
                    className={`${styles.optimizerToggle} ${
                      optimizerStatus?.running ? styles.active : ""
                    }`}
                    onClick={toggleOptimizer}
                  >
                    {optimizerStatus?.running
                      ? "⏸️ Arrêter monitoring"
                      : "▶️ Démarrer monitoring"}
                  </button>

                  <button
                    onClick={fetchDelays}
                    disabled={loading}
                    className={styles.refreshBtn}
                  >
                    🔄 Rafraîchir
                  </button>
                </div>
              </div>
            </div>

            {optimizerStatus?.running && (
              <div className={styles.optimizerStatusBanner}>
                🤖 Monitoring automatique actif - Dernière vérification:{" "}
                {optimizerStatus.last_check
                  ? new Date(optimizerStatus.last_check).toLocaleTimeString(
                      "fr-FR"
                    )
                  : "Jamais"}
                {optimizerStatus.opportunities_count > 0 && (
                  <span className={styles.opportunitiesCount}>
                    {optimizerStatus.opportunities_count} opportunité(s)
                    détectée(s)
                  </span>
                )}
              </div>
            )}

            {error && <div className={styles.errorBanner}>❌ {error}</div>}

            {renderSummary()}

            <div className={styles.delaysList}>
              {loading && delays.length === 0 ? (
                <div className={styles.loadingSpinner}>
                  <div className={styles.spinner}></div>
                  <p>Chargement des données...</p>
                </div>
              ) : delays.length === 0 ? (
                <div className={styles.emptyState}>
                  <div className={styles.emptyIcon}>✨</div>
                  <h3>Aucun retard détecté</h3>
                  <p>Toutes les courses sont à l'heure !</p>
                </div>
              ) : (
                delays.map(renderDelay)
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default LiveDispatchMonitor;
