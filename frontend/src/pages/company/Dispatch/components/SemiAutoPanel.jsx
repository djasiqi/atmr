import React, { useMemo } from 'react';
import DispatchTable from '../../Dashboard/components/DispatchTable';
import EmptyState from '../../../../components/EmptyState';
import ModeBanner from './ModeBanner';
import { showSuccess, showError } from '../../../../utils/toast';
import useRLSuggestions from '../../../../hooks/useRLSuggestions';
import { hasScheduledPickupTime } from '../../../../utils/bookingScheduling';
import RLSuggestionCard from '../../../../components/RL/RLSuggestionCard';

/**
 * Composant SIMPLIFIÉ pour le mode semi-automatique
 * Interface épurée : Tableau + Suggestions importantes seulement
 */
const SemiAutoPanel = ({
  dispatches = [],
  loading: _loading,
  error: _error,
  onDeleteReservation,
  onDispatchNow,
  onAssign, // 🆕 Ajouter onAssign pour permettre l'assignation manuelle
  currentDate,
  styles = {},
  currentCompanyId, // ✅ ID de l'entreprise pour la direction des transferts
  delayMap = {},
}) => {
  // Charger suggestions RL (filtrage strict côté hook)
  const {
    suggestions: allSuggestions,
    loading: mdiLoading,
    error: mdiError,
    applySuggestion,
  } = useRLSuggestions(currentDate, {
    autoRefresh: true,
    refreshInterval: 30000,
    minConfidence: 0.75, // ✅ Seulement haute confiance (75%+)
    limit: 3, // ✅ Max 3 suggestions (les meilleures)
  });

  // ✅ Filtrer suggestions IMPORTANTES seulement (gain > 15 min)
  const importantSuggestions = useMemo(() => {
    return (allSuggestions || []).filter(
      (s) => (s.expected_gain_minutes || 0) >= 15 && (s.confidence || 0) >= 0.75
    );
  }, [allSuggestions]);

  // ✅ Trier les dispatches : d'abord par heure croissante, puis les heures à définir à la fin
  const sortedDispatches = useMemo(() => {
    return [...dispatches].sort((a, b) => {
      const aIsUndefined = !hasScheduledPickupTime(a);
      const bIsUndefined = !hasScheduledPickupTime(b);
      const aTime = a.scheduled_time ? new Date(a.scheduled_time).getTime() : null;
      const bTime = b.scheduled_time ? new Date(b.scheduled_time).getTime() : null;

      if (!aIsUndefined && !bIsUndefined) {
        return aTime - bTime;
      }
      if (aIsUndefined && !bIsUndefined) return 1;
      if (!aIsUndefined && bIsUndefined) return -1;
      return 0;
    });
  }, [dispatches]);

  const formatTime = (timeString) => {
    if (!timeString) return '—';
    const date = new Date(timeString);
    if (isNaN(date.getTime())) return '—';
    return date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
  };

  // Handler pour appliquer suggestion
  const handleApplySuggestion = async (suggestion) => {
    try {
      const result = await applySuggestion(suggestion);

      if (result.success) {
        showSuccess(
          `✅ Suggestion appliquée !\nGain: +${suggestion.expected_gain_minutes || 0} min`
        );
      } else {
        showError(`❌ ${result.error || "Erreur lors de l'application"}`);
      }
    } catch (err) {
      console.error('[SemiAutoPanel] Error:', err);
      showError(`❌ Erreur: ${err.message}`);
    }
  };

  return (
    <>
      {/* Tableau des courses - PRINCIPAL */}
      {dispatches.length === 0 ? (
        <EmptyState
          icon="📋"
          title="Aucune course pour cette date"
          message="Cliquez sur 🚀 Lancer Dispatch ci-dessus pour assigner automatiquement vos chauffeurs."
        />
      ) : (
        <>
          {/* Statut du planning */}
          <div className={styles.planningStatus}>
            {!mdiLoading && importantSuggestions.length === 0 ? (
              <div className={styles.statusOptimal}>
                <span className={styles.statusIcon}>✅</span>
                <div className={styles.statusText}>
                  <strong>Planning optimal</strong>
                  <p>{dispatches.length} courses assignées - Aucune amélioration nécessaire</p>
                </div>
              </div>
            ) : !mdiLoading && importantSuggestions.length > 0 ? (
              <div className={styles.statusWithSuggestions}>
                <span className={styles.statusIcon}>💡</span>
                <div className={styles.statusText}>
                  <strong>Planning créé</strong>
                  <p>
                    {dispatches.length} courses assignées • {importantSuggestions.length}{' '}
                    amélioration(s) suggérée(s)
                  </p>
                </div>
              </div>
            ) : null}
          </div>

          {/* Tableau principal */}
          <DispatchTable
            dispatches={sortedDispatches}
            delayMap={delayMap}
            onDelete={onDeleteReservation}
            onDispatchNow={onDispatchNow}
            onAssign={onAssign} // 🆕 Permettre l'assignation manuelle pour les courses transférées
            hideSchedule={true}
            hideUrgent={false}
            hideEdit={true}
            hideDelete={false}
            hideAssign={false} // 🆕 Afficher le bouton d'assignation pour les courses transférées
            formatTime={formatTime}
            showSuggestions={false}
            currentCompanyId={currentCompanyId}
          />

          {/* Suggestions IMPORTANTES seulement (si gain > 15 min ET confiance > 75%) */}
          {!mdiLoading && !mdiError && importantSuggestions.length > 0 && (
            <div className={styles.suggestionsSection}>
              <div className={styles.suggestionsHeader}>
                <h4>💡 Améliorations suggérées</h4>
                <p className={styles.suggestionsSubtitle}>
                  Le système a détecté {importantSuggestions.length} optimisation(s) possible(s)
                </p>
              </div>

              <div className={styles.suggestionsGrid}>
                {importantSuggestions.map((suggestion, idx) => (
                  <RLSuggestionCard
                    key={idx}
                    suggestion={suggestion}
                    onApply={handleApplySuggestion}
                    readOnly={false}
                  />
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* Bannière informative (discrète) */}
      <ModeBanner
        icon="⚙️"
        title="Mode Semi-Automatique"
        description="Le dispatch s'effectue automatiquement. Vous pouvez appliquer les suggestions d'amélioration en un clic."
        variant="semiAuto"
        styles={styles}
      />
    </>
  );
};

export default SemiAutoPanel;
