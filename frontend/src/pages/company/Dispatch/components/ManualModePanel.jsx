import React, { useMemo, useCallback, useState } from 'react';
import DispatchTable from '../../Dashboard/components/DispatchTable';
import DispatchTableSkeleton from '../../../../components/SkeletonLoaders/DispatchTableSkeleton';
import EmptyState from '../../../../components/EmptyState';
import ModeBanner from './ModeBanner';
import ProTip from './ProTip';
import useRLSuggestions from '../../../../hooks/useRLSuggestions';
import RLSuggestionCard from '../../../../components/RL/RLSuggestionCard';

/**
 * Composant pour le mode manuel de dispatch avec optimisations de performance
 * 🆕 Enrichi avec suggestions MDI en readonly (informatives)
 */
const ManualModePanel = ({
  dispatches = [],
  loading,
  error,
  sortBy,
  setSortBy,
  sortOrder,
  setSortOrder,
  selectedReservationForAssignment: _selectedReservationForAssignment, // Conservé pour compatibilité mais non utilisé
  setSelectedReservationForAssignment, // Fonction pour ouvrir la modale d'assignation
  onSchedule, // 🆕 Handler pour planifier l'heure
  onDispatchNow, // 🆕 Handler pour dispatch urgent
  onDelete, // 🆕 Handler pour supprimer (ouvre la modale)
  currentDate, // 🆕 Date actuelle pour charger suggestions
  drivers: _drivers = [], // 🆕 Liste des chauffeurs pour l'assignation (utilisée dans UnifiedDispatchRefactored)
  styles = {},
}) => {
  // 🆕 État pour collapsible suggestions
  const [suggestionsExpanded, setSuggestionsExpanded] = useState(true);

  // 🆕 Charger suggestions MDI (readonly, pas d'auto-refresh)
  const {
    suggestions,
    highConfidenceSuggestions,
    avgConfidence,
    totalExpectedGain,
    loading: suggestionsLoading,
  } = useRLSuggestions(currentDate, {
    autoRefresh: false, // Mode manuel: pas d'auto-refresh
    minConfidence: 0.5, // Seulement suggestions >50%
    limit: 10, // Max 10 suggestions
  });
  // Mémoisation de la fonction formatTime
  const formatTime = useCallback((timeString) => {
    if (!timeString) return '—';
    const date = new Date(timeString);
    if (isNaN(date.getTime())) return '—';
    return date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
  }, []);

  // Mémoisation du tri des dispatches (évite le tri à chaque render)
  const sortedDispatches = useMemo(() => {
    return [...dispatches].sort((a, b) => {
      let aValue, bValue;

      switch (sortBy) {
        case 'time':
          aValue = new Date(a.scheduled_time || 0);
          bValue = new Date(b.scheduled_time || 0);
          break;
        case 'client':
          aValue = a.customer_name || '';
          bValue = b.customer_name || '';
          break;
        case 'status':
          aValue = a.status || '';
          bValue = b.status || '';
          break;
        default:
          return 0;
      }

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });
  }, [dispatches, sortBy, sortOrder]);

  if (loading) {
    return <DispatchTableSkeleton rows={8} />;
  }

  if (error) {
    return <div className={styles.error}>Erreur: {error}</div>;
  }

  return (
    <>
      {/* Header avec contrôles de tri dans un panel */}
      <div className={styles.manualPanel}>
        <div className={styles.panelHeader}>
          <h3>Mode Manuel - Assignation des chauffeurs</h3>
          <div className={styles.sortControls}>
            <label>
              Trier par:
              <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
                <option value="time">Heure</option>
                <option value="client">Client</option>
                <option value="status">Statut</option>
              </select>
            </label>
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className={styles.sortButton}
            >
              {sortOrder === 'asc' ? '↑' : '↓'}
            </button>
          </div>
        </div>
      </div>

      {/* Tableau en-dehors du panel ou Empty State */}
      {sortedDispatches.length === 0 ? (
        <EmptyState
          icon="📦"
          title="Aucune course pour cette date"
          message="Créez de nouvelles réservations pour commencer l'assignation manuelle."
        />
      ) : (
        <DispatchTable
          dispatches={sortedDispatches}
          onAssign={
            setSelectedReservationForAssignment
              ? (reservationId) => {
                  // Si c'est une fonction, l'appeler directement
                  if (typeof setSelectedReservationForAssignment === 'function') {
                    setSelectedReservationForAssignment(reservationId);
                  }
                }
              : undefined
          }
          onSchedule={onSchedule}
          onDispatchNow={onDispatchNow}
          onDelete={onDelete}
          formatTime={formatTime}
          hideEdit={true}
          hideDelete={true}
        />
      )}

      {/* 🆕 Section Suggestions MDI (Informatives - Readonly) */}
      {!suggestionsLoading && suggestions.length > 0 && (
        <div className={styles.rlSuggestionsSection}>
          <div
            className={styles.suggestionsSectionHeader}
            onClick={() => setSuggestionsExpanded(!suggestionsExpanded)}
            style={{ cursor: 'pointer' }}
          >
            <div className={styles.suggestionsTitle}>
              <h3>
                💡 Suggestions IA (MDI) - Informatives
                {suggestionsExpanded ? ' ▼' : ' ▶'}
              </h3>
              <div className={styles.suggestionsStats}>
                <span className={styles.statBadge}>
                  {suggestions.length} suggestion{suggestions.length > 1 ? 's' : ''}
                </span>
                <span className={styles.statBadge}>
                  {highConfidenceSuggestions.length} haute confiance
                </span>
                <span className={styles.statBadge}>
                  Confiance moy: {(avgConfidence * 100).toFixed(0)}%
                </span>
                {totalExpectedGain > 0 && (
                  <span className={styles.statBadgeGain}>
                    Gain potentiel: +{totalExpectedGain.toFixed(0)} min
                  </span>
                )}
              </div>
            </div>
          </div>

          {suggestionsExpanded && (
            <div className={styles.suggestionsContent}>
              <p className={styles.suggestionsIntro}>
                Le système MDI (Multi-Driver Intelligence) utilise le Reinforcement Learning pour
                suggérer les assignations optimales. Ces suggestions sont{' '}
                <strong>informatives uniquement</strong> en mode Manual - vous gardez le contrôle
                total des décisions.
              </p>

              <div className={styles.suggestionsGrid}>
                {suggestions.slice(0, 5).map((suggestion, idx) => (
                  <RLSuggestionCard key={idx} suggestion={suggestion} readOnly={true} />
                ))}
              </div>

              {suggestions.length > 5 && (
                <p className={styles.moreSuggestions}>
                  ... et {suggestions.length - 5} autre{suggestions.length - 5 > 1 ? 's' : ''}{' '}
                  suggestion{suggestions.length - 5 > 1 ? 's' : ''} disponible
                  {suggestions.length - 5 > 1 ? 's' : ''}.
                  <br />
                  💡 Passez en mode <strong>Semi-Auto</strong> pour appliquer ces suggestions en un
                  clic.
                </p>
              )}

              <div className={styles.suggestionsTip}>
                <strong>💡 Astuce:</strong> Les suggestions haute confiance (&gt;80%) sont
                généralement très fiables. Le MDI a été entraîné sur des milliers de scénarios réels
                pour optimiser distance, temps et satisfaction client.
              </div>
            </div>
          )}
        </div>
      )}

      {/* Bannière Mode Manuel */}
      <ModeBanner
        icon="🔧"
        title="Mode Manuel Activé"
        description="Vous contrôlez entièrement l'assignation des courses. Aucune action automatique n'est effectuée."
        variant="manual"
        styles={styles}
        action={
          <button
            onClick={() => {
              const companyId = window.location.pathname.split('/')[3] || '';
              window.location.href = `/dashboard/company/${companyId}/settings#operations`;
            }}
            className={styles.settingsLink}
          >
            ⚙️ Activer l'automatisation
          </button>
        }
      />

      {/* Conseil Pro */}
      <ProTip
        message={
          <>
            💡 Vous voyez ci-dessus les suggestions MDI (IA). Passez au mode{' '}
            <strong>Semi-Automatique</strong> pour pouvoir les appliquer en un clic et gagner encore
            plus de temps.
          </>
        }
        styles={styles}
      />

      {/* Modal d'assignation - Gérée par ReservationModals dans UnifiedDispatchRefactored */}
    </>
  );
};

export default ManualModePanel;
