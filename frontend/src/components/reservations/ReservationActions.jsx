// src/components/reservations/ReservationActions.jsx
import React from 'react';
import { FiClock, FiZap, FiUserPlus, FiShare2 } from 'react-icons/fi';
import { hasConfirmedPickupTime, hasScheduledPickupTime } from '../../utils/bookingScheduling';
import styles from './ReservationActions.module.css';

/** Statuts où ce bloc n’affiche pas d’actions secondaires (normal en dispatch). */
const NO_SECONDARY_ACTION_STATUSES = new Set([
  'en_route',
  'in_progress',
  'completed',
  'canceled',
  'cancelled',
  'return_completed',
]);

/**
 * Composant centralisé pour les actions sur les réservations
 * Utilisé dans ReservationTable, DispatchTable, etc.
 *
 * Les callbacks doivent ouvrir les modales appropriées :
 * - onSchedule(reservation) : Ouvre la modal de planification
 * - onDispatchNow(reservation) : Action directe (pas de modal)
 * - onAssign(reservation) : Ouvre la modal d'assignation
 * - onTransfer(reservation) : Ouvre la modal de transfert à un partenaire
 * - onDelete(reservation) : Ouvre la modal de confirmation de suppression
 */
const ReservationActions = ({
  reservation,
  onSchedule,
  onDispatchNow,
  onAssign,
  onEdit, // 🆕 Action pour éditer la réservation
  onTransfer, // 🆕 Action pour transférer à un partenaire
  onDelete,
  hideAssign = false, // Si true, cache le bouton assigner
  hideSchedule = false, // Si true, cache le bouton planifier l'heure
  hideUrgent = false, // Si true, cache le bouton urgent
  hideEdit = false, // Si true, cache le bouton éditer
  hideTransfer = false, // Si true, cache le bouton transférer
  hideDelete = false, // Si true, cache le bouton supprimer
  showAll = false, // Si true, affiche toutes les actions disponibles
  className = '',
  needsTimeConfirmationOverride,
}) => {
  // Vérifier si c'est un retour sans heure définie (à confirmer)
  // Support plusieurs façons d'identifier un retour
  const isReturn = !!(
    reservation?.is_return ||
    reservation?.booking_type === 'return' ||
    reservation?.type === 'return'
  );

  const computedNeedsTimeConfirmation = isReturn && !hasConfirmedPickupTime(reservation);
  const needsTimeConfirmation =
    typeof needsTimeConfirmationOverride === 'boolean'
      ? needsTimeConfirmationOverride
      : computedNeedsTimeConfirmation;

  const canMarkUrgent = !hasScheduledPickupTime(reservation);

  const status = reservation?.status?.toLowerCase() || 'unknown';
  const deletableStatuses = ['pending', 'accepted', 'assigned'];
  const isDeletable = deletableStatuses.includes(status);

  // Déterminer quelles actions afficher
  // Planifier et Urgent : uniquement pour les retours nécessitant confirmation
  const showSchedule = !hideSchedule && needsTimeConfirmation && !!onSchedule;
  // « Urgent (+15 min) » utilise onDispatchNow (endpoint /dispatch-now générique :
  // fixe scheduled_time = maintenant + offset). Valable aussi bien pour un retour
  // que pour un leg aller sans heure définie.
  const showUrgent = !hideUrgent && canMarkUrgent && needsTimeConfirmation && !!onDispatchNow;
  // Assigner : pour les retours à confirmer OU pour accepted/assigned normaux (peut être caché)
  const showAssign =
    !hideAssign &&
    (needsTimeConfirmation || ['accepted', 'assigned'].includes(status)) &&
    !!onAssign;
  // Éditer : pour les statuts modifiables (pending, accepted, assigned)
  const editableStatuses = ['pending', 'accepted', 'assigned'];
  const showEdit = !hideEdit && editableStatuses.includes(status) && !!onEdit;
  // Transférer : pour les statuts transférables (pending, accepted, assigned)
  const transferableStatuses = ['pending', 'accepted', 'assigned'];
  const showTransfer = !hideTransfer && transferableStatuses.includes(status) && !!onTransfer;
  // Supprimer : pour les retours à confirmer OU pour les statuts supprimables
  const showDelete = !hideDelete && (needsTimeConfirmation || isDeletable) && !!onDelete;

  // Si aucune action à afficher
  if (!showSchedule && !showUrgent && !showAssign && !showEdit && !showTransfer && !showDelete && !showAll) {
    if (process.env.NODE_ENV === 'development') {
      if (NO_SECONDARY_ACTION_STATUSES.has(status)) {
        // Cas attendu : pas de log (évite le spam à chaque refetch / focus).
      } else if (status === 'pending') {
        console.debug(
          `[ReservationActions] Aucune action secondaire pour #${reservation?.id} (pending : voir boutons parent)`
        );
      } else {
        // accepted / assigned sans callbacks, etc. : debug seulement (pas un warning).
        console.debug(
          `[ReservationActions] Aucune action secondaire pour #${reservation?.id} (statut « ${status} »)`
        );
      }
    }
    return null;
  }

  return (
    <div className={`${styles.actionsContainer} ${className}`}>
      {/* Planifier l'heure */}
      {showSchedule && (
        <button
          onClick={() => onSchedule?.(reservation)}
          title={isReturn ? "Planifier l'heure de retour" : "Planifier l'heure"}
          className={styles.actionButton}
        >
          <FiClock />
        </button>
      )}

      {/* Urgent */}
      {showUrgent && (
        <button
          onClick={() => onDispatchNow?.(reservation)}
          title="Urgent (+15 min)"
          className={`${styles.actionButton} ${styles.urgentButton}`}
        >
          <FiZap />
        </button>
      )}

      {/* Assigner un chauffeur */}
      {showAssign && (
        <button
          data-tour-id="assigned-assign-action"
          onClick={() => onAssign?.(reservation)}
          title="Assigner un chauffeur"
          className={styles.actionButton}
        >
          <FiUserPlus />
        </button>
      )}

      {/* Transférer à un partenaire */}
      {showTransfer && (
        <button
          onClick={() => onTransfer?.(reservation)}
          title="Transférer à un partenaire"
          className={styles.actionButton}
        >
          <FiShare2 />
        </button>
      )}
    </div>
  );
};

export default ReservationActions;
