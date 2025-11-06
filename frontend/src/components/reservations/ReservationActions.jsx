// src/components/reservations/ReservationActions.jsx
import React from 'react';
import { FiClock, FiZap, FiUserPlus, FiTrash2, FiEdit } from 'react-icons/fi';
import styles from './ReservationActions.module.css';

/**
 * Composant centralisé pour les actions sur les réservations
 * Utilisé dans ReservationTable, DispatchTable, etc.
 *
 * Les callbacks doivent ouvrir les modales appropriées :
 * - onSchedule(reservation) : Ouvre la modal de planification
 * - onDispatchNow(reservation) : Action directe (pas de modal)
 * - onAssign(reservation) : Ouvre la modal d'assignation
 * - onDelete(reservation) : Ouvre la modal de confirmation de suppression
 */
const ReservationActions = ({
  reservation,
  onSchedule,
  onDispatchNow,
  onAssign,
  onEdit, // 🆕 Action pour éditer la réservation
  onDelete,
  hideAssign = false, // Si true, cache le bouton assigner
  hideSchedule = false, // Si true, cache le bouton planifier l'heure
  hideUrgent = false, // Si true, cache le bouton urgent
  hideEdit = false, // Si true, cache le bouton éditer
  hideDelete = false, // Si true, cache le bouton supprimer
  showAll = false, // Si true, affiche toutes les actions disponibles
  className = '',
}) => {
  // Vérifier si c'est un retour sans heure définie (à confirmer)
  // Support plusieurs façons d'identifier un retour
  const isReturn = !!(
    reservation?.is_return ||
    reservation?.booking_type === 'return' ||
    reservation?.type === 'return'
  );

  // Vérifier si l'heure doit être confirmée
  // Cas 1: time_confirmed est explicitement false OU null/undefined
  // Cas 2: scheduled_time est manquant/null
  // Cas 3: L'heure est 00:00 (heure par défaut, souvent utilisée pour "à confirmer")
  const hasScheduledTime = !!reservation?.scheduled_time;

  // ⚡ Logique stricte : time_confirmed doit être explicitement true
  // Si null, undefined, ou false, on considère qu'il faut confirmer
  const timeConfirmed = reservation?.time_confirmed === true;

  // Vérifier si l'heure est à 00:00 (indicateur d'heure à confirmer)
  let isDefaultTime = false;
  if (reservation?.scheduled_time) {
    const timeStr = reservation.scheduled_time.toString();
    // Format ISO: "2025-11-03T00:00:00" ou similaire
    isDefaultTime = timeStr.includes('T00:00:00') || timeStr.includes(' 00:00:00');
  }

  const needsTimeConfirmation = isReturn && (!timeConfirmed || !hasScheduledTime || isDefaultTime);

  const status = reservation?.status?.toLowerCase() || 'unknown';
  const deletableStatuses = ['pending', 'accepted', 'assigned'];
  const isDeletable = deletableStatuses.includes(status);

  // Déterminer quelles actions afficher
  // Planifier et Urgent : uniquement pour les retours nécessitant confirmation
  const showSchedule = !hideSchedule && needsTimeConfirmation && !!onSchedule;
  const showUrgent = !hideUrgent && needsTimeConfirmation && !!onDispatchNow;
  // Assigner : pour les retours à confirmer OU pour accepted/assigned normaux (peut être caché)
  const showAssign =
    !hideAssign &&
    (needsTimeConfirmation || ['accepted', 'assigned'].includes(status)) &&
    !!onAssign;
  // Éditer : pour les statuts modifiables (pending, accepted, assigned)
  const editableStatuses = ['pending', 'accepted', 'assigned'];
  const showEdit = !hideEdit && editableStatuses.includes(status) && !!onEdit;
  // Supprimer : pour les retours à confirmer OU pour les statuts supprimables
  const showDelete = !hideDelete && (needsTimeConfirmation || isDeletable) && !!onDelete;

  // Debug log pour comprendre pourquoi les boutons ne s'affichent pas
  if (process.env.NODE_ENV === 'development' && isReturn) {
    console.debug('[ReservationActions]', {
      reservationId: reservation?.id,
      isReturn,
      time_confirmed: reservation?.time_confirmed,
      scheduled_time: reservation?.scheduled_time,
      needsTimeConfirmation,
      showSchedule,
      showUrgent,
      hasOnSchedule: !!onSchedule,
      hasOnDispatchNow: !!onDispatchNow,
    });
  }

  // Si aucune action à afficher
  if (!showSchedule && !showUrgent && !showAssign && !showEdit && !showDelete && !showAll) {
    return null;
  }

  const handleDelete = (e) => {
    e.stopPropagation();
    // Le callback doit ouvrir la modal de confirmation
    onDelete?.(reservation);
  };

  return (
    <div className={`${styles.actionsContainer} ${className}`}>
      {/* Planifier l'heure */}
      {showSchedule && (
        <button
          onClick={() => onSchedule?.(reservation)}
          title="Planifier l'heure de retour"
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
          onClick={() => onAssign?.(reservation)}
          title="Assigner un chauffeur"
          className={styles.actionButton}
        >
          <FiUserPlus />
        </button>
      )}

      {/* Éditer */}
      {showEdit && (
        <button
          onClick={() => onEdit?.(reservation)}
          title="Éditer la réservation"
          className={styles.actionButton}
        >
          <FiEdit />
        </button>
      )}

      {/* Supprimer */}
      {showDelete && (
        <button
          onClick={handleDelete}
          title="Annuler/Supprimer"
          className={`${styles.actionButton} ${styles.deleteButton}`}
        >
          <FiTrash2 />
        </button>
      )}
    </div>
  );
};

export default ReservationActions;
