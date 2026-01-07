// src/pages/company/Dashboard/components/DispatchTable.jsx
import React from 'react';
import styles from './ReservationTable.module.css';
import { FiCheckCircle, FiXCircle } from 'react-icons/fi';
import { renderBookingDateTime } from '../../../../utils/formatDate';
import ReservationActions from '../../../../components/reservations/ReservationActions';

/**
 * Tableau spécifique pour la page Dispatch
 * Affiche la colonne Chauffeur au lieu de Montant
 */
const DispatchTable = ({
  reservations = [],
  dispatches,
  delays = [], // ✅ Retards détectés pour affichage visuel
  onRowClick,
  onAccept,
  onReject,
  onAssign,
  onDelete,
  onSchedule,
  onDispatchNow,
  hideSchedule = false,
  hideUrgent = false,
  hideEdit = false,
  hideDelete = false,
}) => {
  const deletableStatuses = ['pending', 'accepted', 'assigned'];

  // Support des deux noms de prop pour compatibilité
  const data = dispatches || reservations || [];

  // ✅ Fonction helper pour trouver le retard d'une course
  const getDelayForBooking = (bookingId) => {
    if (!delays || delays.length === 0 || !bookingId) return null;
    
    // Gérer les cas où booking_id pourrait être string ou number
    const found = delays.find(d => 
      d.booking_id === bookingId || 
      Number(d.booking_id) === Number(bookingId)
    );
    
    return found;
  };

  // ✅ Fonction pour déterminer le type de retard (3 niveaux)
  const getDelayStatus = (delayInfo) => {
    if (!delayInfo) return null;
    
    // ✅ Priorité 1: Utiliser delay_severity du backend si disponible
    const delaySeverity = delayInfo.delay_severity;
    if (delaySeverity) {
      // Mapper les statuts backend vers les classes CSS frontend
      if (delaySeverity === 'reasonable') return 'reasonable';
      if (delaySeverity === 'moderate') return 'moderate';
      if (delaySeverity === 'critical') return 'critical';
    }
    
    // ✅ Priorité 2: Utiliser delay_minutes pour déterminer la sévérité
    const delayMinutes = delayInfo.delay_minutes || 
                         delayInfo.pickup_delay_minutes || 
                         delayInfo.dropoff_delay_minutes || 
                         0;
    
    if (delayMinutes <= 0) return null;
    if (delayMinutes <= 5) return 'reasonable';  // 1-5 min : raisonnable
    if (delayMinutes <= 10) return 'moderate';   // 5-10 min : modéré
    return 'critical';  // >10 min : critique
  };

  return (
    <div className={styles.tableContainer}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Client</th>
            <th>Date / Heure</th>
            <th>Lieu</th>
            <th>Chauffeur</th>
            <th>Statut</th>
            <th className={styles.actionsCell}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {data.map((r) => {
            const status = r.status?.toLowerCase() || 'unknown';
            const _isDeletable = deletableStatuses.includes(status); // Conservé pour référence future
            const isReturn = !!r.is_return;

            // ❌ Aucune action pour les statuts terminaux (canceled, completed, rejected, etc.)
            const noActionStatuses = [
              'canceled',
              'cancelled',
              'completed',
              'return_completed',
              'rejected',
              'no_show',
            ];
            const hasActions = !noActionStatuses.includes(status);

            // Vérifier si c'est un retour sans heure définie (à confirmer)
            // Utiliser le champ time_confirmed pour déterminer si l'heure est à confirmer
            // Conservé pour référence future (géré par ReservationActions)
            const _needsTimeConfirmation =
              isReturn && (r.time_confirmed === false || !r.scheduled_time);

            // ✅ Obtenir les infos de retard pour cette course
            const delayInfo = getDelayForBooking(r.id);
            const delayStatus = getDelayStatus(delayInfo);
            const delayMinutes = delayInfo 
              ? Math.round(delayInfo.delay_minutes || delayInfo.pickup_delay_minutes || delayInfo.dropoff_delay_minutes || 0)
              : 0;

            return (
              <tr 
                key={r.id} 
                onClick={() => onRowClick?.(r)} 
                className={`${styles.tableRow} ${
                  delayStatus === 'critical' ? styles.rowDelayed : 
                  delayStatus === 'moderate' ? styles.rowSlightDelay : 
                  delayStatus === 'reasonable' ? styles.rowReasonableDelay : ''
                }`}
              >
                <td className={styles.clientCell}>
                  {r.client?.full_name || r.client_name}
                  {delayInfo && delayStatus && (
                    <span 
                      className={`${styles.delayBadge} ${
                        delayStatus === 'critical' ? styles.delayBadgeCritical :
                        delayStatus === 'moderate' ? styles.delayBadgeModerate :
                        styles.delayBadgeReasonable
                      }`} 
                      title={`Retard de ${delayMinutes} minutes (${delayStatus === 'reasonable' ? 'raisonnable' : delayStatus === 'moderate' ? 'modéré' : 'critique'})`}
                    >
                      {delayStatus === 'critical' ? '🚨' : '⚠️'} {delayMinutes > 0 ? `${delayMinutes} min` : 'Retard'}
                    </span>
                  )}
                </td>
                <td>{renderBookingDateTime(r)}</td>
                <td className={styles.locationCell}>
                  <div>
                    <strong>De:</strong> {r.pickup_location}
                  </div>
                  <div>
                    <strong>À:</strong> {r.dropoff_location}
                  </div>
                </td>
                <td className={styles.driverCell}>
                  {r.driver?.full_name ||
                    r.driver?.name ||
                    r.driver?.username ||
                    r.assignment?.driver?.full_name ||
                    r.assignment?.driver?.name ||
                    (r.driver_id ? `Chauffeur #${r.driver_id}` : 'Non assigné')}
                </td>
                <td>
                  <span className={`${styles.statusBadge} ${styles[status] || ''}`}>
                    {(r.status || '').replace('_', ' ') || status}
                  </span>
                  {delayInfo && delayStatus && (
                    <div className={styles.delayIndicator}>
                      ⚠️ Retard: {delayMinutes > 0 ? `${delayMinutes} min` : 'Détecté'}
                    </div>
                  )}
                </td>
                <td
                  className={styles.actionsCell}
                  onClick={(e) => e.stopPropagation()} // Empêche d'ouvrir le modal en cliquant sur un bouton
                >
                  {/* ❌ Aucune action pour les statuts terminaux */}
                  {!hasActions ? (
                    <span
                      style={{
                        color: '#94a3b8',
                        fontSize: '0.85rem',
                        fontStyle: 'italic',
                      }}
                    >
                      Aucune action
                    </span>
                  ) : (
                    <>
                      {/* B) Courses PENDING normales => Accepter + Rejeter */}
                      {status === 'pending' && !isReturn && (
                        <>
                          <button
                            onClick={() => onAccept?.(r.id)}
                            title="Accepter"
                            className={`${styles.actionButton} ${styles.acceptButton}`}
                          >
                            <FiCheckCircle />
                          </button>
                          <button
                            onClick={() => onReject?.(r.id)}
                            title="Rejeter"
                            className={`${styles.actionButton} ${styles.rejectButton}`}
                          >
                            <FiXCircle />
                          </button>
                        </>
                      )}

                      {/* Actions centralisées : Planifier, Urgent, Assigner, Supprimer */}
                      <ReservationActions
                        reservation={r}
                        onSchedule={onSchedule}
                        onDispatchNow={onDispatchNow}
                        onAssign={onAssign}
                        onDelete={onDelete}
                        hideSchedule={hideSchedule}
                        hideUrgent={hideUrgent}
                        hideEdit={hideEdit}
                        hideDelete={hideDelete}
                      />
                    </>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default React.memo(DispatchTable);
