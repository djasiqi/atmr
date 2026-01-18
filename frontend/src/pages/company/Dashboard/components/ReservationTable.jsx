// src/pages/company/Dashboard/components/ReservationTable.jsx
import React from 'react';
import styles from './ReservationTable.module.css';
import { FiCheckCircle, FiXCircle } from 'react-icons/fi';
import { renderBookingDateTime } from '../../../../utils/formatDate';
import ReservationActions from '../../../../components/reservations/ReservationActions';
const ReservationTable = ({
  reservations,
  onRowClick,
  onAccept,
  onReject,
  onAssign,
  onEdit,
  onTransfer,
  onDelete,
  onSchedule,
  onDispatchNow,
  hideAssign = false,
  hideSchedule = false,
  hideUrgent = false,
  hideEdit = false,
  hideTransfer = false,
  hideDelete = false,
  currentCompanyId, // ✅ ID de l'entreprise connectée pour déterminer la direction du transfert
}) => {
  const deletableStatuses = ['pending', 'accepted', 'assigned'];

  return (
    <div className={styles.tableContainer}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Client</th>
            <th>Date / Heure</th>
            <th>Lieu</th>
            <th>Montant</th>
            <th>Statut</th>
            <th className={styles.actionsCell}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {reservations.map((r) => {
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

            // ✅ Déterminer si l'entreprise actuelle peut gérer cette réservation transférée
            // Après acceptation, company_id est mis à jour pour être l'entreprise receveuse
            // Donc on utilise active_transfer.owner_company_id pour déterminer l'émettrice
            const isTransferredSender = currentCompanyId && r.is_transferred && r.active_transfer && r.active_transfer.owner_company_id === currentCompanyId;
            const _isTransferredReceiver = currentCompanyId && r.is_transferred && r.active_transfer && r.active_transfer.executing_company_id === currentCompanyId;
            
            // L'entreprise émettrice (A) ne peut PAS assigner/modifier une course transférée acceptée
            // Seule l'entreprise receveuse (B) peut la gérer
            const canManageReservation = !isTransferredSender || status === 'pending';

            // Vérifier si c'est un retour sans heure définie (à confirmer)
            // Utiliser le champ time_confirmed pour déterminer si l'heure est à confirmer
            // Conservé pour référence future (géré par ReservationActions)
            const _needsTimeConfirmation =
              isReturn && (r.time_confirmed === false || !r.scheduled_time);

            return (
              <tr key={r.id} onClick={() => onRowClick?.(r)} className={styles.tableRow}>
                <td className={styles.clientCell}>{r.client?.full_name || r.client_name}</td>
                <td>{renderBookingDateTime(r)}</td>
                <td className={styles.locationCell}>
                  <div>
                    <strong>De:</strong> {r.pickup_location}
                  </div>
                  <div>
                    <strong>À:</strong> {r.dropoff_location}
                  </div>
                </td>
                <td>{Number(r.amount || 0).toFixed(2)} CHF</td>
                <td>
                  {/* 3. Utiliser les badges de statut */}
                  <span className={`${styles.statusBadge} ${styles[status] || ''}`}>
                    {(r.status || '').replace('_', ' ') || status}
                  </span>
                  {/* ✅ Badge transfert partenaire avec direction */}
                  {r.is_transferred && r.active_transfer && (() => {
                    // Déterminer si je suis l'émetteur (A) ou le receveur (B)
                    // Après acceptation, company_id est mis à jour, donc on utilise active_transfer
                    const isSender = currentCompanyId && r.active_transfer.owner_company_id === currentCompanyId;
                    const isReceiver = currentCompanyId && r.active_transfer.executing_company_id === currentCompanyId;
                    
                    let direction = '';
                    let partnerName = '';
                    
                    if (isSender) {
                      direction = 'à';
                      // Utiliser active_transfer pour obtenir le nom de l'entreprise receveuse
                      partnerName = r.active_transfer.executing_company_name || r.executing_company_name || 'partenaire';
                    } else if (isReceiver) {
                      direction = 'de';
                      // Utiliser active_transfer pour obtenir le nom de l'entreprise émettrice (pas company_name qui devient B après acceptation)
                      partnerName = r.active_transfer.owner_company_name || 'partenaire';
                    } else {
                      // Fallback si currentCompanyId n'est pas fourni
                      direction = 'vers';
                      partnerName = r.executing_company_name || r.company_name || 'partenaire';
                    }
                    
                    return (
                      <span 
                        className={styles.transferBadge}
                        title={`Transférée ${direction} ${partnerName}`}
                      >
                        🔄 Transférée
                      </span>
                    );
                  })()}
                </td>
                <td
                  className={styles.actionsCell}
                  onClick={(e) => e.stopPropagation()} // Empêche d'ouvrir le modal en cliquant sur un bouton
                >
                  {/* --- 4. Logique des boutons d'action simplifiée --- */}

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
                  ) : !canManageReservation ? (
                    /* ✅ Entreprise émettrice : Lecture seule après transfert accepté */
                    <span
                      style={{
                        color: '#6b7280',
                        fontSize: '0.85rem',
                        fontStyle: 'italic',
                      }}
                      title="Cette course est gérée par l'entreprise partenaire"
                    >
                      👁️ Lecture seule
                    </span>
                  ) : (
                    <>
                      {/* B) Courses PENDING => Toujours afficher Accepter/Rejeter */}
                      {status === 'pending' && (
                        <>
                          <button
                            onClick={() => onAccept?.(r.id)}
                            title={r.is_transferred ? "Accepter (récupère ou prend en charge)" : "Accepter"}
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

                      {/* Actions centralisées : Transférer pour pending, autres actions pour accepted/assigned */}
                      <ReservationActions
                        reservation={r}
                        onSchedule={onSchedule}
                        onDispatchNow={onDispatchNow}
                        onAssign={onAssign}
                        onEdit={onEdit}
                        onTransfer={onTransfer}
                        onDelete={onDelete}
                        hideAssign={hideAssign}
                        hideSchedule={status === 'pending' ? true : hideSchedule}
                        hideUrgent={status === 'pending' ? true : hideUrgent}
                        hideEdit={status === 'pending' ? true : hideEdit}
                        hideTransfer={hideTransfer}
                        hideDelete={status === 'pending' ? true : hideDelete}
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

export default React.memo(ReservationTable);
