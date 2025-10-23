// src/pages/company/Dashboard/components/ReservationTable.jsx
import React from 'react';
// 1. Importer le nouveau fichier de style et les icônes
import styles from './ReservationTable.module.css';
import { FiCheckCircle, FiXCircle, FiUserPlus, FiTrash2, FiClock, FiZap } from 'react-icons/fi';
import { renderBookingDateTime } from '../../../../utils/formatDate';
const ReservationTable = ({
  reservations,
  onRowClick,
  onAccept,
  onReject,
  onAssign,
  onDelete,
  onSchedule,
  onDispatchNow, // 2. Nouvelle prop pour l'urgence
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
            const isDeletable = deletableStatuses.includes(status);
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
            const needsTimeConfirmation =
              isReturn && (r.time_confirmed === false || !r.scheduled_time);

            return (
              <tr key={r.id} onClick={() => onRowClick?.(r)} className={styles.tableRow}>
                <td className={styles.clientCell}>{r.client?.full_name || r.customer_name}</td>
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
                  ) : (
                    <>
                      {/* A) Retour avec heure à confirmer => Planifier + Urgent + Assigner + Supprimer */}
                      {needsTimeConfirmation ? (
                        <>
                          <button
                            onClick={() => onSchedule?.(r)}
                            title="Planifier l'heure de retour"
                            className={styles.actionButton}
                          >
                            <FiClock />
                          </button>
                          <button
                            onClick={() => onDispatchNow?.(r)}
                            title="Urgent (+15 min)"
                            className={`${styles.actionButton} ${styles.urgentButton || ''}`}
                          >
                            <FiZap />
                          </button>
                          <button
                            onClick={() => onAssign?.(r)}
                            title="Assigner un chauffeur"
                            className={styles.actionButton}
                          >
                            <FiUserPlus />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              console.log('🗑️ Suppression de la réservation:', r.id);
                              if (!window.confirm(`Supprimer la réservation #${r.id} ?`)) return;
                              onDelete?.(r);
                            }}
                            title="Annuler/Supprimer"
                            className={`${styles.actionButton} ${styles.deleteButton}`}
                          >
                            <FiTrash2 />
                          </button>
                        </>
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

                          {/* C) Courses ACCEPTED/ASSIGNED => Assigner */}
                          {['accepted', 'assigned'].includes(status) && !needsTimeConfirmation && (
                            <button
                              onClick={() => onAssign?.(r)}
                              title="Assigner un chauffeur"
                              className={styles.actionButton}
                            >
                              <FiUserPlus />
                            </button>
                          )}

                          {/* D) Bouton Supprimer pour les autres cas */}
                          {isDeletable && !needsTimeConfirmation && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                console.log('🗑️ Suppression de la réservation:', r.id);
                                if (!window.confirm(`Supprimer la réservation #${r.id} ?`)) return;
                                onDelete?.(r);
                              }}
                              title="Supprimer/Annuler la réservation"
                              className={`${styles.actionButton} ${styles.deleteButton}`}
                            >
                              <FiTrash2 />
                            </button>
                          )}
                        </>
                      )}
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
