// src/pages/company/Dashboard/components/ReservationTable.jsx
import React from "react";
// 1. Importer le nouveau fichier de style et les icônes
import styles from "./ReservationTable.module.css";
import { FiCheckCircle, FiXCircle, FiUserPlus, FiTrash2, FiClock, FiZap } from 'react-icons/fi';
import { renderBookingDateTime } from "../../../../utils/formatDate";
const ReservationTable = ({
  reservations,
  onRowClick,
  onAccept,
  onReject,
  onAssign,
  onDelete,
  onSchedule,
  onDispatchNow // 2. Nouvelle prop pour l'urgence
}) => {

  const deletableStatuses = ["pending", "accepted", "assigned"];

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
            const isReturn = !!r.is_return; // besoin côté back/serialize
            // Si pas d'UTC machine, on considère que l'heure n'est pas planifiée
            const isTimeMissing = !r.scheduled_time;
            return (
              <tr
                key={r.id}
                onClick={() => onRowClick?.(r)}
                className={styles.tableRow}
              >
                <td>{r.client?.full_name || r.customer_name}</td>
                <td>{renderBookingDateTime(r)}</td>
                <td>
                  <div><strong>De:</strong> {r.pickup_location}</div>
                  <div><strong>À:</strong> {r.dropoff_location}</div>
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
                  onClick={e => e.stopPropagation()} // Empêche d'ouvrir le modal en cliquant sur un bouton
                >
                  {/* --- 4. Logique des boutons d'action simplifiée --- */}

                  {/* A) Si retour SANS horaire => proposer Planifier & Urgent */}
                  {isReturn && isTimeMissing && (
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
                    </>
                  )}
                  
                  {status === 'pending' && (
                    <>
                      <button onClick={() => onAccept?.(r.id)} title="Accepter" className={`${styles.actionButton} ${styles.acceptButton}`}>
                        <FiCheckCircle />
                      </button>
                      <button onClick={() => onReject?.(r.id)} title="Rejeter" className={`${styles.actionButton} ${styles.rejectButton}`}>
                        <FiXCircle />
                      </button>
                    </>
                  )}

                  {['accepted', 'assigned'].includes(status) && (
                    <button onClick={() => onAssign?.(r)} title="Assigner un chauffeur" className={styles.actionButton}>
                      <FiUserPlus />
                    </button>
                  )}
                  
                  {/* 5. Intégration du bouton Supprimer */}
                  {isDeletable && (
                     <button onClick={() => onDelete?.(r)} title="Supprimer la réservation" className={`${styles.actionButton} ${styles.deleteButton}`}>
                       <FiTrash2 />
                     </button>
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