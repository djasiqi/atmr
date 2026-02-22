// src/pages/company/Dashboard/components/DispatchTable.jsx
import React, { useState, useCallback } from 'react';
import styles from './ReservationTable.module.css';
import { FiCheckCircle, FiXCircle, FiAlertTriangle, FiRefreshCw, FiEye, FiClock } from 'react-icons/fi';
import ReservationActions from '../../../../components/reservations/ReservationActions';
import DriverInlineSelect from '../../Dispatch/components/DriverInlineSelect';

// V11: Cle unifiee
const getDispatchKey = (d) => d.booking_id ?? d.id;

// V16: Statuts FR — aligne avec STATUS_LABELS de ReservationTable.jsx (accents corrects)
export const STATUS_FR = {
  pending: 'En attente',
  accepted: 'Acceptée',
  assigned: 'Assignée',
  en_route: 'En route',
  in_progress: 'En cours',
  completed: 'Terminée',
  canceled: 'Annulée',
  cancelled: 'Annulée',
  rejected: 'Refusée',
  no_show: 'Non présenté',
  return_completed: 'Retour terminé',
};

// V7: Statut composite — coherent avec le dashboard
const getCompositeStatus = (r) => {
  const status = r.status?.toLowerCase() || 'unknown';
  return STATUS_FR[status] || (r.status || '').replace('_', ' ') || status;
};

// V6: Retards 3 niveaux (seuils etendus CT1: <=5 / 5-15 / >15)
const getDelayLevel = (minutes) => {
  if (!minutes || minutes <= 0) return null;
  if (minutes <= 5) return 'light';
  if (minutes <= 15) return 'moderate';
  return 'critical';
};

// Detecter si un retour necessite confirmation d'heure avant assignation
const checkNeedsTimeConfirmation = (r) => {
  const isReturn = !!(r.is_return || r.booking_type === 'return' || r.type === 'return');
  if (!isReturn) return false;

  const hasScheduledTime = !!r.scheduled_time;
  const timeConfirmed = r.time_confirmed === true;

  let isDefaultTime = false;
  if (r.scheduled_time) {
    const timeStr = r.scheduled_time.toString();
    isDefaultTime = timeStr.includes('T00:00:00') || timeStr.includes(' 00:00:00');
  }

  return !timeConfirmed || !hasScheduledTime || isDefaultTime;
};

// V17: Hierarchie visuelle stricte danger > warning
const getRowPriority = (d, delayMap) => {
  const key = getDispatchKey(d);
  const delay = delayMap?.[key]?.minutes || 0;
  const isUnassigned = !d.driver_id && !d.driver;
  if (delay > 15) return 'critical';
  if (delay > 5) return 'moderate';
  if (delay > 0) return 'light';
  if (isUnassigned) return 'unassigned';
  return 'normal';
};

const ROW_PRIORITY_CLASS = {
  critical: styles.rowDelayedCritical || styles.rowDelayed,
  moderate: styles.rowDelayedModerate || styles.rowSlightDelay,
  light: styles.rowDelayedLight || styles.rowReasonableDelay,
  unassigned: styles.rowUnassigned,
  normal: '',
};

const DELAY_BADGE_CLASS = {
  light: styles.delayBadgeLight || styles.delayBadgeReasonable,
  moderate: styles.delayBadgeModerate,
  critical: styles.delayBadgeCritical,
};

/**
 * Tableau Dispatch refactore (Zone D)
 * 6 colonnes : Client, Heure, Trajet, Chauffeur, Statut, Actions
 */
const DispatchTable = ({
  reservations = [],
  dispatches,
  delays = [],
  delayMap: externalDelayMap,
  onRowClick,
  onAccept,
  onReject,
  onAssign,
  onAssignDirect,
  onTransfer,
  onDelete,
  onSchedule,
  onDispatchNow,
  hideSchedule = false,
  hideUrgent = false,
  hideEdit = false,
  hideDelete = false,
  currentCompanyId,
  activeDrivers = [],
  autoOpenId = null,
  onAutoOpenReset: _onAutoOpenReset,
}) => {
  const [localAutoOpenId, _setLocalAutoOpenId] = useState(null);
  const effectiveAutoOpenId = autoOpenId ?? localAutoOpenId;

  const data = dispatches || reservations || [];

  // Construire delayMap interne si pas fourni en prop (compatibilite)
  const delayMap = externalDelayMap || (() => {
    const map = {};
    if (delays && delays.length > 0) {
      delays.forEach((d) => {
        const key = d.booking_id ?? d.id;
        map[key] = {
          minutes: Math.round(d.delay_minutes || d.pickup_delay_minutes || d.dropoff_delay_minutes || 0),
          severity: d.delay_severity || 'reasonable',
        };
      });
    }
    return map;
  })();

  // Formater heure
  const formatTime = useCallback((timeString) => {
    if (!timeString) return '\u2014';
    const date = new Date(timeString);
    if (isNaN(date.getTime())) return '\u2014';
    return date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
  }, []);

  // Handler clic sur ligne -> ouvre le panel lateral de details
  const handleRowClick = useCallback((r) => {
    onRowClick?.(r);
  }, [onRowClick]);

  // Obtenir le nom du chauffeur
  const getDriverName = (r) => {
    return r.driver?.full_name ||
      r.driver?.name ||
      r.driver?.username ||
      r.assignment?.driver?.full_name ||
      r.assignment?.driver?.name ||
      null;
  };

  return (
    <div className={styles.tableContainer}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Client</th>
            <th>Heure</th>
            <th>Trajet</th>
            <th>Chauffeur</th>
            <th>Statut</th>
            <th className={styles.actionsCell}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {data.map((r) => {
            const status = r.status?.toLowerCase() || 'unknown';
            const key = getDispatchKey(r);
            const priority = getRowPriority(r, delayMap);
            const delayInfo = delayMap[key];
            const delayMinutes = delayInfo?.minutes || 0;
            const delayLevel = getDelayLevel(delayMinutes);
            const driverName = getDriverName(r);
            const needsTimeConfirmation = checkNeedsTimeConfirmation(r);

            const noActionStatuses = ['canceled', 'cancelled', 'completed', 'return_completed', 'rejected', 'no_show'];
            const hasActions = !noActionStatuses.includes(status);

            const isTransferredSender = currentCompanyId && r.is_transferred && r.active_transfer && r.active_transfer.owner_company_id === currentCompanyId;
            const canManageReservation = !isTransferredSender || status === 'pending';

            return (
              <tr
                key={key}
                onClick={(e) => handleRowClick(r, e)}
                className={`${styles.tableRow} ${ROW_PRIORITY_CLASS[priority] || ''} ${onRowClick ? styles.rowClickable : ''}`}
              >
                {/* Colonne Client */}
                <td className={styles.clientCell}>
                  <span className={styles.clientName}>{r.client?.full_name || r.client_name || '\u2014'}</span>
                  {r.client?.institution_name && (
                    <span className={styles.clientInstitution}>{r.client.institution_name}</span>
                  )}
                  {r.is_return && <span className={styles.clientSub}>Retour</span>}
                  {r.is_transferred && <span className={styles.clientSub}>
                    <FiRefreshCw size={10} /> Transfert
                  </span>}
                </td>

                {/* Colonne Heure + badge retard */}
                <td className={styles.timeCell}>
                  {needsTimeConfirmation ? (
                    <span className={styles.timeToDefine} title="Heure de retour a definir">
                      <FiClock size={12} /> A definir
                    </span>
                  ) : (
                    <>
                      <span className={styles.timeBold}>{formatTime(r.scheduled_time)}</span>
                      {delayLevel && (
                        <span
                          className={`${styles.delayBadge} ${DELAY_BADGE_CLASS[delayLevel] || ''}`}
                          title={`Retard de ${delayMinutes} min`}
                        >
                          {delayLevel === 'critical' && <FiAlertTriangle size={10} />}
                          +{delayMinutes}min
                        </span>
                      )}
                    </>
                  )}
                </td>

                {/* Colonne Trajet — aligne avec ReservationTable (locationCell) */}
                <td className={styles.locationCell}>
                  <div className={styles.locationRow}>
                    <span className={`${styles.locationDot} ${styles.locationDotPickup}`} />
                    <span className={styles.locationText} title={r.pickup_location}>
                      {r.pickup_location || '\u2014'}
                    </span>
                  </div>
                  <div className={styles.locationRow}>
                    <span className={`${styles.locationDot} ${styles.locationDotDropoff}`} />
                    <span className={styles.locationText} title={r.dropoff_location}>
                      {r.dropoff_location || '\u2014'}
                    </span>
                  </div>
                </td>

                {/* Colonne Chauffeur V3: dropdown inline (sauf retour sans heure) */}
                <td
                  className={styles.driverCellInline}
                  onClick={(e) => e.stopPropagation()}
                >
                  {needsTimeConfirmation ? (
                    <span className={styles.timeRequiredHint} title="Definir l'heure de retour avant d'assigner">
                      <FiClock size={11} /> Heure requise
                    </span>
                  ) : onAssignDirect ? (
                    <DriverInlineSelect
                      drivers={activeDrivers}
                      reservationId={key}
                      onAssign={onAssignDirect}
                      currentDriverName={driverName}
                      autoOpen={effectiveAutoOpenId === key}
                      disabled={false}
                    />
                  ) : (
                    <span className={driverName ? styles.driverNameText : styles.unassignedText}>
                      {driverName || (r.driver_id ? `Chauffeur #${r.driver_id}` : 'Non assigne')}
                    </span>
                  )}
                </td>

                {/* Colonne Statut FR V7 */}
                <td>
                  <span className={`${styles.statusBadge} ${styles[status] || ''}`}>
                    {getCompositeStatus(r)}
                  </span>
                  {r.is_transferred && r.active_transfer && (() => {
                    const isSender = currentCompanyId && r.active_transfer.owner_company_id === currentCompanyId;
                    const isReceiver = currentCompanyId && r.active_transfer.executing_company_id === currentCompanyId;
                    let direction = '';
                    let partnerName = '';
                    if (isSender) {
                      direction = 'a';
                      partnerName = r.active_transfer.executing_company_name || r.executing_company_name || 'partenaire';
                    } else if (isReceiver) {
                      direction = 'de';
                      partnerName = r.active_transfer.owner_company_name || 'partenaire';
                    } else {
                      direction = 'vers';
                      partnerName = r.executing_company_name || r.company_name || 'partenaire';
                    }
                    return (
                      <span
                        className={styles.transferBadge}
                        title={`Transferee ${direction} ${partnerName}`}
                      >
                        <FiRefreshCw size={10} /> Transferee
                      </span>
                    );
                  })()}
                </td>

                {/* Colonne Actions */}
                <td
                  className={styles.actionsCell}
                  onClick={(e) => e.stopPropagation()}
                >
                  {!hasActions ? (
                    <span className={styles.noActionLabel}>Aucune action</span>
                  ) : !canManageReservation ? (
                    <span className={styles.readOnlyLabel} title="Cette course est geree par l'entreprise partenaire">
                      <FiEye size={12} /> Lecture seule
                    </span>
                  ) : (
                    <>
                      {status === 'pending' && (
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
                      <ReservationActions
                        reservation={r}
                        onSchedule={onSchedule}
                        onDispatchNow={onDispatchNow}
                        onAssign={onAssign}
                        onTransfer={onTransfer}
                        onDelete={onDelete}
                        hideAssign={!!onAssignDirect}
                        hideSchedule={status === 'pending' ? true : hideSchedule}
                        hideUrgent={status === 'pending' ? true : hideUrgent}
                        hideEdit={status === 'pending' ? true : hideEdit}
                        hideTransfer={false}
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

export default React.memo(DispatchTable);
