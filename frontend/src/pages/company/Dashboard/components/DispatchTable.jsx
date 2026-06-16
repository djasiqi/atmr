// src/pages/company/Dashboard/components/DispatchTable.jsx
import React, { useState, useCallback } from 'react';
import styles from './ReservationTable.module.css';
import { FiCheckCircle, FiXCircle, FiAlertTriangle, FiEye, FiClock } from 'react-icons/fi';
import ReservationActions from '../../../../components/reservations/ReservationActions';
import BookingIdentityCell from '../../../../components/booking/BookingIdentityCell';
import BookingTripBadges from '../../../../components/booking/BookingTripBadges';
import BookingStatusBadge from '../../../../components/booking/BookingStatusBadge';
import {
  isReturnLegNeedingTime,
  needsTimeBeforeDriverAssign,
} from '../../../../utils/bookingScheduling';
import BookingScheduleCell from '../../../../components/booking/BookingScheduleCell';
import DriverInlineSelect from '../../Dispatch/components/DriverInlineSelect';
import { pickupArrivalHint } from '../../../../utils/formatPickupEta';
import {
  getDispatchRowDelayInfo,
  normalizeDispatchDelayMapKey,
} from '../../../../utils/dispatchDelayMapKey';

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

// V7: Statut composite — conservé pour exports / compatibilité externe
export const getCompositeStatus = (r) => {
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

// V17: Hierarchie visuelle stricte danger > warning
const getRowPriority = (d, delayMap) => {
  const key = getDispatchKey(d);
  const delay =
    getDispatchRowDelayInfo(delayMap, d)?.minutes ?? delayMap?.[key]?.minutes ?? 0;
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

const EMPTY_ROWS = [];

/**
 * Tableau Dispatch refactore (Zone D)
 * 6 colonnes : Passager, Heure, Trajet, Chauffeur, Statut, Actions
 */
const DispatchTable = ({
  reservations = EMPTY_ROWS,
  dispatches = null,
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

  const data = React.useMemo(
    () => dispatches || reservations || EMPTY_ROWS,
    [dispatches, reservations]
  );
  const firstUnassignedIndex = data.findIndex((r) => !r.driver_id && !r.driver);
  const fallbackDriverAnchorIndex = firstUnassignedIndex >= 0 ? firstUnassignedIndex : 0;
  const firstStatusAnchorIndex = data.length > 0 ? 0 : -1;
  const firstRowClickableIndex = data.length > 0 ? 0 : -1;

  // Construire delayMap interne si pas fourni en prop (compatibilite)
  const delayMap = externalDelayMap || (() => {
    const map = {};
    if (delays && delays.length > 0) {
      delays.forEach((d) => {
        const key = normalizeDispatchDelayMapKey(d.booking_id ?? d.id);
        if (key == null) return;
        map[key] = {
          minutes: Math.round(d.delay_minutes || d.pickup_delay_minutes || d.dropoff_delay_minutes || 0),
          severity: d.delay_severity || 'reasonable',
          current_eta: d.current_eta || null,
          pickup_eta: d.pickup_eta || d.current_eta || null,
        };
      });
    }
    return map;
  })();

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

  const routeGroupSizes = React.useMemo(() => {
    const sizes = {};
    data.forEach((r) => {
      const g = r?.route_group_id;
      if (g) sizes[g] = (sizes[g] || 0) + 1;
    });
    return sizes;
  }, [data]);

  return (
    <div className={styles.tableContainer} data-tour-id="dispatch-table">
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Passager</th>
            <th>Heure</th>
            <th>Trajet</th>
            <th>Chauffeur</th>
            <th>Statut</th>
            <th className={styles.actionsCell}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {data.map((r, index) => {
            const status = r.status?.toLowerCase() || 'unknown';
            const key = getDispatchKey(r);
            const priority = getRowPriority(r, delayMap);
            const delayInfo = getDispatchRowDelayInfo(delayMap, r) ?? delayMap?.[key];
            const delayMinutes = delayInfo?.minutes || 0;
            const delayLevel = getDelayLevel(delayMinutes);
            const etaIso =
              delayInfo?.pickup_eta ||
              delayInfo?.current_eta ||
              r?.assignment?.estimated_pickup_arrival ||
              r?.assignment?.eta_pickup_at ||
              r?.assignment?.pickup_eta ||
              null;
            const etaStatuses = ['accepted', 'assigned', 'en_route'];
            const pickupArrivalLabel =
              etaStatuses.includes(status) && etaIso ? pickupArrivalHint(etaIso) : null;
            const driverName = getDriverName(r);
            const needsTimeBeforeAssign = needsTimeBeforeDriverAssign(r);
            const isReturnUnscheduled = isReturnLegNeedingTime(r);

            const noActionStatuses = ['canceled', 'cancelled', 'completed', 'return_completed', 'rejected', 'no_show'];
            const hasActions = !noActionStatuses.includes(status);

            const needsTimeScheduling = needsTimeBeforeAssign;
            const timeRequiredHintTitle = isReturnUnscheduled
              ? "Definir l'heure de retour avant d'assigner"
              : "Definir l'heure du trajet avant d'assigner";
            const timeToDefineTitle = isReturnUnscheduled
              ? 'Heure de retour a definir'
              : 'Heure du trajet a definir';

            const isTransferredSender = currentCompanyId && r.is_transferred && r.active_transfer && r.active_transfer.owner_company_id === currentCompanyId;
            const canManageReservation = !isTransferredSender || status === 'pending';

            return (
              <tr
                key={key}
                onClick={(e) => handleRowClick(r, e)}
                className={`${styles.tableRow} ${ROW_PRIORITY_CLASS[priority] || ''} ${onRowClick ? styles.rowClickable : ''}`}
                data-tour-id={index === firstRowClickableIndex ? 'dispatch-row-clickable' : undefined}
              >
                <td className={styles.clientCell}>
                  <BookingIdentityCell booking={r} />
                  <BookingTripBadges booking={r} routeGroupSizes={routeGroupSizes} />
                </td>

                {/* Colonne Heure + badge retard */}
                <td className={styles.timeCell}>
                  {needsTimeBeforeAssign ? (
                    <span className={styles.timeToDefine} title={timeToDefineTitle}>
                      <FiClock size={12} /> A definir
                    </span>
                  ) : (
                    <div className={styles.timeCellStack}>
                      <div className={styles.timePrimaryRow}>
                        <BookingScheduleCell booking={r} mode="time" className={styles.timeBold} />
                        {delayLevel && (
                          <span
                            className={`${styles.delayBadge} ${DELAY_BADGE_CLASS[delayLevel] || ''}`}
                            title={`Retard de ${delayMinutes} min`}
                          >
                            {delayLevel === 'critical' && <FiAlertTriangle size={10} />}
                            +{delayMinutes}min
                          </span>
                        )}
                      </div>
                      {pickupArrivalLabel && (
                        <span className={styles.pickupEtaHint} title={pickupArrivalLabel.title}>
                          {pickupArrivalLabel.text}
                        </span>
                      )}
                    </div>
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
                  data-tour-id={index === fallbackDriverAnchorIndex ? 'dispatch-driver-anchor' : undefined}
                >
                  {needsTimeBeforeAssign ? (
                    <span className={styles.timeRequiredHint} title={timeRequiredHintTitle}>
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
                  <span data-tour-id={index === firstStatusAnchorIndex ? 'dispatch-status-anchor' : undefined}>
                    <BookingStatusBadge status={status} />
                  </span>
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
                        needsTimeConfirmationOverride={needsTimeScheduling}
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
