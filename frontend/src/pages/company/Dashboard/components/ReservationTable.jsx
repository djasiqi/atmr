// src/pages/company/Dashboard/components/ReservationTable.jsx
import React, { useState } from 'react';
import { FiCheckCircle, FiXCircle, FiInbox, FiChevronDown } from 'react-icons/fi';
import styles from './ReservationTable.module.css';
import { renderBookingDateTime } from '../../../../utils/formatDate';
import { formatDelay } from '../../../../utils/formatDelay';
import ReservationActions from '../../../../components/reservations/ReservationActions';

const STATUS_LABELS = {
  pending: 'En attente',
  accepted: 'Acceptée',
  assigned: 'Assignée',
  en_route: 'En route',
  in_progress: 'En cours',
  completed: 'Terminée',
  return_completed: 'Retour terminé',
  canceled: 'Annulée',
  cancelled: 'Annulée',
  rejected: 'Refusée',
  no_show: 'Non présenté',
};

const DISPLAY_INCREMENT = 50;

const getDelayRowClass = (delayMinutes) => {
  if (!delayMinutes || delayMinutes <= 0) return '';
  if (delayMinutes >= 15) return 'rowDelayed';
  if (delayMinutes >= 5) return 'rowSlightDelay';
  return 'rowReasonableDelay';
};

const ReservationTable = ({
  reservations,
  loading,
  delays,
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
  currentCompanyId,
}) => {
  const deletableStatuses = ['pending', 'accepted', 'assigned'];
  const delaysMap = delays || {};

  const [displayLimit, setDisplayLimit] = useState(DISPLAY_INCREMENT);

  if (!loading && (!reservations || reservations.length === 0)) {
    return (
      <div className={styles.emptyState}>
        <FiInbox className={styles.emptyIcon} size={40} />
        <p className={styles.emptyTitle}>Aucune course dans cette catégorie</p>
        <p className={styles.emptySubtitle}>Les nouvelles courses apparaitront ici automatiquement</p>
      </div>
    );
  }

  const displayedReservations = reservations.slice(0, displayLimit);
  const hasMore = reservations.length > displayLimit;
  const remainingCount = reservations.length - displayLimit;

  const renderRow = (r) => {
    const status = r.status?.toLowerCase() || 'unknown';
    const _isDeletable = deletableStatuses.includes(status);
    const isReturn = !!r.is_return;
    const noActionStatuses = ['canceled', 'cancelled', 'completed', 'return_completed', 'rejected', 'no_show'];
    const hasActions = !noActionStatuses.includes(status);
    const isTransferredSender = currentCompanyId && r.is_transferred && r.active_transfer && r.active_transfer.owner_company_id === currentCompanyId;
    const _isTransferredReceiver = currentCompanyId && r.is_transferred && r.active_transfer && r.active_transfer.executing_company_id === currentCompanyId;
    const canManageReservation = !isTransferredSender || status === 'pending';
    const _needsTimeConfirmation = isReturn && (r.time_confirmed === false || !r.scheduled_time);
    const bookingDelay = delaysMap[r.id];
    const delayMinutes = bookingDelay?.delay_minutes;
    const delayRowClass = getDelayRowClass(delayMinutes);

    return { status, hasActions, canManageReservation, delayMinutes, delayRowClass };
  };

  return (
    <>
      {/* Desktop table */}
      <div className={styles.tableContainer}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Client</th>
              <th>Date / Heure</th>
              <th>Trajet</th>
              <th>Montant</th>
              <th>Statut</th>
              <th className={styles.actionsCell}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {displayedReservations.map((r, index) => {
              const { status, hasActions, canManageReservation, delayMinutes, delayRowClass } = renderRow(r);

              return (
                <tr
                  key={r.id}
                  data-tour-id={status === 'pending' && index === 0 ? 'pending-row-overview' : undefined}
                  onClick={() => onRowClick?.(r)}
                  className={`${styles.tableRow} ${delayRowClass ? styles[delayRowClass] : ''}`}
                >
                  <td className={styles.clientCell}>
                    <span className={styles.clientName}>
                      {r.client?.full_name || r.client_name}
                    </span>
                    {r.client?.institution_name && (
                      <span className={styles.clientInstitution}>
                        {r.client.institution_name}
                      </span>
                    )}
                    {delayMinutes > 0 && formatDelay(delayMinutes) && (
                      <span className={`${styles.delayBadge} ${
                        delayMinutes >= 15 ? styles.delayBadgeCritical
                          : delayMinutes >= 5 ? styles.delayBadgeModerate
                          : styles.delayBadgeReasonable
                      }`}>
                        {formatDelay(delayMinutes)}
                      </span>
                    )}
                  </td>
                  <td className={styles.dateCell}>
                    {renderBookingDateTime(r)}
                  </td>
                  <td className={styles.locationCell}>
                    <div className={styles.locationRow}>
                      <span className={`${styles.locationDot} ${styles.locationDotPickup}`} />
                      <span className={styles.locationText} title={r.pickup_location}>{r.pickup_location}</span>
                    </div>
                    <div className={styles.locationRow}>
                      <span className={`${styles.locationDot} ${styles.locationDotDropoff}`} />
                      <span className={styles.locationText} title={r.dropoff_location}>{r.dropoff_location}</span>
                    </div>
                  </td>
                  <td>
                    <span className={styles.amountValue}>
                      {Number(r.amount || 0).toFixed(2)}
                    </span>
                    <span className={styles.amountCurrency}> CHF</span>
                    {(() => {
                      const meta = r.metadata_json || {};
                      const billingStatus = meta.billing_resolution_status;
                      if (!billingStatus) return null;
                      const isFailed = billingStatus.startsWith('failed');
                      return (
                        <span
                          className={`${styles.billingBadge} ${isFailed ? styles.billingFailed : styles.billingResolved}`}
                          title={isFailed
                            ? `Destinataire à compléter (${billingStatus.replace('failed_', '').replace(/_/g, ' ')})`
                            : 'Destinataire de facturation résolu'
                          }
                        >
                          {isFailed ? 'Dest. manquant' : 'Dest. résolu'}
                        </span>
                      );
                    })()}
                  </td>
                  <td>
                    <span className={`${styles.statusBadge} ${styles[status] || ''}`}>
                      {STATUS_LABELS[status] || (r.status || '').replace('_', ' ') || status}
                    </span>
                    {r.is_transferred && r.active_transfer && (() => {
                      const isSender = currentCompanyId && r.active_transfer.owner_company_id === currentCompanyId;
                      const isReceiver = currentCompanyId && r.active_transfer.executing_company_id === currentCompanyId;
                      let direction = '';
                      let partnerName = '';
                      if (isSender) {
                        direction = 'à';
                        partnerName = r.active_transfer.executing_company_name || r.executing_company_name || 'partenaire';
                      } else if (isReceiver) {
                        direction = 'de';
                        partnerName = r.active_transfer.owner_company_name || 'partenaire';
                      } else {
                        direction = 'vers';
                        partnerName = r.executing_company_name || r.company_name || 'partenaire';
                      }
                      return (
                        <span className={styles.transferBadge} title={`Transférée ${direction} ${partnerName}`}>
                          Transférée
                        </span>
                      );
                    })()}
                  </td>
                  <td className={styles.actionsCell} onClick={(e) => e.stopPropagation()}>
                    {!hasActions ? (
                      <span className={styles.noActionLabel}>Terminée</span>
                    ) : !canManageReservation ? (
                      <span className={styles.noActionLabel} title="Cette course est gérée par l'entreprise partenaire">Lecture seule</span>
                    ) : (
                      <>
                        {status === 'pending' && (
                          <>
                            <button
                              data-tour-id="pending-accept-action"
                              onClick={() => onAccept?.(r.id)}
                              title={r.is_transferred ? "Accepter (prendre en charge)" : "Accepter"}
                              className={`${styles.actionButton} ${styles.acceptButton}`}
                            >
                              <FiCheckCircle size={16} />
                            </button>
                            <button onClick={() => onReject?.(r.id)} title="Rejeter" className={`${styles.actionButton} ${styles.rejectButton}`}>
                              <FiXCircle size={16} />
                            </button>
                          </>
                        )}
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

      {/* Mobile cards */}
      <div className={styles.mobileCards}>
        {displayedReservations.map((r) => {
          const { status, hasActions, canManageReservation, delayMinutes } = renderRow(r);

          return (
            <div
              key={r.id}
              className={styles.mobileCard}
              onClick={() => onRowClick?.(r)}
            >
              <div className={styles.mobileCardHeader}>
                <div className={styles.mobileCardTitleGroup}>
                  <span className={styles.mobileCardTitle}>
                    {r.client?.full_name || r.client_name}
                  </span>
                  {r.client?.institution_name && (
                    <span className={styles.mobileCardInstitution}>{r.client.institution_name}</span>
                  )}
                </div>
                <span className={`${styles.statusBadge} ${styles[status] || ''}`}>
                  {STATUS_LABELS[status] || status}
                </span>
              </div>
              <div className={styles.mobileCardBody}>
                <div className={styles.mobileCardRow}>
                  <span className={styles.mobileCardLabel}>Horaire</span>
                  <span className={styles.mobileCardValue}>{renderBookingDateTime(r)}</span>
                </div>
                <div className={styles.mobileCardRoute}>
                  <div className={styles.locationRow}>
                    <span className={`${styles.locationDot} ${styles.locationDotPickup}`} />
                    <span className={styles.mobileCardRouteText}>{r.pickup_location || '-'}</span>
                  </div>
                  <div className={styles.locationRow}>
                    <span className={`${styles.locationDot} ${styles.locationDotDropoff}`} />
                    <span className={styles.mobileCardRouteText}>{r.dropoff_location || '-'}</span>
                  </div>
                </div>
                <div className={styles.mobileCardRow}>
                  <span className={styles.mobileCardLabel}>Montant</span>
                  <span className={styles.mobileCardValue}>
                    <span className={styles.amountValue}>{Number(r.amount || 0).toFixed(2)}</span>
                    <span className={styles.amountCurrency}> CHF</span>
                  </span>
                </div>
                {delayMinutes > 0 && formatDelay(delayMinutes) && (
                  <div className={styles.mobileCardRow}>
                    <span className={styles.mobileCardLabel}>Retard</span>
                    <span className={`${styles.delayBadge} ${
                      delayMinutes >= 15 ? styles.delayBadgeCritical
                        : delayMinutes >= 5 ? styles.delayBadgeModerate
                        : styles.delayBadgeReasonable
                    }`}>
                      {formatDelay(delayMinutes)}
                    </span>
                  </div>
                )}
              </div>
              {hasActions && canManageReservation && (
                <div className={styles.mobileCardActions} onClick={(e) => e.stopPropagation()}>
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
                </div>
              )}
            </div>
          );
        })}
      </div>

      {hasMore && (
        <div className={styles.loadMore}>
          <button
            className={styles.loadMoreBtn}
            onClick={() => setDisplayLimit((prev) => prev + DISPLAY_INCREMENT)}
          >
            <FiChevronDown size={16} />
            Afficher {Math.min(DISPLAY_INCREMENT, remainingCount)} courses supplémentaires
            <span className={styles.loadMoreCount}>
              ({remainingCount} restante{remainingCount !== 1 ? 's' : ''})
            </span>
          </button>
        </div>
      )}
    </>
  );
};

export default React.memo(ReservationTable);
