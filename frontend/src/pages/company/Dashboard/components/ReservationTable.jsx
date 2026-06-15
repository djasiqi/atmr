// src/pages/company/Dashboard/components/ReservationTable.jsx
import React, { useState } from 'react';
import { FiCheckCircle, FiXCircle, FiInbox, FiChevronDown, FiClock } from 'react-icons/fi';
import styles from './ReservationTable.module.css';
import { formatDelay } from '../../../../utils/formatDelay';
import { pickupArrivalHint } from '../../../../utils/formatPickupEta';
import BookingScheduleCell from '../../../../components/booking/BookingScheduleCell';
import ReservationActions from '../../../../components/reservations/ReservationActions';
import BookingIdentityCell from '../../../../components/booking/BookingIdentityCell';
import BookingTripBadges from '../../../../components/booking/BookingTripBadges';
import BookingStatusBadge from '../../../../components/booking/BookingStatusBadge';
import { canRespondToInstitutionOffer, isInstitutionOfferExpired } from '../../../../utils/institutionOfferResponse';

const DISPLAY_INCREMENT = 50;

/** Aller lié à un retour (drapeau persisté ou relation `return_trip`). */
function reservationIsRoundTripOutbound(r) {
  if (!r || r.is_return) return false;
  return Boolean(r.is_round_trip || r.has_return);
}

function findParentBooking(allReservations, parentId) {
  if (!parentId || !Array.isArray(allReservations)) return null;
  return allReservations.find((x) => x.id === parentId) || null;
}

function findReturnBookingForOutbound(allReservations, outboundId) {
  if (!outboundId || !Array.isArray(allReservations)) return null;
  return (
    allReservations.find(
      (x) => x && x.is_return && Number(x.parent_booking_id) === Number(outboundId)
    ) || null
  );
}

/** Montants aller-retour : total sur l’aller, libellé explicite sur le retour si montant 0. */
function renderAmountCell(r, allReservations) {
  const amt = Number(r.amount || 0);
  const parent = r.is_return ? findParentBooking(allReservations, r.parent_booking_id) : null;
  const parentAmt = parent != null ? Number(parent.amount || 0) : null;

  if (r.is_return && amt === 0 && parentAmt != null && parentAmt > 0) {
    return (
      <div className={styles.amountCellStack}>
        <span
          className={styles.amountLegLabel}
          title="Le tarif est porté par la course aller ; ce segment ne facture pas en plus."
        >
          Inclus dans l&apos;aller
        </span>
        <span className={styles.amountSub}>
          {parentAmt.toFixed(2)} CHF au total (aller + retour)
        </span>
      </div>
    );
  }

  if (!r.is_return && reservationIsRoundTripOutbound(r)) {
    const ret = findReturnBookingForOutbound(allReservations, r.id);
    const retAmt = ret != null ? Number(ret.amount || 0) : 0;
    if (retAmt > 0) {
      return (
        <span>
          <span className={styles.amountValue}>{amt.toFixed(2)}</span>
          <span className={styles.amountCurrency}> CHF</span>
        </span>
      );
    }
    return (
      <div className={styles.amountCellStack}>
        <span>
          <span className={styles.amountValue}>{amt.toFixed(2)}</span>
          <span className={styles.amountCurrency}> CHF</span>
        </span>
        <span className={styles.amountSub}>Total aller-retour</span>
      </div>
    );
  }

  return (
    <span>
      <span className={styles.amountValue}>{amt.toFixed(2)}</span>
      <span className={styles.amountCurrency}> CHF</span>
    </span>
  );
}

/** Tarif d'une offre institution en attente : estimation (préférentiel/profil) ou "À définir". */
function renderOfferAmount(r) {
  const est = r.__priceEstimate;
  const amount = est ? Number(est.amount) : NaN;
  if (est && !Number.isNaN(amount) && amount > 0) {
    const sourceLabel =
      est.source === 'preferential'
        ? 'Tarif préférentiel'
        : est.source === 'profile'
          ? 'Tarif estimé selon le profil tarifaire'
          : 'Tarif estimé';
    return (
      <span className={styles.amountSub} title={sourceLabel}>
        {amount.toFixed(2)} {est.currency || 'CHF'}
      </span>
    );
  }
  return (
    <span className={styles.amountSub} title="Tarif défini à l'acceptation de la demande">
      À définir
    </span>
  );
}

/** Formate un instant absolu (ex. expiration d'offre). */
const formatInstantDateTime = (isoString) => {
  if (!isoString) return { date: '—', time: '' };
  const d = new Date(isoString);
  if (Number.isNaN(d.getTime())) return { date: '—', time: '' };
  const pad = (n) => String(n).padStart(2, '0');
  return {
    date: `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()}`,
    time: `${pad(d.getHours())}:${pad(d.getMinutes())}`,
  };
};

const getDelayRowClass = (delayMinutes) => {
  if (!delayMinutes || delayMinutes <= 0) return '';
  if (delayMinutes >= 15) return 'rowDelayed';
  if (delayMinutes >= 5) return 'rowSlightDelay';
  return 'rowReasonableDelay';
};

/**
 * Regroupe les legs d'un même parcours multi-destinations (route_group_id) de
 * façon contiguë et ordonnée par route_sequence_number, tout en conservant la
 * position globale des autres lignes (insertion à la 1re occurrence du groupe).
 */
function clusterRouteGroups(list) {
  if (!Array.isArray(list) || list.length === 0) return list;
  const byGroup = new Map();
  list.forEach((r) => {
    const g = r?.route_group_id;
    if (!g) return;
    if (!byGroup.has(g)) byGroup.set(g, []);
    byGroup.get(g).push(r);
  });
  if (byGroup.size === 0) return list;

  const seen = new Set();
  const result = [];
  list.forEach((r) => {
    const g = r?.route_group_id;
    if (!g) {
      result.push(r);
      return;
    }
    if (seen.has(g)) return;
    seen.add(g);
    const members = [...byGroup.get(g)].sort(
      (a, b) => (Number(a.route_sequence_number) || 0) - (Number(b.route_sequence_number) || 0)
    );
    result.push(...members);
  });
  return result;
}

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
  onAcceptInstitutionOffer,
  onProposeInstitutionOffer,
  onRejectInstitutionOffer,
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

  const orderedReservations = clusterRouteGroups(reservations);
  const displayedReservations = orderedReservations.slice(0, displayLimit);
  const hasMore = orderedReservations.length > displayLimit;
  const remainingCount = orderedReservations.length - displayLimit;

  // Nombre de legs par parcours multi-destinations (pour le badge "Trajet N/M").
  const routeGroupSizes = {};
  (reservations || []).forEach((r) => {
    if (r?.route_group_id) {
      routeGroupSizes[r.route_group_id] = (routeGroupSizes[r.route_group_id] || 0) + 1;
    }
  });

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
    const pickupEtaIso =
      bookingDelay?.pickup_eta ??
      r?.assignment?.estimated_pickup_arrival ??
      r?.assignment?.eta_pickup_at ??
      r?.assignment?.pickup_eta ??
      null;
    const delayRowClass = getDelayRowClass(delayMinutes);
    const showPickupEtaStatuses = ['accepted', 'assigned', 'en_route'];
    const pickupArrivalLabel =
      showPickupEtaStatuses.includes(status) && pickupEtaIso
        ? pickupArrivalHint(pickupEtaIso)
        : null;
    const isInstitutionOffer = Boolean(r.__institutionOffer);
    const offerCanRespond = isInstitutionOffer
      ? (typeof r.__offerCanRespond === 'boolean'
        ? r.__offerCanRespond
        : canRespondToInstitutionOffer(r.__offer || r))
      : true;
    const offerExpired = isInstitutionOffer
      ? Boolean(r.__offerExpired) || isInstitutionOfferExpired(r.__offer || r)
      : false;

    return {
      status,
      hasActions,
      canManageReservation,
      delayMinutes,
      delayRowClass,
      pickupArrivalLabel,
      offerCanRespond,
      offerExpired,
    };
  };

  return (
    <>
      {/* Desktop table */}
      <div className={styles.tableContainer}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Passager</th>
              <th>Date / Heure</th>
              <th>Trajet</th>
              <th>Montant</th>
              <th>Statut</th>
              <th className={styles.actionsCell}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {displayedReservations.map((r, index) => {
              const {
                status,
                hasActions,
                canManageReservation,
                delayMinutes,
                delayRowClass,
                pickupArrivalLabel,
                offerCanRespond,
                offerExpired,
              } = renderRow(r);

              return (
                <tr
                  key={r.id}
                  data-tour-id={status === 'pending' && index === 0 ? 'pending-row-overview' : undefined}
                  onClick={() => onRowClick?.(r)}
                  className={`${styles.tableRow} ${delayRowClass ? styles[delayRowClass] : ''}`}
                >
                  <td className={styles.clientCell}>
                    <BookingIdentityCell booking={r} />
                    <BookingTripBadges booking={r} routeGroupSizes={routeGroupSizes} />
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
                    <div className={styles.timeCellStack}>
                      <BookingScheduleCell booking={r} undefinedClassName={styles.pickupEtaHint} />
                      {pickupArrivalLabel && (
                        <span className={styles.pickupEtaHint} title={pickupArrivalLabel.title}>
                          {pickupArrivalLabel.text}
                        </span>
                      )}
                    </div>
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
                    {r.__institutionOffer ? (
                      renderOfferAmount(r)
                    ) : (
                    <>
                    {renderAmountCell(r, reservations)}
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
                    </>
                    )}
                  </td>
                  <td>
                    {r.__institutionOffer ? (
                      <>
                        <span className={`${styles.statusBadge} ${styles.pending}`}>
                          {offerCanRespond ? 'En attente' : offerExpired ? 'Expiré' : 'Indisponible'}
                        </span>
                        {r.expires_at && (
                          <div className={styles.cellMeta}>
                            Exp: {formatInstantDateTime(r.expires_at).date}{' '}
                            {formatInstantDateTime(r.expires_at).time}
                          </div>
                        )}
                      </>
                    ) : (
                      <>
                        <BookingStatusBadge status={status} />
                        {r.active_change_request?.status === 'pending' && (
                          <span
                            className={styles.transferBadge}
                            title="Modification institution en attente de validation"
                            style={{ background: '#fef3c7', color: '#92400e' }}
                          >
                            Modif. en attente
                          </span>
                        )}
                        {r.active_change_request?.status === 'escalation_required' && (
                          <span
                            className={styles.transferBadge}
                            title="Demande de modification expirée — action institution requise"
                            style={{ background: '#ffedd5', color: '#c2410c' }}
                          >
                            Escalade
                          </span>
                        )}
                        {r.active_change_request?.status === 'expired' && (
                          <span
                            className={styles.transferBadge}
                            title="Demande de modification expirée"
                            style={{ background: '#f1f5f9', color: '#475569' }}
                          >
                            Modif. expirée
                          </span>
                        )}
                      </>
                    )}
                  </td>
                  <td className={styles.actionsCell} onClick={(e) => e.stopPropagation()}>
                    {r.__institutionOffer ? (
                      offerCanRespond ? (
                        <>
                          <button
                            onClick={() => onAcceptInstitutionOffer?.(r)}
                            title="Accepter (horaire demandé)"
                            className={`${styles.actionButton} ${styles.acceptButton}`}
                          >
                            <FiCheckCircle size={16} />
                          </button>
                          <button
                            onClick={() => onProposeInstitutionOffer?.(r)}
                            title="Accepter avec un horaire différent"
                            className={styles.actionButton}
                            style={{ color: 'var(--brand-primary)' }}
                          >
                            <FiClock size={16} />
                          </button>
                          <button
                            onClick={() => onRejectInstitutionOffer?.(r.__offerId, r.__offer)}
                            title="Refuser la demande"
                            className={`${styles.actionButton} ${styles.rejectButton}`}
                          >
                            <FiXCircle size={16} />
                          </button>
                        </>
                      ) : (
                        <span className={styles.noActionLabel}>
                          {offerExpired
                            ? 'Offre expirée, vous ne pouvez plus répondre.'
                            : 'Aucune action'}
                        </span>
                      )
                    ) : !hasActions ? (
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
          const {
            status,
            hasActions,
            canManageReservation,
            delayMinutes,
            pickupArrivalLabel,
            offerCanRespond,
            offerExpired,
          } = renderRow(r);

          return (
            <div
              key={r.id}
              className={styles.mobileCard}
              onClick={() => onRowClick?.(r)}
            >
              <div className={styles.mobileCardHeader}>
                <div className={styles.mobileCardTitleGroup}>
                  <BookingIdentityCell booking={r} layout="compact" />
                  <BookingTripBadges booking={r} routeGroupSizes={routeGroupSizes} />
                </div>
                {r.__institutionOffer ? (
                  <span className={`${styles.statusBadge} ${styles.pending}`}>
                    {offerCanRespond ? 'En attente' : offerExpired ? 'Expiré' : 'Indisponible'}
                  </span>
                ) : (
                  <BookingStatusBadge status={status} />
                )}
                {r.active_change_request?.status === 'pending' && (
                  <span style={{ marginLeft: 6, fontSize: 10, color: '#92400e' }}>Modif. en attente</span>
                )}
                {r.active_change_request?.status === 'escalation_required' && (
                  <span style={{ marginLeft: 6, fontSize: 10, color: '#c2410c' }}>Escalade</span>
                )}
              </div>
              <div className={styles.mobileCardBody}>
                <div className={styles.mobileCardRow}>
                  <span className={styles.mobileCardLabel}>Horaire</span>
                  <span className={styles.mobileCardValue}>
                    <div className={styles.timeCellStack}>
                      <BookingScheduleCell booking={r} undefinedClassName={styles.pickupEtaHint} />
                      {pickupArrivalLabel && (
                        <span className={styles.pickupEtaHint} title={pickupArrivalLabel.title}>
                          {pickupArrivalLabel.text}
                        </span>
                      )}
                    </div>
                  </span>
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
                    {r.__institutionOffer
                      ? renderOfferAmount(r)
                      : renderAmountCell(r, reservations)}
                  </span>
                </div>
                {r.__institutionOffer && r.expires_at && (
                  <div className={styles.mobileCardRow}>
                    <span className={styles.mobileCardLabel}>Expiration</span>
                    <span className={styles.mobileCardValue}>
                      Exp: {formatInstantDateTime(r.expires_at).date}{' '}
                      {formatInstantDateTime(r.expires_at).time}
                    </span>
                  </div>
                )}
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
              {r.__institutionOffer ? (
                <div className={styles.mobileCardActions} onClick={(e) => e.stopPropagation()}>
                  {offerCanRespond ? (
                    <>
                      <button
                        onClick={() => onAcceptInstitutionOffer?.(r)}
                        title="Accepter (horaire demandé)"
                        className={`${styles.actionButton} ${styles.acceptButton}`}
                      >
                        <FiCheckCircle size={16} />
                      </button>
                      <button
                        onClick={() => onProposeInstitutionOffer?.(r)}
                        title="Accepter avec un horaire différent"
                        className={styles.actionButton}
                        style={{ color: 'var(--brand-primary)' }}
                      >
                        <FiClock size={16} />
                      </button>
                      <button
                        onClick={() => onRejectInstitutionOffer?.(r.__offerId, r.__offer)}
                        title="Refuser la demande"
                        className={`${styles.actionButton} ${styles.rejectButton}`}
                      >
                        <FiXCircle size={16} />
                      </button>
                    </>
                  ) : (
                    <span className={styles.noActionLabel}>
                      {offerExpired
                        ? 'Offre expirée, vous ne pouvez plus répondre.'
                        : 'Aucune action'}
                    </span>
                  )}
                </div>
              ) : hasActions && canManageReservation && (
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
