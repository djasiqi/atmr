// src/pages/company/Dashboard/components/ReservationTable.jsx
import React, { useState } from 'react';
import { FiCheckCircle, FiXCircle, FiInbox, FiChevronDown } from 'react-icons/fi';
import styles from './ReservationTable.module.css';
import { formatDelay } from '../../../../utils/formatDelay';
import { pickupArrivalHint } from '../../../../utils/formatPickupEta';
import BookingScheduleCell from '../../../../components/booking/BookingScheduleCell';
import {
  journeyPlaces,
  otherJourneyScheduleLines,
  pickupNeedsCompanyConfirmation,
} from '../../../../utils/routeGroupItinerary';
import ReservationActions from '../../../../components/reservations/ReservationActions';
import BookingIdentityCell from '../../../../components/booking/BookingIdentityCell';
import BookingTripBadges from '../../../../components/booking/BookingTripBadges';
import BookingStatusBadge from '../../../../components/booking/BookingStatusBadge';
import InstitutionOfferActions from './InstitutionOfferActions';
import { canRespondToInstitutionOffer, isInstitutionOfferExpired } from '../../../../utils/institutionOfferResponse';
import { institutionOfferEstimateLabel } from '../../../../utils/institutionOfferEstimateLabel';
import {
  getPendingActionBadge,
  indexPendingActionsByRouteGroup,
  resolvePendingTransportAction,
  resolveRespondTargetBooking,
} from '../../../../utils/transportActionPending';
import { useSearchParams } from 'react-router-dom';
import { portalCarrierFacingAmountDisplay, portalCarrierAcceptButtonLabel, portalCarrierRouteGroupAmount } from '../../../../utils/portalDoubleValidationUi';

/** Boutons Valider/Départ immédiat/Planifier/Refuser pour une offre institution (branche `__institutionOffer`). */
const renderInstitutionOfferActionButtons = (r, handlers) => (
  <InstitutionOfferActions
    offer={r.__offer || r}
    onValidate={() => handlers.onValidate?.(r)}
    onAcceptNow={() => handlers.onAcceptNow?.(r)}
    onPlan={() => handlers.onPlan?.(r)}
    onReject={() => handlers.onReject?.(r.__offerId, r.__offer)}
  />
);

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

function renderOtherSchedules(booking, allReservations) {
  return otherJourneyScheduleLines(booking, allReservations).map((line) => (
    <span key={line.key} className={styles.scheduleExtra}>
      {line.text}
    </span>
  ));
}

function renderJourney(booking, allReservations, textClassName) {
  const places = journeyPlaces(booking, allReservations);
  return places.map((place, index) => {
    const isFirst = index === 0;
    const isLast = index === places.length - 1;
    const dotClass = isFirst
      ? styles.locationDotPickup
      : isLast
        ? styles.locationDotDropoff
        : styles.locationDotStop;
    return (
      <div className={styles.locationRow} key={`${index}-${place}`}>
        <span className={`${styles.locationDot} ${dotClass}`} aria-hidden />
        <span className={textClassName} title={place}>{place}</span>
      </div>
    );
  });
}

/** Montants aller-retour : total sur l’aller, libellé explicite sur le retour si montant 0. */
const PICKUP_CONFIRM_LABEL = "Accepter et confirmer l'heure de prise en charge";

function acceptActionLabel(r, allReservations) {
  const groupAmt = portalCarrierRouteGroupAmount(r, allReservations);
  if (groupAmt?.isAnchor && groupAmt.total > 0) {
    return `Accepter cette demande à CHF ${groupAmt.total.toFixed(2)}`;
  }
  return portalCarrierAcceptButtonLabel(r);
}

function renderAmountCell(r, allReservations) {
  const groupAmt = portalCarrierRouteGroupAmount(r, allReservations);
  if (groupAmt) {
    if (!groupAmt.isAnchor) {
      return (
        <div className={styles.amountCellStack}>
          <span className={styles.amountLegLabel}>Inclus dans la demande</span>
          <span className={styles.amountSub}>{groupAmt.label}</span>
        </div>
      );
    }
    return (
      <div className={styles.amountCellStack}>
        <span>
          <span className={styles.amountValue}>{groupAmt.total.toFixed(2)}</span>
          <span className={styles.amountCurrency}> CHF</span>
        </span>
        <span className={styles.amountSub}>{groupAmt.label}</span>
      </div>
    );
  }

  const carrierAmt = portalCarrierFacingAmountDisplay(r);
  if (carrierAmt.mode === 'company_quote' || carrierAmt.mode === 'awaiting_quote') {
    return (
      <div className={styles.amountCellStack}>
        {carrierAmt.amount != null ? (
          <>
            <span className={styles.amountValue}>{carrierAmt.amount.toFixed(2)}</span>
            <span className={styles.amountCurrency}> CHF</span>
          </>
        ) : (
          <span className={styles.amountLegLabel}>Selon grille</span>
        )}
        <span className={styles.amountSub}>{carrierAmt.label}</span>
      </div>
    );
  }

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
    const billingIntent = r.__offer?.transport_request?.billing_intent;
    const sourceLabel = institutionOfferEstimateLabel(est, billingIntent);
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
export function clusterRouteGroups(list) {
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
  onValidateInstitutionOffer,
  onPlanInstitutionOffer,
  onAcceptNowInstitutionOffer,
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
  const [, setSearchParams] = useSearchParams();

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
  const pendingByRouteGroup = indexPendingActionsByRouteGroup(reservations);

  const openRespondPanel = (row) => {
    const target = resolveRespondTargetBooking(row, pendingByRouteGroup, reservations);
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.set('focus', 'change_request');
      return next;
    }, { replace: true });
    onRowClick?.(target);
  };

  // Nombre de legs par parcours multi-destinations (pour le badge "Trajet N/M").
  const routeGroupSizes = {};
  (reservations || []).forEach((r) => {
    if (r?.route_group_id) {
      routeGroupSizes[r.route_group_id] = (routeGroupSizes[r.route_group_id] || 0) + 1;
    }
  });

  const renderRow = (r) => {
    const status = r.status?.toLowerCase() || 'unknown';
    const pendingAction = resolvePendingTransportAction(r, pendingByRouteGroup);
    const pendingBadge = getPendingActionBadge(pendingAction);
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
      pendingBadge,
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
                pendingBadge,
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
                      {renderOtherSchedules(r, reservations)}
                      {pickupArrivalLabel && (
                        <span className={styles.pickupEtaHint} title={pickupArrivalLabel.title}>
                          {pickupArrivalLabel.text}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className={styles.locationCell} aria-label="Itinéraire">
                    {renderJourney(r, reservations, styles.locationText)}
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
                    {pendingBadge && canManageReservation && !r.__institutionOffer ? (
                      <button
                        type="button"
                        className={`${styles.actionPendingBadge} ${
                          pendingBadge.isCancellation ? styles.actionPendingBadgeCancel : ''
                        } ${styles.actionPendingBadgeBtn}`}
                        title={pendingBadge.title}
                        onClick={() => openRespondPanel(r)}
                      >
                        Répondre
                      </button>
                    ) : r.__institutionOffer ? (
                      offerCanRespond ? (
                        renderInstitutionOfferActionButtons(r, {
                          onValidate: onValidateInstitutionOffer || onAcceptInstitutionOffer,
                          onPlan: onPlanInstitutionOffer || onProposeInstitutionOffer,
                          onAcceptNow: onAcceptNowInstitutionOffer,
                          onReject: onRejectInstitutionOffer,
                        })
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
                              type="button"
                              data-tour-id="pending-accept-action"
                              onClick={() => (
                                pickupNeedsCompanyConfirmation(r)
                                  ? onSchedule?.(r)
                                  : onAccept?.(r)
                              )}
                              title={
                                pickupNeedsCompanyConfirmation(r)
                                  ? PICKUP_CONFIRM_LABEL
                                  : acceptActionLabel(r, reservations) ||
                                    (r.is_transferred ? 'Accepter (prendre en charge)' : 'Accepter')
                              }
                              aria-label={
                                pickupNeedsCompanyConfirmation(r)
                                  ? PICKUP_CONFIRM_LABEL
                                  : acceptActionLabel(r, reservations) ||
                                    (r.is_transferred ? 'Accepter (prendre en charge)' : 'Accepter')
                              }
                              className={`${styles.actionButton} ${styles.acceptButton} ${styles.touchTarget}`}
                            >
                              <FiCheckCircle size={16} aria-hidden />
                            </button>
                            <button
                              type="button"
                              onClick={() => onReject?.(r.id)}
                              title="Refuser"
                              aria-label="Refuser"
                              className={`${styles.actionButton} ${styles.rejectButton} ${styles.touchTarget}`}
                            >
                              <FiXCircle size={16} aria-hidden />
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
            pendingBadge,
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
                {r.active_change_request?.status === 'escalation_required' && (
                  <span style={{ marginLeft: 6, fontSize: 10, color: '#c2410c' }}>Escalade</span>
                )}
              </div>
              <div className={styles.mobileCardBody}>
                {pendingBadge && canManageReservation && (
                  <button
                    type="button"
                    className={`${styles.actionPendingBadge} ${
                      pendingBadge.isCancellation ? styles.actionPendingBadgeCancel : ''
                    } ${styles.actionPendingBadgeBtn} ${styles.touchTarget}`}
                    title={pendingBadge.title}
                    aria-label={pendingBadge.title || 'Répondre'}
                    onClick={(e) => {
                      e.stopPropagation();
                      openRespondPanel(r);
                    }}
                    style={{ marginBottom: 8 }}
                  >
                    Répondre
                  </button>
                )}
                <div className={styles.mobileCardRow}>
                  <span className={styles.mobileCardLabel}>Horaire</span>
                  <span className={styles.mobileCardValue}>
                    <div className={styles.timeCellStack}>
                      <BookingScheduleCell booking={r} undefinedClassName={styles.pickupEtaHint} />
                      {renderOtherSchedules(r, reservations)}
                      {pickupArrivalLabel && (
                        <span className={styles.pickupEtaHint} title={pickupArrivalLabel.title}>
                          {pickupArrivalLabel.text}
                        </span>
                      )}
                    </div>
                  </span>
                </div>
                <div className={styles.mobileCardRoute} aria-label="Itinéraire">
                  {renderJourney(r, reservations, styles.mobileCardRouteText)}
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
                    renderInstitutionOfferActionButtons(r, {
                      onValidate: onValidateInstitutionOffer || onAcceptInstitutionOffer,
                      onPlan: onPlanInstitutionOffer || onProposeInstitutionOffer,
                      onAcceptNow: onAcceptNowInstitutionOffer,
                      onReject: onRejectInstitutionOffer,
                    })
                  ) : (
                    <span className={styles.noActionLabel}>
                      {offerExpired
                        ? 'Offre expirée, vous ne pouvez plus répondre.'
                        : 'Aucune action'}
                    </span>
                  )}
                </div>
              ) : hasActions && canManageReservation ? (
                <div className={styles.mobileCardActions} onClick={(e) => e.stopPropagation()}>
                  {status === 'pending' && !pendingBadge ? (
                    <>
                      <button
                        type="button"
                        data-tour-id="pending-accept-action-mobile"
                        onClick={() => (
                          pickupNeedsCompanyConfirmation(r)
                            ? onSchedule?.(r)
                            : onAccept?.(r)
                        )}
                        title={
                          pickupNeedsCompanyConfirmation(r)
                            ? PICKUP_CONFIRM_LABEL
                            : acceptActionLabel(r, reservations) ||
                              (r.is_transferred ? 'Accepter (prendre en charge)' : 'Accepter')
                        }
                        aria-label={
                          pickupNeedsCompanyConfirmation(r)
                            ? PICKUP_CONFIRM_LABEL
                            : acceptActionLabel(r, reservations) ||
                              (r.is_transferred ? 'Accepter (prendre en charge)' : 'Accepter')
                        }
                        className={`${styles.actionButton} ${styles.acceptButton} ${styles.touchTarget}`}
                      >
                        <FiCheckCircle size={16} aria-hidden />
                      </button>
                      <button
                        type="button"
                        onClick={() => onReject?.(r.id)}
                        title="Refuser"
                        aria-label="Refuser"
                        className={`${styles.actionButton} ${styles.rejectButton} ${styles.touchTarget}`}
                      >
                        <FiXCircle size={16} aria-hidden />
                      </button>
                    </>
                  ) : null}
                  {!pendingBadge ? (
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
                  ) : null}
                </div>
              ) : null}
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

export { journeyPlaces, otherJourneyScheduleLines };
export default React.memo(ReservationTable);
