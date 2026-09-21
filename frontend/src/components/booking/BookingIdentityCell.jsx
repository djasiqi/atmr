import React from 'react';
import { buildIdentityFromApi } from '../../utils/bookingIdentity';
import { getBookingSourceMeta } from '../../constants/bookingSourceLabels';
import {
  DELIVERY_BENEFICIARY_LABEL,
  isMaterialDelivery,
  MISSION_DELIVERY_BADGE,
} from '../../utils/missionTypeDisplay';
import styles from './BookingIdentityCell.module.css';

/**
 * Affichage canonique : passager (ligne 1) + origine commerciale (ligne 2).
 * Pour une livraison, le badge LIVRAISON et l'origine deviennent le signal principal.
 */
const BookingIdentityCell = ({
  booking,
  identity: identityProp,
  layout = 'stacked',
  showRequester = false,
  showOriginIcon = false,
  passengerSubLabel,
}) => {
  const identity = identityProp || buildIdentityFromApi(booking);
  const apiIdentity = booking?.identity;
  const passengerLabel = apiIdentity?.primary_label || identity.passengerLabel;
  const secondaryLabel = apiIdentity?.secondary_label || identity.source?.name;
  const sourceMeta = getBookingSourceMeta(identity.source?.type);
  const delivery = isMaterialDelivery(booking || identityProp);
  const showSourceLine = Boolean(secondaryLabel) && !delivery;

  const requesterLine = showRequester && identity.requester?.name
    ? identity.requester.name
    : null;

  const subLabel = passengerSubLabel || null;
  const primaryLabel = delivery
    ? (secondaryLabel || MISSION_DELIVERY_BADGE)
    : passengerLabel;

  return (
    <div className={`${styles.root} ${styles[layout] || ''}`}>
      {delivery && (
        <span className={styles.deliveryBadge} data-testid="booking-identity-delivery-badge">
          {MISSION_DELIVERY_BADGE}
        </span>
      )}
      <span className={styles.passenger}>{primaryLabel}</span>
      {delivery && passengerLabel ? (
        <span className={styles.passengerSub}>
          {DELIVERY_BENEFICIARY_LABEL} · {passengerLabel}
        </span>
      ) : null}
      {subLabel && <span className={styles.passengerSub}>{subLabel}</span>}
      {showSourceLine && (
        <span className={styles.source}>
          {showOriginIcon && sourceMeta.icon ? (
            <span className={styles.sourceIcon} aria-hidden>{sourceMeta.icon}</span>
          ) : null}
          <span className={styles.sourceName}>{secondaryLabel}</span>
        </span>
      )}
      {requesterLine && (
        <span className={styles.requester} title="Demandeur">
          {requesterLine}
        </span>
      )}
    </div>
  );
};

export default BookingIdentityCell;
