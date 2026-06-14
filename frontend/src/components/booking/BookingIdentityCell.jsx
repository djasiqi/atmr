import React from 'react';
import { buildIdentityFromApi } from '../../utils/bookingIdentity';
import { getBookingSourceMeta } from '../../constants/bookingSourceLabels';
import styles from './BookingIdentityCell.module.css';

/**
 * Affichage canonique : passager (ligne 1) + origine commerciale (ligne 2).
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
  const primaryLabel = apiIdentity?.primary_label || identity.passengerLabel;
  const secondaryLabel = apiIdentity?.secondary_label || identity.source?.name;
  const sourceMeta = getBookingSourceMeta(identity.source?.type);
  const showSourceLine = Boolean(secondaryLabel);

  const requesterLine = showRequester && identity.requester?.name
    ? identity.requester.name
    : null;

  const subLabel = passengerSubLabel || null;

  return (
    <div className={`${styles.root} ${styles[layout] || ''}`}>
      <span className={styles.passenger}>{primaryLabel}</span>
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
