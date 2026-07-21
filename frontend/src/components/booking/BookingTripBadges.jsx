import React from 'react';
import {
  buildTripBadgeDescriptors,
  resolveTripFlagsFromBooking,
} from '../../utils/bookingTripFlags';
import tableStyles from '../../pages/company/Dashboard/components/ReservationTable.module.css';

/**
 * Badges parcours data-driven (trip_flags API).
 */
const BookingTripBadges = ({ booking, routeGroupSizes = {}, className = '' }) => {
  const flags = resolveTripFlagsFromBooking(booking, routeGroupSizes);
  const badges = buildTripBadgeDescriptors(flags);
  if (!badges.length) return null;

  const variantClass = {
    roundTrip: tableStyles.roundTripBadge,
    returnLeg: tableStyles.returnLegBadge,
    routeLeg: tableStyles.routeLegBadge,
    transfer: tableStyles.transferBadge,
  };

  return (
    <span className={className}>
      {badges.map((badge) => (
        <span
          key={badge.key}
          className={variantClass[badge.variant] || tableStyles.roundTripBadge}
          title={badge.title || badge.label}
        >
          {badge.label}
        </span>
      ))}
    </span>
  );
};

export default BookingTripBadges;
