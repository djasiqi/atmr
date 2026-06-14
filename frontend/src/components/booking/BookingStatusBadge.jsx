import React from 'react';
import { getBookingStatusLabel } from '../../constants/bookingStatusLabels';
import tableStyles from '../../pages/company/Dashboard/components/ReservationTable.module.css';

const BookingStatusBadge = ({ status, className = '' }) => {
  const key = String(status || '').toLowerCase();
  const label = getBookingStatusLabel(status);

  return (
    <span className={`${tableStyles.statusBadge} ${tableStyles[key] || ''} ${className}`}>
      {label}
    </span>
  );
};

export default BookingStatusBadge;
