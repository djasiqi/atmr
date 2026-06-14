import React from 'react';
import { formatAppointmentTime } from '../../utils/bookingScheduling';
import styles from './BookingScheduleCell.module.css';

/**
 * Affichage horaire canonique — lit scheduling.display_* (INV-1, INV-7).
 */
const BookingScheduleCell = ({
  booking,
  mode = 'datetime',
  className = '',
  undefinedClassName = '',
}) => {
  const scheduling = booking?.scheduling;
  let text;

  if (scheduling) {
    if (mode === 'time') {
      text = scheduling.display_time ?? 'À définir';
    } else {
      text = scheduling.display_datetime ?? scheduling.display_time ?? 'À définir';
    }
  } else {
    text = formatAppointmentTime(booking, { dateAndTime: mode !== 'time' });
  }

  const isUndefined = scheduling
    ? scheduling.time_defined === false
    : booking?.time_confirmed === false;

  return (
    <span
      className={`${styles.root} ${className} ${isUndefined ? `${styles.undefined} ${undefinedClassName}` : ''}`}
    >
      {text}
    </span>
  );
};

export default BookingScheduleCell;
