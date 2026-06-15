import React from 'react';
import PropTypes from 'prop-types';
import { getMissionScheduleCardDisplay } from '../../utils/formatLegTime';
import s from '../../pages/institution/Requests/InstitutionRequests.module.css';

/**
 * Bloc date/heure compact pour carte liste — 2 lignes max, sans débordement.
 */
const MissionScheduleCardTime = ({ request, title }) => {
  const { dateLabel, primary, secondary } = getMissionScheduleCardDisplay(request);

  if (!primary && dateLabel === '—') {
    return <span className={s.cardDateTime}>—</span>;
  }

  const isDeparture = primary?.label === 'Départ';

  return (
    <div className={s.cardSchedule} title={title}>
      <div className={s.cardScheduleRow}>
        {dateLabel !== '—' && (
          <span className={s.cardScheduleDate}>{dateLabel}</span>
        )}
        {primary && (
          <span className={isDeparture ? s.cardScheduleDeparture : s.cardScheduleSecondaryInline}>
            <span className={isDeparture ? s.cardSchedulePrimaryTime : s.cardScheduleSecondaryTime}>
              {primary.time}
            </span>
            {' '}
            <span className={isDeparture ? s.cardSchedulePrimaryLabel : s.cardScheduleSecondaryLabel}>
              {primary.label}
            </span>
          </span>
        )}
      </div>
      {secondary.length > 0 && (
        <div className={s.cardScheduleRow}>
          {secondary.map((part, index) => (
            <React.Fragment key={`${part.label}-${part.time}`}>
              {index > 0 && <span className={s.cardScheduleSep} aria-hidden="true">·</span>}
              <span className={s.cardScheduleSecondaryInline}>
                <span className={s.cardScheduleSecondaryLabel}>{part.label}</span>
                {' '}
                <span className={s.cardScheduleSecondaryTime}>{part.time}</span>
              </span>
            </React.Fragment>
          ))}
        </div>
      )}
    </div>
  );
};

MissionScheduleCardTime.propTypes = {
  request: PropTypes.object,
  title: PropTypes.string,
};

export default MissionScheduleCardTime;
