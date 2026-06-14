import React, { useMemo } from 'react';
import { useBookingTimeline } from '../../../hooks/useInstitutionData';
import s from './RequestDetailPanel.module.css';

const fmtShort = (date) => {
  if (!date) return '';
  return new Date(date).toLocaleString('fr-CH', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const mapTimelineEvent = (ev) => ({
  event: ev.label || ev.event_type,
  date: ev.created_at,
  type: ev.event_type === 'cancelled' ? 'cancel' : undefined,
  eventId: ev.id,
});

const InstitutionBookingHistory = ({ bookingId }) => {
  const { data, isLoading } = useBookingTimeline(bookingId, Boolean(bookingId));

  const items = useMemo(() => {
    const events = data?.events || [];
    return events
      .map(mapTimelineEvent)
      .filter((it) => it && it.date)
      .sort((a, b) => new Date(b.date) - new Date(a.date));
  }, [data]);

  if (!items.length && !isLoading) {
    return (
      <div className={s.timeline}>
        <p className={s.billingMuted}>Aucun événement.</p>
      </div>
    );
  }

  return (
    <div className={s.timeline}>
      {isLoading && (
        <p className={s.billingMuted}>Chargement historique…</p>
      )}
      {items.map((item) => (
        <div
          key={item.eventId || `${item.date}-${item.event}`}
          className={`${s.timelineItem} ${item.type === 'cancel' ? s.timelineItemCancel : ''}`}
        >
          <div className={s.timelineEvent}>{item.event}</div>
          <div className={s.timelineDate}>{fmtShort(item.date)}</div>
        </div>
      ))}
    </div>
  );
};

export default InstitutionBookingHistory;
