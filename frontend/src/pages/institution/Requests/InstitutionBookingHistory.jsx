import React, { useMemo } from 'react';
import { useBookingChangeEvents } from '../../../hooks/useInstitutionData';
import s from './RequestDetailPanel.module.css';

const SOURCE_LABELS = {
  institution_portal: 'institution',
  driver_app: 'chauffeur',
  company_dispatch: 'dispatch',
  system: 'système',
};

const FIELD_LABELS = {
  scheduled_time: 'horaire',
  pickup_location: 'adresse de départ',
  dropoff_location: 'adresse d’arrivée',
  pickup_floor: 'étage départ',
  pickup_door_code: 'code départ',
  pickup_access_notes: 'consignes départ',
  dropoff_floor: 'étage arrivée',
  dropoff_door_code: 'code arrivée',
  dropoff_access_notes: 'consignes arrivée',
  medical_facility: 'établissement',
  hospital_service: 'service',
  doctor_name: 'médecin',
  notes_medical: 'notes patient',
  wheelchair_need: 'fauteuil requis',
  wheelchair_client_has: 'fauteuil du patient',
  customer_name: 'patient',
  customer_phone: 'téléphone',
  delivery_description: 'description',
  mission_type: 'type de mission',
};

const HIDDEN_FIELDS = new Set([
  'edit_version',
  'status',
  'pickup_lat',
  'pickup_lon',
  'dropoff_lat',
  'dropoff_lon',
  'boarded_at',
]);

const fmtShort = (date) => {
  if (!date) return '';
  return new Date(date).toLocaleString('fr-CH', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const humanFieldsList = (changedFields) => {
  if (!changedFields || typeof changedFields !== 'object') return [];
  return Object.keys(changedFields)
    .filter((k) => changedFields[k] && !HIDDEN_FIELDS.has(k))
    .map((k) => FIELD_LABELS[k] || k.replace(/_/g, ' '));
};

const buildChangeEventLabel = (ev) => {
  const actor = ev.actor_display_name || ev.actor_role || 'utilisateur';
  const sourceLabel = SOURCE_LABELS[ev.source] || ev.source || '';
  const action = ev.action_type;

  if (action === 'cancelled') {
    return `Annulation par ${actor}${ev.reason ? ` — ${ev.reason}` : ''}`;
  }
  if (action === 'notification_sent') {
    return null;
  }
  if (action === 'ack_received') {
    return `Accusé de réception (${SOURCE_LABELS[ev.source] || ev.source})`;
  }
  if (action === 'field_updated') {
    const fields = humanFieldsList(ev.changed_fields);
    const prefix = sourceLabel ? `Modification ${sourceLabel}` : 'Modification';
    const suffix = fields.length ? ` — ${fields.join(', ')}` : '';
    return `${prefix} par ${actor}${suffix}${ev.reason ? ` · motif : ${ev.reason}` : ''}`;
  }
  if (action === 'status_changed') {
    return `Changement de statut par ${actor}`;
  }
  return `${action} par ${actor}`;
};

const InstitutionBookingHistory = ({
  bookingId,
  lifecycleTimeline = [],
}) => {
  const { data, isLoading } = useBookingChangeEvents(bookingId, Boolean(bookingId));

  const merged = useMemo(() => {
    const items = [...(lifecycleTimeline || [])];
    const activity = data?.activity || [];
    for (const ev of activity) {
      const label = buildChangeEventLabel(ev);
      if (!label) continue;
      items.push({
        event: label,
        date: ev.created_at,
        type: ev.severity === 'CRITICAL'
          ? 'critical'
          : ev.action_type === 'cancelled'
            ? 'cancel'
            : 'change',
      });
    }
    return items
      .filter((it) => it && it.date)
      .sort((a, b) => new Date(b.date) - new Date(a.date));
  }, [lifecycleTimeline, data]);

  if (!merged.length && !isLoading) {
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
      {merged.map((item, i) => (
        <div
          key={i}
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
