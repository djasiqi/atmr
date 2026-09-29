import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { toast } from 'sonner';
import { submitContactRequest } from '../../../services/contactService';
import { getApiErrorMessage } from '../../../utils/apiErrorMessage';
import {
  getClientBookingUx,
  resolveClientBookingDisplayStatus,
} from '../../../utils/clientBookingUx';
import styles from './Reservations.module.css';

const EVENT_KINDS = [
  { value: 'delay', label: 'Retard' },
  { value: 'during_trip', label: 'Problème pendant le trajet' },
  { value: 'forgotten', label: 'Objet oublié' },
  { value: 'billing', label: 'Facturation' },
  { value: 'other', label: 'Autre' },
];

function clientIdentity(client) {
  const user = client?.user && typeof client.user === 'object' ? client.user : {};
  const name = String(
    client?.full_name ||
      `${client?.first_name || user.first_name || ''} ${client?.last_name || user.last_name || ''}`.trim()
  ).trim();
  const email = String(user.email || client?.contact_email || client?.email || '').trim();
  const phoneRaw = String(client?.phone || user.phone || '').trim();
  const phone = /^\+?[0-9][0-9\s().-]{7,}$/.test(phoneRaw) ? phoneRaw : '';
  return { name, email, phone };
}

function transportReference(booking) {
  const ids =
    Array.isArray(booking?.route_request_ids) && booking.route_request_ids.length
      ? booking.route_request_ids
      : [booking?.id];
  return ids
    .filter((id) => id != null && String(id).trim())
    .map((id) => `#${id}`)
    .join(', ')
    .slice(0, 120);
}

function formatTicketMoment(iso) {
  const parsed = Date.parse(iso);
  if (!Number.isFinite(parsed)) return '';
  const date = new Date(parsed);
  const day = date.toLocaleDateString('fr-CH', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
  const time = date.toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });
  return `${day} à ${time}`;
}

function formatTicketAmount(booking) {
  const amount = Number(booking?.route_request_amount ?? booking?.amount);
  if (!Number.isFinite(amount) || amount <= 0) return '';
  return `${amount.toFixed(2)} CHF`;
}

function ticketStops(booking) {
  const folded = Array.isArray(booking?.route_request_stops) ? booking.route_request_stops : [];
  if (folded.length > 1) return folded;
  const pickup = String(booking?.pickup_location || '').trim();
  const dropoff = String(booking?.dropoff_location || '').trim();
  const detail = [booking?.hospital_service, booking?.doctor_name]
    .map((value) => String(value || '').trim())
    .filter((value) => value && value !== 'Non spécifié' && value !== 'Aucune note')
    .join(' · ');
  return [
    {
      label: 'Prise en charge',
      place: pickup,
      detail: '',
      boarded_at: booking?.boarded_at || null,
    },
    {
      label: 'Destination',
      place: dropoff,
      detail,
      completed_at: booking?.completed_at || null,
    },
  ].filter((stop) => stop.place || stop.detail);
}

function transportDossier(booking) {
  const reference = transportReference(booking);
  const status = getClientBookingUx(resolveClientBookingDisplayStatus(booking)).label;
  const planned = formatTicketMoment(booking?.scheduled_time || booking?.route_request_latest_time);
  const amount = formatTicketAmount(booking);
  const company = String(booking?.company_name || '').trim();
  const driver = String(booking?.driver_name || '').trim();
  const lines = [
    'Transport',
    reference ? `Référence : ${reference}` : '',
    status ? `Statut : ${status}` : '',
    planned ? `Prévu : ${planned}` : '',
    amount ? `Montant : ${amount}` : '',
    company ? `Entreprise : ${company}` : '',
    driver ? `Chauffeur : ${driver}` : '',
    '',
    'Trajet',
  ];
  ticketStops(booking).forEach((stop, index) => {
    const place = [stop.place, stop.detail].filter(Boolean).join(' · ');
    lines.push(`${index + 1}. ${stop.label || 'Étape'}${place ? ` — ${place}` : ''}`);
    const boarded = formatTicketMoment(stop.boarded_at);
    const dropped = formatTicketMoment(stop.completed_at);
    if (boarded) lines.push(`   Prise en charge confirmée : ${boarded}`);
    if (dropped) lines.push(`   Dépôt confirmé : ${dropped}`);
  });
  return lines.filter((line, index, all) => line !== '' || all[index - 1] !== '').join('\n');
}

export default function ClientTripEventReportModal({ booking, client, open, onClose }) {
  const [portalEl, setPortalEl] = useState(null);
  const [kind, setKind] = useState('delay');
  const [message, setMessage] = useState('');
  const [consent, setConsent] = useState(false);
  const [error, setError] = useState('');
  const [sending, setSending] = useState(false);
  const identity = useMemo(() => clientIdentity(client), [client]);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    setPortalEl(typeof document !== 'undefined' ? document.body : null);
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    setKind('delay');
    setMessage('');
    setConsent(false);
    setError('');
    setSending(false);
    const onKey = (event) => {
      if (event.key === 'Escape') onCloseRef.current();
    };
    window.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
    };
  }, [open, booking?.id]);

  if (!open || !booking || !portalEl) return null;

  const reference = transportReference(booking);
  const dossier = transportDossier(booking);

  const onSubmit = async (event) => {
    event.preventDefault();
    setError('');
    const text = message.trim();
    if (!identity.email || identity.name.length < 2) {
      setError('Le nom ou l’e-mail du compte est incomplet. Complétez votre profil, puis réessayez.');
      return;
    }
    if (text.length < 5) {
      setError('Décrivez l’événement en quelques mots.');
      return;
    }
    if (!consent) {
      setError('Le consentement est nécessaire pour transmettre le signalement.');
      return;
    }
    const kindLabel = EVENT_KINDS.find((item) => item.value === kind)?.label || 'Autre';
    setSending(true);
    try {
      await submitContactRequest({
        category: 'support',
        name: identity.name,
        email: identity.email,
        ...(identity.phone ? { phone: identity.phone } : {}),
        subject_detail: kind,
        reference: reference || undefined,
        urgency: 'normal',
        message: `${dossier}\n\nÉvénement : ${kindLabel}\n\n${text}`,
        privacy_consent: true,
        website: '',
      });
      toast.success('Signalement envoyé à Lirie.');
      onClose();
    } catch (err) {
      setError(getApiErrorMessage(err, "Le signalement n'a pas pu être envoyé. Réessayez dans un instant."));
    } finally {
      setSending(false);
    }
  };

  return createPortal(
    <div className={styles.contactModalBackdrop} role="presentation" onClick={onClose}>
      <div
        className={styles.contactModal}
        role="dialog"
        aria-modal="true"
        aria-labelledby="client-trip-event-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className={styles.contactModalHeader}>
          <h2 id="client-trip-event-title" className={styles.contactModalTitle}>
            Signaler un événement
          </h2>
          <button type="button" className={styles.contactModalClose} onClick={onClose} aria-label="Fermer">
            ×
          </button>
        </div>
        <form className={styles.contactModalBody} onSubmit={onSubmit}>
          <p className={styles.eventReportDossier}>{dossier}</p>
          <p className={styles.contactModalMuted}>
            Une copie de ce ticket, avec l’ensemble du transport, est envoyée à info@lirie.ch.
          </p>
          <label className={styles.eventReportLabel} htmlFor="client-trip-event-kind">
            Événement
          </label>
          <select
            id="client-trip-event-kind"
            className={styles.eventReportField}
            value={kind}
            onChange={(event) => setKind(event.target.value)}
          >
            {EVENT_KINDS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
          <label className={styles.eventReportLabel} htmlFor="client-trip-event-message">
            Que s’est-il passé ?
          </label>
          <textarea
            id="client-trip-event-message"
            className={styles.eventReportField}
            rows={4}
            maxLength={3500}
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            placeholder="Décrivez ce qui s’est passé."
          />
          <label className={styles.eventReportConsent}>
            <input
              type="checkbox"
              checked={consent}
              onChange={(event) => setConsent(event.target.checked)}
            />
            <span>J’accepte que Lirie traite ce message pour répondre à ce signalement.</span>
          </label>
          {error ? <p className={styles.eventReportError}>{error}</p> : null}
          <div className={styles.eventReportActions}>
            <button type="submit" className={styles.eventReportSubmit} disabled={sending}>
              {sending ? 'Envoi…' : 'Envoyer à Lirie'}
            </button>
          </div>
        </form>
      </div>
    </div>,
    portalEl
  );
}
