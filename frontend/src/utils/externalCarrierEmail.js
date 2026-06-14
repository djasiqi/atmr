/**
 * Compose un e-mail pré-rempli destiné à un transporteur externe.
 *
 * Limitation technique : le protocole `mailto:` ne permet pas de joindre
 * automatiquement un fichier. Le flux retenu est donc :
 *   1. télécharger le bon de transport (PDF) sur le poste de l'utilisateur ;
 *   2. ouvrir le client de messagerie (Outlook, Gmail, Apple Mail…) avec le
 *      destinataire, l'objet et le corps déjà remplis ;
 *   3. l'utilisateur joint manuellement le PDF téléchargé avant l'envoi.
 */

import { formatDepartureTime, formatLegTime } from './formatLegTime';

/**
 * Formate une date mission sans risque de décalage horaire.
 * Parse explicitement YYYY-MM-DD sans passer par new Date('YYYY-MM-DD').
 */
export const formatMissionDate = (value) => {
  const raw = String(value || '').slice(0, 10);
  const match = raw.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return '';
  const [, year, month, day] = match;
  return `${day}.${month}.${year}`;
};

const getPatientName = (request) => {
  if (request?.patient) {
    return `${request.patient.first_name || ''} ${request.patient.last_name || ''}`.trim();
  }
  return request?.booking_summary?.customer_name || '';
};

/** Date de naissance du patient (déjà présente dans le bon PDF). */
const getPatientDob = (request) => formatMissionDate(request?.patient?.dob);

/** Contact demandeur null-safe (contact_on_site peut être null ou {}). */
export const getRequesterContact = (request) => {
  const contact = request?.contact_on_site || {};
  return {
    requester_name: String(contact.requester_name || '').trim(),
    requester_phone: String(contact.requester_phone || '').trim(),
    requester_service: String(contact.requester_service || '').trim(),
  };
};

/** Numéro de demande affiché : external_reference prioritaire, sinon id numérique. */
export const getRequestNumber = (request) => (
  request?.external_reference || request?.id || ''
);

/** Référence distincte du numéro affiché (évite le doublon N° / Référence). */
export const shouldDisplayReference = (request) => {
  const ref = request?.external_reference;
  if (!ref) return false;
  const requestNumber = getRequestNumber(request);
  return String(ref) !== String(requestNumber);
};

const sortedLegs = (request) => (
  Array.isArray(request?.legs)
    ? [...request.legs].sort((a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0))
    : []
);

/** Nettoie une heure formatée : ignore « À définir », garde « (non confirmé) ». */
const cleanTime = (value) => {
  if (!value || value === 'À définir') return '';
  return value;
};

/**
 * Construit les lignes de trajet avec étiquette horaire :
 *  - heure de départ → « départ prévu à HH:MM »
 *  - heure de destination/étape → « rendez-vous prévu à HH:MM »
 */
const resolveRouteLines = (request) => {
  const legs = sortedLegs(request);
  const lines = [];

  const departureAddress = legs.length > 0
    ? (legs[0].pickup_location || '')
    : (request?.pickup_location || '');
  const departureTime = cleanTime(formatDepartureTime(request));
  if (departureAddress) {
    lines.push(
      `Départ : ${departureAddress}`
      + (departureTime ? ` (départ à ${departureTime})` : ''),
    );
  }

  if (legs.length > 0) {
    const realDestinations = legs.length - (request?.return_to_institution ? 1 : 0);
    legs.forEach((leg, index) => {
      const isReturn = Boolean(request?.return_to_institution) && index === legs.length - 1;
      const address = leg.dropoff_location || '';
      if (!address) return;
      const time = cleanTime(formatLegTime(leg));
      let label;
      if (isReturn) {
        label = 'Retour';
      } else if (realDestinations > 1) {
        label = `Étape ${index + 1}`;
      } else {
        label = 'Destination';
      }
      lines.push(
        `${label} : ${address}`
        + (time ? ` (rendez-vous à ${time})` : ''),
      );
    });
  } else if (request?.dropoff_location) {
    const time = request?.scheduled_time_type === 'arrival' && request?.scheduled_time
      ? cleanTime(formatLegTime({ scheduled_time: request.scheduled_time, time_confirmed: true }))
      : '';
    lines.push(
      `Destination : ${request.dropoff_location}`
      + (time ? ` (rendez-vous à ${time})` : ''),
    );
  }

  return lines;
};

const resolveMissionDateLabel = (request) => {
  const fromMissionDate = formatMissionDate(request?.mission_date);
  if (fromMissionDate) return fromMissionDate;
  return formatMissionDate(request?.scheduled_time);
};

/**
 * Construit l'objet et le corps de l'e-mail.
 * @param {Object} request - Demande de transport sérialisée.
 * @param {Object} [options]
 * @param {string} [options.institutionName] - Nom de l'institution (signature).
 * @param {string} [options.institutionPhone] - Téléphone institution (repli contact).
 * @returns {{ subject: string, body: string }}
 */
export const buildCarrierEmail = (request, { institutionName, institutionPhone } = {}) => {
  const patientName = getPatientName(request);
  const patientDob = getPatientDob(request);
  const dateLabel = resolveMissionDateLabel(request);
  const routeLines = resolveRouteLines(request);
  const requestNumber = getRequestNumber(request);
  const contact = getRequesterContact(request);
  const phone = contact.requester_phone || String(institutionPhone || '').trim();

  // Objet enrichi (utile pour le tri/recherche dans la boîte de réception).
  const subject = [
    requestNumber ? `Bon de transport #${requestNumber}` : 'Bon de transport',
    patientName,
    institutionName,
    dateLabel,
  ].filter(Boolean).join(' — ');

  // Corps volontairement court : le PDF joint contient l'ensemble des détails.
  // L'e-mail répond à 3 questions : qui ? où ? qui appeler ?
  const lines = [
    'Bonjour,',
    '',
    'Veuillez trouver ci-joint le bon de transport concernant :',
  ];

  if (patientName) {
    lines.push('');
    lines.push(patientDob ? `Patient : ${patientName} (${patientDob})` : `Patient : ${patientName}`);
  }

  if (routeLines.length > 0) {
    lines.push('');
    lines.push('Trajet :');
    routeLines.forEach((line) => lines.push(`• ${line}`));
  }

  if (phone) {
    lines.push('');
    lines.push(`Contact : ${phone}`);
  }

  lines.push('');
  lines.push('Cordialement,');
  if (institutionName) {
    lines.push('');
    lines.push(institutionName);
  }

  return { subject, body: lines.join('\n') };
};

/**
 * Construit un lien `mailto:` pré-rempli pour le transporteur.
 * @param {string} email - Adresse du transporteur externe.
 * @param {Object} request - Demande de transport sérialisée.
 * @param {Object} [options] - Voir {@link buildCarrierEmail}.
 * @returns {string} URL mailto.
 */
export const buildCarrierMailto = (email, request, options = {}) => {
  const { subject, body } = buildCarrierEmail(request, options);
  const params = `subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
  return `mailto:${encodeURIComponent(email)}?${params}`;
};
