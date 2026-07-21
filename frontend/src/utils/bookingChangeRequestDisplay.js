/** Libellés FR des champs opérationnels d'une demande de modification institution. */
export const BOOKING_CHANGE_FIELD_LABELS = {
  pickup_location: 'Lieu de prise en charge',
  dropoff_location: 'Destination',
  dropoff_establishment: 'Établissement',
  dropoff_service: 'Service',
  dropoff_doctor: 'Médecin',
  medical_facility: 'Établissement',
  hospital_service: 'Service',
  doctor_name: 'Médecin',
  intermediate_stops: 'Étapes intermédiaires',
  multi_stop: 'Parcours multi-étapes',
  return_to_institution: 'Retour institution',
  is_round_trip: 'Aller-retour',
  mission_date: 'Date de mission',
  scheduled_time: 'Horaire prévu',
  scheduled_time_type: 'Type d\'horaire',
  pickup_time_confirmed: 'Heure de prise en charge confirmée',
  appointment_time: 'Heure de rendez-vous',
  return_time: 'Heure de retour',
  return_date: 'Date de retour',
  return_time_confirmed: 'Retour confirmé',
  return_scheduled_time: 'Horaire retour',
  mobility: 'Mobilité',
  requires_wheelchair: 'Fauteuil roulant',
  requires_assistance: 'Assistance requise',
  wheelchair_need: 'Besoin fauteuil',
  wheelchair_client_has: 'Fauteuil client',
  notes: 'Notes',
  notes_medical: 'Notes médicales',
  pickup_access_notes: 'Consignes départ',
  dropoff_access_notes: 'Consignes arrivée',
  pickup_floor: 'Étage départ',
  pickup_door_code: 'Code porte départ',
  dropoff_floor: 'Étage arrivée',
  dropoff_door_code: 'Code porte arrivée',
  delivery_description: 'Description livraison',
  patient_id: 'Patient',
  customer_name: 'Nom patient',
  external_reference: 'Référence externe',
  billing_intent: 'Facturation',
  billing_details: 'Détails facturation',
  phone: 'Téléphone',
  instructions: 'Instructions',
  amount: 'Montant',
};

/** Champs techniques masqués dans le résumé transporteur. */
const HIDDEN_SNAPSHOT_FIELDS = new Set([
  'pickup_lat',
  'pickup_lon',
  'dropoff_lat',
  'dropoff_lon',
  'edit_version',
  'status',
  'boarded_at',
  'mission_type',
]);

/** Ordre d'affichage des modifications (du plus impactant au plus détail). */
const FIELD_DISPLAY_ORDER = [
  'scheduled_time',
  'pickup_location',
  'dropoff_location',
  'medical_facility',
  'hospital_service',
  'doctor_name',
  'customer_name',
  'pickup_floor',
  'pickup_door_code',
  'dropoff_floor',
  'dropoff_door_code',
  'notes_medical',
  'pickup_access_notes',
  'dropoff_access_notes',
  'wheelchair_need',
  'wheelchair_client_has',
  'delivery_description',
  'external_reference',
  'amount',
];

const MAX_VALUE_LENGTH = 60;

/**
 * @param {Record<string, boolean>|string[]|null|undefined} changedFields
 * @returns {string[]}
 */
export const extractChangedFieldKeys = (changedFields) => {
  if (!changedFields) return [];
  if (Array.isArray(changedFields)) {
    return changedFields.map(String).filter(Boolean);
  }
  if (typeof changedFields === 'object') {
    return Object.entries(changedFields)
      .filter(([, value]) => value)
      .map(([key]) => String(key));
  }
  return [];
};

/**
 * @param {Record<string, boolean>|string[]|null|undefined} changedFields
 * @returns {string[]}
 */
export const formatChangedFieldLabels = (changedFields) => {
  const keys = extractChangedFieldKeys(changedFields)
    .filter((key) => !HIDDEN_SNAPSHOT_FIELDS.has(key));
  const labels = keys.map((key) => BOOKING_CHANGE_FIELD_LABELS[key] || key);
  return [...new Set(labels)];
};

/**
 * @param {string} key
 * @param {unknown} value
 * @returns {string|null}
 */
export const formatSnapshotValue = (key, value) => {
  if (value === null || value === undefined) return null;
  if (typeof value === 'boolean') return value ? 'Oui' : 'Non';
  if (typeof value === 'number') return String(value);

  const text = String(value).trim();
  if (!text) return null;

  if (key === 'scheduled_time') {
    const date = new Date(text);
    if (!Number.isNaN(date.getTime())) {
      const pad = (n) => String(n).padStart(2, '0');
      return `${pad(date.getDate())}/${pad(date.getMonth() + 1)} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
    }
  }

  return text;
};

const truncateValue = (value) => {
  if (!value) return value;
  if (value.length <= MAX_VALUE_LENGTH) return value;
  return `${value.slice(0, MAX_VALUE_LENGTH - 1)}…`;
};

/**
 * @param {string} key
 * @param {string} label
 * @param {unknown} beforeVal
 * @param {unknown} afterVal
 * @returns {string|null}
 */
export const formatFieldChangeLine = (key, label, beforeVal, afterVal) => {
  const before = formatSnapshotValue(key, beforeVal);
  const after = formatSnapshotValue(key, afterVal);

  if (before === null && after === null) return null;
  if (before === null && after !== null) {
    return `${label} : ajout « ${truncateValue(after)} »`;
  }
  if (before !== null && after === null) {
    return `${label} : suppression « ${truncateValue(before)} »`;
  }
  if (before === after) return `${label} modifié`;
  return `${label} : ${truncateValue(before)} → ${truncateValue(after)}`;
};

const sortChangedFieldKeys = (keys) => [...keys].sort((a, b) => {
  const indexA = FIELD_DISPLAY_ORDER.indexOf(a);
  const indexB = FIELD_DISPLAY_ORDER.indexOf(b);
  if (indexA === -1 && indexB === -1) return a.localeCompare(b, 'fr');
  if (indexA === -1) return 1;
  if (indexB === -1) return -1;
  return indexA - indexB;
});

/**
 * Détail lisible champ par champ (avant → après).
 * @param {object|null|undefined} changeRequest
 * @returns {{ key: string, text: string }[]}
 */
export const buildChangeRequestDetailLines = (changeRequest) => {
  if (!changeRequest) return [];

  const keys = sortChangedFieldKeys(
    extractChangedFieldKeys(changeRequest.changed_fields)
      .filter((key) => !HIDDEN_SNAPSHOT_FIELDS.has(key)),
  );
  if (!keys.length) return [];

  const before = changeRequest.before_snapshot || {};
  const after = changeRequest.after_snapshot || {};
  const hasSnapshots = keys.some(
    (key) => formatSnapshotValue(key, before[key]) !== null
      || formatSnapshotValue(key, after[key]) !== null,
  );

  return keys
    .map((key) => {
      const label = BOOKING_CHANGE_FIELD_LABELS[key] || key;
      if (!hasSnapshots) {
        return { key, text: label };
      }
      const text = formatFieldChangeLine(key, label, before[key], after[key]);
      return text ? { key, text } : null;
    })
    .filter(Boolean);
};

/**
 * Résumé lisible pour la bannière de validation transporteur.
 * @param {object|null|undefined} changeRequest
 * @returns {{ fieldLabels: string[], changeLines: { key: string, text: string }[], reason: string|null, expiresAt: string|null }}
 */
export const summarizeBookingChangeRequest = (changeRequest) => {
  if (!changeRequest) {
    return { fieldLabels: [], changeLines: [], reason: null, expiresAt: null };
  }
  return {
    fieldLabels: formatChangedFieldLabels(changeRequest.changed_fields),
    changeLines: buildChangeRequestDetailLines(changeRequest),
    reason: changeRequest.reason?.trim() || null,
    expiresAt: changeRequest.expires_at || null,
  };
};

/**
 * Fusionne les champs acceptés d'une TransportAction dans l'objet réservation local
 * (mise à jour ciblée du panneau, sans refetch complet).
 * @param {object} reservation
 * @param {object|null|undefined} changeRequest
 * @returns {object}
 */
export const mergeAcceptedChangeIntoReservation = (reservation, changeRequest) => {
  if (!reservation || !changeRequest) return reservation;

  const after = {
    ...(changeRequest.after_snapshot || {}),
    ...(changeRequest.proposed_patch || {}),
  };

  let keys = extractChangedFieldKeys(changeRequest.changed_fields)
    .filter((key) => !HIDDEN_SNAPSHOT_FIELDS.has(key));

  if (!keys.length) {
    keys = Object.keys(changeRequest.proposed_patch || {})
      .filter((key) => !HIDDEN_SNAPSHOT_FIELDS.has(key));
  }

  if (!keys.length) return reservation;

  const updates = {};
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(after, key)) {
      updates[key] = after[key];
    }
  }

  if (!Object.keys(updates).length) return reservation;
  return { ...reservation, ...updates };
};

/**
 * @param {string|null|undefined} iso
 * @returns {string|null}
 */
export const formatChangeRequestExpiry = (iso) => {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(date.getDate())}/${pad(date.getMonth() + 1)} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
};
