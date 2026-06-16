/** Libellés FR des champs opérationnels d'une demande de modification institution. */
export const BOOKING_CHANGE_FIELD_LABELS = {
  pickup_location: 'Lieu de prise en charge',
  dropoff_location: 'Destination',
  dropoff_establishment: 'Établissement',
  dropoff_service: 'Service',
  dropoff_doctor: 'Médecin',
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
  pickup_access_notes: 'Accès départ',
  dropoff_access_notes: 'Accès arrivée',
  patient_id: 'Patient',
  customer_name: 'Nom patient',
  external_reference: 'Référence externe',
  billing_intent: 'Facturation',
  billing_details: 'Détails facturation',
  phone: 'Téléphone',
  instructions: 'Instructions',
  amount: 'Montant',
};

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
  const keys = extractChangedFieldKeys(changedFields);
  const labels = keys.map((key) => BOOKING_CHANGE_FIELD_LABELS[key] || key);
  return [...new Set(labels)];
};

/**
 * Résumé lisible pour la bannière de validation transporteur.
 * @param {object|null|undefined} changeRequest
 * @returns {{ fieldLabels: string[], reason: string|null, expiresAt: string|null }}
 */
export const summarizeBookingChangeRequest = (changeRequest) => {
  if (!changeRequest) {
    return { fieldLabels: [], reason: null, expiresAt: null };
  }
  return {
    fieldLabels: formatChangedFieldLabels(changeRequest.changed_fields),
    reason: changeRequest.reason?.trim() || null,
    expiresAt: changeRequest.expires_at || null,
  };
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
