/** Présentation du workflow de contestation — aucune logique financière. */

export const CARRIER_STANCES = [
  {
    value: 'institution_right',
    label: "L'institution a raison",
  },
  {
    value: 'mission_done',
    label: 'La mission a bien été effectuée',
  },
  {
    value: 'needs_correction',
    label: 'Mission effectuée mais informations à corriger',
  },
];

export const EXCLUSION_REASONS = [
  { value: 'created_by_error', label: 'Course créée par erreur' },
  { value: 'mission_cancelled', label: 'Mission annulée' },
  { value: 'duplicate', label: 'Doublon' },
  { value: 'other', label: 'Autre' },
];

export const EVIDENCE_KINDS = [
  { value: 'signed_transport_sheet', label: 'Feuille / bon de transport signé' },
  { value: 'pickup_proof', label: 'Preuve de prise en charge' },
  { value: 'gps_history', label: 'Preuve GPS / historique mission' },
  { value: 'actual_times', label: 'Heures réelles départ / arrivée' },
  { value: 'institution_written', label: "Confirmation écrite de l'institution" },
  { value: 'patient_confirmation', label: 'Confirmation du patient / représentant' },
  { value: 'other', label: 'Autre document justificatif' },
];

export const OPEN_DISPUTE_STATUSES = [
  'disputed',
  'awaiting_carrier_response',
  'evidence_submitted',
  'awaiting_correction',
];

const INSTITUTION_REASON_LABELS = {
  TRANSPORT_DISPUTED: 'Course non reconnue',
  PAYER_NOT_FOUND: 'Mauvais payeur',
  FINANCIAL_INCONSISTENCY: 'Incohérence financière',
  MISSING_BLOCKING_DATA: 'Données bloquantes manquantes',
  OTHER: 'Autre',
};

export const institutionReasonLabel = (code) => {
  const raw = String(code || '').trim().toUpperCase();
  if (!raw) return 'Autre';
  return INSTITUTION_REASON_LABELS[raw] || raw;
};

/** Commentaire institution, sans répéter le code technique (`OTHER: …`). */
export const institutionReasonComment = (code, text) => {
  const raw = String(text || '').trim();
  if (!raw) return '';
  const prefix = String(code || '').trim();
  if (prefix) {
    const re = new RegExp(`^${prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*[:—-]\\s*`, 'i');
    const stripped = raw.replace(re, '').trim();
    if (stripped && stripped.toUpperCase() !== prefix.toUpperCase()) return stripped;
  }
  const generic = raw.replace(/^[A-Z][A-Z0-9_]*\s*[:—-]\s*/, '').trim();
  if (generic && generic.toUpperCase() !== raw.toUpperCase()) return generic;
  if (prefix && raw.toUpperCase() === prefix.toUpperCase()) return '';
  return raw;
};

export const presentInstitutionDisputeReason = (code, text) => ({
  category: institutionReasonLabel(code),
  comment: institutionReasonComment(code, text),
});

export const canTreatDispute = (row) => {
  const status = String(row?.disputeStatus || '').toLowerCase();
  if (status === 'resolved_institution' || status === 'resolved_carrier') return false;
  if (row?.disputeTreatable) return true;
  if (status && OPEN_DISPUTE_STATUSES.includes(status)) return true;
  const bucket = String(row?.invoiceBucket || '').toLowerCase();
  const validation = String(row?.validationStatus || '').toLowerCase();
  return (
    bucket === 'disputed_blocked' ||
    validation === 'disputed' ||
    validation === 'anomaly'
  );
};

export const unwrapDisputePayload = (response) =>
  response?.data?.data || response?.data?.dispute || response?.data || response?.dispute || response || null;

export const hasUploadedEvidence = (dispute) =>
  (dispute?.evidence || []).some((row) => row.source === 'uploaded');

export const systemFactsLines = (facts = {}) => {
  const lines = [];
  if (facts.driver_id) lines.push(`Chauffeur : #${facts.driver_id}`);
  if (facts.completed_at) {
    lines.push(`Prise en charge / arrivée enregistrée : ${facts.completed_at}`);
  }
  if (facts.scheduled_time) lines.push(`Horaire prévu : ${facts.scheduled_time}`);
  lines.push(`GPS mission : ${facts.gps_available ? 'disponible' : 'non disponible'}`);
  if (facts.status) lines.push(`Statut chauffeur : ${facts.status}`);
  return lines;
};
