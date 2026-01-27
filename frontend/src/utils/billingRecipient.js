// Utils for Billing Review recipient display.

const normalize = (value) => (value == null ? '' : String(value).trim());

// ✅ Constantes centralisées pour billing_source (alignées avec backend BillingSource enum)
export const BILLING_SOURCE = {
  DEFAULT_CLIENT: 'default_client',
  TRANSPORT_VOUCHER: 'transport_voucher',
  CLIENT_STAY: 'client_stay',
  MANUAL_OVERRIDE: 'manual_override',
  IMPORT: 'import',
  SYSTEM_RULE: 'system_rule',
};

const SOURCE_LABELS = {
  [BILLING_SOURCE.TRANSPORT_VOUCHER]: 'Bon de transport',
  [BILLING_SOURCE.CLIENT_STAY]: "Séjour hospitalier",
  [BILLING_SOURCE.DEFAULT_CLIENT]: 'Tiers payeur par défaut',
  [BILLING_SOURCE.MANUAL_OVERRIDE]: 'Override manuel',
  [BILLING_SOURCE.IMPORT]: 'Import',
  [BILLING_SOURCE.SYSTEM_RULE]: 'Règle système',
};

const PAYER_TYPE_LABELS = {
  patient: 'Patient',
  billing_party: 'Tiers payeur',
  company: 'Clinique',
};

export const getRecipientLabel = (booking) => {
  if (!booking || typeof booking !== 'object') return 'Payeur inconnu';
  const payerName = normalize(booking.payer_name);
  if (payerName) return payerName;
  const payerType = normalize(booking.payer_type).toLowerCase();
  return PAYER_TYPE_LABELS[payerType] || 'Payeur inconnu';
};

export const getRecipientSourceLabel = (booking) => {
  if (!booking || typeof booking !== 'object') return 'Source inconnue';
  const source = normalize(booking.billing_source).toLowerCase();
  if (!source) return 'Source inconnue';
  return SOURCE_LABELS[source] || source;
};

export const getRecipientStatus = (booking) => {
  if (!booking || typeof booking !== 'object') return 'unknown';
  if (booking.has_unvalidated_voucher) return 'voucher';
  if (booking.has_conflict) return 'conflict';
  if (booking.missing_recipient) return 'missing';
  if (normalize(booking.status) === 'needs_review') return 'review';
  return 'ok';
};

export const getRecipientWarningText = (booking) => {
  if (!booking || typeof booking !== 'object') return null;
  const source = normalize(booking.billing_source).toLowerCase();
  const payerType = normalize(booking.payer_type).toLowerCase();
  const mappingMissing = booking.missing_recipient && (source === 'client_stay' || payerType === 'company');

  if (mappingMissing) return 'Clinique : destinataire de facturation non configuré';
  if (booking.has_unvalidated_voucher) return 'Bon de transport non validé';
  if (booking.has_conflict) return 'Conflit : séjour actif vs payeur';
  if (booking.missing_recipient) return 'Destinataire manquant';
  if (normalize(booking.status) === 'needs_review') return 'À vérifier';
  return null;
};
