/**
 * Type de mission (transport patient / livraison matériel) — affichage partagé.
 */

export const MISSION_TYPE_PATIENT_TRANSPORT = 'patient_transport';
export const MISSION_TYPE_MATERIAL_DELIVERY = 'material_delivery';

export const MISSION_TYPE_LABELS = {
  [MISSION_TYPE_PATIENT_TRANSPORT]: 'Transport patient',
  [MISSION_TYPE_MATERIAL_DELIVERY]: 'Livraison',
};

export const MISSION_DELIVERY_BADGE = 'LIVRAISON';
export const MISSING_DELIVERY_DESCRIPTION = 'Description de livraison non renseignée';
export const DELIVERY_BENEFICIARY_LABEL = 'Bénéficiaire';
export const PATIENT_TRANSPORT_CREATE_LABEL = 'Transport de personne';

export function normalizeMissionType(value) {
  const key = String(value || '').trim().toLowerCase();
  return key || MISSION_TYPE_PATIENT_TRANSPORT;
}

export function formatMissionTypeLabel(value) {
  const key = normalizeMissionType(value);
  if (MISSION_TYPE_LABELS[key]) return MISSION_TYPE_LABELS[key];
  const readable = String(value || '').replace(/_/g, ' ').trim();
  if (!readable) return MISSION_TYPE_LABELS[MISSION_TYPE_PATIENT_TRANSPORT];
  return readable.charAt(0).toUpperCase() + readable.slice(1);
}

function readMissionTypeFromEntity(entity) {
  if (entity == null) return '';
  if (typeof entity === 'string') return entity;
  return (
    entity.mission_type
    || entity.__offer?.transport_request?.mission_type
    || entity.transport_request?.mission_type
    || ''
  );
}

export function isMaterialDelivery(entity) {
  return normalizeMissionType(readMissionTypeFromEntity(entity))
    === MISSION_TYPE_MATERIAL_DELIVERY;
}

export function getDeliveryDescription(entity) {
  if (!entity || typeof entity !== 'object') return '';
  const raw = entity.delivery_description
    ?? entity.__offer?.transport_request?.delivery_description
    ?? entity.transport_request?.delivery_description
    ?? '';
  return String(raw).trim();
}

export function formatDeliveryDescriptionDisplay(entity) {
  return getDeliveryDescription(entity) || MISSING_DELIVERY_DESCRIPTION;
}

export function formatDeliveryBeneficiaryLabel() {
  return DELIVERY_BENEFICIARY_LABEL;
}

export function buildDeliveryPresentation(entity) {
  if (!isMaterialDelivery(entity)) return null;
  const description = formatDeliveryDescriptionDisplay(entity);
  const beneficiary = (() => {
    if (!entity || typeof entity !== 'object') return '';
    const patient = entity.patient || entity.client || {};
    const first = String(patient.first_name || entity.customer_first_name || '').trim();
    const last = String(patient.last_name || entity.customer_last_name || '').trim();
    const full = `${first} ${last}`.trim()
      || String(patient.full_name || entity.customer_name || entity.client_name || '').trim();
    return full;
  })();
  return {
    badge: MISSION_DELIVERY_BADGE,
    typeLabel: formatMissionTypeLabel(MISSION_TYPE_MATERIAL_DELIVERY),
    description,
    beneficiaryLabel: DELIVERY_BENEFICIARY_LABEL,
    beneficiary: beneficiary || null,
    cargoLabel: 'À transporter',
  };
}
