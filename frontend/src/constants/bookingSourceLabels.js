/** Libellés et icônes pour identity.source.type */
export const BOOKING_SOURCE_LABELS = {
  institution: { label: 'Institution', icon: '🏥' },
  partner_company: { label: 'Partenaire', icon: '🤝' },
  company_client: { label: 'Portefeuille', icon: '🏢' },
  company_account: { label: 'Compte entreprise', icon: '🏢' },
  lirie_client: { label: 'Plateforme LIRIE', icon: '🌐' },
  lirie_guest: { label: 'Invité LIRIE', icon: '🎫' },
  legacy: { label: 'Course', icon: null },
};

export function getBookingSourceMeta(sourceType) {
  const key = String(sourceType || 'legacy').toLowerCase();
  return BOOKING_SOURCE_LABELS[key] || BOOKING_SOURCE_LABELS.legacy;
}
