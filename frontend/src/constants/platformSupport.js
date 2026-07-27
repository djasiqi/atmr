/** Coordonnées support plateforme LIRIE (surchargeables via variables d'environnement CRA). */
export const LIRIE_SUPPORT_PHONE =
  String(process.env.REACT_APP_LIRIE_SUPPORT_PHONE || '022 512 02 03').trim();

export const LIRIE_SUPPORT_EMAIL =
  String(process.env.REACT_APP_LIRIE_SUPPORT_EMAIL || 'info@lirie.ch').trim();

/**
 * Message utilisateur lorsque l'API est injoignable (maintenance, panne, proxy).
 * @returns {string}
 */
export function getServiceUnavailableMessage() {
  const parts = [
    'Le service est momentanément indisponible (maintenance ou incident).',
    'Merci de contacter le support',
  ];

  if (LIRIE_SUPPORT_PHONE) {
    parts.push(`au ${LIRIE_SUPPORT_PHONE}`);
  }
  if (LIRIE_SUPPORT_EMAIL) {
    parts.push(LIRIE_SUPPORT_PHONE ? `ou par e-mail à ${LIRIE_SUPPORT_EMAIL}` : `à ${LIRIE_SUPPORT_EMAIL}`);
  }

  return `${parts.join(' ')}.`;
}
