/**
 * Helpers partagés d'affichage pour les factures LIRIE émises (registre + relevés).
 * Statuts alignés sur `ui_status` retourné par l'API (services/platform_billing/issued_status.py).
 */

export const STATUS_LABELS = {
  ISSUED: 'Émise',
  SENT: 'Envoyée',
  PARTIALLY_PAID: 'Partiellement payée',
  PAID: 'Payée',
  OVERDUE: 'En retard',
  CANCELLED: 'Annulée',
  CREDITED: 'Créditée',
};

/**
 * Résout la classe CSS de badge pour un `ui_status`, à partir d'un module CSS
 * exposant les clés badgeIssued/badgeSent/badgePartiallyPaid/badgePaid/badgeOverdue/
 * badgeCancelled/badgeCredited.
 * @param {string} uiStatus
 * @param {Record<string, string>} styleMap
 */
export const statusBadgeClass = (uiStatus, styleMap) => {
  switch (uiStatus) {
    case 'PAID':
      return styleMap.badgePaid;
    case 'SENT':
      return styleMap.badgeSent;
    case 'PARTIALLY_PAID':
      return styleMap.badgePartiallyPaid;
    case 'OVERDUE':
      return styleMap.badgeOverdue;
    case 'CANCELLED':
      return styleMap.badgeCancelled;
    case 'CREDITED':
      return styleMap.badgeCredited;
    default:
      return styleMap.badgeIssued;
  }
};

export const fmtMoney = (n) => {
  if (n == null || n === '') return '—';
  return `${String(n)} CHF`;
};

export const fmtDate = (iso) => {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  const day = String(d.getDate()).padStart(2, '0');
  const month = String(d.getMonth() + 1).padStart(2, '0');
  return `${day}.${month}.${d.getFullYear()}`;
};
