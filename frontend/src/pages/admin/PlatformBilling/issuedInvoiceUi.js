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

export const fmtDateTime = (iso) => {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  const day = String(d.getDate()).padStart(2, '0');
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  return `${day}.${month}.${d.getFullYear()} ${h}:${m}`;
};

const _fmtLabelNum = (v) => {
  const n = Number(String(v ?? '').replace(',', '.'));
  if (!Number.isFinite(n)) return null;
  const rounded = Math.round(n * 100) / 100;
  return String(rounded).replace(/\.?0+$/, '') || '0';
};

/**
 * Libellé d'affichage cohérent avec qté × prix (évite « 2 h » si qté = 1).
 */
export const displayInvoiceLineLabel = (line) => {
  const raw = (line?.label || line?.line_type || 'Ligne').trim();
  const lt = String(line?.line_type || '').toLowerCase();
  const isSupport = lt.includes('support') || /^support/i.test(raw);
  if (!isSupport) return raw || 'Ligne';
  const hours = _fmtLabelNum(line?.quantity);
  const rate = _fmtLabelNum(line?.unit_amount);
  if (hours == null) return raw || 'Ligne';
  if (rate != null) return `Support plateforme — ${hours} h à ${rate} CHF/h`;
  return `Support plateforme — ${hours} h`;
};
