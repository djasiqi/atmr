/**
 * Saisie JJ.MM.AAAA / JJ-MM-AAAA / JJ/MM/AAAA / 8 chiffres / ISO → `YYYY-MM-DD` pour l’API factures.
 */
export function normalizeServiceDateToIsoForApi(raw) {
  if (raw == null) return null;
  const s0 = String(raw)
    .trim()
    .replace(/[\u200b\ufeff]/g, '')
    .replace(/\u00a0/g, ' ')
    .replace(/,/g, '.');
  if (!s0) return null;
  if (/^\d{4}-\d{2}-\d{2}/.test(s0)) {
    const head = s0.slice(0, 10);
    if (/^\d{4}-\d{2}-\d{2}$/.test(head)) return head;
  }
  const tryYmd = (d, mo, y) => {
    if (mo < 1 || mo > 12 || d < 1 || d > 31) return null;
    const dt = new Date(y, mo - 1, d);
    if (dt.getFullYear() !== y || dt.getMonth() !== mo - 1 || dt.getDate() !== d) return null;
    const dd = String(d).padStart(2, '0');
    const mm = String(mo).padStart(2, '0');
    return `${y}-${mm}-${dd}`;
  };
  const mDot = /^(\d{1,2})[./](\d{1,2})[./](\d{4})$/.exec(s0);
  if (mDot) {
    const iso = tryYmd(parseInt(mDot[1], 10), parseInt(mDot[2], 10), parseInt(mDot[3], 10));
    if (iso) return iso;
  }
  const mDash = /^(\d{1,2})-(\d{1,2})-(\d{4})$/.exec(s0);
  if (mDash) {
    const iso = tryYmd(parseInt(mDash[1], 10), parseInt(mDash[2], 10), parseInt(mDash[3], 10));
    if (iso) return iso;
  }
  const dig = s0.replace(/\D/g, '');
  if (dig.length === 8) {
    const d = parseInt(dig.slice(0, 2), 10);
    const mo = parseInt(dig.slice(2, 4), 10);
    const y = parseInt(dig.slice(4, 8), 10);
    return tryYmd(d, mo, y);
  }
  return null;
}
