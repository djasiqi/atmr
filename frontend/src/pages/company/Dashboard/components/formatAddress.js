/**
 * P2.1: Format court d'adresse pour la confirm bar (footer).
 * Heuristiques sans géocoding.
 *
 * @param {string} raw - Adresse brute
 * @returns {string} Adresse courte
 */
export function shortAddress(raw) {
  if (!raw || typeof raw !== 'string') return raw || '';

  let s = raw.trim().replace(/\s+/g, ' ').replace(/\s*,\s*/g, ', ');

  // Supprimer doublons courants : ", Suisse", ", Switzerland"
  s = s.replace(/,?\s*Suisse\s*$/i, '').replace(/,?\s*Switzerland\s*$/i, '').trim();

  // Si contient " - " ou " — " => garder la partie avant le séparateur
  const dashIdx = s.search(/\s+[-–—]\s+/);
  if (dashIdx > 0) {
    s = s.slice(0, dashIdx).trim();
  }

  // Si contient "," => prendre les 2 premiers segments significatifs max
  if (s.includes(',')) {
    const parts = s.split(',').map((p) => p.trim()).filter(Boolean);
    if (parts.length > 2) {
      s = [parts[0], parts[1]].join(', ');
    }
  }

  // Si longueur > 48 chars => tronquer avec ellipsis
  const maxLen = 48;
  if (s.length > maxLen) {
    s = s.slice(0, maxLen - 1).trim() + '…';
  }

  return s || raw;
}
