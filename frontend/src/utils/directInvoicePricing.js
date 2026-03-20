/**
 * Facturation directe — règles de montants et remise globale (frontend uniquement).
 * Aligné sur round_to_5_cents backend (multiples 0,05 CHF, arrondi demi-au-supérieur).
 */

/**
 * @param {unknown} n
 * @returns {number}
 */
export function roundTo005(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return 0;
  const rounded = Math.round(x * 20) / 20;
  return Math.round(rounded * 100) / 100;
}

/**
 * @param {unknown} a
 * @param {unknown} b
 * @param {number} [tolerance=0.01]
 */
export function isSameMoney(a, b, tolerance = 0.01) {
  const x = Number(a);
  const y = Number(b);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
  return Math.abs(x - y) <= tolerance;
}

/**
 * Montant affiché (une seule définition produit).
 * 1) override.amount si nombre fini
 * 2) sinon reservation.amount
 * 3) sinon reservation.estimated_amount
 * 4) sinon 0
 *
 * @param {object} reservation
 * @param {object} [override]
 */
export function getDisplayedLineAmount(reservation, override = {}) {
  const o = override || {};
  if (o.amount !== undefined && o.amount !== null && o.amount !== '') {
    const v = Number(o.amount);
    if (Number.isFinite(v)) return v;
  }
  if (reservation?.amount !== undefined && reservation?.amount !== null && reservation?.amount !== '') {
    const v = Number(reservation.amount);
    if (Number.isFinite(v)) return v;
  }
  if (
    reservation?.estimated_amount !== undefined &&
    reservation?.estimated_amount !== null &&
    reservation?.estimated_amount !== ''
  ) {
    const v = Number(reservation.estimated_amount);
    if (Number.isFinite(v)) return v;
  }
  return 0;
}

/**
 * @param {object} args
 * @param {object} args.reservation
 * @param {object} [args.prevOverride]
 * @param {'adjusted'|'catalog'} args.discountBaseMode
 * @returns {{ catalogAmount: number, chosenBaseAmount: number, source: string }}
 */
export function resolveDiscountBaseForLine({ reservation, prevOverride, discountBaseMode }) {
  const r = reservation || {};
  const prev = prevOverride || {};
  const catalogRaw = r.amount ?? r.estimated_amount ?? 0;
  const catalogAmount = Number.isFinite(Number(catalogRaw)) ? Number(catalogRaw) : 0;

  if (discountBaseMode === 'catalog') {
    return {
      catalogAmount,
      chosenBaseAmount: catalogAmount,
      source: 'catalog',
    };
  }

  const meta = prev.pricingMeta || {};
  const metaBase = meta.baseBeforeDiscount;
  if (metaBase !== undefined && metaBase !== null && Number.isFinite(Number(metaBase))) {
    return {
      catalogAmount,
      chosenBaseAmount: Number(metaBase),
      source: 'pricingMeta.baseBeforeDiscount',
    };
  }

  if (prev.amount !== undefined && prev.amount !== null && prev.amount !== '') {
    const pv = Number(prev.amount);
    if (Number.isFinite(pv)) {
      return {
        catalogAmount,
        chosenBaseAmount: pv,
        source: 'prevOverride.amount',
      };
    }
  }

  if (r.amount !== undefined && r.amount !== null && r.amount !== '') {
    const v = Number(r.amount);
    if (Number.isFinite(v)) {
      return {
        catalogAmount,
        chosenBaseAmount: v,
        source: 'reservation.amount',
      };
    }
  }

  if (r.estimated_amount !== undefined && r.estimated_amount !== null && r.estimated_amount !== '') {
    const v = Number(r.estimated_amount);
    if (Number.isFinite(v)) {
      return {
        catalogAmount,
        chosenBaseAmount: v,
        source: 'reservation.estimated_amount',
      };
    }
  }

  return {
    catalogAmount,
    chosenBaseAmount: 0,
    source: 'zero',
  };
}

/**
 * @param {number} baseAmount
 * @param {number} percent
 */
export function applyPercentDiscount(baseAmount, percent) {
  const base = Number(baseAmount);
  const p = Number(percent);
  if (!Number.isFinite(base) || !Number.isFinite(p)) return 0;
  return roundTo005(base * (1 - p / 100));
}

/**
 * Remise globale sur le sous-total HT (lignes au tarif normal, pas par ligne).
 *
 * Arrondi CHF (0,05) : le **pourcentage n’est pas arrondi**. On calcule
 * `subtotalHt × percent / 100`, puis **un seul** `roundTo005` sur le montant de remise.
 * Le net HT est `subtotal − remise`, puis arrondi 0,05 (aligné backend `round_to_5_cents`).
 * Aucun arrondi « par ligne » pour la remise globale.
 *
 * @param {number} subtotalHt
 * @param {number} percent
 * @returns {{ discountAmountHt: number, netHtAfter: number }}
 */
/**
 * Interprète le champ « Remise % » pendant la saisie : null si vide, incomplet ou hors plage.
 * Permet une mise à jour automatique des totaux sans bouton « Appliquer ».
 *
 * @param {unknown} value
 * @returns {number|null}
 */
export function parseGlobalDiscountPercentField(value) {
  const s = String(value ?? '')
    .replace(/,/g, '.')
    .trim();
  if (s === '' || s === '.') return null;
  // Saisie en cours : « 25. » — on n’applique pas encore
  if (/^\d+\.$/.test(s)) return null;
  const p = parseFloat(s);
  if (!Number.isFinite(p) || p <= 0 || p > 100) return null;
  return p;
}

export function computeGlobalPercentDiscountOnSubtotal(subtotalHt, percent) {
  const sub = Number(subtotalHt);
  const p = Number(percent);
  if (!Number.isFinite(sub) || sub <= 0 || !Number.isFinite(p) || p <= 0) {
    return { discountAmountHt: 0, netHtAfter: roundTo005(sub) };
  }
  const rawDisc = sub * (p / 100);
  const discountAmountHt = roundTo005(rawDisc);
  let netHtAfter = roundTo005(sub - discountAmountHt);
  if (netHtAfter < 0) netHtAfter = 0;
  return { discountAmountHt, netHtAfter };
}

/**
 * Badges source prix (facturation directe).
 * @param {object} reservation
 * @param {object} [override]
 * @param {{ suspectThresholdChf?: number }} [opts]
 * @returns {{ key: string, label: string }[]}
 */
export function getDirectLinePriceBadges(reservation, override = {}, opts = {}) {
  const threshold = opts.suspectThresholdChf ?? 5;
  if (!reservation || reservation.id == null) return [];

  const displayed = getDisplayedLineAmount(reservation, override);
  const ovAmt = override?.amount;
  const hasOv =
    ovAmt !== undefined && ovAmt !== null && ovAmt !== '' && Number.isFinite(Number(ovAmt));
  const resAmt = reservation.amount;
  const resOk =
    resAmt !== undefined && resAmt !== null && resAmt !== '' && Number.isFinite(Number(resAmt));
  const est = reservation.estimated_amount;
  const estOk =
    est !== undefined && est !== null && est !== '' && Number.isFinite(Number(est));

  const tags = [];

  if (hasOv && resOk && !isSameMoney(Number(ovAmt), Number(resAmt))) {
    tags.push({ key: 'corrected', label: 'Corrigé' });
  } else if (
    resOk &&
    (!hasOv || isSameMoney(Number(ovAmt), Number(resAmt))) &&
    isSameMoney(displayed, Number(resAmt))
  ) {
    tags.push({ key: 'catalog', label: 'Catalogue' });
  } else if (
    estOk &&
    isSameMoney(displayed, Number(est)) &&
    !(resOk && isSameMoney(displayed, Number(resAmt)))
  ) {
    tags.push({ key: 'estimate', label: 'Estimé' });
  }

  if (displayed > 0 && displayed < threshold) {
    tags.push({ key: 'suspect_low', label: 'Montant suspect' });
  }

  return tags;
}

export function directLineHasSuspectAmount(reservation, override, opts) {
  return getDirectLinePriceBadges(reservation, override, opts).some((b) => b.key === 'suspect_low');
}

/**
 * Ancienne remise au niveau ligne (metadata) ou note explicite : risque de double remise
 * si l’utilisateur ajoute une remise globale en synthèse.
 *
 * @param {object} [override]
 * @returns {boolean}
 */
export function lineOverrideMaySuggestPriorDiscount(override) {
  if (!override || typeof override !== 'object') return false;
  const meta = override.pricingMeta;
  if (meta && typeof meta === 'object') {
    if (meta.discountPercent != null && String(meta.discountPercent).trim() !== '') {
      return true;
    }
    if (meta.baseBeforeDiscount != null && Number.isFinite(Number(meta.baseBeforeDiscount))) {
      return true;
    }
  }
  const note = String(override.note || '').toLowerCase();
  if (!note) return false;
  if (/\bremise\b/.test(note) || /\bremisé/.test(note) || /\bremises\b/.test(note)) {
    return true;
  }
  if (/\b\d+\s*%/.test(note) && /\b(remise|rabais|ristourne)\b/.test(note)) {
    return true;
  }
  return false;
}

/**
 * @param {object[]} reservations
 * @param {Record<string, object>} overrides
 */
export function directSelectionMaySuggestLineLevelDiscount(reservations, overrides) {
  if (!Array.isArray(reservations) || !overrides) return false;
  for (const r of reservations) {
    if (!r || r.id == null) continue;
    const o = overrides[String(r.id)] || overrides[r.id];
    if (lineOverrideMaySuggestPriorDiscount(o)) return true;
  }
  return false;
}
