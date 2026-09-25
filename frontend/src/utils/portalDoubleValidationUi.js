/**
 * Helpers UI double validation PORTAL (7B / 7B.1).
 * Le flag serveur reste la source de vérité.
 */

export function isPortalDoubleValidationFlow(bookingLike) {
  const flow = String(bookingLike?.portal_contract_flow || '')
    .trim()
    .toLowerCase();
  return flow === 'double_validation_v2';
}

export function isPortalConditionalOrderFlow(bookingLike) {
  const flow = String(bookingLike?.portal_contract_flow || '')
    .trim()
    .toLowerCase();
  return flow === 'conditional_order_v1';
}

/** Plafond / pas Saferpay : DV ou commande conditionnelle. */
export function isPortalContractFlow(bookingLike) {
  return (
    isPortalDoubleValidationFlow(bookingLike) ||
    isPortalConditionalOrderFlow(bookingLike)
  );
}

export function isPortalDoubleValidationUiReady({
  enabled,
  estimate,
  maximum,
} = {}) {
  if (!enabled) {
    return { ok: true, estimateEqualsMaximum: false };
  }
  const est = Number(estimate);
  const max = Number(maximum);
  if (!Number.isFinite(max) || max <= 0) {
    return { ok: false, estimateEqualsMaximum: false };
  }
  if (Number.isFinite(est) && est > 0 && max < est) {
    return { ok: false, estimateEqualsMaximum: false };
  }
  return {
    ok: true,
    estimateEqualsMaximum: Number.isFinite(est) && est === max,
  };
}

/** Offre confirmable côté UI : ≤ plafond, hash présent, pas stale. */
export function isPortalOfferConfirmable(offer) {
  if (!offer || offer.id == null) return false;
  if (!offer.offer_content_hash) return false;
  const offered = Number(offer.offered_amount);
  if (!Number.isFinite(offered) || offered <= 0) return false;
  const max =
    offer.maximum_accepted_amount != null
      ? Number(offer.maximum_accepted_amount)
      : null;
  if (max != null && Number.isFinite(max) && offered > max) return false;
  return true;
}

export const PORTAL_DV_COPY = {
  firstClick:
    'Votre demande de transport a été enregistrée. Elle a été transmise aux entreprises de transport disponibles.',
  carrierOfferedTitle: 'Une proposition de transport est disponible',
  transportConfirmed: 'Votre transport est confirmé.',
};

/**
 * Affichage montant carte réservations.
 * DV v2 : ne jamais présenter l’estimation comme un tarif parallèle au plafond
 * (risque de confusion avec le prix à payer). Une fois le transport confirmé,
 * afficher le montant contractuel.
 *
 * @param {Record<string, unknown> | null | undefined} booking
 * @returns {{
 *   isCeiling: boolean,
 *   label: string,
 *   amount: number | null,
 *   estimate: number | null,
 *   ceiling: number | null,
 *   lines: Array<{ key: string, label: string, amount: number, primary?: boolean }>
 * }}
 */
export function portalReservationAmountDisplay(booking) {
  const estimateRaw = Number(
    booking?.estimated_amount_snapshot ?? booking?.amount
  );
  const estimate = Number.isFinite(estimateRaw) ? estimateRaw : null;

  if (isPortalDoubleValidationFlow(booking) || isPortalConditionalOrderFlow(booking)) {
    const contractualRaw = Number(
      booking?.contractual_amount ?? booking?.contractual_amount_snapshot
    );
    const contractual =
      Number.isFinite(contractualRaw) && contractualRaw > 0 ? contractualRaw : null;
    if (contractual != null) {
      return {
        isCeiling: false,
        label: 'Montant confirmé',
        amount: contractual,
        estimate,
        ceiling: null,
        lines: [
          {
            key: 'contractual',
            label: 'Montant confirmé',
            amount: contractual,
            primary: true,
          },
        ],
      };
    }

    const maxRaw = Number(
      booking?.maximum_accepted_amount ??
        booking?.maximum_accepted_amount_snapshot
    );
    const ceiling = Number.isFinite(maxRaw) && maxRaw > 0 ? maxRaw : null;
    if (ceiling != null) {
      // Uniquement le plafond : pas de ligne « Estimation indicative »
      // (induit en erreur à côté d’un prix maximum).
      return {
        isCeiling: true,
        label: 'Prix maximum de la demande',
        amount: ceiling,
        estimate,
        ceiling,
        lines: [
          {
            key: 'ceiling',
            label: 'Prix maximum accepté (pas le prix final)',
            amount: ceiling,
            primary: true,
          },
        ],
      };
    }
  }

  return {
    isCeiling: false,
    label: 'Montant',
    amount: estimate,
    estimate: null,
    ceiling: null,
    lines:
      estimate != null
        ? [{ key: 'amount', label: 'Montant', amount: estimate, primary: true }]
        : [],
  };
}

export function firstClickToastMessage({ doubleValidationEnabled } = {}) {
  if (doubleValidationEnabled) return PORTAL_DV_COPY.firstClick;
  return null;
}

/**
 * Affichage montant côté entreprise (marché ouvert PORTAL DV).
 * Ne jamais montrer l'estimation client ni le plafond.
 *
 * @param {Record<string, unknown> | null | undefined} reservation
 * @returns {{ mode: 'company_quote' | 'awaiting_quote' | 'standard', label: string, amount: number | null }}
 */
export function portalCarrierFacingAmountDisplay(reservation) {
  if (!isPortalContractFlow(reservation)) {
    const amt = Number(reservation?.amount);
    return {
      mode: 'standard',
      label: 'Montant',
      amount: Number.isFinite(amt) ? amt : null,
    };
  }
  const assigned =
    reservation?.company_id != null && Number(reservation.company_id) > 0;
  if (assigned) {
    const contractual = Number(
      reservation?.contractual_amount ?? reservation?.amount
    );
    return {
      mode: 'standard',
      label: 'Montant contractuel',
      amount: Number.isFinite(contractual) ? contractual : null,
    };
  }
  const suggested = Number(
    reservation?.carrier_quote ?? reservation?.company_suggested_amount
  );
  if (Number.isFinite(suggested) && suggested > 0) {
    return {
      mode: 'company_quote',
      label: 'Votre tarif (grille)',
      amount: suggested,
    };
  }
  return {
    mode: 'awaiting_quote',
    label: 'Votre tarif',
    amount: null,
  };
}

/**
 * Montant d'offre à envoyer à l'acceptation PORTAL DV (tarif grille viewer).
 * @param {Record<string, unknown> | null | undefined} reservation
 * @returns {number | null}
 */
export function portalCarrierAcceptOfferedAmount(reservation) {
  const disp = portalCarrierFacingAmountDisplay(reservation);
  if (disp.mode === 'company_quote' && disp.amount != null && disp.amount > 0) {
    return disp.amount;
  }
  return null;
}

/**
 * Libellé CTA acceptation entreprise.
 * @param {Record<string, unknown> | null | undefined} reservation
 * @returns {string}
 */
export function portalCarrierAcceptButtonLabel(reservation) {
  const amount = portalCarrierAcceptOfferedAmount(reservation);
  if (amount != null) {
    return `Accepter cette course à CHF ${amount.toFixed(2)}`;
  }
  if (isPortalContractFlow(reservation) && !reservation?.company_id) {
    return 'Accepter (tarif grille)';
  }
  return 'Accepter';
}

/**
 * Texte conditions d'annulation pour affichage client (sans pied technique).
 * @param {unknown} text
 * @returns {string}
 */
export function formatPortalCancellationPolicyForDisplay(text) {
  if (text == null) return '';
  return String(text)
    .replace(/\n*\[Réf\. config [^\]]+\]\s*$/i, '')
    .trim();
}

/**
 * Montant CHF formaté (2 décimales) ou chaîne vide si invalide.
 * @param {unknown} value
 * @returns {string}
 */
export function formatPortalOfferChf(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '';
  return `CHF ${n.toFixed(2)}`;
}
