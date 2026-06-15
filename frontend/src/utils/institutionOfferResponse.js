const PENDING_STATUS = 'PENDING';

/** Durée d'affichage d'une offre expirée avant masquage automatique (1h). */
export const EXPIRED_OFFER_VISIBLE_MS = 60 * 60 * 1000;

export const isInstitutionOfferExpired = (offer, now = new Date()) => {
  if (!offer?.expires_at) return false;

  const expiresAt = new Date(offer.expires_at);
  if (Number.isNaN(expiresAt.getTime())) return false;

  return expiresAt <= now;
};

/** Fallback client si can_respond est absent ou incohérent avec expires_at. */
export const canRespondToInstitutionOffer = (offer, now = new Date()) => {
  if (typeof offer?.can_respond === 'boolean') {
    if (!offer.can_respond) return false;
  } else if (offer?.status && String(offer.status).toUpperCase() !== PENDING_STATUS) {
    return false;
  }

  if (offer?.expires_at) {
    const expiresAt = new Date(offer.expires_at);
    if (!Number.isNaN(expiresAt.getTime())) {
      return expiresAt > now;
    }
  }

  return offer?.can_respond !== false;
};

/** Offre encore affichable dans le tableau entreprise (non masquée après expiration). */
export const isInstitutionOfferVisible = (offer, nowMs = Date.now()) => {
  if (!offer?.expires_at) return true;

  const expiresAt = new Date(offer.expires_at).getTime();
  if (Number.isNaN(expiresAt)) return true;
  if (expiresAt > nowMs) return true;

  return nowMs - expiresAt <= EXPIRED_OFFER_VISIBLE_MS;
};

export const filterVisibleInstitutionOffers = (offers, nowMs = Date.now()) =>
  (offers || []).filter((offer) => isInstitutionOfferVisible(offer, nowMs));
