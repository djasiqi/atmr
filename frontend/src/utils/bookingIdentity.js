/**
 * Lit le bloc API `identity` ou reconstruit un fallback minimal (transition).
 */

const DEFAULT_PASSENGER = 'Non spécifié';

function warnMissingDisplayModel(booking) {
  if (process.env.NODE_ENV === 'production') return;
  if (!booking || booking.display_model === 'booking') return;
  // eslint-disable-next-line no-console
  console.warn(
    '[canonical-display] payload sans display_model booking — fallback legacy',
    { id: booking?.id },
  );
}

function legacyPassengerName(booking) {
  return (
    booking?.identity?.passenger?.name
    || booking?.client?.full_name
    || booking?.client_name
    || booking?.patient_name
    || DEFAULT_PASSENGER
  );
}

function legacySource(booking) {
  if (booking?.identity?.source) {
    return booking.identity.source;
  }
  const institutionName = booking?.client?.institution_name
    || booking?.institution_timeline?.institution_name;
  if (institutionName) {
    return {
      type: 'institution',
      id: booking?.client?.linked_institution_id ?? null,
      code: null,
      name: institutionName,
    };
  }
  // Origine portail institution : ne jamais retomber sur « Portefeuille propre ».
  if (booking?.created_via === 'institution_portal') {
    return {
      type: 'institution',
      id: booking?.client?.linked_institution_id ?? null,
      code: null,
      name: booking?.medical_facility || 'Institution',
    };
  }
  if (booking?.created_via === 'public_guest') {
    return { type: 'lirie_guest', id: null, code: 'GUEST', name: 'Invité LIRIE' };
  }
  if (booking?.client?.client_type === 'PORTAL' || booking?.created_via === 'client_app') {
    return { type: 'lirie_client', id: booking?.client?.id ?? null, code: 'LIRIE', name: 'LIRIE' };
  }
  if (booking?.client?.is_institution && booking?.client?.institution_name) {
    return {
      type: 'company_account',
      id: booking?.client?.id ?? null,
      code: null,
      name: booking.client.institution_name,
    };
  }
  return {
    type: 'company_client',
    id: booking?.client?.id ?? null,
    code: null,
    name: 'Portefeuille propre',
  };
}

/** @returns {import('./bookingIdentity.types').BookingIdentityView} */
export function buildIdentityFromApi(booking) {
  if (!booking) {
    return {
      passengerLabel: DEFAULT_PASSENGER,
      source: { type: 'legacy', id: null, code: null, name: DEFAULT_PASSENGER },
      requester: null,
      ownership: null,
      execution: null,
      upstream: null,
      originChannel: 'legacy',
    };
  }

  const identity = booking.identity;
  if (identity) {
    if (!booking.display_model) {
      warnMissingDisplayModel(booking);
    }
    return {
      passengerLabel: identity.primary_label || identity.passenger?.name || DEFAULT_PASSENGER,
      source: identity.source || legacySource(booking),
      requester: identity.requester || null,
      ownership: identity.ownership || null,
      execution: identity.execution || null,
      upstream: identity.upstream || null,
      originChannel: identity.origin_channel || booking.created_via || 'legacy',
    };
  }

  warnMissingDisplayModel(booking);
  return {
    passengerLabel: legacyPassengerName(booking),
    source: legacySource(booking),
    requester: booking?.institution_timeline?.created_by_name
      ? { id: null, name: booking.institution_timeline.created_by_name }
      : null,
    ownership: booking?.company_id
      ? {
          owner_company_id: booking.company_id,
          owner_company_name: booking.company_name || null,
        }
      : null,
    execution: {
      executing_company_id: booking.executing_company_id || booking.company_id || null,
      executing_company_name: booking.executing_company_name || booking.company_name || null,
    },
    upstream: null,
    originChannel: booking?.created_via || 'legacy',
  };
}

export function buildOfferIdentity(offer) {
  const req = offer?.transport_request || {};
  const apiIdentity = req.identity;
  if (apiIdentity?.primary_label) {
    return {
      passengerLabel: apiIdentity.primary_label,
      source: apiIdentity.source || {
        type: 'institution',
        id: req.institution_id ?? null,
        code: null,
        name: apiIdentity.secondary_label || req.institution_name || 'Institution',
      },
      requester: apiIdentity.requester || null,
      ownership: null,
      execution: null,
      upstream: null,
      originChannel: 'institution_portal',
    };
  }
  const passenger = req.patient_name || null;
  const institution = req.institution_name || 'Institution';
  const requesterName = req.contact_on_site?.requester_name || null;

  return {
    passengerLabel: passenger || institution,
    source: {
      type: 'institution',
      id: req.institution_id ?? null,
      code: null,
      name: passenger ? institution : null,
    },
    requester: requesterName ? { id: null, name: requesterName } : null,
    ownership: null,
    execution: null,
    upstream: null,
    originChannel: 'institution_portal',
  };
}

export function matchesSearchIndex(booking, query) {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return true;
  const index = booking?.search_index;
  if (Array.isArray(index) && index.length > 0) {
    return index.some((token) => String(token).toLowerCase().includes(q));
  }
  const identity = buildIdentityFromApi(booking);
  const haystack = [
    identity.passengerLabel,
    identity.source?.name,
    identity.upstream?.name,
    identity.requester?.name,
  ].filter(Boolean).join(' ').toLowerCase();
  return haystack.includes(q);
}
