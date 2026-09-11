/**
 * Mise à jour locale des listes de demandes institution.
 * Ne pas attendre un GET liste pour afficher create/send.
 */

/** Applique le payload PATCH booking sur le cache détail (RDV + reconfirmation). */
export function applyOperationalBookingPatchToRequest(request, payload = {}, response = {}) {
  if (!request) return request;

  const legs = Array.isArray(request.legs)
    ? request.legs.map((leg) => ({ ...leg }))
    : [];
  const sorted = [...legs].sort(
    (a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0),
  );
  const destCount = request.return_to_institution && sorted.length > 1
    ? sorted.length - 1
    : sorted.length;
  const destLegs = sorted.slice(0, destCount);
  const appointments = Array.isArray(payload.leg_appointments)
    ? payload.leg_appointments
    : [];

  appointments.forEach((item) => {
    const dest = destLegs[item?.index];
    if (!dest) return;
    if (Object.prototype.hasOwnProperty.call(item, 'scheduled_time')) {
      dest.scheduled_time = item.scheduled_time;
      dest.time_confirmed = item.scheduled_time != null;
    }
  });
  if (payload.appointment_time && destLegs[0] && appointments.length === 0) {
    destLegs[0].scheduled_time = payload.appointment_time;
    destLegs[0].time_confirmed = true;
  }

  const next = { ...request, legs };
  const reconfirm = Boolean(response?.pickup_reconfirmation_required);
  if (next.booking_summary) {
    next.booking_summary = {
      ...next.booking_summary,
      time_confirmed: reconfirm ? false : next.booking_summary.time_confirmed,
      edit_version: response?.edit_version ?? next.booking_summary.edit_version,
    };
  }
  if (reconfirm) {
    next.pickup_time_confirmed = false;
  }
  return next;
}

export function upsertInstitutionRequestInLists(oldData, request) {
  if (!request?.id || oldData == null) return oldData;

  if (Array.isArray(oldData)) {
    const idx = oldData.findIndex((row) => row?.id === request.id);
    if (idx >= 0) {
      const next = oldData.slice();
      next[idx] = { ...next[idx], ...request };
      return next;
    }
    return [request, ...oldData];
  }

  if (oldData.id === request.id && !Array.isArray(oldData.requests) && !Array.isArray(oldData.items)) {
    return { ...oldData, ...request };
  }

  const listKey = Array.isArray(oldData.requests)
    ? 'requests'
    : Array.isArray(oldData.items)
      ? 'items'
      : null;
  if (!listKey) return oldData;

  const list = oldData[listKey];
  const idx = list.findIndex((row) => row?.id === request.id);
  if (idx >= 0) {
    const next = list.slice();
    next[idx] = { ...next[idx], ...request };
    return { ...oldData, [listKey]: next };
  }

  return {
    ...oldData,
    [listKey]: [request, ...list],
    total: typeof oldData.total === 'number' ? oldData.total + 1 : oldData.total,
  };
}
