/**
 * Résout la cible de navigation pour une notification entreprise.
 * Parcours validation modification : cloche → dispatch → panneau latéral ouvert.
 */
export const resolveCompanyNotificationLink = ({
  notif,
  dashboardRoot,
  companyPublicId,
}) => {
  const meta = notif?.metadata || {};
  const base = `${dashboardRoot}/company/${companyPublicId}`;
  const bookingId = meta.booking_id;

  if (bookingId) {
    const params = new URLSearchParams({ booking: String(bookingId) });
    if (
      notif.event_type === 'institution_change_request'
      || meta.change_request_id
      || meta.focus === 'change_request'
    ) {
      params.set('focus', 'change_request');
    }
    return `${base}/dispatch?${params.toString()}`;
  }

  if (notif.event_type === 'new_request' || notif.event_type === 'request_updated') {
    const params = new URLSearchParams({ tab: 'institution' });
    if (meta.request_id) params.set('request', String(meta.request_id));
    if (meta.offer_id) params.set('offer', String(meta.offer_id));
    return `${base}?${params.toString()}`;
  }

  return base;
};
