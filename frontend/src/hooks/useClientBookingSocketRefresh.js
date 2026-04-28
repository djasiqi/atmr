import { useEffect, useRef } from 'react';
import { toast } from 'sonner';
import { ensureClientPortalSocket } from '../services/clientPortalSocket';

function showClientBookingMilestoneToast(payload) {
  const title = typeof payload?.title === 'string' ? payload.title.trim() : '';
  const body = typeof payload?.body === 'string' ? payload.body.trim() : '';
  if (title && body) {
    toast.info(title, { description: body, duration: 7000 });
    return;
  }
  const m = payload?.milestone;
  if (m === 'company_accepted') {
    toast.info('Transport confirmé', {
      description: 'Une entreprise a accepté votre demande de transport.',
      duration: 7000,
    });
    return;
  }
  if (m === 'driver_assigned') {
    toast.info('Chauffeur désigné', {
      description: 'Un chauffeur a été assigné à votre course.',
      duration: 7000,
    });
    return;
  }
  if (m === 'en_route') {
    toast.info('Chauffeur en route', {
      description: 'Le chauffeur est en route vers le lieu de prise en charge.',
      duration: 7000,
    });
    return;
  }
  toast.info('Votre course a été mise à jour.', { duration: 5000 });
}

/**
 * Écoute `client_booking_updated` (Socket.IO room client) et rafraîchit la liste des réservations.
 * @param {(quiet?: boolean) => Promise<unknown>} reloadBookings — ex. `reloadBookings` / `loadBookings`
 * @param {boolean} enabled — ex. `Boolean(effectivePublicId)`
 */
export function useClientBookingSocketRefresh(reloadBookings, enabled) {
  const reloadRef = useRef(reloadBookings);
  reloadRef.current = reloadBookings;

  useEffect(() => {
    if (!enabled) return undefined;

    let cancelled = false;
    let sock = null;

    const handler = (payload) => {
      showClientBookingMilestoneToast(payload);
      void Promise.resolve(reloadRef.current(true)).catch(() => {});
    };

    (async () => {
      sock = await ensureClientPortalSocket();
      if (cancelled || !sock) return;
      sock.on('client_booking_updated', handler);
    })();

    return () => {
      cancelled = true;
      if (sock) sock.off('client_booking_updated', handler);
    };
  }, [enabled]);
}
