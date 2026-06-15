import { isConvertedLirie, isExternalRequest } from './requestStatus';

/**
 * Indique si l'institution peut relancer la diffusion LIRIE
 * (aucune offre en attente, pas encore acceptée).
 */
export function canRelaunchInstitutionRequest(req) {
  if (!req || isExternalRequest(req) || isConvertedLirie(req)) return false;
  if (req.booking_id || req.accepted_by_company) return false;

  const dispatch = req?.dispatch;
  if (dispatch?.has_only_expired_pending) return true;
  if (typeof dispatch?.can_relaunch === 'boolean') {
    return dispatch.can_relaunch;
  }

  return ['SENT', 'EXPIRED'].includes(req.status);
}
