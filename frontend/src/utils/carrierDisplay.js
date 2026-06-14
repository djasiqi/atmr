/**
 * Affichage unifié du transporteur (LIRIE ou externe).
 */

import { isExternalRequest } from './requestStatus';

const EMPTY = {
  type: 'none',
  name: null,
  phone: null,
  reference: null,
  email: null,
  badge: null,
  reason: null,
};

/**
 * @param {object|null|undefined} req
 * @returns {{
 *   type: 'none'|'lirie'|'external',
 *   name: string|null,
 *   phone: string|null,
 *   reference: string|null,
 *   email: string|null,
 *   badge: string|null,
 *   reason: string|null,
 * }}
 */
export function getCarrierDisplay(req) {
  if (!req) return { ...EMPTY };

  if (isExternalRequest(req)) {
    const ext = req.external_carrier || {};
    return {
      type: 'external',
      name: ext.name || null,
      phone: ext.phone || null,
      reference: ext.reference || null,
      email: null,
      badge: 'EXTERNE',
      reason: ext.reason || null,
    };
  }

  const lirie = req.accepted_by_company || {};
  return {
    type: 'lirie',
    name: lirie.name || null,
    phone: lirie.contact_phone || null,
    reference: null,
    email: lirie.contact_email || null,
    badge: null,
    reason: null,
  };
}
