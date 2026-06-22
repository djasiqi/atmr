/**
 * Libellé d'estimation tarifaire pour une offre institution (entreprise).
 */

/**
 * @param {{ source?: string } | null | undefined} estimate
 * @param {string | null | undefined} billingIntent
 * @returns {string}
 */
export function institutionOfferEstimateLabel(estimate, billingIntent) {
  const intent = String(billingIntent || 'patient').toLowerCase();
  const source = estimate?.source;

  if (source === 'preferential' && intent === 'institution') {
    return 'Tarif préférentiel';
  }
  if (source === 'company_profile' || source === 'profile') {
    return 'Tarif estimé (profil tarifaire)';
  }
  if (source === 'mixed') {
    return 'Tarif estimé (multi-payeurs)';
  }
  return 'Tarif estimé';
}
