/**
 * Constantes partagées frontend (alignées avec backend shared.constants.ErrorCodes).
 * Évite les fautes de frappe lors du traitement des erreurs API.
 */

export const ErrorCodes = {
  /** Prix fixe livraison non configuré (création ou génération facture) */
  MATERIAL_DELIVERY_PRICE_NOT_CONFIGURED: 'MATERIAL_DELIVERY_PRICE_NOT_CONFIGURED',
  /** Description livraison manquante (création) */
  MATERIAL_DELIVERY_DESCRIPTION_REQUIRED: 'MATERIAL_DELIVERY_DESCRIPTION_REQUIRED',
};
