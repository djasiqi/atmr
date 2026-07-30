/**
 * Vide les caches client scopés tenant (RQ + autocomplete adresses).
 * À appeler à la déconnexion et au changement d'entreprise.
 */
export function clearTenantScopedClientCaches() {
  try {
    // eslint-disable-next-line global-require
    const { queryClient } = require('../App');
    queryClient.clear();
  } catch (_) {
    // App pas encore chargé
  }
  try {
    // eslint-disable-next-line global-require
    const { clearAddressAutocompleteCache } = require('../components/common/AddressAutocomplete');
    clearAddressAutocompleteCache();
  } catch (_) {
    // ignore
  }
}
