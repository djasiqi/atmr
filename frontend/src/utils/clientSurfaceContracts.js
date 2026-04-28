import { trackClientKpiEvent } from './clientKpi';

export const CLIENT_SURFACE_CONTRACTS = {
  statusDictionaryVersion: '1.0.0',
  pricingContractVersion: '1.0.0',
  canonicalAddressContractVersion: '1.0.0',
};

export function reportContractMismatch({
  contract,
  expected,
  received,
  surface = 'web',
}) {
  trackClientKpiEvent('status_dictionary_mismatch_event', {
    contract,
    expected,
    received: received || null,
    surface,
  });
}

