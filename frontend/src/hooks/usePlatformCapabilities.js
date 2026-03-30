import { useQuery } from '@tanstack/react-query';
import { fetchPlatformMe } from '../services/adminService';

/** Segments sous /platform-ops/:segment */
export const PLATFORM_SEGMENTS = [
  'overview',
  'tenants',
  'runbooks',
  'audit',
  'runtime',
  'reconciliation',
  'investigation',
];

/**
 * Bundles requis par segment (tous doivent être présents dans bundles_effective pour accès).
 * Si bundles_effective est absent/vide, accès complet (compat admin legacy).
 */
const SEGMENT_REQUIRED_BUNDLES = {
  overview: ['observe_core'],
  tenants: ['operate_tenant_controls'],
  runbooks: ['operate_tenant_controls'],
  audit: ['observe_core'],
  runtime: ['observe_core'],
  reconciliation: ['operate_tenant_controls'],
  investigation: ['observe_core'],
};

function hasFullPlatformAccess(data) {
  const bundles = data?.platform?.bundles_effective;
  if (!bundles || !Array.isArray(bundles) || bundles.length === 0) {
    return true;
  }
  return false;
}

export function canAccessPlatformSegment(meData, segment) {
  if (!PLATFORM_SEGMENTS.includes(segment)) {
    return false;
  }
  if (hasFullPlatformAccess(meData)) {
    return true;
  }
  const bundles = meData?.platform?.bundles_effective || [];
  const required = SEGMENT_REQUIRED_BUNDLES[segment] || [];
  return required.every((b) => bundles.includes(b));
}

export function usePlatformCapabilities() {
  const query = useQuery({
    queryKey: ['platform', 'me'],
    queryFn: fetchPlatformMe,
    staleTime: 3 * 60 * 1000,
    retry: 1,
  });

  const canAccess = (segment) => canAccessPlatformSegment(query.data, segment);

  return {
    ...query,
    canAccess,
    bundlesEffective: query.data?.platform?.bundles_effective ?? null,
    permissionsEffective: query.data?.platform?.permissions_effective ?? null,
  };
}
