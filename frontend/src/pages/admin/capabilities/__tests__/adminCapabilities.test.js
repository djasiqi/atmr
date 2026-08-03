import { ADMIN_CAP, hasAdminCapability } from '../adminCapabilities';

describe('adminCapabilities', () => {
  it('autorise pendant le chargement (liste vide)', () => {
    expect(hasAdminCapability(null, ADMIN_CAP.LABS_READ)).toBe(true);
    expect(hasAdminCapability([], ADMIN_CAP.LABS_READ)).toBe(true);
  });

  it('respecte la liste effective', () => {
    expect(hasAdminCapability([ADMIN_CAP.LABS_READ], ADMIN_CAP.LABS_READ)).toBe(true);
    expect(hasAdminCapability([ADMIN_CAP.BILLING_LOCK], ADMIN_CAP.LABS_READ)).toBe(false);
  });
});
