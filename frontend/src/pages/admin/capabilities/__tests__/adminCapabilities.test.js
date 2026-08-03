import { ADMIN_CAP, hasAdminCapability } from '../adminCapabilities';

describe('adminCapabilities', () => {
  it('enforced=false autorise même avec liste partielle', () => {
    expect(
      hasAdminCapability([ADMIN_CAP.BILLING_LOCK], ADMIN_CAP.LABS_READ, {
        enforced: false,
      })
    ).toBe(true);
    expect(hasAdminCapability(null, ADMIN_CAP.LABS_READ, { enforced: false })).toBe(true);
  });

  it('enforced=true avec capacité présente', () => {
    expect(
      hasAdminCapability([ADMIN_CAP.LABS_READ], ADMIN_CAP.LABS_READ, { enforced: true })
    ).toBe(true);
  });

  it('enforced=true avec capacité absente', () => {
    expect(
      hasAdminCapability([ADMIN_CAP.BILLING_LOCK], ADMIN_CAP.LABS_READ, {
        enforced: true,
      })
    ).toBe(false);
    expect(hasAdminCapability([], ADMIN_CAP.LABS_READ, { enforced: true })).toBe(false);
    expect(hasAdminCapability(null, ADMIN_CAP.LABS_READ, { enforced: true })).toBe(false);
  });
});
