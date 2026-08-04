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

  it('enforced=true : alias partners déjà développé côté backend dans capabilities_effective', () => {
    // Le backend développe CAPABILITY_ALIASES dans capabilities_effective ;
    // le frontend vérifie une correspondance exacte sur la liste reçue.
    const fromBackend = [
      ADMIN_CAP.PARTNERS_READ,
      ADMIN_CAP.ORGANIZATIONS_READ,
      ADMIN_CAP.ACCOUNTS_READ,
    ];
    expect(
      hasAdminCapability(fromBackend, ADMIN_CAP.ORGANIZATIONS_READ, { enforced: true })
    ).toBe(true);
    expect(
      hasAdminCapability(fromBackend, ADMIN_CAP.ACCOUNTS_READ, { enforced: true })
    ).toBe(true);
    expect(
      hasAdminCapability([ADMIN_CAP.PARTNERS_READ], ADMIN_CAP.ORGANIZATIONS_READ, {
        enforced: true,
      })
    ).toBe(false);
  });
});
