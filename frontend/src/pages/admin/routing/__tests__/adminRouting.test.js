import { adminPaths, adminBasePath } from '../adminRoutePaths';
import {
  ADMIN_LEGACY_REDIRECT_SPECS,
  resolveLegacyRelativePath,
} from '../adminLegacyRedirects';

describe('adminRoutePaths', () => {
  const id = 'pub-1';

  it('construit les chemins cibles', () => {
    expect(adminBasePath(id)).toBe('/dashboard/admin/pub-1');
    expect(adminPaths.overview(id)).toBe('/dashboard/admin/pub-1');
    expect(adminPaths.operationsBookings(id)).toBe('/dashboard/admin/pub-1/operations/bookings');
    expect(adminPaths.operationsBooking(id, 'b42')).toBe(
      '/dashboard/admin/pub-1/operations/bookings/b42'
    );
    expect(adminPaths.partnersUsers(id)).toBe('/dashboard/admin/pub-1/partners/users');
    expect(adminPaths.partnersOrganizations(id)).toBe(
      '/dashboard/admin/pub-1/partners/organizations'
    );
    expect(adminPaths.financeReleves(id)).toBe('/dashboard/admin/pub-1/finance/releves');
    expect(adminPaths.financeFactures(id)).toBe('/dashboard/admin/pub-1/finance/factures');
    expect(adminPaths.configuration(id)).toBe('/dashboard/admin/pub-1/configuration');
    expect(adminPaths.advancedPlatform(id, 'runtime')).toBe(
      '/dashboard/admin/pub-1/advanced/platform/runtime'
    );
    expect(adminPaths.advancedLabsOptuna(id)).toBe(
      '/dashboard/admin/pub-1/advanced/labs/optuna'
    );
  });
});

describe('adminLegacyRedirects', () => {
  it('expose les specs de redirection', () => {
    expect(ADMIN_LEGACY_REDIRECT_SPECS.length).toBeGreaterThanOrEqual(10);
  });

  it('résout les deep links legacy', () => {
    expect(resolveLegacyRelativePath('reservations')).toBe('operations/bookings');
    expect(resolveLegacyRelativePath('reservations/bk-9')).toBe('operations/bookings/bk-9');
    expect(resolveLegacyRelativePath('platform-ops/runtime')).toBe('advanced/platform/runtime');
    expect(resolveLegacyRelativePath('platform-ops/investigation')).toBe(
      'advanced/platform/investigation'
    );
    expect(resolveLegacyRelativePath('platform-ops')).toBe('advanced/platform/overview');
    expect(resolveLegacyRelativePath('billing/releves')).toBe('finance/factures');
    expect(resolveLegacyRelativePath('billing/config')).toBe('finance/config');
    expect(resolveLegacyRelativePath('platform-billing')).toBe('finance/factures');
    expect(resolveLegacyRelativePath('finance/releves')).toBe('finance/factures');
    expect(resolveLegacyRelativePath('users')).toBe('partners/users');
    expect(resolveLegacyRelativePath('settings')).toBe('configuration');
    expect(resolveLegacyRelativePath('shadow-mode')).toBe('advanced/labs/shadow-mode');
    expect(resolveLegacyRelativePath('optuna')).toBe('advanced/labs/optuna');
  });

  it('ne fabrique pas de route billing/:periodId', () => {
    expect(resolveLegacyRelativePath('billing/123')).toBeNull();
  });
});
