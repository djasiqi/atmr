import {
  ADMIN_WORKSPACES,
  getAdminRelativePath,
  resolveActiveWorkspace,
} from '../adminNavRegistry';

describe('adminNavRegistry', () => {
  it('expose six workspaces', () => {
    expect(ADMIN_WORKSPACES).toHaveLength(6);
    expect(ADMIN_WORKSPACES.map((w) => w.id)).toEqual([
      'overview',
      'operations',
      'partners',
      'finance',
      'configuration',
      'advanced',
    ]);
  });

  it('résout le workspace depuis le chemin relatif', () => {
    expect(resolveActiveWorkspace('').id).toBe('overview');
    expect(resolveActiveWorkspace('operations/bookings').id).toBe('operations');
    expect(resolveActiveWorkspace('partners/users').id).toBe('partners');
    expect(resolveActiveWorkspace('finance/releves').id).toBe('finance');
    expect(resolveActiveWorkspace('configuration').id).toBe('configuration');
    expect(resolveActiveWorkspace('advanced/platform/runtime').id).toBe('advanced');
  });

  it('extrait le chemin relatif admin', () => {
    expect(getAdminRelativePath('/dashboard/admin/abc/operations/bookings', 'abc')).toBe(
      'operations/bookings'
    );
    expect(getAdminRelativePath('/dashboard/admin/abc', 'abc')).toBe('');
  });

    it('porte platformCapability uniquement sur les enfants Platform Ops', () => {
    const advanced = ADMIN_WORKSPACES.find((w) => w.id === 'advanced');
    const platformChildren = advanced.children.filter((c) => c.platformCapability);
    const labs = advanced.children.filter((c) => c.adminCapability);
    expect(platformChildren.length).toBe(7);
    expect(labs.map((c) => c.id)).toEqual(['labs-shadow', 'labs-optuna']);
    expect(
      ADMIN_WORKSPACES.filter((w) => w.id !== 'advanced').every(
        (w) => !(w.children || []).some((c) => c.platformCapability)
      )
    ).toBe(true);
  });
});
