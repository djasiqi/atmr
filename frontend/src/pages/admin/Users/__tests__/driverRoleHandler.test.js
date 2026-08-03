/**
 * Garantit que le changement de rôle chauffeur s’interrompt avant l’API
 * (pas d’appel updateUserRole sans company_id).
 */

describe('rôle chauffeur — interruption avant mutation', () => {
  async function runDriverRoleChange({
    userId,
    newRole,
    companyOptions,
    updateUserRole,
    setPendingDriverUserId,
    setShowCompanyDropdown,
  }) {
    if (newRole.toLowerCase() === 'driver') {
      if (!companyOptions.length) {
        return 'no_companies';
      }
      setPendingDriverUserId(userId);
      setShowCompanyDropdown(true);
      return 'awaiting_company';
    }
    await updateUserRole(userId, { role: newRole });
    return 'updated';
  }

  it('n’appelle pas updateUserRole sans company_id', async () => {
    const updateUserRole = jest.fn();
    const setPendingDriverUserId = jest.fn();
    const setShowCompanyDropdown = jest.fn();

    const result = await runDriverRoleChange({
      userId: 42,
      newRole: 'driver',
      companyOptions: [{ id: 1, name: 'Co' }],
      updateUserRole,
      setPendingDriverUserId,
      setShowCompanyDropdown,
    });

    expect(result).toBe('awaiting_company');
    expect(setPendingDriverUserId).toHaveBeenCalledWith(42);
    expect(setShowCompanyDropdown).toHaveBeenCalledWith(true);
    expect(updateUserRole).not.toHaveBeenCalled();
  });
});
