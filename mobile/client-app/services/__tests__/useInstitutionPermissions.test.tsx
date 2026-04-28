/* eslint-disable import/first */
import React from 'react';
import renderer, { act } from 'react-test-renderer';

jest.mock('@tanstack/react-query', () => ({
  useQuery: jest.fn(),
}));

jest.mock('@/services/featureFlags', () => ({
  featureFlags: {
    institutionMobileRoleGuardsEnabled: true,
  },
}));

import { useQuery } from '@tanstack/react-query';
import { useInstitutionPermissions } from '../useInstitutionPermissions';

const mockedUseQuery = useQuery as jest.Mock;

describe('useInstitutionPermissions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  function renderAndRead(role: string) {
    let latest: ReturnType<typeof useInstitutionPermissions> | undefined;
    mockedUseQuery.mockReturnValue({
      data: { institution_role: role },
      isLoading: false,
    });

    function Probe() {
      latest = useInstitutionPermissions();
      return null;
    }

    act(() => {
      renderer.create(<Probe />);
    });
    return latest as ReturnType<typeof useInstitutionPermissions>;
  }

  it('grants create/send for requester', () => {
    const out = renderAndRead('institution_requester');
    expect(out.canCreateRequest).toBe(true);
    expect(out.canSendRequest).toBe(true);
    expect(out.isReader).toBe(false);
  });

  it('blocks write actions for reader', () => {
    const out = renderAndRead('institution_reader');
    expect(out.canCreateRequest).toBe(false);
    expect(out.canSendRequest).toBe(false);
    expect(out.isReader).toBe(true);
  });

  it('allows notifications edit for billing only', () => {
    const billing = renderAndRead('institution_billing');
    const reader = renderAndRead('institution_reader');
    expect(billing.canEditNotifications).toBe(true);
    expect(reader.canEditNotifications).toBe(false);
  });
});
