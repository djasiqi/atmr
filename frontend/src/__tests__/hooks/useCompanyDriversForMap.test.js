import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import useCompanyDriversForMap from 'hooks/useCompanyDriversForMap';
import { fetchCompanyDriversCanonical } from 'services/companyService';

jest.mock('services/companyService');
jest.mock('hooks/useCompanySocket', () => ({
  useSocketConnected: jest.fn(() => false),
}));

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }) {
    return React.createElement(QueryClientProvider, { client: queryClient }, children);
  }
  return Wrapper;
}

describe('useCompanyDriversForMap', () => {
  beforeEach(() => {
    fetchCompanyDriversCanonical.mockResolvedValue([
      {
        id: 1,
        full_name: 'Alice Driver',
        email: 'alice@example.com',
        latitude: 46.2,
        longitude: 6.1,
        status: 'available',
      },
    ]);
  });

  it('projette les champs minimaux pour la carte', async () => {
    const { result } = renderHook(() => useCompanyDriversForMap(1), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.driversForMap).toHaveLength(1);
    });

    expect(result.current.driversForMap[0]).toMatchObject({
      id: 1,
      latitude: 46.2,
      longitude: 6.1,
      status: 'available',
    });
    expect(result.current.driversForMap[0].email).toBe('alice@example.com');
  });
});
