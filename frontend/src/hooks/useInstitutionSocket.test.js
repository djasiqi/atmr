/** @jest-environment jsdom */

import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';

jest.mock('../services/institutionSocket', () => ({
  ensureInstitutionSocket: jest.fn().mockResolvedValue(undefined),
  getInstitutionSocket: jest.fn().mockReturnValue(null),
  joinInstitutionRoom: jest.fn().mockResolvedValue(undefined),
  leaveInstitutionRoom: jest.fn(),
  on: jest.fn().mockResolvedValue(undefined),
  off: jest.fn(),
  disconnectInstitutionSocket: jest.fn(),
}));

jest.mock('sonner', () => ({
  toast: { info: jest.fn() },
}));

import { useInstitutionSocket } from './useInstitutionSocket';
import { institutionQueryKeys } from './useInstitutionData';

function wrapper(queryClient) {
  return function Wrapper({ children }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
}

describe('useInstitutionSocket notifications merge', () => {
  it('merge une notification socket sans refetch et ignore les doublons', async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    queryClient.setQueryData(institutionQueryKeys.notifications(), {
      notifications: [{ id: 1, title: 'Existing', is_read: true, metadata: {} }],
      unread_count: 0,
    });

    const { result } = renderHook(() => useInstitutionSocket(99), {
      wrapper: wrapper(queryClient),
    });

    expect(result.current.chatActiveStatuses).toBeDefined();

    const { on } = require('../services/institutionSocket');
    await waitFor(() => expect(on).toHaveBeenCalled());

    const handlerCall = on.mock.calls.find((call) => call[0] === 'new_notification');
    expect(handlerCall).toBeTruthy();
    const handler = handlerCall[1];

    await act(async () => {
      handler({
        id: 2,
        title: 'Nouvelle',
        message: 'Message',
        is_read: false,
        metadata: {},
        event_type: 'request_sent',
      });
    });

    let cached = queryClient.getQueryData(institutionQueryKeys.notifications());
    expect(cached.notifications[0].id).toBe(2);
    expect(cached.unread_count).toBe(1);

    await act(async () => {
      handler({
        id: 2,
        title: 'Nouvelle',
        message: 'Message',
        is_read: false,
        metadata: {},
        event_type: 'request_sent',
      });
    });

    cached = queryClient.getQueryData(institutionQueryKeys.notifications());
    expect(cached.notifications.filter((n) => n.id === 2)).toHaveLength(1);
    expect(cached.unread_count).toBe(1);
  });
});
