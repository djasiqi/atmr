import React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  institutionQueryKeys,
  useChangeBillingControlPayer,
  useReopenBillingControlBooking,
  useValidateBillingControlBooking,
} from '../useInstitutionData';
import institutionBillingControlService from '../../services/institutionBillingControlService';

jest.mock('../../services/institutionBillingControlService', () => ({
  __esModule: true,
  default: {
    validateBillingControlBooking: jest.fn(),
    changeBillingControlPayer: jest.fn(),
    reopenBillingControlBooking: jest.fn(),
    markBillingControlAnomaly: jest.fn(),
    listBillingControlBookings: jest.fn(),
  },
}));

jest.mock('../../services/institutionService', () => ({
  __esModule: true,
  default: {},
}));

function createWrapper(client) {
  return function Wrapper({ children }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

describe('useValidateBillingControlBooking', () => {
  it('met à jour le cache avant la réponse serveur', async () => {
    let resolveMut;
    institutionBillingControlService.validateBillingControlBooking.mockImplementation(
      () => new Promise((resolve) => { resolveMut = resolve; }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const key = institutionQueryKeys.billingControlList({ period: '2026-09' });
    client.setQueryData(key, {
      items: [{ booking_id: 45708, control: { effective_status: 'pending_review' } }],
      summary: { validated: 0, pending_review: 1, anomaly: 0 },
    });

    const { result } = renderHook(() => useValidateBillingControlBooking(), {
      wrapper: createWrapper(client),
    });

    act(() => {
      result.current.mutate({ bookingId: 45708, data: {} });
    });

    await waitFor(() => {
      const cached = client.getQueryData(key);
      expect(cached.items[0].control.effective_status).toBe('validated');
      expect(cached.summary.validated).toBe(1);
      expect(cached.summary.pending_review).toBe(0);
    });

    await act(async () => {
      resolveMut({
        success: true,
        control: { control_status: 'validated', validated_by_display_name: 'Marc' },
      });
    });
  });

  it('change le payeur dans le cache avant la réponse serveur', async () => {
    let resolveMut;
    institutionBillingControlService.changeBillingControlPayer.mockImplementation(
      () => new Promise((resolve) => { resolveMut = resolve; }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const key = institutionQueryKeys.billingControlList({ period: '2026-09' });
    client.setQueryData(key, {
      items: [{
        booking_id: 45708,
        payer: { type: 'patient' },
        control: { effective_status: 'pending_review' },
      }],
      summary: { payer_clinic: 0, payer_patient: 1, validated: 0, pending_review: 1 },
    });

    const { result } = renderHook(() => useChangeBillingControlPayer(), {
      wrapper: createWrapper(client),
    });

    act(() => {
      result.current.mutate({
        bookingId: 45708,
        payerType: 'clinic',
        generation: 1,
        data: { billing_intent: 'institution' },
      });
    });

    await waitFor(() => {
      const cached = client.getQueryData(key);
      expect(cached.items[0].payer.type).toBe('clinic');
      expect(cached.summary.payer_clinic).toBe(1);
      expect(cached.summary.payer_patient).toBe(0);
    });

    await act(async () => {
      resolveMut({ success: true });
    });
  });

  it('réouvre un booking validé dans le cache avant la réponse serveur', async () => {
    let resolveMut;
    institutionBillingControlService.reopenBillingControlBooking.mockImplementation(
      () => new Promise((resolve) => { resolveMut = resolve; }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const key = institutionQueryKeys.billingControlList({ period: '2026-09' });
    client.setQueryData(key, {
      items: [{
        booking_id: 202,
        control: { effective_status: 'validated', validated_by_display_name: 'Marc' },
      }],
      summary: { validated: 1, pending_review: 0, anomaly: 0 },
    });

    const { result } = renderHook(() => useReopenBillingControlBooking(), {
      wrapper: createWrapper(client),
    });

    act(() => {
      result.current.mutate({ bookingId: 202, data: {} });
    });

    await waitFor(() => {
      const cached = client.getQueryData(key);
      expect(cached.items[0].control.effective_status).toBe('pending_review');
      expect(cached.summary.validated).toBe(0);
      expect(cached.summary.pending_review).toBe(1);
    });

    await act(async () => {
      resolveMut({ success: true, control: { control_status: 'pending_review' } });
    });
  });
});
