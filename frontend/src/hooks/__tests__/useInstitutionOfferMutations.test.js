import React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { lirieKeys } from '../../queryKeys/lirie';
import { useInstitutionOfferMutations } from '../useInstitutionOfferMutations';
import {
  countVisibleInstitutionOffers,
  resetLocalInstitutionOfferDecisions,
} from '../../utils/institutionOffersCache';
import * as companyService from '../../services/companyService';

jest.mock('../../services/companyService', () => ({
  acceptRequestOffer: jest.fn(),
  rejectRequestOffer: jest.fn(),
  fetchRequestOffer: jest.fn(),
}));

jest.mock('sonner', () => ({
  toast: { error: jest.fn(), success: jest.fn(), info: jest.fn() },
}));

function createWrapper(client) {
  return function Wrapper({ children }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

const offer = (id) => ({
  id,
  status: 'PENDING',
  can_respond: true,
  expires_at: '2030-01-01T12:00:00Z',
  transport_request: {
    patient_name: `Patient ${id}`,
    pickup_location: 'A',
    dropoff_location: 'B',
    mission_date: '2026-09-11',
    scheduled_time: '2026-09-11T14:00:00',
  },
});

describe('useInstitutionOfferMutations', () => {
  let client;

  beforeEach(() => {
    resetLocalInstitutionOfferDecisions();
    jest.clearAllMocks();
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(lirieKeys.institutionOffers(), {
      offers: [offer(10), offer(11), offer(12)],
      total: 3,
    });
  });

  it('refuse : retire immédiatement et décrémente le badge', async () => {
    let resolveReject;
    companyService.rejectRequestOffer.mockImplementation(
      () => new Promise((resolve) => { resolveReject = resolve; }),
    );

    const { result } = renderHook(
      () => useInstitutionOfferMutations({ dispatchDay: '2026-09-11', companyId: 1 }),
      { wrapper: createWrapper(client) },
    );

    act(() => {
      result.current.rejectOffer(11, offer(11));
    });

    const afterClick = client.getQueryData(lirieKeys.institutionOffers());
    expect(afterClick.offers.map((o) => o.id)).toEqual([10, 12]);
    expect(countVisibleInstitutionOffers(afterClick.offers)).toBe(2);

    await act(async () => {
      resolveReject({ success: true });
    });
    expect(countVisibleInstitutionOffers(
      client.getQueryData(lirieKeys.institutionOffers()).offers,
    )).toBe(2);
  });

  it('refuse en échec : restaure l\'offre et le badge', async () => {
    companyService.rejectRequestOffer.mockRejectedValue(new Error('500'));
    const { result } = renderHook(
      () => useInstitutionOfferMutations({ dispatchDay: '2026-09-11', companyId: 1 }),
      { wrapper: createWrapper(client) },
    );

    await act(async () => {
      await result.current.rejectOffer(11, offer(11)).catch(() => {});
    });

    const restored = client.getQueryData(lirieKeys.institutionOffers());
    expect(restored.offers.map((o) => o.id)).toEqual([10, 11, 12]);
    expect(countVisibleInstitutionOffers(restored.offers)).toBe(3);
  });

  it('accepte : retire l\'offre en attente dès la confirmation', async () => {
    let resolveAccept;
    companyService.acceptRequestOffer.mockImplementation(
      () => new Promise((resolve) => { resolveAccept = resolve; }),
    );
    const { result } = renderHook(
      () => useInstitutionOfferMutations({ dispatchDay: '2026-09-11', companyId: 1 }),
      { wrapper: createWrapper(client) },
    );

    act(() => {
      result.current.acceptOffer(10, '2026-09-11T13:30:00', offer(10));
    });

    expect(
      client.getQueryData(lirieKeys.institutionOffers()).offers.map((o) => o.id),
    ).toEqual([11, 12]);

    await act(async () => {
      resolveAccept({ success: true, booking_id: 9001, offer_id: 10 });
    });
    await waitFor(() => {
      expect(countVisibleInstitutionOffers(
        client.getQueryData(lirieKeys.institutionOffers()).offers,
      )).toBe(2);
    });
  });

  it('refus A + accept B + refus C : badge final exact', async () => {
    companyService.rejectRequestOffer.mockResolvedValue({ success: true });
    companyService.acceptRequestOffer.mockResolvedValue({ success: true, booking_id: 1 });
    const { result } = renderHook(
      () => useInstitutionOfferMutations({ dispatchDay: '2026-09-11', companyId: 1 }),
      { wrapper: createWrapper(client) },
    );

    await act(async () => {
      await Promise.all([
        result.current.rejectOffer(10, offer(10)),
        result.current.acceptOffer(11, undefined, offer(11)),
        result.current.rejectOffer(12, offer(12)),
      ]);
    });

    const next = client.getQueryData(lirieKeys.institutionOffers());
    expect(next.offers).toEqual([]);
    expect(countVisibleInstitutionOffers(next.offers)).toBe(0);
  });

  it('événement PENDING tardif ne réinsère pas une offre refusée', async () => {
    companyService.rejectRequestOffer.mockResolvedValue({ success: true });
    companyService.fetchRequestOffer.mockResolvedValue(offer(11));
    const { result } = renderHook(
      () => useInstitutionOfferMutations({ dispatchDay: '2026-09-11', companyId: 1 }),
      { wrapper: createWrapper(client) },
    );

    await act(async () => {
      await result.current.rejectOffer(11, offer(11));
    });
    await act(async () => {
      await result.current.applyRealtimeOfferEvent({ offer_id: 11, status: 'PENDING' });
    });

    expect(
      client.getQueryData(lirieKeys.institutionOffers()).offers.map((o) => o.id),
    ).toEqual([10, 12]);
    expect(companyService.fetchRequestOffer).not.toHaveBeenCalled();
  });
});
