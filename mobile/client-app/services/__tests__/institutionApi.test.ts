/* eslint-disable import/first */
import type { CreateInstitutionRequestPayload } from '@/types/api';

jest.mock('@/services/api', () => ({
  api: {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
  },
}));

import { api } from '@/services/api';
import {
  cancelRequest,
  createPatient,
  createRequest,
  listRequests,
  sendRequest,
} from '../institutionApi';

const mockedApi = api as jest.Mocked<typeof api>;

describe('institutionApi', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('normalizes paginated requests list', async () => {
    mockedApi.get.mockResolvedValueOnce({
      data: {
        requests: [{ id: 1 }, { id: 2 }],
        total: 42,
        page: 2,
        per_page: 20,
      },
    } as never);

    const out = await listRequests({ status: 'SENT', page: 2, per_page: 20 });

    expect(mockedApi.get).toHaveBeenCalledWith('/institutions/requests', {
      params: { status: 'SENT', page: 2, per_page: 20 },
    });
    expect(out.items).toHaveLength(2);
    expect(out.total).toBe(42);
    expect(out.page).toBe(2);
    expect(out.perPage).toBe(20);
  });

  it('creates and unwraps transport request payload', async () => {
    const payload: CreateInstitutionRequestPayload = {
      external_reference: 'REQ-001',
      scheduled_time: '2026-04-19T10:00:00',
      pickup_location: 'Hopital A',
      dropoff_location: 'Clinique B',
      billing_intent: 'institution',
      mobility: { wheelchair: true, walking: false },
    };
    mockedApi.post.mockResolvedValueOnce({
      data: { data: { id: 11, status: 'DRAFT' } },
    } as never);

    const out = await createRequest(payload);

    expect(mockedApi.post).toHaveBeenCalledWith('/institutions/requests', payload);
    expect(out.id).toBe(11);
    expect(out.status).toBe('DRAFT');
  });

  it('sends and cancels request with correct endpoints', async () => {
    mockedApi.post.mockResolvedValueOnce({ data: { data: { id: 9, status: 'SENT' } } } as never);
    mockedApi.post.mockResolvedValueOnce({
      data: { data: { id: 9, status: 'CANCELLED' } },
    } as never);

    const sent = await sendRequest(9);
    const cancelled = await cancelRequest(9);

    expect(mockedApi.post).toHaveBeenNthCalledWith(1, '/institutions/requests/9/send', {});
    expect(mockedApi.post).toHaveBeenNthCalledWith(2, '/institutions/requests/9/cancel', {});
    expect(sent.status).toBe('SENT');
    expect(cancelled.status).toBe('CANCELLED');
  });

  it('creates patient with terrain fields', async () => {
    mockedApi.post.mockResolvedValueOnce({
      data: { data: { id: 99, first_name: 'Ada', door_code: '1234' } },
    } as never);

    const out = await createPatient({
      first_name: 'Ada',
      last_name: 'Lovelace',
      door_code: '1234',
      floor: '3',
      access_notes: 'Interphone gauche',
    });

    expect(mockedApi.post).toHaveBeenCalledWith('/institutions/patients', {
      first_name: 'Ada',
      last_name: 'Lovelace',
      door_code: '1234',
      floor: '3',
      access_notes: 'Interphone gauche',
    });
    expect(out.id).toBe(99);
  });
});
