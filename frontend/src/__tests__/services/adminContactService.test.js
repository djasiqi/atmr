import {
  fetchAdminContactRequests,
  retryAdminContactNotification,
} from '../../services/adminContactService';
import apiClient from '../../utils/apiClient';

jest.mock('../../utils/apiClient', () => ({
  get: jest.fn(),
  post: jest.fn(),
}));

describe('adminContactService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('charge les demandes de contact en échec', async () => {
    apiClient.get.mockResolvedValue({
      data: { items: [{ id: 1, trace_id: 'ct_1' }], failed_count: 1 },
    });
    const result = await fetchAdminContactRequests('failed');
    expect(apiClient.get).toHaveBeenCalledWith(
      '/admin/contact_requests',
      expect.objectContaining({ params: { status: 'failed' } })
    );
    expect(result.items).toHaveLength(1);
    expect(result.failedCount).toBe(1);
  });

  it('relance une notification interne', async () => {
    apiClient.post.mockResolvedValue({ data: { ok: true } });
    const result = await retryAdminContactNotification(12);
    expect(apiClient.post).toHaveBeenCalledWith(
      '/admin/contact_requests/12/retry-notification',
      {},
      expect.any(Object)
    );
    expect(result.ok).toBe(true);
  });
});
