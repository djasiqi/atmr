import { submitContactRequest } from '../../services/contactService';
import apiClient from '../../utils/apiClient';

jest.mock('../../utils/apiClient');

describe('contactService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('soumet une demande de contact', async () => {
    apiClient.post.mockResolvedValue({ data: { ok: true } });
    const payload = {
      category: 'support',
      name: 'Jean Dupont',
      email: 'jean@example.com',
      message: 'Bonjour',
      privacy_consent: true,
      client_request_id: 'req-1',
    };

    const result = await submitContactRequest(payload);

    expect(apiClient.post).toHaveBeenCalledWith('/contact/requests', payload, {
      baseURL: '/api/v1',
      skipCsrf: true,
      skipAuthRedirect: true,
    });
    expect(result).toEqual({ ok: true });
  });

  it('rejette une categorie invalide', async () => {
    await expect(
      submitContactRequest({
        category: 'invalid',
      })
    ).rejects.toThrow('Invalid contact category');
  });
});
