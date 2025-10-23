// frontend/tests/services/clientService.test.js
import { fetchClient } from 'services/clientService';
import apiClient from 'utils/apiClient';

// Mock apiClient
jest.mock('utils/apiClient');

describe('clientService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    console.error = jest.fn(); // Suppress console.error
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe('fetchClient', () => {
    it('devrait récupérer le profil client avec public_id', async () => {
      localStorage.setItem('public_id', 'client-abc123');

      const mockProfile = {
        id: 42,
        public_id: 'client-abc123',
        user: {
          first_name: 'Jean',
          last_name: 'Dupont',
          email: 'jean.dupont@example.com',
        },
        billing_address: 'Rue de Lausanne 1, 1201 Genève',
        preferential_rate: 45.0,
      };

      apiClient.get.mockResolvedValue({ data: mockProfile });

      const result = await fetchClient();

      expect(apiClient.get).toHaveBeenCalledWith('/clients/client-abc123');
      expect(result).toEqual(mockProfile);
    });

    it('devrait lever une erreur si public_id est manquant', async () => {
      // public_id non défini dans localStorage

      await expect(fetchClient()).rejects.toThrow(
        "Aucun public_id trouvé pour l'utilisateur connecté."
      );

      expect(apiClient.get).not.toHaveBeenCalled();
    });

    it('devrait propager les erreurs API', async () => {
      localStorage.setItem('public_id', 'client-xyz');

      apiClient.get.mockRejectedValue({
        response: { status: 404, data: { error: 'Client not found' } },
      });

      await expect(fetchClient()).rejects.toEqual({
        response: { status: 404, data: { error: 'Client not found' } },
      });

      expect(console.error).toHaveBeenCalledWith(
        'Erreur lors du chargement du profil client :',
        expect.any(Object)
      );
    });

    it('devrait gérer les erreurs réseau', async () => {
      localStorage.setItem('public_id', 'client-123');

      const networkError = new Error('Network failure');
      apiClient.get.mockRejectedValue(networkError);

      await expect(fetchClient()).rejects.toThrow('Network failure');
    });

    it('devrait gérer erreur 401 (non authentifié)', async () => {
      localStorage.setItem('public_id', 'client-456');

      apiClient.get.mockRejectedValue({
        response: { status: 401, data: { error: 'Unauthorized' } },
      });

      await expect(fetchClient()).rejects.toEqual({
        response: { status: 401, data: { error: 'Unauthorized' } },
      });

      expect(console.error).toHaveBeenCalled();
    });

    it('devrait fonctionner avec différents formats de public_id', async () => {
      localStorage.setItem('public_id', 'pub-uuid-12345-67890');

      const mockProfile = { id: 100, public_id: 'pub-uuid-12345-67890' };
      apiClient.get.mockResolvedValue({ data: mockProfile });

      const result = await fetchClient();

      expect(apiClient.get).toHaveBeenCalledWith('/clients/pub-uuid-12345-67890');
      expect(result).toEqual(mockProfile);
    });
  });
});
