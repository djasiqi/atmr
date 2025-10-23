// frontend/tests/services/bookingService.test.js
import { fetchBookings, cancelBooking, exportBookingsPDF } from 'services/bookingService';
import apiClient from 'utils/apiClient';

// Mock apiClient
jest.mock('utils/apiClient');

// Mock window.alert
global.alert = jest.fn();

describe('bookingService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.alert.mockClear();
  });

  describe('fetchBookings', () => {
    it("devrait récupérer les réservations d'un client", async () => {
      const mockBookings = [
        {
          id: 1,
          pickup_location: 'Rue de Lausanne 1',
          dropoff_location: 'HUG',
          status: 'PENDING',
          company_id: 5,
          driver_id: null,
        },
        {
          id: 2,
          pickup_location: 'Chemin des Acacias',
          dropoff_location: 'Grangettes',
          status: 'COMPLETED',
          company_id: 5,
          driver_id: 12,
        },
      ];

      apiClient.get.mockResolvedValue({ data: mockBookings });

      const result = await fetchBookings('client-123');

      expect(apiClient.get).toHaveBeenCalledWith('/clients/client-123/bookings');
      expect(result).toHaveLength(2);
      expect(result[0].company_name).toBe('Entreprise 5');
      expect(result[1].driver_name).toBe('Chauffeur 12');
    });

    it('devrait gérer les données non-array', async () => {
      apiClient.get.mockResolvedValue({ data: { error: 'Invalid format' } });

      const result = await fetchBookings('client-456');

      expect(result).toEqual([]);
    });

    it("devrait retourner un tableau vide en cas d'erreur", async () => {
      apiClient.get.mockRejectedValue(new Error('Network error'));

      const result = await fetchBookings('client-789');

      expect(result).toEqual([]);
    });
  });

  describe('cancelBooking', () => {
    const API_URL = process.env.REACT_APP_API_URL;

    beforeEach(() => {
      localStorage.setItem('authToken', 'fake-token');
      global.fetch = jest.fn();
    });

    afterEach(() => {
      localStorage.clear();
    });

    it('devrait annuler une réservation avec succès', async () => {
      const mockResponse = {
        ok: true,
        json: async () => ({ message: 'Booking canceled', booking_id: 123 }),
      };

      global.fetch.mockResolvedValue(mockResponse);

      const result = await cancelBooking(123);

      expect(global.fetch).toHaveBeenCalledWith(
        `${API_URL}/bookings/123`,
        expect.objectContaining({
          method: 'DELETE',
          headers: {
            Authorization: 'Bearer fake-token',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ status: 'canceled' }),
        })
      );

      expect(result).toEqual({ message: 'Booking canceled', booking_id: 123 });
    });

    it('devrait lever une erreur si la requête échoue', async () => {
      const mockResponse = {
        ok: false,
        json: async () => ({ message: 'Cannot cancel completed booking' }),
      };

      global.fetch.mockResolvedValue(mockResponse);

      await expect(cancelBooking(456)).rejects.toThrow('Cannot cancel completed booking');
    });

    it('devrait gérer les erreurs réseau', async () => {
      global.fetch.mockRejectedValue(new Error('Network failure'));

      await expect(cancelBooking(789)).rejects.toThrow('Network failure');
    });
  });

  describe('exportBookingsPDF', () => {
    it('devrait afficher une alerte si aucune réservation', async () => {
      await exportBookingsPDF('Janvier', [], {}, {});

      expect(global.alert).toHaveBeenCalledWith('Aucune réservation trouvée pour ce mois.');
    });

    it("devrait logger un message TODO pour l'export PDF", async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();

      const bookings = [{ id: 1 }, { id: 2 }];
      const client = { id: 5, first_name: 'Jean' };
      const company = { id: 10, name: 'ATMR' };

      await exportBookingsPDF('Février', bookings, client, company);

      expect(consoleSpy).toHaveBeenCalledWith('📂 Génération PDF en cours sur le frontend...');
      expect(consoleSpy).toHaveBeenCalledWith(
        'PDF generation moved to backend API - To be implemented'
      );

      consoleSpy.mockRestore();
    });
  });
});
