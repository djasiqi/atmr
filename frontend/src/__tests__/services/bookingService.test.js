// frontend/tests/services/bookingService.test.js
import { fetchBookings, cancelBooking, exportBookingsPDF } from 'services/bookingService';
import apiClient from 'utils/apiClient';

// Mock apiClient
jest.mock('utils/apiClient');

describe('bookingService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.URL.createObjectURL = jest.fn(() => 'blob:mock-url');
    global.URL.revokeObjectURL = jest.fn();
    document.body.innerHTML = '';
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
          company_name: 'Taxi Test',
          driver_id: 12,
          driver_name: 'Marie Conductrice',
        },
      ];

      apiClient.get.mockResolvedValue({ data: mockBookings });

      const result = await fetchBookings('client-123');

      expect(apiClient.get).toHaveBeenCalledWith('/clients/client-123/bookings');
      expect(result).toHaveLength(2);
      expect(result[0].company_name).toBe('Entreprise 5');
      expect(result[1].company_name).toBe('Taxi Test');
      expect(result[1].driver_name).toBe('Marie Conductrice');
    });

    it('complète driver_name depuis driver.full_name si absent', async () => {
      const mockBookings = [
        {
          id: 3,
          status: 'ASSIGNED',
          company_id: 1,
          driver_id: 99,
          driver: { full_name: 'Khalid Alaoui' },
        },
      ];
      apiClient.get.mockResolvedValue({ data: mockBookings });
      const result = await fetchBookings('client-xyz');
      expect(result[0].driver_name).toBe('Khalid Alaoui');
    });

    it('devrait gérer les données non-array', async () => {
      apiClient.get.mockResolvedValue({ data: { error: 'Invalid format' } });

      const result = await fetchBookings('client-456');

      expect(result).toEqual([]);
    });

    it("devrait propager l'erreur en cas d'échec API", async () => {
      apiClient.get.mockRejectedValue(new Error('Network error'));
      await expect(fetchBookings('client-789')).rejects.toThrow('Network error');
    });
  });

  describe('cancelBooking', () => {
    it('devrait annuler une réservation avec succès', async () => {
      apiClient.delete.mockResolvedValue({ data: { message: 'Booking canceled', booking_id: 123 } });

      const result = await cancelBooking(123);

      expect(apiClient.delete).toHaveBeenCalledWith(
        '/bookings/123',
        expect.objectContaining({
          data: { status: 'canceled' },
        })
      );

      expect(result).toEqual({ message: 'Booking canceled', booking_id: 123 });
    });

    it('devrait lever une erreur si la requête échoue', async () => {
      apiClient.delete.mockRejectedValue(new Error('Cannot cancel completed booking'));
      await expect(cancelBooking(456)).rejects.toThrow('Cannot cancel completed booking');
    });

    it('devrait gérer les erreurs réseau', async () => {
      apiClient.delete.mockRejectedValue(new Error('Network failure'));
      await expect(cancelBooking(789)).rejects.toThrow('Network failure');
    });
  });

  describe('exportBookingsPDF', () => {
    it('devrait échouer si la période est vide', async () => {
      await expect(exportBookingsPDF('this_month', [], {}, {})).rejects.toMatchObject({
        code: 'empty_period',
      });
    });

    it('devrait télécharger un blob PDF', async () => {
      const appendSpy = jest.spyOn(document.body, 'appendChild');
      const clickSpy = jest.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

      apiClient.post.mockResolvedValue({
        data: new Blob(['fake-pdf'], { type: 'application/pdf' }),
        headers: {
          'content-type': 'application/pdf',
          'x-period-label': 'Ce mois',
          'x-total-amount': '120.00',
          'x-rows-count': '2',
        },
      });

      const bookings = [{ scheduled_time: new Date().toISOString() }];
      const result = await exportBookingsPDF('this_month', bookings, { id: 5 }, null);

      expect(apiClient.post).toHaveBeenCalledWith(
        '/bookings/clients/me/bookings/export-pdf',
        expect.objectContaining({ period: 'this_month' }),
        expect.objectContaining({ responseType: 'blob' })
      );
      expect(appendSpy).toHaveBeenCalled();
      expect(clickSpy).toHaveBeenCalled();
      expect(result).toMatchObject({ periodLabel: 'Ce mois', totalAmount: '120.00', rowsCount: '2' });
      clickSpy.mockRestore();
    });

    it('devrait gérer une réponse JSON avec pdf_url', async () => {
      const dataBlob = {
        text: async () =>
          JSON.stringify({
            pdf_url: 'https://example.com/export.pdf',
            period_label: 'Cette année',
            total_amount: 100.5,
            rows_count: 4,
          }),
      };
      apiClient.post.mockResolvedValue({
        data: dataBlob,
        headers: { 'content-type': 'application/json' },
      });
      const bookings = [{ scheduled_time: new Date().toISOString() }];
      const result = await exportBookingsPDF('this_year', bookings, { id: 5 }, null);
      expect(result.pdfUrl).toBe('https://example.com/export.pdf');
      expect(result.rowsCount).toBe(4);
    });

    it('propage une erreur backend 404', async () => {
      const error = new Error('Not found');
      error.response = { status: 404 };
      apiClient.post.mockRejectedValue(error);
      const bookings = [{ scheduled_time: new Date().toISOString() }];
      await expect(exportBookingsPDF('this_month', bookings, { id: 5 }, null)).rejects.toMatchObject(
        {
          response: { status: 404 },
        }
      );
    });

    it('devrait échouer si période custom invalide', async () => {
      await expect(
        exportBookingsPDF('custom', [{ scheduled_time: 'invalid-date' }], { id: 5 }, null)
      ).rejects.toMatchObject({ code: 'custom_period_invalid' });
    });
  });
});
