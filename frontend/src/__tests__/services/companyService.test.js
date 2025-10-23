// frontend/tests/services/companyService.test.js
import {
  fetchCompanyReservations,
  acceptReservation,
  rejectReservation,
  assignDriver,
  deleteReservation,
  scheduleReservation,
  dispatchNowForReservation,
  triggerReturnBooking,
  fetchCompanyDriver,
  updateDriverStatus,
  toggleDriverType,
  deleteDriver,
  createManualBooking,
  searchClients,
  runDispatchForDay,
} from 'services/companyService';
import apiClient from 'utils/apiClient';

// Mock apiClient
jest.mock('utils/apiClient');

describe('companyService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    console.error = jest.fn(); // Suppress console.error in tests
  });

  describe('fetchCompanyReservations', () => {
    it("devrait récupérer les réservations de l'entreprise", async () => {
      const mockReservations = [
        { id: 1, pickup_location: 'Genève', status: 'PENDING' },
        { id: 2, pickup_location: 'Lausanne', status: 'ASSIGNED' },
      ];

      apiClient.get.mockResolvedValue({ data: mockReservations });

      const result = await fetchCompanyReservations('2025-10-16');

      expect(apiClient.get).toHaveBeenCalledWith('/companies/me/reservations', {
        params: { flat: true, date: '2025-10-16' },
      });
      expect(result).toEqual(mockReservations);
    });

    it('devrait gérer le format data.reservations', async () => {
      const mockReservations = [{ id: 1, status: 'PENDING' }];
      apiClient.get.mockResolvedValue({ data: { reservations: mockReservations } });

      const result = await fetchCompanyReservations();

      expect(result).toEqual(mockReservations);
    });

    it("devrait retourner tableau vide en cas d'erreur 401", async () => {
      apiClient.get.mockRejectedValue({
        response: { status: 401, data: { error: 'Unauthorized' } },
      });

      const result = await fetchCompanyReservations();

      expect(result).toEqual([]);
      expect(console.error).toHaveBeenCalled();
    });

    it("devrait retourner tableau vide en cas d'erreur réseau", async () => {
      apiClient.get.mockRejectedValue(new Error('Network error'));

      const result = await fetchCompanyReservations();

      expect(result).toEqual([]);
    });
  });

  describe('acceptReservation', () => {
    it('devrait accepter une réservation', async () => {
      const mockResponse = { id: 123, status: 'CONFIRMED' };
      apiClient.post.mockResolvedValue({ data: mockResponse });

      const result = await acceptReservation(123);

      expect(apiClient.post).toHaveBeenCalledWith('/companies/me/reservations/123/accept');
      expect(result).toEqual(mockResponse);
    });
  });

  describe('rejectReservation', () => {
    it('devrait rejeter une réservation', async () => {
      const mockResponse = { id: 123, status: 'REJECTED' };
      apiClient.post.mockResolvedValue({ data: mockResponse });

      const result = await rejectReservation(123);

      expect(apiClient.post).toHaveBeenCalledWith('/companies/me/reservations/123/reject');
      expect(result).toEqual(mockResponse);
    });
  });

  describe('assignDriver', () => {
    it('devrait assigner un chauffeur à une réservation', async () => {
      const mockResponse = { id: 123, driver_id: 5, status: 'ASSIGNED' };
      apiClient.post.mockResolvedValue({ data: mockResponse });

      const result = await assignDriver(123, 5);

      expect(apiClient.post).toHaveBeenCalledWith('/companies/me/reservations/123/assign', {
        driver_id: 5,
      });
      expect(result).toEqual(mockResponse);
    });
  });

  describe('deleteReservation', () => {
    it('devrait supprimer une réservation', async () => {
      const mockResponse = { message: 'Deleted' };
      apiClient.delete.mockResolvedValue({ data: mockResponse });

      const result = await deleteReservation(123);

      expect(apiClient.delete).toHaveBeenCalledWith('/companies/me/reservations/123');
      expect(result).toEqual(mockResponse);
    });
  });

  describe('scheduleReservation', () => {
    it('devrait planifier une réservation à une date précise', async () => {
      const mockResponse = { id: 123, scheduled_time: '2025-10-16T14:30:00' };
      apiClient.put.mockResolvedValue({ data: mockResponse });

      const result = await scheduleReservation(123, '2025-10-16T14:30:00');

      expect(apiClient.put).toHaveBeenCalledWith('/companies/me/reservations/123/schedule', {
        scheduled_time: '2025-10-16T14:30:00',
      });
      expect(result).toEqual(mockResponse);
    });
  });

  describe('dispatchNowForReservation', () => {
    it('devrait dispatcher maintenant avec offset par défaut', async () => {
      const mockResponse = { id: 123, status: 'DISPATCHED' };
      apiClient.post.mockResolvedValue({ data: mockResponse });

      const result = await dispatchNowForReservation(123);

      expect(apiClient.post).toHaveBeenCalledWith('/companies/me/reservations/123/dispatch-now', {
        minutes_offset: 15,
      });
      expect(result).toEqual(mockResponse);
    });

    it('devrait dispatcher avec offset personnalisé', async () => {
      const mockResponse = { id: 123, status: 'DISPATCHED' };
      apiClient.post.mockResolvedValue({ data: mockResponse });

      const result = await dispatchNowForReservation(123, 30);

      expect(apiClient.post).toHaveBeenCalledWith('/companies/me/reservations/123/dispatch-now', {
        minutes_offset: 30,
      });
      expect(result).toEqual(mockResponse);
    });
  });

  describe('triggerReturnBooking', () => {
    it('devrait déclencher un retour sans payload', async () => {
      const mockResponse = { id: 456, return_booking_id: 789 };
      apiClient.post.mockResolvedValue({ data: mockResponse });

      const result = await triggerReturnBooking(456);

      expect(apiClient.post).toHaveBeenCalledWith(
        '/companies/me/reservations/456/trigger-return',
        {}
      );
      expect(result).toEqual(mockResponse);
    });

    it('devrait déclencher un retour urgent', async () => {
      const mockResponse = { id: 456, return_booking_id: 789 };
      apiClient.post.mockResolvedValue({ data: mockResponse });

      const result = await triggerReturnBooking(456, { urgent: true, minutes_offset: 20 });

      expect(apiClient.post).toHaveBeenCalledWith('/companies/me/reservations/456/trigger-return', {
        urgent: true,
        minutes_offset: 20,
      });
      expect(result).toEqual(mockResponse);
    });
  });

  describe('fetchCompanyDriver', () => {
    it("devrait récupérer les chauffeurs de l'entreprise", async () => {
      const mockDrivers = [
        { id: 1, user: { first_name: 'Pierre' }, is_available: true },
        { id: 2, user: { first_name: 'Marie' }, is_available: false },
      ];

      apiClient.get.mockResolvedValue({ data: { driver: mockDrivers } });

      const result = await fetchCompanyDriver();

      expect(apiClient.get).toHaveBeenCalledWith('/companies/me/drivers');
      expect(result).toEqual(mockDrivers);
    });

    it('devrait gérer format tableau direct', async () => {
      const mockDrivers = [{ id: 1 }];
      apiClient.get.mockResolvedValue({ data: mockDrivers });

      const result = await fetchCompanyDriver();

      expect(result).toEqual(mockDrivers);
    });

    it("devrait retourner tableau vide en cas d'erreur", async () => {
      apiClient.get.mockRejectedValue(new Error('Network error'));

      const result = await fetchCompanyDriver();

      expect(result).toEqual([]);
    });
  });

  describe('updateDriverStatus', () => {
    it('devrait mettre à jour le statut du chauffeur', async () => {
      const mockResponse = { id: 5, is_available: false };
      apiClient.put.mockResolvedValue({ data: mockResponse });

      const result = await updateDriverStatus(5, { is_available: false });

      expect(apiClient.put).toHaveBeenCalledWith('/companies/me/drivers/5', {
        is_available: false,
      });
      expect(result).toEqual(mockResponse);
    });
  });

  describe('toggleDriverType', () => {
    it('devrait changer le type de chauffeur', async () => {
      const mockResponse = { id: 5, is_employee: true };
      apiClient.put.mockResolvedValue({ data: mockResponse });

      const result = await toggleDriverType(5);

      expect(apiClient.put).toHaveBeenCalledWith('/companies/me/drivers/5/toggle-type');
      expect(result).toEqual(mockResponse);
    });
  });

  describe('deleteDriver', () => {
    it('devrait supprimer un chauffeur', async () => {
      const mockResponse = { message: 'Driver deleted' };
      apiClient.delete.mockResolvedValue({ data: mockResponse });

      const result = await deleteDriver(5);

      expect(apiClient.delete).toHaveBeenCalledWith('/companies/me/drivers/5');
      expect(result).toEqual(mockResponse);
    });
  });

  describe('createManualBooking', () => {
    it('devrait créer une réservation manuelle', async () => {
      const bookingData = {
        client_id: 10,
        pickup_location: 'Genève',
        dropoff_location: 'Lausanne',
        scheduled_time: '2025-10-16T10:00:00',
        amount: 50,
      };

      const mockResponse = { id: 999, ...bookingData };
      apiClient.post.mockResolvedValue({ data: mockResponse });

      const result = await createManualBooking(bookingData);

      expect(apiClient.post).toHaveBeenCalledWith('/companies/me/reservations/manual', bookingData);
      expect(result).toEqual(mockResponse);
    });
  });

  describe('searchClients', () => {
    it('devrait rechercher des clients', async () => {
      const mockClients = [
        { id: 1, first_name: 'Jean', last_name: 'Dupont' },
        { id: 2, first_name: 'Marie', last_name: 'Martin' },
      ];

      apiClient.get.mockResolvedValue({ data: mockClients });

      const result = await searchClients('Jean');

      expect(apiClient.get).toHaveBeenCalledWith('/companies/me/clients?search=Jean');
      expect(result).toEqual(mockClients);
    });

    it('devrait gérer recherche vide', async () => {
      const mockClients = [{ id: 1 }];
      apiClient.get.mockResolvedValue({ data: mockClients });

      const result = await searchClients('');

      expect(apiClient.get).toHaveBeenCalledWith('/companies/me/clients?search=');
      expect(result).toEqual(mockClients);
    });

    it("devrait retourner tableau vide en cas d'erreur", async () => {
      apiClient.get.mockRejectedValue(new Error('Search failed'));

      const result = await searchClients('test');

      expect(result).toEqual([]);
    });
  });

  describe('runDispatchForDay', () => {
    it('devrait lancer le dispatch pour un jour', async () => {
      const dispatchParams = {
        forDate: '2025-10-16',
        regularFirst: true,
        allowEmergency: true,
        mode: 'auto',
        runAsync: true,
      };

      const mockResponse = { success: true, assignments: 10 };
      apiClient.post.mockResolvedValue({ data: mockResponse });

      const result = await runDispatchForDay(dispatchParams);

      expect(apiClient.post).toHaveBeenCalledWith(
        '/company_dispatch/run',
        expect.objectContaining({
          for_date: '2025-10-16',
          regular_first: true,
          allow_emergency: true,
          async: true,
          mode: 'auto',
        })
      );
      expect(result).toEqual(
        expect.objectContaining({
          success: true,
          assignments: 10,
        })
      );
    });
  });
});
