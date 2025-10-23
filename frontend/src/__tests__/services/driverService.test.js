// frontend/tests/services/driverService.test.js
import {
  fetchDriverProfile,
  fetchDriverBookings,
  updateDriverLocation,
  fetchDriverBookingDetails,
  updateBookingStatus,
  rejectBooking,
  updateDriverAvailability,
  startBooking,
  completeBooking,
  reportBookingIssue,
} from 'services/driverService';
import apiClient from 'utils/apiClient';

// Mock apiClient
jest.mock('utils/apiClient');

describe('driverService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    localStorage.setItem('authToken', 'fake-driver-token');
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe('fetchDriverProfile', () => {
    it('devrait récupérer le profil du chauffeur', async () => {
      const mockProfile = {
        id: 5,
        first_name: 'Pierre',
        last_name: 'Martin',
        license_number: 'ABC123',
      };

      apiClient.get.mockResolvedValue({ data: { profile: mockProfile } });

      const result = await fetchDriverProfile();

      expect(apiClient.get).toHaveBeenCalledWith('/driver/me/profile');
      expect(result).toEqual(mockProfile);
    });

    it('devrait propager les erreurs', async () => {
      apiClient.get.mockRejectedValue(new Error('Profile not found'));

      await expect(fetchDriverProfile()).rejects.toThrow('Profile not found');
    });
  });

  describe('fetchDriverBookings', () => {
    it('devrait récupérer les réservations du chauffeur', async () => {
      const mockBookings = [
        { id: 1, pickup_location: 'Genève', status: 'ASSIGNED' },
        { id: 2, pickup_location: 'Lausanne', status: 'IN_PROGRESS' },
      ];

      apiClient.get.mockResolvedValue({ data: mockBookings });

      const result = await fetchDriverBookings();

      expect(apiClient.get).toHaveBeenCalledWith('/driver/me/bookings', {
        headers: { Authorization: 'Bearer fake-driver-token' },
      });
      expect(result).toEqual(mockBookings);
    });
  });

  describe('updateDriverLocation', () => {
    it('devrait mettre à jour la localisation du chauffeur', async () => {
      const mockResponse = {
        latitude: 46.2044,
        longitude: 6.1432,
        updated_at: '2025-10-16T10:00:00',
      };

      apiClient.put.mockResolvedValue({ data: mockResponse });

      const result = await updateDriverLocation(46.2044, 6.1432);

      expect(apiClient.put).toHaveBeenCalledWith(
        '/driver/me/location',
        { latitude: 46.2044, longitude: 6.1432 },
        { headers: { Authorization: 'Bearer fake-driver-token' } }
      );
      expect(result).toEqual(mockResponse);
    });

    it('devrait gérer les erreurs de localisation', async () => {
      apiClient.put.mockRejectedValue(new Error('Location update failed'));

      await expect(updateDriverLocation(0, 0)).rejects.toThrow('Location update failed');
    });
  });

  describe('fetchDriverBookingDetails', () => {
    it("devrait récupérer les détails d'une réservation", async () => {
      const mockDetails = {
        id: 123,
        client: { first_name: 'Jean', last_name: 'Dupont' },
        pickup_location: 'Genève',
        dropoff_location: 'Lausanne',
        status: 'ASSIGNED',
      };

      apiClient.get.mockResolvedValue({ data: mockDetails });

      const result = await fetchDriverBookingDetails(123);

      expect(apiClient.get).toHaveBeenCalledWith('/driver/me/bookings/123', {
        headers: { Authorization: 'Bearer fake-driver-token' },
      });
      expect(result).toEqual(mockDetails);
    });
  });

  describe('updateBookingStatus', () => {
    it("devrait mettre à jour le statut d'une réservation", async () => {
      const mockResponse = { id: 123, status: 'IN_PROGRESS' };

      apiClient.put.mockResolvedValue({ data: mockResponse });

      const result = await updateBookingStatus(123, 'IN_PROGRESS');

      expect(apiClient.put).toHaveBeenCalledWith(
        '/driver/me/bookings/123/status',
        { status: 'IN_PROGRESS' },
        { headers: { Authorization: 'Bearer fake-driver-token' } }
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe('rejectBooking', () => {
    it('devrait rejeter une réservation', async () => {
      const mockResponse = { message: 'Booking rejected' };

      apiClient.delete.mockResolvedValue({ data: mockResponse });

      const result = await rejectBooking(123);

      expect(apiClient.delete).toHaveBeenCalledWith('/driver/me/bookings/123', {
        headers: { Authorization: 'Bearer fake-driver-token' },
      });
      expect(result).toEqual(mockResponse);
    });
  });

  describe('updateDriverAvailability', () => {
    it('devrait mettre à jour la disponibilité (disponible)', async () => {
      const mockResponse = { is_available: true };

      apiClient.put.mockResolvedValue({ data: mockResponse });

      const result = await updateDriverAvailability(true);

      expect(apiClient.put).toHaveBeenCalledWith(
        '/driver/me/availability',
        { is_available: true },
        { headers: { Authorization: 'Bearer fake-driver-token' } }
      );
      expect(result).toEqual(mockResponse);
    });

    it('devrait mettre à jour la disponibilité (indisponible)', async () => {
      const mockResponse = { is_available: false };

      apiClient.put.mockResolvedValue({ data: mockResponse });

      const result = await updateDriverAvailability(false);

      expect(apiClient.put).toHaveBeenCalledWith(
        '/driver/me/availability',
        { is_available: false },
        { headers: { Authorization: 'Bearer fake-driver-token' } }
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe('startBooking', () => {
    it('devrait démarrer une réservation', async () => {
      const mockResponse = { id: 123, status: 'IN_PROGRESS' };

      apiClient.put.mockResolvedValue({ data: mockResponse });

      const result = await startBooking(123);

      expect(apiClient.put).toHaveBeenCalledWith(
        '/driver/me/bookings/123/status',
        { status: 'in_progress' },
        { headers: { Authorization: 'Bearer fake-driver-token' } }
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe('completeBooking', () => {
    it('devrait compléter une réservation', async () => {
      const mockResponse = { id: 123, status: 'COMPLETED' };

      apiClient.put.mockResolvedValue({ data: mockResponse });

      const result = await completeBooking(123);

      expect(apiClient.put).toHaveBeenCalledWith(
        '/driver/me/bookings/123/status',
        { status: 'completed' },
        { headers: { Authorization: 'Bearer fake-driver-token' } }
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe('reportBookingIssue', () => {
    it('devrait signaler un problème sur une réservation', async () => {
      const mockResponse = { id: 123, issue_reported: true };

      apiClient.post.mockResolvedValue({ data: mockResponse });

      const result = await reportBookingIssue(123, 'Client introuvable');

      expect(apiClient.post).toHaveBeenCalledWith(
        '/driver/me/bookings/123/report',
        { issue: 'Client introuvable' },
        { headers: { Authorization: 'Bearer fake-driver-token' } }
      );
      expect(result).toEqual(mockResponse);
    });
  });
});
