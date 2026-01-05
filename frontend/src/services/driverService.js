// src/services/driverService.js
import apiClient from '../utils/apiClient';

export const fetchDriverProfile = async () => {
  try {
    const response = await apiClient.get(`/driver/me/profile`);
    return response.data.profile;
  } catch (error) {
    throw error;
  }
};

// src/services/driverService.js
export const updateDriverPhoto = async (photoData) => {
  // ✅ Utiliser apiClient au lieu de axios directement pour bénéficier des cookies httpOnly
  try {
    const response = await apiClient.put(
      `/driver/me/photo`,
      { photo: photoData }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const fetchDriverBookings = async () => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const response = await apiClient.get('/driver/me/bookings');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const updateDriverLocation = async (latitude, longitude) => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const response = await apiClient.put(
      '/driver/me/location',
      { latitude, longitude }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const fetchDriverBookingDetails = async (bookingId) => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const response = await apiClient.get(`/driver/me/bookings/${bookingId}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const updateBookingStatus = async (bookingId, newStatus) => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const response = await apiClient.put(
      `/driver/me/bookings/${bookingId}/status`,
      { status: newStatus }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const rejectBooking = async (bookingId) => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const response = await apiClient.delete(`/driver/me/bookings/${bookingId}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const updateDriverAvailability = async (isAvailable) => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const response = await apiClient.put(
      '/driver/me/availability',
      { is_available: isAvailable }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const updateDriverProfile = async (profileData) => {
  // ✅ Utiliser apiClient au lieu de axios directement pour bénéficier des cookies httpOnly
  try {
    const response = await apiClient.put(
      `/driver/me/profile`,
      profileData
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const fetchDriverAssignments = async () => {
  try {
    const response = await apiClient.get('/driver/me/bookings');
    return response.data; // Assurez-vous que l'API renvoie un tableau de courses
  } catch (error) {
    throw error;
  }
};

export const startBooking = async (bookingId) => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const response = await apiClient.put(
      `/driver/me/bookings/${bookingId}/status`,
      {
        status: 'in_progress',
      }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const reportBookingIssue = async (bookingId, issueMessage) => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const response = await apiClient.post(
      `/driver/me/bookings/${bookingId}/report`,
      {
        issue: issueMessage,
      }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const completeBooking = async (bookingId) => {
  try {
    // ✅ apiClient gère automatiquement l'authentification (token dans localStorage ou cookies httpOnly)
    const response = await apiClient.put(
      `/driver/me/bookings/${bookingId}/status`,
      {
        status: 'completed',
      }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};
