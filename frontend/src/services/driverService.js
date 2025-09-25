// src/services/driverService.js
import apiClient from "../utils/apiClient";
import axios from "axios";

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
  const token = localStorage.getItem("authToken");
  try {
    const response = await axios.put(
      `${process.env.REACT_APP_API_BASE_URL}/driver/me/photo`,
      { photo: photoData },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const fetchDriverBookings = async () => {
  try {
    const token = localStorage.getItem("authToken");
    const response = await apiClient.get("/driver/me/bookings", {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const updateDriverLocation = async (latitude, longitude) => {
  try {
    const token = localStorage.getItem("authToken");
    const response = await apiClient.put(
      "/driver/me/location",
      { latitude, longitude },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const fetchDriverBookingDetails = async (bookingId) => {
  try {
    const token = localStorage.getItem("authToken");
    const response = await apiClient.get(`/driver/me/bookings/${bookingId}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const updateBookingStatus = async (bookingId, newStatus) => {
  try {
    const token = localStorage.getItem("authToken");
    const response = await apiClient.put(
      `/driver/me/bookings/${bookingId}/status`,
      { status: newStatus },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const rejectBooking = async (bookingId) => {
  try {
    const token = localStorage.getItem("authToken");
    const response = await apiClient.delete(`/driver/me/bookings/${bookingId}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const updateDriverAvailability = async (isAvailable) => {
  try {
    const token = localStorage.getItem("authToken");
    const response = await apiClient.put(
      "/driver/me/availability",
      { is_available: isAvailable },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};


export const updateDriverProfile = async (profileData) => {
  // Utilisez la clé "authToken" si c'est celle qui est utilisée pour stocker le token
  const token = localStorage.getItem("authToken");
  try {
    const response = await axios.put(
      `${process.env.REACT_APP_API_BASE_URL}/driver/me/profile`,
      profileData,
      {
        headers: { Authorization: `Bearer ${token}` },
      }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const fetchDriverAssignments = async () => {
  try {
    const response = await apiClient.get("/driver/me/bookings");
    return response.data; // Assurez-vous que l'API renvoie un tableau de courses
  } catch (error) {
    throw error;
  }
};

export const startBooking = async (bookingId) => {
  try {
    const token = localStorage.getItem("authToken");
    const response = await apiClient.put(`/driver/me/bookings/${bookingId}/status`, {
      status: "in_progress"
    }, {
      headers: { Authorization: `Bearer ${token}` }
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const reportBookingIssue = async (bookingId, issueMessage) => {
  try {
    const token = localStorage.getItem("authToken");
    const response = await apiClient.post(`/driver/me/bookings/${bookingId}/report`, {
      issue: issueMessage
    }, {
      headers: { Authorization: `Bearer ${token}` }
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};


export const completeBooking = async (bookingId) => {
  try {
    const token = localStorage.getItem("authToken");
    const response = await apiClient.put(`/driver/me/bookings/${bookingId}/status`, {
      status: "completed"
    }, {
      headers: { Authorization: `Bearer ${token}` }
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};



