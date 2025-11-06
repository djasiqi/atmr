export const API_BASE_URL = 'https://api.yourdomain.com';

export const API_ENDPOINTS = {
  LOGIN: '/auth/login',
  FORGOT_PASSWORD: '/auth/forgot-password',
  REFRESH_TOKEN: '/auth/refresh-token',
  DRIVER_PROFILE: (driverId: number) => `/drivers/${driverId}/profile`,
  DRIVER_TOGGLE_AVAILABILITY: (driverId: number) => `/drivers/${driverId}/toggle-availability`,
  DRIVER_ASSIGNED_TRIPS: (driverId: number) => `/drivers/${driverId}/assigned-trips`,
  DRIVER_COMPLETED_TRIPS: (driverId: number) => `/drivers/${driverId}/completed-trips`,
  UPDATE_DRIVER_PROFILE: (driverId: number) => `/drivers/${driverId}/update-profile`,
};