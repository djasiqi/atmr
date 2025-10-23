import apiClient from '../utils/apiClient';

export const fetchShifts = async ({ from, to, driverId, status, page = 1, pageSize = 100 }) => {
  const params = new URLSearchParams();
  if (from) params.append('from', from);
  if (to) params.append('to', to);
  if (driverId) params.append('driver_id', driverId);
  if (status) params.append('status', status);
  params.append('page', String(page));
  params.append('page_size', String(pageSize));
  const { data } = await apiClient.get(`/companies/me/planning/shifts?${params.toString()}`);
  return data;
};

export const createShift = async (_companyId, payload) => {
  const { data } = await apiClient.post(`/companies/me/planning/shifts`, payload);
  return data;
};

export const updateShift = async (_companyId, id, payload) => {
  const { data } = await apiClient.put(`/companies/me/planning/shifts/${id}`, payload);
  return data;
};

export const deleteShift = async (_companyId, id) => {
  const { data } = await apiClient.delete(`/companies/me/planning/shifts/${id}`);
  return data;
};

// --- Unavailability ---
export const fetchUnavailability = async ({ from, to, driverId }) => {
  const params = new URLSearchParams();
  if (from) params.append('from', from);
  if (to) params.append('to', to);
  if (driverId) params.append('driver_id', driverId);
  const { data } = await apiClient.get(
    `/companies/me/planning/unavailability?${params.toString()}`
  );
  return data;
};

export const createUnavailability = async (payload) => {
  const { data } = await apiClient.post(`/companies/me/planning/unavailability`, payload);
  return data;
};

export const deleteUnavailability = async (id) => {
  const { data } = await apiClient.delete(`/companies/me/planning/unavailability/${id}`);
  return data;
};

// --- Weekly template ---
export const fetchWeeklyTemplate = async () => {
  const { data } = await apiClient.get(`/companies/me/planning/templates/weekly`);
  return data;
};

export const saveWeeklyTemplate = async (payload) => {
  const { data } = await apiClient.post(`/companies/me/planning/templates/weekly`, payload);
  return data;
};

export const materializeTemplate = async (payload) => {
  const { data } = await apiClient.post(`/companies/me/planning/templates/materialize`, payload);
  return data;
};

// --- Settings ---
export const fetchPlanningSettings = async () => {
  const { data } = await apiClient.get(`/companies/me/planning/settings`);
  return data;
};

export const updatePlanningSettings = async (payload) => {
  const { data } = await apiClient.put(`/companies/me/planning/settings`, payload);
  return data;
};

// --- Assignments overlay (reservations) ---
export const fetchAssignments = async ({ from, to, driverId }) => {
  const params = new URLSearchParams();
  if (from) params.append('from', from);
  if (to) params.append('to', to);
  if (driverId) params.append('driver_id', driverId);
  const { data } = await apiClient.get(`/companies/me/planning/assignments?${params.toString()}`);
  return data;
};

// --- Exports ---
export const exportICS = async ({ driverId, from, to }) => {
  const params = new URLSearchParams();
  if (driverId) params.append('driver_id', driverId);
  if (from) params.append('from', from);
  if (to) params.append('to', to);
  const { data } = await apiClient.get(`/companies/me/planning/export/ics?${params.toString()}`);
  return data;
};

export const exportCSV = async (paramsObj) => {
  const params = new URLSearchParams(paramsObj || {});
  const { data } = await apiClient.get(`/companies/me/planning/export/csv?${params.toString()}`);
  return data;
};

export const exportPDF = async (paramsObj) => {
  const params = new URLSearchParams(paramsObj || {});
  const { data } = await apiClient.get(`/companies/me/planning/export/pdf?${params.toString()}`);
  return data;
};

// --- Compat helpers used by CompanyDriverPlanning.jsx ---
export const fetchCompanyDriver = async () => {
  const { data } = await apiClient.get(`/companies/me/drivers`);
  return Array.isArray(data) ? data : data?.drivers || [];
};

export const fetchDriverVacations = async (driverId) => {
  // Endpoint à implémenter côté back; placeholder pour compiler l'UI
  const { data } = await apiClient.get(`/companies/me/drivers/${driverId}/vacations`);
  return Array.isArray(data) ? data : data?.vacations || [];
};

const driverPlanningService = {
  fetchShifts,
  createShift,
  updateShift,
  deleteShift,
  fetchUnavailability,
  createUnavailability,
  deleteUnavailability,
  fetchWeeklyTemplate,
  saveWeeklyTemplate,
  materializeTemplate,
  fetchPlanningSettings,
  updatePlanningSettings,
  fetchAssignments,
  exportICS,
  exportCSV,
  exportPDF,
  fetchCompanyDriver,
  fetchDriverVacations,
};

export default driverPlanningService;
