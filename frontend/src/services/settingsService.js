// frontend/src/services/settingsService.js
import apiClient from '../utils/apiClient';

/**
 * Service pour gérer les paramètres avancés de l'entreprise
 */

// ==================== Paramètres Opérationnels ====================

export const fetchOperationalSettings = async () => {
  const response = await apiClient.get('/company-settings/operational');
  return response.data.data;
};

export const updateOperationalSettings = async (settings) => {
  const response = await apiClient.put('/company-settings/operational', settings);
  return response.data;
};

// ==================== Paramètres de Facturation ====================

export const fetchBillingSettings = async () => {
  const response = await apiClient.get('/company-settings/billing');
  return response.data;
};

export const updateBillingSettings = async (settings) => {
  const response = await apiClient.put('/company-settings/billing', settings);
  return response.data;
};

// ==================== Mappings clinique → BillingParty (P1) ====================

export const fetchClinicBillingMappings = async () => {
  const response = await apiClient.get('/company-settings/billing/clinic-mappings');
  return response.data;
};

export const fetchClinicBillingMapping = async (clinicCompanyId) => {
  const response = await apiClient.get(
    `/company-settings/billing/clinic-mappings/${clinicCompanyId}`
  );
  return response.data;
};

export const upsertClinicBillingMapping = async ({
  clinic_company_id,
  billing_party_id,
  is_active = true,
}) => {
  const response = await apiClient.put('/company-settings/billing/clinic-mappings', {
    clinic_company_id,
    billing_party_id,
    is_active,
  });
  return response.data;
};

export const fetchBillingParties = async ({ active = true } = {}) => {
  const response = await apiClient.get('/company-settings/billing/parties', {
    params: { active },
  });
  return response.data;
};

export const createBillingParty = async (payload) => {
  const response = await apiClient.post('/company-settings/billing/parties', payload);
  return response.data;
};

export const updateBillingParty = async (billingPartyId, payload) => {
  const response = await apiClient.put(
    `/company-settings/billing/parties/${billingPartyId}`,
    payload
  );
  return response.data;
};

// ==================== Paramètres de Planning ====================

export const fetchPlanningSettings = async () => {
  const response = await apiClient.get('/company-settings/planning');
  return response.data.data;
};

export const updatePlanningSettings = async (settings) => {
  const response = await apiClient.put('/company-settings/planning', { settings });
  return response.data;
};
