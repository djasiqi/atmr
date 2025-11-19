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

// ==================== Paramètres de Planning ====================

export const fetchPlanningSettings = async () => {
  const response = await apiClient.get('/company-settings/planning');
  return response.data.data;
};

export const updatePlanningSettings = async (settings) => {
  const response = await apiClient.put('/company-settings/planning', { settings });
  return response.data;
};
