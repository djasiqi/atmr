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

// ==================== Service Areas (V1) ====================

export const fetchServiceAreas = async () => {
  const response = await apiClient.get('/company-settings/service-areas');
  return response.data;
};

export const createServiceArea = async (payload) => {
  const response = await apiClient.post('/company-settings/service-areas', payload);
  return response.data;
};

export const updateServiceArea = async (id, payload) => {
  const response = await apiClient.put(`/company-settings/service-areas/${id}`, payload);
  return response.data;
};

export const deleteServiceArea = async (id) => {
  const response = await apiClient.delete(`/company-settings/service-areas/${id}`);
  return response.data;
};

export const fetchServiceAreaZones = async ({
  q = '',
  lang = 'fr',
  types = 'commune,canton',
  limit = 20,
  ids = [],
  tokens = [],
  cantonCode = '',
  includeMeta = false,
  includeGeometry = false,
} = {}) => {
  const response = await apiClient.get('/geocode/zones', {
    params: {
      q: q || undefined,
      lang,
      types,
      limit,
      canton_code: cantonCode || undefined,
      include_geometry: includeGeometry ? 1 : undefined,
      ids: Array.isArray(ids) && ids.length > 0 ? ids.join(',') : undefined,
      tokens: Array.isArray(tokens) && tokens.length > 0 ? tokens.join(',') : undefined,
    },
  });
  const items = Array.isArray(response.data?.items) ? response.data.items : [];
  const meta = response.data?.meta || {};
  return includeMeta ? { items, meta } : items;
};

export const fetchPricingZoneSets = async ({ scope = '', active = true, limit = 100 } = {}) => {
  const response = await apiClient.get('/pricing/zone-sets', {
    params: {
      scope: scope || undefined,
      active: active ? 1 : 0,
      limit,
    },
  });
  return Array.isArray(response.data?.items) ? response.data.items : [];
};

export const fetchPricingZoneSetsMap = async ({
  scope = '',
  active = true,
  includeGeometry = true,
  geometryLevel = 'simplified',
  limit = 200,
} = {}) => {
  const response = await apiClient.get('/pricing/zone-sets-map', {
    params: {
      scope: scope || undefined,
      active: active ? 1 : 0,
      include_geometry: includeGeometry ? 1 : 0,
      geometry_level: includeGeometry ? geometryLevel : undefined,
      limit,
    },
  });
  return Array.isArray(response.data?.items) ? response.data.items : [];
};

export const fetchPricingZoneSetByKey = async (
  key,
  { includeGeometry = false, geometryLevel = 'simplified' } = {}
) => {
  const response = await apiClient.get(`/pricing/zone-sets/${encodeURIComponent(key)}`, {
    params: {
      include_geometry: includeGeometry ? 1 : undefined,
      geometry_level: includeGeometry ? geometryLevel : undefined,
    },
  });
  return response.data?.item || null;
};

// ==================== Pricing Zone Sets (Admin Platform) ====================

export const fetchAdminPricingZoneSets = async ({ scope = '', active = null, limit = 200 } = {}) => {
  const response = await apiClient.get('/pricing/admin/zone-sets', {
    params: {
      scope: scope || undefined,
      active: active == null ? undefined : (active ? 1 : 0),
      limit,
    },
  });
  return Array.isArray(response.data?.items) ? response.data.items : [];
};

export const fetchAdminPricingZoneSetByKey = async (key) => {
  const response = await apiClient.get(`/pricing/admin/zone-sets/${encodeURIComponent(key)}`);
  return response.data?.item || null;
};

export const createAdminPricingZoneSet = async (payload) => {
  const response = await apiClient.post('/pricing/admin/zone-sets', payload);
  return response.data?.item || null;
};

export const updateAdminPricingZoneSet = async (key, payload) => {
  const response = await apiClient.put(`/pricing/admin/zone-sets/${encodeURIComponent(key)}`, payload);
  return response.data?.item || null;
};

// ==================== Pricing Simulation (V1) ====================

export const simulatePricing = async (payload, config = {}) => {
  const response = await apiClient.post('/pricing/simulate', payload, config);
  return response.data;
};
