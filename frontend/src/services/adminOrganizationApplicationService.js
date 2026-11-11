import apiClient from '../utils/apiClient';

const APP_ADMIN_REQUEST_CONFIG = {
  baseURL: process.env.NODE_ENV === 'development' ? '/api/app' : '/api/v1',
  _targetEnv: 'app',
  skipAuthRedirect: true,
  skipEnvRouting: true,
};

function readError(error, fallback) {
  return error?.response?.data?.message || fallback;
}

export async function fetchOrganizationApplications(params) {
  const response = await apiClient.get('/admin/organization-applications', {
    ...APP_ADMIN_REQUEST_CONFIG,
    params,
  });
  return response.data;
}

export async function fetchOrganizationApplication(publicId) {
  const response = await apiClient.get(
    `/admin/organization-applications/${publicId}`,
    APP_ADMIN_REQUEST_CONFIG
  );
  return response.data;
}

export async function searchExistingOrganizations(publicId, query) {
  const response = await apiClient.get(
    `/admin/organization-applications/${publicId}/organizations`,
    { ...APP_ADMIN_REQUEST_CONFIG, params: { q: query } }
  );
  return response.data;
}

export async function approveOrganizationApplication(publicId) {
  try {
    const response = await apiClient.post(
      `/admin/organization-applications/${publicId}/approve`,
      {},
      APP_ADMIN_REQUEST_CONFIG
    );
    return response.data;
  } catch (error) {
    throw new Error(readError(error, 'L’approbation n’a pas pu être enregistrée.'));
  }
}

export async function decideOrganizationApplication(publicId, payload) {
  try {
    const response = await apiClient.post(
      `/admin/organization-applications/${publicId}/decisions`,
      payload,
      APP_ADMIN_REQUEST_CONFIG
    );
    return response.data;
  } catch (error) {
    throw new Error(readError(error, 'La décision n’a pas pu être enregistrée.'));
  }
}

export { readError as organizationApplicationError };
