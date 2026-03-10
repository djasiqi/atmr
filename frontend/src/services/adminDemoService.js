import apiClient from '../utils/apiClient';

// Dev: /api/app (setupProxy). Prod: /api/v1 (api/app inexistant)
const APP_ADMIN_REQUEST_CONFIG = {
  baseURL: process.env.NODE_ENV === 'development' ? '/api/app' : '/api/v1',
  _targetEnv: 'app',
  skipAuthRedirect: true,
  skipEnvRouting: true,
};

export const fetchAdminDemoRequests = async () => {
  const response = await apiClient.get('/admin/demo_requests', APP_ADMIN_REQUEST_CONFIG);
  return response.data?.items || [];
};

export const provisionDemoAccess = async (demoRequestId, provisionProfile) => {
  const response = await apiClient.post(
    `/admin/demo_requests/${demoRequestId}/provision-access`,
    provisionProfile
      ? { provision_profile: provisionProfile }
      : {},
    APP_ADMIN_REQUEST_CONFIG
  );
  return response.data;
};

export const updateDemoRequestStatus = async (demoRequestId, status) => {
  const response = await apiClient.post(
    `/admin/demo_requests/${demoRequestId}/status`,
    { status },
    APP_ADMIN_REQUEST_CONFIG
  );
  return response.data;
};

export const resendDemoAccess = async (accessId) => {
  const response = await apiClient.post(
    `/admin/demo_accesses/${accessId}/resend`,
    {},
    APP_ADMIN_REQUEST_CONFIG
  );
  return response.data;
};

export const revokeDemoAccess = async (accessId) => {
  const response = await apiClient.post(
    `/admin/demo_accesses/${accessId}/revoke`,
    {},
    APP_ADMIN_REQUEST_CONFIG
  );
  return response.data;
};
