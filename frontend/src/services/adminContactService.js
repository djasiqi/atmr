import apiClient from '../utils/apiClient';

const APP_ADMIN_REQUEST_CONFIG = {
  baseURL: process.env.NODE_ENV === 'development' ? '/api/app' : '/api/v1',
  _targetEnv: 'app',
  skipAuthRedirect: true,
  skipEnvRouting: true,
};

export const fetchAdminContactRequests = async (status) => {
  const params = status && status !== 'all' ? { status } : undefined;
  const response = await apiClient.get('/admin/contact_requests', {
    ...APP_ADMIN_REQUEST_CONFIG,
    params,
  });
  return {
    items: response.data?.items || [],
    failedCount: response.data?.failed_count || 0,
  };
};

export const retryAdminContactNotification = async (contactRequestId) => {
  const response = await apiClient.post(
    `/admin/contact_requests/${contactRequestId}/retry-notification`,
    {},
    APP_ADMIN_REQUEST_CONFIG
  );
  return response.data;
};
