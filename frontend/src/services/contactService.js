import apiClient from '../utils/apiClient';

const CATEGORIES = ['support', 'institution', 'transport', 'demo', 'billing', 'family'];

const ensureClientRequestId = (payload) => {
  if (payload.client_request_id) {
    return payload;
  }
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return {
      ...payload,
      client_request_id: crypto.randomUUID(),
    };
  }
  return payload;
};

export const submitContactRequest = async (payload) => {
  if (!payload?.category || !CATEGORIES.includes(payload.category)) {
    throw new Error('Invalid contact category');
  }
  const prepared = ensureClientRequestId(payload);
  try {
    const response = await apiClient.post('/contact/requests', prepared, {
      baseURL: '/api/v1',
      skipCsrf: true,
      skipAuthRedirect: true,
      skipEnvRouting: true,
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};
