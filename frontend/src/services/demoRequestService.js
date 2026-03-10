import apiClient from '../utils/apiClient';

export const DEMO_ORGANIZATION_TYPES = [
  'transport_company',
  'institution',
  'ems',
  'clinic',
  'hospital',
  'curatorship',
  'other',
];

export const DEMO_USE_CASES = [
  'planning_dispatch',
  'billing',
  'transport_tracking',
  'multi_company_coordination',
  'reporting',
  'si_integration',
  'other',
];

export const submitDemoRequest = async (payload, options = {}) => {
  // Dev: /api/app (setupProxy). Prod: /api/v1 (api/app inexistant)
  const requestOptions = options.publicRequest
    ? {
        baseURL: process.env.NODE_ENV === 'development' ? '/api/app' : '/api/v1',
        _targetEnv: 'app',
        skipCsrf: true,
        skipAuthRedirect: true,
        skipEnvRouting: true,
      }
    : {};
  const response = await apiClient.post('/demo-requests', payload, requestOptions);
  return response.data;
};
