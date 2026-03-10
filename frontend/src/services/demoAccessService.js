import apiClient from '../utils/apiClient';

export const consumeDemoMagicLink = async (token) => {
  try {
    const response = await apiClient.post(
      '/demo_access/consume-magic-link',
      { token },
      {
        baseURL: '/api/demo',
        _targetEnv: 'demo',
        skipCsrf: true,
        skipAuthRedirect: true,
        skipEnvRouting: true,
      }
    );
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const setDemoPassword = async (newPassword) => {
  const response = await apiClient.post(
    '/demo_access/set-password',
    { new_password: newPassword },
    {
      baseURL: '/api/demo',
      _targetEnv: 'demo',
      skipCsrf: true,
      skipAuthRedirect: true,
      skipEnvRouting: true,
    }
  );
  return response.data;
};

