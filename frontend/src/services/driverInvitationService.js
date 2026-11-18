import apiClient from '../utils/apiClient';

const publicOptions = {
  baseURL: process.env.NODE_ENV === 'development' ? '/api/app' : '/api/v1',
  _targetEnv: 'app',
  skipCsrf: true,
  skipAuthRedirect: true,
  skipEnvRouting: true,
};

export const INVITE_NETWORK_MESSAGE =
  'La connexion a été interrompue. Les informations saisies restent affichées. '
  + 'LIRIE a peut-être déjà reçu cette tentative : consultez la liste avant de renvoyer, '
  + 'et n’utilisez que le lien du message le plus récent.';

export const ACCEPT_NETWORK_MESSAGE =
  'La connexion a été interrompue. Rouvrez le dernier lien reçu. '
  + 'Un second essai n’ouvre pas un deuxième accès.';

export function invitationErrorMessage(error, fallback) {
  if (!error?.response) {
    return INVITE_NETWORK_MESSAGE;
  }
  return error.response.data?.message || fallback;
}

export async function listDriverInvitations() {
  const response = await apiClient.get('/companies/me/driver-invitations');
  return response.data;
}

export async function createDriverInvitation(payload) {
  const response = await apiClient.post('/companies/me/driver-invitations', payload);
  return response.data;
}

export async function resendDriverInvitation(publicId) {
  const response = await apiClient.post(
    `/companies/me/driver-invitations/${publicId}/resend`
  );
  return response.data;
}

export async function cancelDriverInvitation(publicId) {
  const response = await apiClient.post(
    `/companies/me/driver-invitations/${publicId}/cancel`
  );
  return response.data;
}

export async function revokeDriverInvitation(publicId) {
  const response = await apiClient.post(
    `/companies/me/driver-invitations/${publicId}/revoke`
  );
  return response.data;
}

export async function openDriverInvitation(token) {
  const response = await apiClient.post(
    '/driver/invitations/open',
    { token },
    publicOptions
  );
  return response.data;
}

export async function acceptDriverInvitation(payload) {
  const response = await apiClient.post(
    '/driver/invitations/accept',
    payload,
    publicOptions
  );
  return response.data;
}
