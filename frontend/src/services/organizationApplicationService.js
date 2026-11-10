import apiClient from '../utils/apiClient';

const publicOptions = {
  baseURL: process.env.NODE_ENV === 'development' ? '/api/app' : '/api/v1',
  _targetEnv: 'app',
  skipCsrf: true,
  skipAuthRedirect: true,
  skipEnvRouting: true,
};

export const SLA_NOTICE =
  'Votre demande sera examinée dans un délai maximal de 24 heures administratives cumulées, à compter de la réception de votre dossier complet.';

export const TRACKING_STORAGE_KEY = 'lirie_organization_tracking';

export const NETWORK_INTERRUPTION_MESSAGE =
  'La connexion a été interrompue. Les informations saisies restent dans le formulaire. LIRIE a peut-être déjà reçu cette tentative : le lien précédent n’est alors plus utilisable. Renvoyez la demande, puis ouvrez seulement le lien du message le plus récent.';

export function submissionErrorMessage(error) {
  if (!error?.response) {
    return NETWORK_INTERRUPTION_MESSAGE;
  }
  if (error.response.status === 429) {
    return 'Trop de dépôts depuis cette connexion. Réessayez dans un moment. Vos réponses sont conservées.';
  }
  return error.response.data?.message
    || 'Le dossier n’a pas pu être envoyé. Vos réponses sont conservées.';
}

export async function submitOrganizationApplication(payload) {
  const response = await apiClient.post('/organization-applications', payload, publicOptions);
  return response.data;
}

export async function verifyOrganizationEmail(token) {
  const response = await apiClient.post(
    '/organization-applications/verify',
    { token },
    publicOptions
  );
  return response.data;
}

export async function fetchMyOrganizationApplication(trackingToken) {
  const response = await apiClient.get('/organization-applications/mine', {
    ...publicOptions,
    headers: trackingToken ? { Authorization: `Bearer ${trackingToken}` } : {},
  });
  return response.data;
}
