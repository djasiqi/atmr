import apiClient from '../utils/apiClient'; // Chemin correct

export const getUserAccount = async (public_id) => {
  console.log("🔍 Envoi de la requête à l'API avec public_id :", public_id);

  try {
    const response = await apiClient.get(`/users/${public_id}`);
    console.log('✅ Réponse API reçue :', response.data);
    return response;
  } catch (error) {
    console.error('❌ Erreur API :', error.response?.data || error.message);
    throw error;
  }
};
