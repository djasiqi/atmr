// frontend/src/services/authService.js
import apiClient, { logoutUser as coreLogoutUser } from '../utils/apiClient';
import { getAuthEnv, setEnvPublicId, setEnvUser } from '../utils/webAuthSession';

// ✅ Inscription d'un utilisateur
export const registerUser = async (userData) => {
  try {
    const response = await apiClient.post('/auth/register', userData);
    return response.data; // Retourne les données de la réponse
  } catch (error) {
    console.error("❌ Erreur lors de l'inscription :", error);
    throw error; // Remonte l'erreur pour la gérer ultérieurement
  }
};

// ✅ Connexion d'un utilisateur
export const loginUser = async (credentials) => {
  try {
    const response = await apiClient.post('/auth/login', credentials);
    // ✅ P1-1: Le token est dans les cookies httpOnly, pas besoin de le récupérer
    const { user } = response.data;

    console.log('🔐 Connexion réussie. Données reçues :', response.data);

    if (!user || !user.public_id) {
      throw new Error('Public ID manquant');
    }

    // ✅ P1-1: Standardisation sur cookies httpOnly uniquement
    // Les tokens sont stockés dans des cookies httpOnly définis par le backend
    // On stocke uniquement les infos utilisateur pour l'affichage (pas les tokens)
    const env = getAuthEnv();
    setEnvUser(user, env, { mirrorLegacy: true });
    setEnvPublicId(user.public_id, env, { mirrorLegacy: true });
    // ❌ NE PAS stocker authToken ou refreshToken dans localStorage (sécurité)

    // ✅ Active automatiquement le Shadow Mode pour les admins
    if (String(user?.role || '').toLowerCase() === 'admin') {
      try {
        await apiClient.post(
          '/shadow-mode/session',
          {},
          {
            baseURL: '/api',
            skipAuthRedirect: true,
          }
        );
      } catch (shadowError) {
        console.warn(
          "⚠️ Impossible d'activer le Shadow Mode lors de la connexion admin:",
          shadowError?.response?.data || shadowError?.message || shadowError
        );
      }
    }

    // ✅ Vérifie si l'utilisateur doit changer son mot de passe
    if (user.force_password_change) {
      return { redirectToReset: true }; // ✅ Retourne un flag pour redirection
    }

    return { success: true }; // ✅ Connexion réussie
  } catch (error) {
    console.error('❌ Erreur lors de la connexion :', error);
    throw error;
  }
};

// ✅ Déconnexion d'un utilisateur (proxy vers apiClient.logoutUser)
export const logoutUser = async (options = { redirect: true }) => {
  await coreLogoutUser(options);
};

// ✅ Réinitialisation du mot de passe (page utilisateur)
export const resetPassword = async (newPassword) => {
  try {
    const response = await apiClient.post('/auth/update-password', {
      new_password: newPassword,
    });

    console.log('🔑 Mot de passe mis à jour :', response.data);
    return response.data;
  } catch (error) {
    console.error('❌ Erreur lors de la mise à jour du mot de passe :', error);
    throw error;
  }
};

// ✅ Obtenir un token "fresh" en vérifiant le mot de passe
export const getFreshToken = async (password) => {
  try {
    const response = await apiClient.post('/auth/fresh-token', {
      password,
    });

    // ✅ P1-1: Le token fresh est dans les cookies httpOnly
    // Pas besoin de le stocker dans localStorage
    console.log('✅ Token fresh obtenu avec succès (dans cookies httpOnly)');

    return response.data;
  } catch (error) {
    console.error('❌ Erreur lors de l\'obtention du token fresh :', error);
    throw error;
  }
};