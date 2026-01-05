// frontend/src/services/authService.js
import apiClient, { logoutUser as coreLogoutUser, cleanLocalSession } from '../utils/apiClient';

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
    const { token, user } = response.data;

    console.log('🔐 Connexion réussie. Données reçues :', response.data);

    if (!user || !user.public_id) {
      throw new Error('Public ID manquant');
    }

    // ✅ Stocke les informations utilisateur
    localStorage.setItem('authToken', token);
    localStorage.setItem('user', JSON.stringify(user));
    localStorage.setItem('public_id', user.public_id);

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
  try {
    await coreLogoutUser({ redirect: false });
  } finally {
    cleanLocalSession();

    if (options?.redirect !== false) {
      window.location.href = '/login';
    }
  }
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

    const { access_token } = response.data;
    if (access_token) {
      // Mettre à jour le token dans localStorage (pour mode mobile/compatibilité)
      localStorage.setItem('authToken', access_token);
      console.log('✅ Token fresh obtenu avec succès');
    }

    return response.data;
  } catch (error) {
    console.error('❌ Erreur lors de l\'obtention du token fresh :', error);
    throw error;
  }
};