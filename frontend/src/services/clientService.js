// src/services/clientService.js

import apiClient from '../utils/apiClient';
import { getFreshToken } from './authService';
import { getActivePublicId } from '../utils/webAuthSession';

/**
 * Récupère les informations du profil client pour le user connecté.
 * Résout le public_id via la session web active (env + fallback legacy).
 */
export const fetchClient = async (publicIdOverride = null) => {
  const publicId = publicIdOverride || getActivePublicId();
  if (!publicId) {
    throw new Error("Aucun public_id trouvé pour l'utilisateur connecté.");
  }
  try {
    const response = await apiClient.get(`/clients/${publicId}`);
    return response.data;
  } catch (error) {
    console.error('Erreur lors du chargement du profil client :', error);
    throw error;
  }
};

/**
 * Change le mot de passe du compte client connecté.
 * Le backend exige un JWT « fresh » : obtention via le mot de passe actuel, puis POST reset-password.
 */
export const changeClientPassword = async (publicId, { oldPassword, newPassword, confirmPassword }) => {
  const pid = String(publicId || '').trim();
  if (!pid) {
    throw new Error("Identifiant client manquant.");
  }
  await getFreshToken(oldPassword);
  const { data } = await apiClient.post(`/clients/${pid}/reset-password`, {
    old_password: oldPassword,
    new_password: newPassword,
    confirm_password: confirmPassword,
  });
  return data;
};
