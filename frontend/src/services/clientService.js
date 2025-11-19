// src/services/clientService.js

import apiClient from '../utils/apiClient';

/**
 * Récupère les informations du profil client pour le user connecté.
 * On suppose que le public_id est stocké dans localStorage.
 */
export const fetchClient = async () => {
  const publicId = localStorage.getItem('public_id');
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
