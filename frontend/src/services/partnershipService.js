// frontend/src/services/partnershipService.js
import apiClient from '../utils/apiClient';

const API_BASE = '/partnerships';

/**
 * Récupère la liste des partenariats de l'entreprise connectée
 */
export const fetchPartnerships = async () => {
  try {
    const { data } = await apiClient.get(API_BASE);
    // La réponse est {data: [...]}, donc on accède à data.data
    const partnershipsList = data?.data || data || [];
    return Array.isArray(partnershipsList) ? partnershipsList : [];
  } catch (error) {
    console.error('Erreur lors de la récupération des partenariats:', error);
    throw error;
  }
};

/**
 * Récupère la liste des partenariats disponibles pour le transfert de courses
 * (uniquement ceux où l'entreprise est propriétaire)
 */
export const fetchPartnershipsForTransfer = async () => {
  try {
    const { data } = await apiClient.get(`${API_BASE}/for-transfer`);
    // La réponse est {data: [...]}, donc on accède à data.data
    const partnershipsList = data?.data || data || [];
    return Array.isArray(partnershipsList) ? partnershipsList : [];
  } catch (error) {
    console.error('Erreur lors de la récupération des partenariats pour transfert:', error);
    throw error;
  }
};

/**
 * Récupère un partenariat par son ID
 */
export const fetchPartnership = async (partnershipId) => {
  try {
    const { data } = await apiClient.get(`${API_BASE}/${partnershipId}`);
    return data;
  } catch (error) {
    console.error('Erreur lors de la récupération du partenariat:', error);
    throw error;
  }
};

/**
 * Crée un nouveau partenariat
 */
export const createPartnership = async (partnershipData) => {
  try {
    const { data } = await apiClient.post(API_BASE, partnershipData);
    return data;
  } catch (error) {
    console.error('Erreur lors de la création du partenariat:', error);
    throw error;
  }
};

/**
 * Met à jour un partenariat
 */
export const updatePartnership = async (partnershipId, partnershipData) => {
  try {
    const { data } = await apiClient.put(`${API_BASE}/${partnershipId}`, partnershipData);
    return data;
  } catch (error) {
    console.error('Erreur lors de la mise à jour du partenariat:', error);
    throw error;
  }
};

/**
 * Désactive un partenariat
 */
export const deactivatePartnership = async (partnershipId) => {
  try {
    const { data } = await apiClient.delete(`${API_BASE}/${partnershipId}`);
    return data;
  } catch (error) {
    console.error('Erreur lors de la désactivation du partenariat:', error);
    throw error;
  }
};

/**
 * Récupère les transferts d'un partenariat
 */
export const fetchPartnershipTransfers = async (partnershipId) => {
  try {
    const { data } = await apiClient.get(`${API_BASE}/${partnershipId}/transfers`);
    return Array.isArray(data) ? data : [];
  } catch (error) {
    console.error('Erreur lors de la récupération des transferts:', error);
    throw error;
  }
};

/**
 * Propose un transfert de course à un partenaire
 */
export const proposeTransfer = async (partnershipId, bookingId, transferModel = null) => {
  try {
    const payload = { booking_id: bookingId };
    if (transferModel) {
      payload.transfer_model = transferModel;
    }
    const { data } = await apiClient.post(`${API_BASE}/${partnershipId}/transfers`, payload);
    // La réponse est {data: {...}}, donc on accède à data.data
    return data?.data || data;
  } catch (error) {
    console.error('Erreur lors de la proposition du transfert:', error);
    throw error;
  }
};

/**
 * Accepte un transfert
 */
export const acceptTransfer = async (transferId) => {
  try {
    const { data } = await apiClient.post(`/partnerships/transfers/${transferId}/accept`);
    return data;
  } catch (error) {
    console.error('Erreur lors de l\'acceptation du transfert:', error);
    throw error;
  }
};

/**
 * Refuse un transfert
 */
export const rejectTransfer = async (transferId) => {
  try {
    const { data } = await apiClient.post(`/partnerships/transfers/${transferId}/reject`);
    return data;
  } catch (error) {
    console.error('Erreur lors du refus du transfert:', error);
    throw error;
  }
};

/**
 * Valide la complétion d'un transfert
 */
export const validateTransfer = async (transferId) => {
  try {
    const { data } = await apiClient.post(`/partnerships/transfers/${transferId}/validate`);
    return data;
  } catch (error) {
    console.error('Erreur lors de la validation du transfert:', error);
    throw error;
  }
};

/**
 * Récupère tous les transferts de l'entreprise connectée
 * Note: Pour l'instant, on récupère les transferts via les partenariats
 * Une route dédiée pourrait être ajoutée plus tard
 */
export const fetchAllTransfers = async (status = null) => {
  try {
    // Récupérer tous les partenariats
    const partnerships = await fetchPartnerships();
    
    // Récupérer les transferts de chaque partenariat
    const allTransfers = [];
    for (const partnership of partnerships) {
      try {
        const transfers = await fetchPartnershipTransfers(partnership.id);
        allTransfers.push(...transfers);
      } catch (err) {
        console.warn(`Erreur lors de la récupération des transferts pour le partenariat ${partnership.id}:`, err);
      }
    }
    
    // Filtrer par statut si demandé
    if (status) {
      return allTransfers.filter(t => t.status === status);
    }
    
    return allTransfers;
  } catch (error) {
    console.error('Erreur lors de la récupération des transferts:', error);
    throw error;
  }
};

/**
 * Récupère les statistiques d'un partenariat
 */
export const fetchPartnershipStats = async (partnershipId) => {
  try {
    const { data } = await apiClient.get(`${API_BASE}/${partnershipId}/stats`);
    return data;
  } catch (error) {
    console.error('Erreur lors de la récupération des statistiques:', error);
    throw error;
  }
};

/**
 * Récupère les factures mensuelles d'un partenariat
 */
export const fetchPartnershipInvoices = async (partnershipId) => {
  try {
    const { data } = await apiClient.get(`${API_BASE}/${partnershipId}/invoices`);
    return Array.isArray(data) ? data : [];
  } catch (error) {
    console.error('Erreur lors de la récupération des factures:', error);
    throw error;
  }
};

/**
 * Génère une facture mensuelle consolidée pour un partenariat
 */
export const generateMonthlyInvoice = async (partnershipId, year, month) => {
  try {
    const { data } = await apiClient.post(`${API_BASE}/${partnershipId}/invoices`, {
      year,
      month,
    });
    return data;
  } catch (error) {
    console.error('Erreur lors de la génération de la facture:', error);
    throw error;
  }
};

/**
 * Récupère les détails d'une facture partenaire
 */
export const fetchPartnerInvoice = async (partnershipId, invoiceId) => {
  try {
    const { data } = await apiClient.get(`${API_BASE}/${partnershipId}/invoices/${invoiceId}`);
    return data;
  } catch (error) {
    console.error('Erreur lors de la récupération de la facture:', error);
    throw error;
  }
};

/**
 * Marque une facture partenaire comme payée
 */
export const markInvoiceAsPaid = async (partnershipId, invoiceId) => {
  try {
    const { data } = await apiClient.put(`${API_BASE}/${partnershipId}/invoices/${invoiceId}`);
    return data;
  } catch (error) {
    console.error('Erreur lors de la mise à jour de la facture:', error);
    throw error;
  }
};

/**
 * Recherche des entreprises par nom
 */
export const searchCompanies = async (query) => {
  try {
    const { data } = await apiClient.get(`${API_BASE}/search-companies`, {
      params: { q: query },
    });
    return Array.isArray(data) ? data : [];
  } catch (error) {
    console.error('Erreur lors de la recherche d\'entreprises:', error);
    throw error;
  }
};

/**
 * Récupère les demandes de partenariat en attente
 */
export const fetchPendingRequests = async () => {
  try {
    const { data } = await apiClient.get(`${API_BASE}/requests`);
    return Array.isArray(data) ? data : [];
  } catch (error) {
    console.error('Erreur lors de la récupération des demandes:', error);
    throw error;
  }
};

/**
 * Accepter une demande de partenariat
 */
export const acceptPartnershipRequest = async (partnershipId) => {
  try {
    const { data } = await apiClient.post(`${API_BASE}/${partnershipId}/accept`);
    return data;
  } catch (error) {
    console.error('Erreur lors de l\'acceptation de la demande:', error);
    throw error;
  }
};

/**
 * Refuser une demande de partenariat
 */
export const rejectPartnershipRequest = async (partnershipId) => {
  try {
    const { data } = await apiClient.post(`${API_BASE}/${partnershipId}/reject`);
    return data;
  } catch (error) {
    console.error('Erreur lors du refus de la demande:', error);
    throw error;
  }
};

