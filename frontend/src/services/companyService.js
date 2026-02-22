// src/services/companyService.js
import apiClient from '../utils/apiClient';

// --------------------------------------
// Helpers date/temps
// --------------------------------------
/**
 * Sélectionne le meilleur champ date présent sur un objet "booking/reservation"
 * et renvoie une ISO string (ou null si rien d'exploitable).
 */
const pickBestDateField = (r) => {
  if (!r || typeof r !== 'object') return null;
  // ordre de priorité des champs possibles vus côté backend/front
  const candidates = [
    r.scheduled_time,
    r.pickup_time,
    r.date_time,
    r.datetime,
    r.start_time,
    r.time,
    r.pickup_at,
    r.created_at,
  ];
  for (const v of candidates) {
    if (!v) continue;
    // Conserver les dates NAÏVES telles quelles pour éviter les décalages TZ
    if (typeof v === 'string') {
      const s = v.trim();
      if (s) return s; // ex: "2025-10-08T18:00:00" (sans Z)
    }
    if (v instanceof Date) {
      // Formater en local naïf (YYYY-MM-DDTHH:mm:ss), sans Z ni offset
      const d = v;
      const pad = (n) => String(n).padStart(2, '0');
      return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(
        d.getHours()
      )}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    }
  }
  return null;
};

/* -------------------------- RÉSERVATIONS ENTREPRISE -------------------------- */

export const fetchCompanyReservations = async (date) => {
  try {
    const { data } = await apiClient.get('/companies/me/reservations', {
      params: { flat: true, ...(date ? { date } : {}) },
      // Désactiver le cache pour forcer un rechargement
      headers: {
        'Cache-Control': 'no-cache',
        Pragma: 'no-cache',
      },
    });
    return Array.isArray(data) ? data : Array.isArray(data?.reservations) ? data.reservations : [];
  } catch (e) {
    // Gérer spécifiquement les erreurs d'authentification JWT
    if (e.response?.status === 401 || e.response?.status === 422) {
      console.error("Erreur d'authentification JWT:", e.response.data);
      // Optionnel: déclencher une reconnexion ou un refresh token
    }
    console.error('fetchCompanyReservations failed:', e?.response?.data || e);
    return []; // ✅ safe fallback
  }
};

/**
 * Récupère les réservations avec pagination/filtrage côté serveur.
 */
export const fetchCompanyReservationsPaginated = async ({
  date,
  startDate,
  endDate,
  page = 1,
  perPage = 25,
  status,
  tab,
  search,
  sortOrder = 'desc',
  excludeCanceled = false,
} = {}) => {
  try {
    const params = {
      flat: true,
      page,
      per_page: perPage,
      sort_order: sortOrder,
    };
    if (date) params.date = date;
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    if (status && status !== 'all') params.status = status;
    if (tab && tab !== 'all') params.tab = tab;
    if (search) params.search = search;
    if (excludeCanceled) params.exclude_canceled = true;

    const { data } = await apiClient.get('/companies/me/reservations', {
      params,
      headers: {
        'Cache-Control': 'no-cache',
        Pragma: 'no-cache',
      },
    });
    return data || {};
  } catch (e) {
    if (e.response?.status === 401 || e.response?.status === 422) {
      console.error("Erreur d'authentification JWT:", e.response.data);
    }
    console.error('fetchCompanyReservationsPaginated failed:', e?.response?.data || e);
    return {
      reservations: [],
      total: 0,
      page,
      per_page: perPage,
      total_pages: 0,
      stats: {
        total: 0,
        pending: 0,
        inProgress: 0,
        completed: 0,
        canceled: 0,
        revenue: 0,
      },
    };
  }
};

export const acceptReservation = async (reservationId) => {
  try {
    const { data } = await apiClient.post(`/companies/me/reservations/${reservationId}/accept`);
    return data;
  } catch (error) {
    throw error;
  }
};

export const rejectReservation = async (reservationId) => {
  const { data } = await apiClient.post(`/companies/me/reservations/${reservationId}/reject`);
  return data;
};

/**
 * Assigne un chauffeur à une réservation.
 * body: { driver_id }
 */
export const assignDriver = async (reservationId, driverId) => {
  const { data } = await apiClient.post(`/companies/me/reservations/${reservationId}/assign`, {
    driver_id: driverId,
  });
  return data;
};

export const completeReservation = async (reservationId) => {
  const { data } = await apiClient.post(`/companies/me/reservations/${reservationId}/complete`);
  return data;
};

/**
 * Supprime ou annule une réservation.
 * @param {number} reservationId
 * @param {string|null} reasonCode - Code motif (requis pour ASSIGNED/EN_ROUTE)
 * @param {string|null} reasonText - Texte libre (requis si reasonCode === 'OTHER')
 */
export const deleteReservation = async (reservationId, reasonCode = null, reasonText = null) => {
  const config = {};
  if (reasonCode) {
    const rt = reasonText?.trim();
    config.data = { reason_code: reasonCode, ...(rt ? { reason_text: rt } : {}) };
  }
  const { data } = await apiClient.delete(
    `/companies/me/reservations/${reservationId}`,
    config,
  );
  return data;
};

/**
 * Planifie une réservation à une date/heure précise (ISO local sans Z).
 */
export const scheduleReservation = async (reservationId, isoDatetime) => {
  const { data } = await apiClient.put(`/companies/me/reservations/${reservationId}/schedule`, {
    scheduled_time: isoDatetime,
  });
  return data;
};

/**
 * Met à jour une réservation (adresses, heure, informations médicales, etc.)
 * Utilise l'endpoint spécifique pour les entreprises qui autorise PENDING, ACCEPTED et ASSIGNED
 * @param {number} reservationId - ID de la réservation
 * @param {object} updateData - Données à mettre à jour (pickup_location, dropoff_location, scheduled_time, medical_facility, doctor_name, notes_medical, etc.)
 * @returns {Promise} Réservation mise à jour
 */
export const updateReservation = async (reservationId, updateData) => {
  try {
    const { data } = await apiClient.put(`/companies/me/reservations/${reservationId}`, updateData);
    return data;
  } catch (error) {
    console.error('[CompanyService] Error updating reservation:', error);
    throw error.response?.data || error.message;
  }
};

/**
 * Dispatch maintenant une réservation (+ minutes_offset).
 */
export const dispatchNowForReservation = async (reservationId, minutesOffset = 15) => {
  const { data } = await apiClient.post(
    `/companies/me/reservations/${reservationId}/dispatch-now`,
    { minutes_offset: minutesOffset }
  );
  return data;
};

/**
 * Déclencher un retour (payload: {}, { return_time }, ou { urgent: true, minutes_offset? }).
 */
export const triggerReturnBooking = async (reservationId, payload = {}) => {
  const { data } = await apiClient.post(
    `/companies/me/reservations/${reservationId}/trigger-return`,
    payload
  );
  return data;
};

/* --------------------------------- CHAUFFEURS -------------------------------- */

export const fetchCompanyDriver = async () => {
  try {
    const { data } = await apiClient.get('/companies/me/drivers');
    // backend: { driver: [...] } ou parfois déjà un tableau
    if (Array.isArray(data)) return data;
    if (Array.isArray(data?.drivers)) return data.drivers;
    if (Array.isArray(data?.driver)) return data.driver;
    return []; // fallback sûr
  } catch (e) {
    // Ne pas logger les erreurs 403/404 comme des erreurs critiques (permissions manquantes)
    const status = e?.response?.status;
    if (status !== 403 && status !== 404) {
      console.error('fetchCompanyDriver failed:', e?.response?.data || e);
    }
    return [];
  }
};

/** Positions GPS des chauffeurs (Redis + DB). Pour la carte live. */
export const fetchCompanyDriverLocations = async () => {
  try {
    const { data } = await apiClient.get('/companies/me/drivers/locations');
    return Array.isArray(data?.locations) ? data.locations : [];
  } catch (e) {
    const status = e?.response?.status;
    if (status !== 403 && status !== 404) {
      console.error('fetchCompanyDriverLocations failed:', e?.response?.data || e);
    }
    return [];
  }
};

/**
 * Crée un nouveau chauffeur (Utilisateur + Profil Chauffeur).
 * (nécessaire pour CompanyDriver.jsx)
 */
export const createDriver = async (driverData) => {
  try {
    const { data } = await apiClient.post('/companies/me/drivers/create', driverData);
    return data;
  } catch (error) {
    // Améliorer le message d'erreur pour le débogage
    const errorData = error.response?.data || {};
    const status = error.response?.status;
    
    console.error('[createDriver] Erreur backend:', {
      status,
      data: errorData,
      message: error.message
    });
    
    // Créer un objet Error enrichi avec les données du backend
    const enrichedError = new Error(errorData.message || errorData.error || error.message);
    enrichedError.status = status;
    enrichedError.details = errorData.details;
    enrichedError.error = errorData.error;
    
    throw enrichedError;
  }
};

/**
 * Associe un chauffeur existant par user_id
 */
export const addDriver = async (userId) => {
  const { data } = await apiClient.post('/companies/me/drivers', {
    user_id: userId,
  });
  return data;
};

/**
 * updateDriverStatus(id, true|false) → { is_active }
 * ou bien updateDriverStatus(id, { is_available: true }) → payload tel quel
 */
export const updateDriverStatus = async (driverId, payloadOrIsActive) => {
  const body =
    typeof payloadOrIsActive === 'object' ? payloadOrIsActive : { is_active: !!payloadOrIsActive };
  const { data } = await apiClient.put(`/companies/me/drivers/${driverId}`, body);
  return data;
};

export const updateDriverDetails = async (driverId, driverData) => {
  try {
    const { data } = await apiClient.put(`/companies/me/drivers/${driverId}`, driverData);
    return data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

export const deleteDriver = async (driverId) => {
  const { data } = await apiClient.delete(`/companies/me/drivers/${driverId}`);
  return data;
};

export const fetchDriverCompletedTrips = async (driverId) => {
  const { data } = await apiClient.get(`/companies/me/drivers/${driverId}/completed-trips`);
  return data;
};

export const toggleDriverType = async (driverId) => {
  const { data } = await apiClient.put(`/companies/me/drivers/${driverId}/toggle-type`);
  return data;
};

/* ------------------------------ ENTREPRISE (misc) ----------------------------- */

export const fetchCompanyInvoices = async () => {
  const { data } = await apiClient.get('/companies/me/invoices');
  return data;
};

export const setDispatchEnabled = async (enabled) => {
  const { data } = await apiClient.post('/companies/me/dispatch/activate', {
    enabled,
  });
  return data;
};

export const fetchCompanyInfo = async () => {
  try {
    const { data } = await apiClient.get('/companies/me');
    // ✅ Le backend retourne success_response(data=company.serialize) qui crée { data: {...}, message: "..." }
    // Il faut extraire data.data si la structure est { data: {...} }, sinon retourner data directement
    if (data && typeof data === 'object' && 'data' in data && !('error' in data)) {
      return data.data;
    }
    // Si data n'a pas de structure { data: {...} }, retourner data directement (compatibilité)
    return data;
  } catch (error) {
    // Ne pas logger les erreurs 403/404/401 comme des erreurs critiques (permissions manquantes ou company non trouvée)
    const status = error?.response?.status;
    if (status !== 403 && status !== 404 && status !== 401) {
      console.error('Error fetching company info:', error?.response?.data || error);
    }
    // Return a minimal valid company object to prevent UI from breaking
    return {
      id: null,
      name: 'Error loading company',
      email: '',
      phone: '',
      address: '',
      logo_url: null,
      error: true,
    };
  }
};

export const updateCompanyInfo = async (payload) => {
  const { data } = await apiClient.put('/companies/me', payload);
  return data;
};

export const uploadCompanyLogo = async (file) => {
  const form = new FormData();
  form.append('file', file);
  const { data } = await apiClient.post('/companies/me/logo', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data; // { logo_url, size_bytes }
};

/* ------------------------------ MESSAGERIE / CHAT ----------------------------- */

export const fetchCompanyMessages = async (companyId) => {
  const { data } = await apiClient.get(`/messages/${companyId}`);
  return data;
};

/* --------------------------- CLIENTS / ÉTABLISSEMENTS ------------------------- */

export const fetchCompanyClients = async () => {
  // Récupérer tous les clients en une seule fois (max 1000 par page)
  const { data } = await apiClient.get('/companies/me/clients?per_page=1000');
  // L'API retourne un format paginé : {data: [...], pagination: {...}, links: {...}}
  if (data && data.data && Array.isArray(data.data)) {
    return data.data;
  }
  // Fallback : format ancien {clients: [...]}
  return data.clients || [];
};

/**
 * (nécessaire pour ClientInvoices.jsx)
 */
/**
 * Récupère les détails complets d'un client (pour le drawer)
 * Combine les informations du client avec ses séjours et tiers payeurs
 */
export const fetchClientDetails = async (clientId) => {
  try {
    // Récupérer le client depuis la liste (ou créer un endpoint dédié si nécessaire)
    const clients = await fetchCompanyClients();
    const client = clients.find((c) => c.id === clientId);
    
    if (!client) {
      throw new Error('Client non trouvé');
    }

    // Charger les séjours et billing parties en parallèle
    const [staysResponse, billingPartiesResponse] = await Promise.allSettled([
      fetchClientStays(clientId),
      fetchClientBillingParties(clientId),
    ]);

    const stays = staysResponse.status === 'fulfilled' ? staysResponse.value.data || [] : [];
    const billingParties = billingPartiesResponse.status === 'fulfilled' 
      ? billingPartiesResponse.value.data || [] 
      : [];

    // Détecter si le client a un séjour actif ou un tiers payeur
    const hasActiveStay = stays.some((s) => s.status === 'active' && !s.end_date);
    const hasBillingParty = billingParties.length > 0;

    return {
      ...client,
      stays,
      billingParties,
      has_active_stay: hasActiveStay,
      has_billing_party: hasBillingParty,
    };
  } catch (err) {
    console.error('Erreur lors du chargement des détails client:', err);
    throw err;
  }
};

export const fetchClientReservations = async (
  clientId,
  { limit, includeInvoices = true } = {}
) => {
  const params = {};
  if (typeof includeInvoices === 'boolean') {
    params.include_invoices = includeInvoices;
  }
  if (limit) {
    params.limit = limit;
  }
  const { data } = await apiClient.get(`/companies/me/clients/${clientId}/reservations`, {
    params,
  });
  return data;
};

export const searchClients = async (query) => {
  try {
    const { data } = await apiClient.get(
      `/companies/me/clients?search=${encodeURIComponent(query)}`
    );
    // ✅ Le backend retourne un format paginé : {"data": [...], "pagination": {...}, "links": {...}}
    // Extraire le tableau clients depuis data.data
    if (data && data.data && Array.isArray(data.data)) {
      console.log(`✅ ${data.data.length} client(s) trouvé(s) pour "${query}"`);
      return data.data;
    }
    // Fallback : format ancien {"clients": [...]}
    if (data && Array.isArray(data.clients)) {
      console.log(`✅ ${data.clients.length} client(s) trouvé(s) pour "${query}"`);
      return data.clients;
    }
    // Fallback : si c'est déjà un tableau
    if (Array.isArray(data)) {
      return data;
    }
    console.warn('⚠️ Format de réponse inattendu:', data);
    return [];
  } catch (error) {
    console.error('❌ Error searching clients:', error?.response?.data || error);
    return []; // Return empty array on error
  }
};

export const createClient = async (payload) => {
  const { data } = await apiClient.post('/companies/me/clients', payload);
  return data;
};

/**
 * Met à jour un client (coordonnées, statut, etc.)
 */
export const updateClient = async (clientId, payload) => {
  try {
    const { data } = await apiClient.put(`/companies/me/clients/${clientId}`, payload);
    return data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

/**
 * Supprime un client (soft delete par défaut, hard delete si hardDelete=true)
 */
export const deleteClient = async (clientId, hardDelete = false) => {
  try {
    const params = hardDelete ? { hard: 'true' } : {};
    const { data } = await apiClient.delete(`/companies/me/clients/${clientId}`, { params });
    return data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

// ==================== SÉJOURS D'HOSPITALISATION ====================

/**
 * Récupère les séjours d'hospitalisation d'un client
 * @param {number} clientId - ID du client
 * @returns {Promise} Liste des séjours
 */
export const fetchClientStays = async (clientId) => {
  const response = await apiClient.get(`/clients/${clientId}/stays`);
  return response.data;
};

/**
 * Récupère le séjour actif d'un client avec toutes les informations de la clinique
 * @param {number} clientId - ID du client
 * @returns {Promise} Séjour actif avec détails de la clinique (adresse, tarif préférentiel, etc.)
 */
export const fetchClientActiveStay = async (clientId) => {
  try {
    const response = await apiClient.get(`/clients/${clientId}/active-stay`);
    return response.data;
  } catch (error) {
    // Si 404 ou pas de séjour actif, retourner null
    if (error.response?.status === 404 || error.response?.status === 200) {
      return { success: true, data: null };
    }
    throw error;
  }
};

export const createCompanyForInstitutionClient = async (clientId) => {
  try {
    const { data } = await apiClient.post(`/clients/${clientId}/create-company`);
    return data?.data || data;
  } catch (error) {
    console.error('Error creating company for institution client:', error);
    throw error;
  }
};

/**
 * Crée un nouveau séjour d'hospitalisation
 * @param {number} clientId - ID du client
 * @param {Object} stayData - Données du séjour (company_id, start_date, end_date?, status?, notes?)
 * @returns {Promise} Séjour créé
 */
export const createClientStay = async (clientId, stayData) => {
  const response = await apiClient.post(`/clients/${clientId}/stays`, stayData);
  return response.data;
};

/**
 * Met à jour un séjour d'hospitalisation
 * @param {number} stayId - ID du séjour
 * @param {Object} stayData - Données à mettre à jour
 * @returns {Promise} Séjour mis à jour
 */
export const updateClientStay = async (stayId, stayData) => {
  const response = await apiClient.patch(`/clients/client-stays/${stayId}`, stayData);
  return response.data;
};

/**
 * Ferme un séjour d'hospitalisation (définit end_date)
 * @param {number} stayId - ID du séjour
 * @returns {Promise} Séjour fermé
 */
export const closeClientStay = async (stayId, endDate = null) => {
  const payload = endDate ? { end_date: endDate } : {};
  const response = await apiClient.post(`/clients/client-stays/${stayId}/close`, payload);
  return response.data;
};

// ==================== TIERS PAYEURS / CURATEURS ====================

/**
 * Récupère les tiers payeurs liés à un client
 * @param {number} clientId - ID du client
 * @returns {Promise} Liste des liens ClientBillingParty
 * @note Endpoint backend : GET /api/v1/clients/{client_id}/billing-parties
 */
export const fetchClientBillingParties = async (clientId) => {
  // Endpoint: GET /api/v1/clients/{client_id}/billing-parties
  // Retourne: { success: true, data: [ClientBillingParty avec billing_party] }
  try {
    const response = await apiClient.get(`/clients/${clientId}/billing-parties`);
    return response.data;
  } catch (err) {
    // Si l'endpoint n'existe pas encore, retourner une liste vide
    if (err.response?.status === 404) {
      return { success: true, data: [] };
    }
    throw err;
  }
};

/**
 * Lie un tiers payeur à un client
 * @param {number} clientId - ID du client
 * @param {Object} linkData - Données du lien (billing_party_id, is_default?, role?)
 * @returns {Promise} Lien créé
 * @note Endpoint backend : POST /api/v1/clients/{client_id}/billing-parties
 */
export const linkClientBillingParty = async (clientId, linkData) => {
  // Endpoint: POST /api/v1/clients/{client_id}/billing-parties
  // Body: { billing_party_id, is_default?, role?, contact_name?, contact_email?, contact_phone? }
  const response = await apiClient.post(`/clients/${clientId}/billing-parties`, linkData);
  return response.data;
};

/**
 * Met à jour un lien client-billing party
 * @param {number} linkId - ID du lien
 * @param {Object} linkData - Données à mettre à jour
 * @returns {Promise} Lien mis à jour
 * @note Endpoint backend : PATCH /api/v1/clients/billing-party-links/{link_id}
 */
export const updateClientBillingPartyLink = async (linkId, linkData) => {
  const response = await apiClient.patch(`/clients/billing-party-links/${linkId}`, linkData);
  return response.data;
};

/**
 * Supprime un lien client-billing party
 * @param {number} linkId - ID du lien
 * @returns {Promise}
 * @note Endpoint backend : DELETE /api/v1/clients/billing-party-links/{link_id}
 */
export const unlinkClientBillingParty = async (linkId) => {
  const response = await apiClient.delete(`/clients/billing-party-links/${linkId}`);
  return response.data;
};

export const searchEstablishments = async (q, limit = 8, signal) => {
  const query = (q || '').trim();
  if (query.length < 2) return [];
  const { data } = await apiClient.get('/medical/establishments', {
    params: { q: query, limit: Math.min(Math.max(1, limit || 8), 25) },
    signal,
  });
  return Array.isArray(data) ? data : [];
};

export const listServicesByEstab = async (establishmentId, q = '') => {
  const id = Number(establishmentId);
  if (!Number.isFinite(id) || id <= 0) return [];
  const params = new URLSearchParams({ establishment_id: id });
  if (q?.trim()) params.append('q', q.trim());
  const url = `/medical/services?${params.toString()}`;
  try {
    const res = await apiClient.get(url);
    return Array.isArray(res.data) ? res.data : [];
  } catch {
    return [];
  }
};

/* ------------------------------ DISPATCH / MOTEUR ----------------------------- */

/**
 * ⚙️ État du moteur (compat existant)
 * GET /company_dispatch/status
 */
export const getDispatchStatus = async () => {
  const { data } = await apiClient.get('/company_dispatch/status');
  return data;
};

// --------------------------------------
// Helpers communs (mapping & payload)
// --------------------------------------
const normalizeMode = (m) => {
  const s = String(m ?? 'auto')
    .trim()
    .toLowerCase();
  // ⚡ Support pour semi_auto → convertit en auto (backend ne supporte que auto/heuristic_only/solver_only)
  if (s === 'semi_auto' || s === 'semi-auto') return 'auto';
  if (s === 'heuristic') return 'heuristic_only';
  if (s === 'solver') return 'solver_only';
  if (['auto', 'heuristic_only', 'solver_only'].includes(s)) return s;
  return 'auto';
};

const toRunPayload = ({
  forDate,
  regularFirst = true,
  allowEmergency,
  runAsync = true,
  mode = 'auto',
  overrides,
} = {}) => {
  const payload = {
    for_date: forDate,
    regular_first: !!regularFirst,
    ...(typeof allowEmergency === 'boolean' ? { allow_emergency: !!allowEmergency } : {}),
    async: !!runAsync,
  };
  // si l’API accepte 'mode' au root :
  // payload.mode = normalizeMode(mode);
  // sinon, le passer en overrides (recommandé => rétrocompatible) :
  payload.mode = normalizeMode(mode);
  const ov = { ...(overrides || {}) };
  ov.mode = normalizeMode(mode);
  if (Object.keys(ov).length) payload.overrides = ov;
  return payload;
};

/**
 * Aperçu (compat existant)
 * GET /company_dispatch/preview?for_date=YYYY-MM-DD&regular_first=true&allow_emergency=true|false
 * NB: si allowEmergency est omis (undefined), on laisse le backend hériter des settings.
 */
export const previewDispatch = async ({ forDate, regularFirst = true, allowEmergency } = {}) => {
  if (!forDate) throw new Error('forDate (YYYY-MM-DD) requis');
  const params = new URLSearchParams({
    for_date: forDate,
    regular_first: String(!!regularFirst),
  });
  if (typeof allowEmergency !== 'undefined') {
    params.append('allow_emergency', String(!!allowEmergency));
  }
  const { data } = await apiClient.get(`/company_dispatch/preview?${params.toString()}`);
  return data;
};

/**
 * Déclenche un run async (compat existant)
 * POST /company_dispatch/trigger
 */
export const triggerDispatch = async ({
  forDate,
  regularFirst = true,
  allowEmergency = true,
  overrides,
} = {}) => {
  if (!forDate) throw new Error('forDate (YYYY-MM-DD) requis');
  const payload = toRunPayload({
    forDate,
    regularFirst,
    allowEmergency,
    runAsync: false, // trigger est toujours async côté backend
    overrides,
  });
  // /trigger est l'API "queue" historique → toujours async
  const { data } = await apiClient.post('/company_dispatch/trigger', payload);
  return data; // { status: "queued", job_id }
};
/** * Run (async par défaut) — compat existant /company_dispatch/run
 */
export const runDispatchNow = async ({
  forDate,
  regularFirst = true,
  allowEmergency,
  runAsync = false,
  mode = 'auto',
  overrides,
} = {}) => {
  if (!forDate) throw new Error('forDate (YYYY-MM-DD) requis');
  const payload = {
    for_date: forDate,
    regular_first: !!regularFirst,
    ...(typeof allowEmergency === 'boolean' ? { allow_emergency: !!allowEmergency } : {}), // NullableBoolean → n’envoie rien si indéfini
    async: !!runAsync,
    ...(overrides && Object.keys(overrides).length ? { overrides } : {}),
  };

  // si tu veux quand même passer “mode”
  payload.overrides = {
    ...(payload.overrides || {}),
    mode: normalizeMode(mode),
  };

  try {
    // ▶️ Chemin principal : /run (200 si sync, 202 si async selon runAsync)
    const { data } = await apiClient.post('/company_dispatch/run', payload);
    return {
      ...data,
      dispatch_run_id: data.dispatch_run_id || data.meta?.dispatch_run_id || null,
    };
  } catch (e) {
    // Si c'est une erreur de validation (ex: mismatch de clés) → fallback vers trigger (202 queued)
    const status = e?.response?.status;
    if (status === 400 || status === 422) {
      console.error('RUN 400/422 body:', e?.response?.data || e);
      const { data } = await apiClient.post('/company_dispatch/trigger', payload);
      return data;
    }
    throw e;
  }
};

/**
 * Récupère le statut du dispatch pour une entreprise.
 *
 * @param {string} forDate - Date optionnelle (YYYY-MM-DD) pour obtenir le statut d'un dispatch spécifique
 * @returns {Promise<Object>} Statut du dispatch avec dispatch_run_id, compteurs, etc.
 */
export const fetchDispatchStatus = async (forDate = null) => {
  try {
    const params = forDate ? { date: forDate } : {};
    const { data } = await apiClient.get('/company_dispatch/status', { params });

    console.log(`[Dispatch] Status fetched for date=${forDate}:`, {
      is_running: data.is_running,
      dispatch_run_id: data.dispatch_run_id,
      assignments_count: data.counters?.assignments || 0,
      active_dispatch_run: data.active_dispatch_run,
    });

    return data;
  } catch (error) {
    console.error('[Dispatch] Error fetching status:', error?.response?.data || error);
    throw error;
  }
};

/**
 * Alias clair pour notre UI : "optimiser la journée"
 */
export const runDispatchForDay = async ({
  forDate,
  regularFirst = true,
  allowEmergency,
  mode = 'auto', // auto par défaut
  runAsync = true, // ⚡ Par défaut async, mais peut être forcé à false pour petits dispatchs
  overrides,
} = {}) => {
  if (!forDate) throw new Error('forDate (YYYY-MM-DD) requis');

  console.log(
    `runDispatchForDay called with: forDate=${forDate}, regularFirst=${regularFirst}, allowEmergency=${allowEmergency}, mode=${mode}, runAsync=${runAsync}`
  );

  const payload = toRunPayload({
    forDate,
    regularFirst,
    allowEmergency,
    runAsync, // ✅ correct key for helper → produces { async: true/false }
    mode,
    overrides,
  });

  try {
    // --- bloc principal : /run ---
    console.log('Sending dispatch request with payload:', payload);

    // ⚡ Timeout adaptatif : mode sync = 90s (pour dispatchs complexes), async = 60s
    const timeout = runAsync ? 60000 : 90000; // 60s pour async, 90s pour sync

    // 1) on tente /run (200 si sync, 202 si async)
    const { data } = await apiClient.post('/company_dispatch/run', payload, {
      timeout: timeout,
    });

    console.log('Dispatch response:', data);

    return {
      ...data,
      status: data.status || (runAsync ? 'queued' : 'completed'),
      dispatch_run_id: data.dispatch_run_id || data.meta?.dispatch_run_id || null,
    };
  } catch (e) {
    // --- fallback : /trigger ---
    // ⚡ Si c'est un timeout en mode sync, c'est acceptable (le dispatch continue en arrière-plan)
    if (e.code === 'ECONNABORTED' && !runAsync) {
      console.warn(
        '⚠️ [Dispatch] Timeout en mode sync, le dispatch continue en arrière-plan. Vérifiez via WebSocket ou rafraîchissement.'
      );
    } else {
      console.error('Dispatch request failed:', e);
      console.error('Error details:', e?.response?.data);
    }
    console.log('Falling back to /trigger endpoint');

    try {
      // /trigger = file d'attente → toujours async (timeout 60s)
      const { data } = await apiClient.post('/company_dispatch/trigger', payload, {
        timeout: 60000, // 60s pour async
      });
      return {
        ...data,
        status: data.status || 'queued',
        dispatch_run_id: data.dispatch_run_id || data.meta?.dispatch_run_id || null,
      };
    } catch (triggerError) {
      console.error('Trigger fallback also failed:', triggerError);
      throw triggerError; // on remonte l'erreur
    }
  }
};
const toYMD = (isoString) => {
  if (!isoString) return null;
  const d = new Date(isoString);
  if (isNaN(d.getTime())) return null;
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
};

/**
 * Récupère les réservations assignées (avec détails du chauffeur et de l’assignation)
 * pour une date donnée (YYYY-MM-DD).
 * Si forDate est omis, récupère pour aujourd’hui.
 */
export const fetchAssignedReservations = async (forDate) => {
  console.log(`[Dispatch] Fetching assigned reservations for date: ${forDate}`);

  try {
    // Use separate try/catch blocks to handle each request independently
    let reservations = [];
    let assignments = [];

    try {
      const reservationsRes = await apiClient.get('/companies/me/reservations/', {
        params: { flat: true, ...(forDate ? { date: forDate } : {}) },
      });

      // Normalise la charge utile en tableau
      const payload = reservationsRes.data;
      reservations = Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.reservations)
          ? payload.reservations
          : [];
      console.log(`[Dispatch] Received ${reservations.length} reservations for date=${forDate}`);
    } catch (error) {
      console.error('[Dispatch] Error fetching reservations:', error?.response?.data || error);
      // Continue with empty reservations array
    }

    try {
      const assignmentsRes = await apiClient.get('/company_dispatch/assignments', {
        params: forDate ? { date: forDate } : undefined,
      });

      // /company_dispatch/assignments doit rester un tableau
      assignments = Array.isArray(assignmentsRes.data) ? assignmentsRes.data : [];
      console.log(`[Dispatch] Received ${assignments.length} assignments for date=${forDate}`);

      // ✅ Log détaillé pour debugging (avec driver info)
      if (assignments.length > 0) {
        console.log(
          `[Dispatch] Assignments details:`,
          assignments.map((a) => ({
            id: a.id,
            booking_id: a.booking_id,
            driver_id: a.driver_id,
            driver: a.driver
              ? {
                  id: a.driver.id,
                  full_name: a.driver.full_name,
                  name: a.driver.name,
                  user: a.driver.user
                    ? {
                        first_name: a.driver.user.first_name,
                        last_name: a.driver.user.last_name,
                        username: a.driver.user.username,
                      }
                    : null,
                }
              : null,
            status: a.status,
          }))
        );
      } else {
        console.warn(
          `[Dispatch] No assignments found for date=${forDate} (${reservations.length} reservations exist)`
        );
      }
    } catch (error) {
      console.error('[Dispatch] Error fetching assignments:', error?.response?.data || error);
      // Continue with empty assignments array
    }

    console.log(
      `[Dispatch] Processing ${reservations.length} reservations and ${assignments.length} assignments for date=${forDate}`
    );

    const byBookingId = new Map(assignments.map((a) => [a.booking_id, a]));
    console.log(`[Dispatch] Created map with ${byBookingId.size} assignments by booking ID`);

    // Jour cible : YYYY-MM-DD (local)
    const targetDay =
      forDate ||
      (() => {
        const d = new Date();
        const pad = (n) => String(n).padStart(2, '0');
        return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
      })();

    console.log(`Filtering bookings for target day: ${targetDay}`);

    // 🔧 filtre plus tolérant : accepte scheduled_time, pickup_time, date_time, datetime
    const bookingsOfDay = reservations.filter((r) => {
      try {
        const rawWhen = pickBestDateField(r);
        const ymd = toYMD(rawWhen);
        return !targetDay || ymd === targetDay;
      } catch (e) {
        console.error('Error filtering booking:', e);
        return false;
      }
    });

    console.log(`Found ${bookingsOfDay.length} bookings for the target day`);

    // Construction des lignes
    const rows = bookingsOfDay
      .map((b) => {
        try {
          const a = byBookingId.get(b.id) || null;

          const clientName =
            b.client_name ||
            b.client?.full_name ||
            b.client_name ||
            b.client?.name ||
            'Non spécifié';

          // ✅ Source temporelle unifiée
          const when = pickBestDateField(b);
          const scheduled_time = when;
          const dropoff_time = b.dropoff_time || b.drop_time || null;

          // Create a synthetic assignment if booking has driver_id but no assignment
          const syntheticAssignment =
            !a && b.driver_id
              ? {
                  id: null, // Synthetic assignment has no ID
                  booking_id: b.id,
                  driver_id: b.driver_id,
                  status: 'assigned', // Default status
                  estimated_pickup_arrival: null,
                  estimated_dropoff_arrival: null,
                  is_synthetic: true, // Flag to identify synthetic assignments
                }
              : null;

          return {
            id: b.id,
            // ✅ P1-4 Phase 2.1: Utiliser client_name au lieu de customer_name
            client_name: clientName,
            client: b.client || { full_name: clientName },
            scheduled_time,
            pickup_time: scheduled_time, // compat
            dropoff_time,
            pickup_location: b.pickup_location || b.pickup_address || b.origin || '',
            dropoff_location: b.dropoff_location || b.dropoff_address || b.destination || '',
            // ✅ P1-4 Phase 2.4: Ajouter coordonnées GPS
            pickup_lat: b.pickup_lat || null,
            pickup_lon: b.pickup_lon || null,
            dropoff_lat: b.dropoff_lat || null,
            dropoff_lon: b.dropoff_lon || null,
            amount: b.amount || b.price || 0, // ✅ Ajout du montant
            is_return: !!b.is_return,
            parent_booking_id: b.parent_booking_id || b.outbound_booking_id || null, // ✅ ID de la course aller
            time_confirmed: b.time_confirmed, // ✅ Ajout de time_confirmed pour les retours
            status: b.status || 'scheduled',
            driver_username: b.driver_username || b.driver?.username,
            driver_id: b.driver_id || a?.driver_id || null, // ✅ Ajout du driver_id
            driver: b.driver || null,
            // ✅ P1-4 Phase 2.2: Utiliser company_id et company_name du backend
            company_id: b.company_id || null,
            company_name: b.company_name || null,
            // ✅ P1-4 Phase 2.3: Utiliser timestamps ISO (le backend expose maintenant created_at et updated_at en ISO)
            created_at: b.created_at || null,
            updated_at: b.updated_at || null,
            // ✅ P1-4 Phase 2.3: Utiliser created_at_formatted et updated_at_formatted si disponibles (pour compatibilité)
            created_at_formatted: b.created_at_formatted || null,
            updated_at_formatted: b.updated_at_formatted || null,
            // accepte ancienne/ nouvelle forme (eta_* vs estimated_*)
            assignment: a
              ? {
                  id: a.id,
                  booking_id: a.booking_id,
                  driver_id: a.driver_id,
                  driver: a.driver || null, // ⭐ IMPORTANT : Copier le driver de l'assignment
                  status: a.status,
                  estimated_pickup_arrival:
                    a.estimated_pickup_arrival || a.eta_pickup_at || a.pickup_eta || null,
                  estimated_dropoff_arrival:
                    a.estimated_dropoff_arrival || a.eta_dropoff_at || a.dropoff_eta || null,
                }
              : syntheticAssignment,
          };
        } catch (e) {
          console.error('Error processing booking:', e);
          // Return a minimal valid row to avoid breaking the UI
          return {
            id: b.id || Math.random().toString(36).substring(2, 15),
            client_name: 'Error processing booking',
            scheduled_time: new Date().toISOString(),
            pickup_location: '',
            dropoff_location: '',
            status: 'error',
          };
        }
      })
      .filter(Boolean); // Remove any undefined entries

    console.log(
      `[Dispatch] Returning ${rows.length} formatted rows for dispatch table (date=${forDate})`
    );

    // ✅ Log du nombre de lignes avec assignments
    const rowsWithAssignments = rows.filter((r) => r.assignment);
    console.log(`[Dispatch] Rows with assignments: ${rowsWithAssignments.length}/${rows.length}`);

    return rows;
  } catch (error) {
    console.error('[Dispatch] Error fetching assigned reservations:', error);
    // Return empty array instead of throwing to prevent UI from breaking
    return [];
  }
};

/**
 * ⏱️ Retards courants (monté sous /company_dispatch/delays)
 */
export const fetchDispatchDelays = async (date) => {
  try {
    const { data } = await apiClient.get('/company_dispatch/delays', {
      params: { date },
    });
    // normalize: one item per late leg
    return (Array.isArray(data) ? data : []).flatMap((d) => {
      const rows = [];
      if ((d.pickup_delay_minutes ?? 0) >= 5) {
        rows.push({
          booking_id: d.booking_id,
          delay_minutes: d.pickup_delay_minutes,
          is_pickup: true,
        });
      }
      if ((d.dropoff_delay_minutes ?? 0) >= 5) {
        rows.push({
          booking_id: d.booking_id,
          delay_minutes: d.dropoff_delay_minutes,
          is_pickup: false,
        });
      }
      return rows;
    });
  } catch (e) {
    console.error('fetchDispatchDelays failed:', e?.response?.data || e);
    return [];
  }
};

/**
 * 🗂️ Historique des runs & détail d’un run
 */
export const fetchDispatchRuns = async ({ limit = 50, offset = 0 } = {}) => {
  const { data } = await apiClient.get('/company_dispatch/runs', {
    params: { limit, offset },
  });
  return Array.isArray(data) ? data : [];
};

export const fetchDispatchRunById = async (runId) => {
  if (!runId) throw new Error('runId requis');
  const { data } = await apiClient.get(`/company_dispatch/runs/${encodeURIComponent(runId)}`);
  return data;
};

/**
 * ✏️ MAJ d’une assignation (driver/status) & réassignation
 */
export const patchAssignment = async (assignmentId, payload = {}) => {
  if (!assignmentId) throw new Error('assignmentId requis');
  const { data } = await apiClient.patch(
    `/company_dispatch/assignments/${encodeURIComponent(assignmentId)}`,
    payload
  );
  return data;
};

export const reassignAssignment = async (assignmentId, newDriverId) => {
  if (!assignmentId) throw new Error('assignmentId requis');
  if (!newDriverId) throw new Error('newDriverId requis');
  const { data } = await apiClient.post(
    `/company_dispatch/assignments/${encodeURIComponent(assignmentId)}/reassign`,
    { new_driver_id: Number(newDriverId) }
  );
  return data;
};

/* ------------------------------ VÉHICULES ------------------------------------ */

/**
 * Récupère la liste des véhicules de l'entreprise
 */
export const fetchCompanyVehicles = async () => {
  try {
    const { data } = await apiClient.get('/companies/me/vehicles');
    return Array.isArray(data) ? data : [];
  } catch (error) {
    console.error('Error fetching vehicles:', error?.response?.data || error);
    return [];
  }
};

/**
 * Crée un nouveau véhicule
 */
export const createVehicle = async (vehicleData) => {
  try {
    const { data } = await apiClient.post('/companies/me/vehicles', vehicleData);
    return data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

/**
 * Récupère les détails d'un véhicule spécifique
 */
export const fetchVehicle = async (vehicleId) => {
  try {
    const { data } = await apiClient.get(`/companies/me/vehicles/${vehicleId}`);
    return data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

/**
 * Met à jour un véhicule
 */
export const updateVehicle = async (vehicleId, vehicleData) => {
  try {
    const { data } = await apiClient.put(`/companies/me/vehicles/${vehicleId}`, vehicleData);
    return data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

/**
 * Supprime un véhicule (soft delete par défaut, hard delete si hardDelete=true)
 */
export const deleteVehicle = async (vehicleId, hardDelete = false) => {
  try {
    const params = hardDelete ? { hard: 'true' } : {};
    const { data } = await apiClient.delete(`/companies/me/vehicles/${vehicleId}`, { params });
    return data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

/* ------------------------------- CRÉATION MANUELLE ---------------------------- */

export const createManualBooking = async (bookingData) => {
  try {
    const { data } = await apiClient.post('/companies/me/reservations/manual', bookingData);
    return data; // { message, reservation, return_booking? }
  } catch (error) {
    // ⚡ Préserver toute la structure de l'erreur Axios pour que le frontend puisse accéder à error.response.data
    // Si on lance seulement error.response?.data, on perd la structure { response: { data: {...}, status: 400 } }
    if (error.response) {
      // Erreur HTTP avec réponse (400, 500, etc.) - lancer l'erreur complète
      throw error;
    }
    // Erreur réseau ou autre - lancer avec un format standardisé
    throw new Error(error.message || 'Erreur lors de la création de la réservation');
  }
};

/* -------------------------------- PHASE 2 APIs -------------------------------- */

/**
 * Reprogramme une réservation avec une nouvelle heure
 * @param {number} reservationId - ID de la réservation
 * @param {string} newTime - Nouvelle heure au format HH:MM
 * @param {string} date - Date au format YYYY-MM-DD (optionnel)
 * @returns {Promise} Réservation mise à jour
 */
export const rescheduleBooking = async (reservationId, newTime, date = null) => {
  try {
    const payload = { new_time: newTime };
    if (date) payload.date = date;

    const { data } = await apiClient.put(
      `/companies/me/reservations/${reservationId}/reschedule`,
      payload
    );
    return data;
  } catch (error) {
    console.error('[CompanyService] Error rescheduling booking:', error);
    throw error.response?.data || error.message;
  }
};

/**
 * Planifie l'heure de retour pour une réservation
 * @param {number} reservationId - ID de la réservation
 * @param {string} returnTime - Heure de retour au format ISO ou HH:MM
 * @returns {Promise} Réservation mise à jour
 */
export const scheduleReturn = async (reservationId, returnTime) => {
  try {
    const { data } = await apiClient.put(
      `/companies/me/reservations/${reservationId}/schedule_return`,
      { return_time: returnTime }
    );
    return data;
  } catch (error) {
    console.error('[CompanyService] Error scheduling return:', error);
    throw error.response?.data || error.message;
  }
};

/**
 * Dispatche une réservation immédiatement (priorité urgente)
 * @param {number} reservationId - ID de la réservation
 * @returns {Promise} Résultat du dispatch
 */
export const dispatchBookingNow = async (reservationId) => {
  try {
    const { data } = await apiClient.post(
      `/companies/me/reservations/${reservationId}/dispatch_now`
    );
    return data;
  } catch (error) {
    console.error('[CompanyService] Error dispatching booking now:', error);
    throw error.response?.data || error.message;
  }
};

// --------------------------------------
// Recherche d'institutions officielles
// --------------------------------------

/**
 * Recherche parmi les institutions officielles de la plateforme.
 * Utilisé lors de la création/édition d'un client institution pour le lier
 * à une institution officielle existante.
 * @param {string} query - Texte de recherche (min 2 caractères)
 * @returns {Promise<Array<{id: number, name: string, institution_type: string, address: string, contact_email: string, contact_phone: string}>>}
 */
export const searchOfficialInstitutions = async (query) => {
  try {
    if (!query || query.length < 2) return [];
    const { data } = await apiClient.get('/companies/me/institutions/search', {
      params: { q: query },
    });
    return data.institutions || [];
  } catch (error) {
    console.error('[CompanyService] Error searching institutions:', error);
    return [];
  }
};

// Offres de transport institutionnelles
// --------------------------------------

/**
 * Récupère les offres de transport reçues par l'entreprise.
 * @param {string} [status] - Filtrer par statut (PENDING, ACCEPTED, etc.)
 * @returns {Promise<{offers: Array, total: number}>}
 */
export const fetchRequestOffers = async (status) => {
  try {
    const params = {};
    if (status) params.status = status;
    const { data } = await apiClient.get('/company/request-offers', { params });
    return data;
  } catch (error) {
    console.error('[CompanyService] Error fetching request offers:', error);
    return { offers: [], total: 0 };
  }
};

/**
 * Accepte une offre de transport institutionnelle.
 * @param {number} offerId - ID de l'offre
 * @param {string} [proposedPickupTime] - Horaire de prise en charge proposé (ISO8601)
 * @returns {Promise<{success: boolean, offer_id: number, booking_id: number}>}
 */
export const acceptRequestOffer = async (offerId, proposedPickupTime) => {
  const body = {};
  if (proposedPickupTime) body.proposed_pickup_time = proposedPickupTime;
  const { data } = await apiClient.post(`/company/request-offers/${offerId}/accept`, body);
  return data;
};

/**
 * Refuse une offre de transport institutionnelle.
 * @param {number} offerId - ID de l'offre
 * @param {string} [reason] - Raison du refus
 * @returns {Promise<{success: boolean, offer_id: number}>}
 */
export const rejectRequestOffer = async (offerId, reason) => {
  const { data } = await apiClient.post(`/company/request-offers/${offerId}/reject`, {
    reason,
  });
  return data;
};

// ---------------------------------------------------------------------------
// Mini-chat booking (institution <-> entreprise)
// ---------------------------------------------------------------------------

/**
 * Recupere les messages d'un booking (pagine).
 * @param {number} bookingId
 * @param {{ limit?: number, beforeId?: number }} options
 * @returns {Promise<{ messages: Array, has_more: boolean }>}
 */
export const fetchBookingMessages = async (bookingId, { limit = 30, beforeId } = {}) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (beforeId) params.set('before_id', String(beforeId));
  const { data } = await apiClient.get(`/bookings/${bookingId}/messages?${params}`);
  return data;
};

/**
 * Envoie un message dans le mini-chat d'un booking.
 * @param {number} bookingId
 * @param {string} content
 * @returns {Promise<{ message: Object }>}
 */
export const sendBookingMessage = async (bookingId, content) => {
  const { data } = await apiClient.post(`/bookings/${bookingId}/messages`, { content });
  return data;
};

// ============================================================================
// Company Notifications
// ============================================================================

export const fetchCompanyNotifications = async ({ limit = 30, offset = 0 } = {}) => {
  const { data } = await apiClient.get('/companies/notifications', { params: { limit, offset } });
  return data;
};

export const markCompanyNotificationRead = async (notificationId) => {
  const { data } = await apiClient.put(`/companies/notifications/${notificationId}/read`);
  return data;
};

export const markAllCompanyNotificationsRead = async () => {
  const { data } = await apiClient.put('/companies/notifications/read-all');
  return data;
};
