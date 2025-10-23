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

export const acceptReservation = async (reservationId) => {
  const { data } = await apiClient.post(`/companies/me/reservations/${reservationId}/accept`);
  return data;
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
 * Supprime une réservation.
 */
export const deleteReservation = async (reservationId) => {
  const { data } = await apiClient.delete(`/companies/me/reservations/${reservationId}`);
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
    console.error('fetchCompanyDriver failed:', e?.response?.data || e);
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
    throw error.response?.data || error;
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
    return data;
  } catch (error) {
    console.error('Error fetching company info:', error?.response?.data || error);
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
  // L'API retourne {clients: [...], total: number}
  return data.clients || [];
};

/**
 * (nécessaire pour ClientInvoices.jsx)
 */
export const fetchClientReservations = async (clientId) => {
  const { data } = await apiClient.get(`/companies/me/clients/${clientId}/reservations`);
  return data;
};

export const searchClients = async (query) => {
  try {
    const { data } = await apiClient.get(
      `/companies/me/clients?search=${encodeURIComponent(query)}`
    );
    // ✅ Le backend retourne {"clients": [...], "total": ...}
    // Extraire le tableau clients
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
 * Alias clair pour notre UI : "optimiser la journée"
 */
export const runDispatchForDay = async ({
  forDate,
  regularFirst = true,
  allowEmergency,
  mode = 'auto', // auto par défaut
  runAsync = true, // Changed default to true for reliability
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

    // 1) on tente /run (200 si sync, 202 si async)
    const { data } = await apiClient.post('/company_dispatch/run', payload);

    console.log('Dispatch response:', data);

    return {
      ...data,
      status: data.status || (runAsync ? 'queued' : 'completed'),
      dispatch_run_id: data.dispatch_run_id || data.meta?.dispatch_run_id || null,
    };
  } catch (e) {
    // --- fallback : /trigger ---
    console.error('Dispatch request failed:', e);
    console.error('Error details:', e?.response?.data);
    console.log('Falling back to /trigger endpoint');

    try {
      // /trigger = file d'attente → toujours async
      const { data } = await apiClient.post('/company_dispatch/trigger', payload);
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
  console.log(`Fetching assigned reservations for date: ${forDate}`);

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
      console.log(`Received ${reservations.length} reservations`);
    } catch (error) {
      console.error('Error fetching reservations:', error?.response?.data || error);
      // Continue with empty reservations array
    }

    try {
      const assignmentsRes = await apiClient.get('/company_dispatch/assignments', {
        params: forDate ? { date: forDate } : undefined,
      });

      // /company_dispatch/assignments doit rester un tableau
      assignments = Array.isArray(assignmentsRes.data) ? assignmentsRes.data : [];
      console.log(`Received ${assignments.length} assignments`);
    } catch (error) {
      console.error('Error fetching assignments:', error?.response?.data || error);
      // Continue with empty assignments array
    }

    console.log(
      `Processing ${reservations.length} reservations and ${assignments.length} assignments`
    );

    const byBookingId = new Map(assignments.map((a) => [a.booking_id, a]));
    console.log(`Created map with ${byBookingId.size} assignments by booking ID`);

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
            b.customer_name ||
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
            customer_name: clientName,
            client: b.client || { full_name: clientName },
            scheduled_time,
            pickup_time: scheduled_time, // compat
            dropoff_time,
            pickup_location: b.pickup_location || b.pickup_address || b.origin || '',
            dropoff_location: b.dropoff_location || b.dropoff_address || b.destination || '',
            amount: b.amount || b.price || 0, // ✅ Ajout du montant
            is_return: !!b.is_return,
            parent_booking_id: b.parent_booking_id || b.outbound_booking_id || null, // ✅ ID de la course aller
            time_confirmed: b.time_confirmed, // ✅ Ajout de time_confirmed pour les retours
            status: b.status || 'scheduled',
            driver_username: b.driver_username || b.driver?.username,
            driver_id: b.driver_id || a?.driver_id || null, // ✅ Ajout du driver_id
            driver: b.driver || null,
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
            customer_name: 'Error processing booking',
            scheduled_time: new Date().toISOString(),
            pickup_location: '',
            dropoff_location: '',
            status: 'error',
          };
        }
      })
      .filter(Boolean); // Remove any undefined entries

    console.log(`Returning ${rows.length} formatted rows for dispatch table`);
    return rows;
  } catch (error) {
    console.error('Error fetching assigned reservations:', error);
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
    throw error.response?.data || error.message;
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
    const { data} = await apiClient.put(
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
