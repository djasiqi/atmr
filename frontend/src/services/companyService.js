// src/services/companyService.js
import apiClient from "../utils/apiClient";

/* -------------------------- RÉSERVATIONS ENTREPRISE -------------------------- */

export const fetchCompanyReservations = async () => {
  try {
    const payload = await apiClient.get("/companies/me/reservations?flat=true");
    const reservations = Array.isArray(payload.data)
      ? payload.data
      : (Array.isArray(payload.data?.reservations) ? payload.data.reservations : []);
    return reservations;               // ✅ manquait
  } catch (e) {
    console.error("fetchCompanyReservations failed:", e?.response?.data || e);
    return [];                         // ✅ safe fallback
  }
};

export const acceptReservation = async (reservationId) => {
  const { data } = await apiClient.post(
    `/companies/me/reservations/${reservationId}/accept`
  );
  return data;
};

export const rejectReservation = async (reservationId) => {
  const { data } = await apiClient.post(
    `/companies/me/reservations/${reservationId}/reject`
  );
  return data;
};

/**
 * Assigne un chauffeur à une réservation.
 * body: { driver_id }
 */
export const assignDriver = async (reservationId, driverId) => {
  const { data } = await apiClient.post(
    `/companies/me/reservations/${reservationId}/assign`,
    { driver_id: driverId }
  );
  return data;
};

export const completeReservation = async (reservationId) => {
  const { data } = await apiClient.post(
    `/companies/me/reservations/${reservationId}/complete`
  );
  return data;
};

/**
 * Supprime une réservation.
 */
export const deleteReservation = async (reservationId) => {
  const { data } = await apiClient.delete(
    `/companies/me/reservations/${reservationId}`
  );
  return data;
};

/**
 * Planifie une réservation à une date/heure précise (ISO local sans Z).
 */
export const scheduleReservation = async (reservationId, isoDatetime) => {
  const { data } = await apiClient.put(
    `/companies/me/reservations/${reservationId}/schedule`,
    { scheduled_time: isoDatetime }
  );
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
    const { data } = await apiClient.get("/companies/me/driver");
    // backend: { driver: [...] } ou parfois déjà un tableau
    if (Array.isArray(data)) return data;
    if (Array.isArray(data?.driver)) return data.driver;
    return []; // fallback sûr
  } catch (e) {
    console.error("fetchCompanyDriver failed:", e?.response?.data || e);
    return [];
  }
};

/**
 * Crée un nouveau chauffeur (Utilisateur + Profil Chauffeur).
 * (nécessaire pour CompanyDriver.jsx)
 */
export const createDriver = async (driverData) => {
  try {
    const { data } = await apiClient.post("/companies/me/drivers/create", driverData);
    return data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

/**
 * Associe un chauffeur existant par user_id
 */
export const addDriver = async (userId) => {
  const { data } = await apiClient.post("/companies/me/driver", { user_id: userId });
  return data;
};

/**
 * updateDriverStatus(id, true|false) → { is_active }
 * ou bien updateDriverStatus(id, { is_available: true }) → payload tel quel
 */
export const updateDriverStatus = async (driverId, payloadOrIsActive) => {
  const body = (typeof payloadOrIsActive === "object")
    ? payloadOrIsActive
    : { is_active: !!payloadOrIsActive };
  const { data } = await apiClient.put(`/companies/me/driver/${driverId}`, body);
  return data;
};

export const updateDriverDetails = async (driverId, driverData) => {
  try {
    const { data } = await apiClient.put(`/companies/me/driver/${driverId}`, driverData);
    return data;
  } catch (error) {
    throw error.response?.data || error;
  }
};

export const deleteDriver = async (driverId) => {
  const { data } = await apiClient.delete(`/companies/me/driver/${driverId}`);
  return data;
};

export const fetchDriverCompletedTrips = async (driverId) => {
  const { data } = await apiClient.get(`/companies/me/driver/${driverId}/completed-trips`);
  return data;
};

export const toggleDriverType = async (driverId) => {
  const { data } = await apiClient.put(`/companies/me/driver/${driverId}/toggle-type`);
  return data;
};

/* ------------------------------ ENTREPRISE (misc) ----------------------------- */

export const fetchCompanyInvoices = async () => {
  const { data } = await apiClient.get("/companies/me/invoices");
  return data;
};

export const setDispatchEnabled = async (enabled) => {
  const { data } = await apiClient.post("/companies/me/dispatch/activate", { enabled });
  return data;
};

export const fetchCompanyInfo = async () => {
  const { data } = await apiClient.get("/companies/me");
  return data;
};

export const updateCompanyInfo = async (payload) => {
  const { data } = await apiClient.put("/companies/me", payload);
  return data;
};

export const uploadCompanyLogo = async (file) => {
  const form = new FormData();
  form.append("file", file);
  const { data } = await apiClient.post("/companies/me/logo", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data; // { logo_url }
};

/* ------------------------------ MESSAGERIE / CHAT ----------------------------- */

export const fetchCompanyMessages = async (companyId) => {
  const { data } = await apiClient.get(`/messages/${companyId}`);
  return data;
};

/* --------------------------- CLIENTS / ÉTABLISSEMENTS ------------------------- */

export const fetchCompanyClients = async () => {
  const { data } = await apiClient.get("/companies/me/clients");
  return data;
};

/**
 * (nécessaire pour ClientInvoices.jsx)
 */
export const fetchClientReservations = async (clientId) => {
  const { data } = await apiClient.get(`/companies/me/clients/${clientId}/reservations`);
  return data;
};

export const searchClients = async (query) => {
  const { data } = await apiClient.get(
    `/companies/me/clients?search=${encodeURIComponent(query)}`
  );
  return data;
};

export const createClient = async (payload) => {
  const { data } = await apiClient.post("/companies/me/clients", payload);
  return data;
};

export const searchEstablishments = async (q, limit = 8, signal) => {
  const query = (q || "").trim();
  if (query.length < 2) return [];
  const { data } = await apiClient.get("/medical/establishments", {
    params: { q: query, limit: Math.min(Math.max(1, limit || 8), 25) },
    signal,
  });
  return Array.isArray(data) ? data : [];
};

export const listServicesByEstab = async (establishmentId, q = "") => {
  const id = Number(establishmentId);
  if (!Number.isFinite(id) || id <= 0) return [];
  const params = new URLSearchParams({ establishment_id: id });
  if (q?.trim()) params.append("q", q.trim());
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
  const { data } = await apiClient.get("/company_dispatch/status");
  return data;
};

// --------------------------------------
// Helpers communs (mapping & payload)
// --------------------------------------
const normalizeMode = (m) => {
  const s = String(m ?? "auto").trim().toLowerCase();
  if (s === "heuristic") return "heuristic_only";
  if (s === "solver") return "solver_only";
  if (["auto", "heuristic_only", "solver_only"].includes(s)) return s;
  return "auto";
};

const toRunPayload = ({
  forDate,
  regularFirst = true,
  allowEmergency,
  runAsync = false,
  mode = "auto",
  overrides,
} = {}) => {
  const payload = {
    for_date: forDate,
    regular_first: !!regularFirst,
    run_async: !!runAsync,
    mode: normalizeMode(mode),
    overrides: overrides || {},
  };
  // Tri-state: si null → rester null, si défini → bool, si undefined → omettre
  if (allowEmergency === null) {
    payload.allow_emergency = null;
  } else if (typeof allowEmergency !== "undefined") {
    payload.allow_emergency = !!allowEmergency;
  }
  return payload;
};

 /**
 * Aperçu (compat existant)
 * GET /company_dispatch/preview?for_date=YYYY-MM-DD&regular_first=true&allow_emergency=true|false
 * NB: si allowEmergency est omis (undefined), on laisse le backend hériter des settings.
  */
export const previewDispatch = async ({ forDate, regularFirst = true, allowEmergency } = {}) => {
  if (!forDate) throw new Error("forDate (YYYY-MM-DD) requis");
  const params = new URLSearchParams({
    for_date: forDate,
    regular_first: String(!!regularFirst),
  });
  if (typeof allowEmergency !== "undefined") {
    params.append("allow_emergency", String(!!allowEmergency));
  }
  const { data } = await apiClient.get(`/company_dispatch/preview?${params.toString()}`);
  return data;
};

/**
 * Déclenche un run async (compat existant)
 * POST /company_dispatch/trigger
 */
export const triggerDispatch = async ({ forDate, regularFirst = true, allowEmergency = true, overrides } = {}) => {
  if (!forDate) throw new Error("forDate (YYYY-MM-DD) requis");
  const payload = toRunPayload({
    forDate,
    regularFirst,
    allowEmergency,
    runAsync: false,    // trigger est toujours async côté backend
    overrides,
  });
  const { data } = await apiClient.post("/company_dispatch/trigger", { ...payload, run_async: true });
  return data; // { status: "queued", job_id }
};

/**
 * Run (async par défaut) — compat existant /company_dispatch/run
 */
export const runDispatchNow = async ({
  forDate,
  regularFirst = true,
  allowEmergency,
  runAsync = false,
  mode = "auto",
  overrides,
} = {}) => {
  if (!forDate) throw new Error("forDate (YYYY-MM-DD) requis");
  const payload = toRunPayload({
    forDate,
    regularFirst,
    allowEmergency,
    runAsync,
    mode,
    overrides,
  });
  try {
    const { data } = await apiClient.post("/company_dispatch/trigger", { ...payload, run_async: true });
    return {
      ...data,
      dispatch_run_id: data.dispatch_run_id || data.meta?.dispatch_run_id || null,
    };
  } catch (e) {
    // Si c'est une erreur de validation (ex: mismatch de clés) → fallback vers trigger (202 queued)
    const status = e?.response?.status;
    if (status === 400 || status === 422) {
      console.error("RUN 400/422 body:", e?.response?.data || e);
      const { data } = await apiClient.post("/company_dispatch/trigger", payload);
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
  mode = "auto",     // auto par défaut
  runAsync = true,   // Changed default to true for reliability
  overrides,
} = {}) => {
  if (!forDate) throw new Error("forDate (YYYY-MM-DD) requis");

  console.log(`runDispatchForDay called with: forDate=${forDate}, regularFirst=${regularFirst}, allowEmergency=${allowEmergency}, mode=${mode}, runAsync=${runAsync}`);

  const payload = toRunPayload({
    forDate,
    regularFirst,
    allowEmergency,
    runAsync, 
    mode,
    overrides,
  });
  
  try {
    // --- bloc principal : /run ---
    console.log("Sending dispatch request with payload:", payload);

    // 1) on tente /run (200 si sync, 202 si async)
    const { data } = await apiClient.post("/company_dispatch/run", payload);

    console.log("Dispatch response:", data);

    return {
      ...data,
      status: data.status || (runAsync ? "queued" : "completed"),
      dispatch_run_id: data.dispatch_run_id || data.meta?.dispatch_run_id || null,
    };

  } catch (e) {
    // --- fallback : /trigger ---
    console.error("Dispatch request failed:", e);
    console.error("Error details:", e?.response?.data);
    console.log("Falling back to /trigger endpoint");

    try {
      const { data } = await apiClient.post("/company_dispatch/trigger", {
        ...payload,
        run_async: true, // force async pour cohérence
      });
      console.log("Trigger fallback response:", data);
      return {
        ...data,
        status: data.status || "queued",
        dispatch_run_id: data.dispatch_run_id || data.meta?.dispatch_run_id || null,
      };
    } catch (triggerError) {
      console.error("Trigger fallback also failed:", triggerError);
      throw triggerError; // on remonte l'erreur
    }
  }

};

// --- utilitaires date robustes ---
const pickBestDateField = (r) =>
  r?.scheduled_time ??
  r?.pickup_time ??
  r?.date_time ??
  r?.datetime ??
  null;

// Retourne YYYY-MM-DD sans dépendre du parsing Date (fiable même si pas de 'T' / pas de 'Z')
const toYMD = (raw) => {
  if (!raw) return null;
  if (typeof raw === "string") {
    const s = raw.trim();
    // si ça commence par YYYY-MM-DD, on prend les 10 premiers chars
    const m = s.match(/^(\d{4}-\d{2}-\d{2})/);
    if (m) return m[1];
  }
  // fallback: on tente un Date()
  try {
    const d = new Date(raw);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  } catch {
    return null;
  }
};

export const fetchAssignedReservations = async (forDate) => {
  console.log(`Fetching assigned reservations for date: ${forDate}`);
  
  try {
    const [reservationsRes, assignmentsRes] = await Promise.all([
      apiClient.get("/companies/me/reservations", {
        params: forDate ? { date: forDate, flat: true } : { flat: true },
      }),
      apiClient.get("/company_dispatch/assignments", {
        params: forDate ? { date: forDate } : undefined,
      }),
    ]);

    // Normalise la charge utile en tableau
    const payload = reservationsRes.data;
    const reservations = Array.isArray(payload)
      ? payload
      : (Array.isArray(payload?.reservations) ? payload.reservations : []);
    console.log(`Received ${reservations.length} reservations and ${assignmentsRes.data?.length || 0} assignments`);

    // /company_dispatch/assignments doit rester un tableau
    const assignments = Array.isArray(assignmentsRes.data) ? assignmentsRes.data : [];

    console.log(
      `Received ${reservations.length} reservations and ${assignments.length} assignments`
    );

    const byBookingId = new Map(assignments.map((a) => [a.booking_id, a]));
    console.log(`Created map with ${byBookingId.size} assignments by booking ID`);

    // Jour cible : YYYY-MM-DD (local)
    const targetDay = forDate || (() => {
      const d = new Date();
      const pad = (n) => String(n).padStart(2, "0");
      return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
    })();

    console.log(`Filtering bookings for target day: ${targetDay}`);

    // 🔧 filtre plus tolérant : accepte scheduled_time, pickup_time, date_time, datetime
    const bookingsOfDay = reservations.filter((r) => {
      const rawWhen = pickBestDateField(r);
      const ymd = toYMD(rawWhen);
      return !targetDay || (ymd === targetDay);
    });

    console.log(`Found ${bookingsOfDay.length} bookings for the target day`);

    // Construction des lignes
    const rows = bookingsOfDay.map((b) => {
      const a = byBookingId.get(b.id) || null;

      const clientName =
        b.customer_name ||
        b.client?.full_name ||
        b.client_name ||
        b.client?.name ||
        "Non spécifié";

      // ✅ Source temporelle unifiée
      const when = pickBestDateField(b);
      const scheduled_time = when;
      const dropoff_time = b.dropoff_time || b.drop_time || null;

      // Create a synthetic assignment if booking has driver_id but no assignment
      const syntheticAssignment = !a && b.driver_id ? {
        id: null, // Synthetic assignment has no ID
        booking_id: b.id,
        driver_id: b.driver_id,
        status: "assigned", // Default status
        estimated_pickup_arrival: null,
        estimated_dropoff_arrival: null,
        is_synthetic: true, // Flag to identify synthetic assignments
      } : null;


      return {
        id: b.id,
        customer_name: clientName,
        client: b.client || { full_name: clientName },
        scheduled_time,
        pickup_time: scheduled_time, // compat
        dropoff_time,
        pickup_location: b.pickup_location || b.pickup_address || b.origin || "",
        dropoff_location: b.dropoff_location || b.dropoff_address || b.destination || "",
        is_return: !!b.is_return,
        status: b.status || "scheduled",
        driver_username: b.driver_username || b.driver?.username,
        driver: b.driver || null,
        // accepte ancienne/ nouvelle forme (eta_* vs estimated_*)
        assignment: a ? {
          id: a.id,
          booking_id: a.booking_id,
          driver_id: a.driver_id,
          status: a.status,
          estimated_pickup_arrival: a.estimated_pickup_arrival || a.eta_pickup_at || null,
          estimated_dropoff_arrival: a.estimated_dropoff_arrival || a.eta_dropoff_at || null,
        } : syntheticAssignment,
      };
    });

    console.log(`Returning ${rows.length} formatted rows for dispatch table`);
    return rows;
  } catch (error) {
    console.error("Error fetching assigned reservations:", error);
    throw error;
  }
};


/**
 * ⏱️ Retards courants (monté sous /company_dispatch/delays)
 */
export const fetchDispatchDelays = async (forDate) => {
  const { data } = await apiClient.get("/company_dispatch/delays", {
    params: forDate ? { date: forDate } : undefined,
  });
  return Array.isArray(data) ? data : [];
};

/**
 * 🗂️ Historique des runs & détail d’un run
 */
export const fetchDispatchRuns = async ({ limit = 50, offset = 0 } = {}) => {
  const { data } = await apiClient.get("/company_dispatch/runs", {
    params: { limit, offset },
  });
  return Array.isArray(data) ? data : [];
};

export const fetchDispatchRunById = async (runId) => {
  if (!runId) throw new Error("runId requis");
  const { data } = await apiClient.get(`/company_dispatch/runs/${encodeURIComponent(runId)}`);
  return data;
};

/**
 * ✏️ MAJ d’une assignation (driver/status) & réassignation
 */
export const patchAssignment = async (assignmentId, payload = {}) => {
  if (!assignmentId) throw new Error("assignmentId requis");
  const { data } = await apiClient.patch(`/company_dispatch/assignments/${encodeURIComponent(assignmentId)}`, payload);
  return data;
};

export const reassignAssignment = async (assignmentId, newDriverId) => {
  if (!assignmentId) throw new Error("assignmentId requis");
  if (!newDriverId) throw new Error("newDriverId requis");
  const { data } = await apiClient.post(
    `/company_dispatch/assignments/${encodeURIComponent(assignmentId)}/reassign`,
    { new_driver_id: Number(newDriverId) }
  );
  return data;
};

/* ------------------------------- CRÉATION MANUELLE ---------------------------- */

export const createManualBooking = async (bookingData) => {
  try {
    const { data } = await apiClient.post("/companies/me/reservations/manual", bookingData);
    return data; // { message, reservation, return_booking? }
  } catch (error) {
    throw error.response?.data || error.message;
  }
};