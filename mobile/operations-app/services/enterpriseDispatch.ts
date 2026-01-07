import { enterpriseApi } from "./enterpriseAuth";
import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { ENTERPRISE_TOKEN_KEY, ENTERPRISE_SESSION_KEY } from "./enterpriseAuth";
import {
  DispatchRunResponse,
  DispatchStatus,
  PaginatedRides,
  RideDetail,
  AssignRequestPayload,
  AssignResponsePayload,
  ModeResponse,
  IncidentPayload,
  DispatchSettings,
  DispatchSettingsUpdate,
  ScheduleRidePayload,
  MarkUrgentPayload,
  DispatchMessage,
  RideEditPayload,
  RideCreatePayload,
  AddressSuggestion,
  ClientOption,
} from "@/types/enterpriseDispatch";

export const getDispatchStatus = async (date?: string): Promise<DispatchStatus> => {
  const params: { date?: string } = {};
  if (date) {
    params.date = date;
  }
  const response = await enterpriseApi.get<DispatchStatus>(
    "/dispatch/v1/status",
    { params }
  );
  return response.data;
};

interface ListRidesParams {
  date: string;
  status?: "assigned" | "unassigned" | "urgent" | "cancelled";
  query?: string;
  page?: number;
  page_size?: number;
}

export const getDispatchRides = async (
  params: ListRidesParams
): Promise<PaginatedRides> => {
  const response = await enterpriseApi.get<PaginatedRides>(
    "/dispatch/v1/rides",
    {
      params: {
        date: params.date,
        status: params.status,
        q: params.query,
        page: params.page ?? 1,
        page_size: params.page_size ?? 20,
      },
    }
  );
  return response.data;
};

export const getDispatchRideDetails = async (
  rideId: string
): Promise<RideDetail> => {
  const response = await enterpriseApi.get<RideDetail>(
    `/dispatch/v1/rides/${rideId}`
  );
  return response.data;
};

export const assignRide = async (
  rideId: string,
  payload: AssignRequestPayload
): Promise<AssignResponsePayload> => {
  const response = await enterpriseApi.post<AssignResponsePayload>(
    `/dispatch/v1/rides/${rideId}/assign`,
    payload
  );
  return response.data;
};

export const reassignRide = async (
  rideId: string,
  payload: AssignRequestPayload
): Promise<AssignResponsePayload> => {
  const response = await enterpriseApi.post<AssignResponsePayload>(
    `/dispatch/v1/rides/${rideId}/reassign`,
    payload
  );
  return response.data;
};

export const cancelRide = async (
  rideId: string,
  reason_code: string,
  note?: string
) => {
  await enterpriseApi.post(`/dispatch/v1/rides/${rideId}/cancel`, {
    reason_code,
    note,
  });
};

export const switchDispatchMode = async (
  target_mode: "manual" | "semi_auto" | "fully_auto",
  reason?: string
): Promise<ModeResponse> => {
  const response = await enterpriseApi.put("/dispatch/v1/mode", {
    dispatch_mode: target_mode,
    reason,
  });

  const payload = response.data as {
    dispatch_mode?: "manual" | "semi_auto" | "fully_auto";
    previous_mode?: "manual" | "semi_auto" | "fully_auto";
    message?: string;
  };

  const nowIso = new Date().toISOString();

  return {
    mode_before: payload.previous_mode ?? target_mode,
    mode_after: payload.dispatch_mode ?? target_mode,
    effective_at: nowIso,
    requires_approval: false,
    audit_event_id: payload.message ?? "",
  };
};

export const getDispatchModes = async () => {
  const response = await enterpriseApi.get("/dispatch/v1/mode");
  return response.data;
};

export const createIncident = async (payload: IncidentPayload) => {
  const response = await enterpriseApi.post("/dispatch/v1/incidents", payload);
  return response.data;
};

export const scheduleRide = async (
  rideId: string,
  payload: ScheduleRidePayload
) => {
  await enterpriseApi.post(`/dispatch/v1/rides/${rideId}/schedule`, payload);
};

export const markRideUrgent = async (
  rideId: string,
  payload: MarkUrgentPayload
) => {
  await enterpriseApi.post(`/dispatch/v1/rides/${rideId}/urgent`, payload);
};

export const runDispatch = async (
  forDate?: string
): Promise<DispatchRunResponse> => {
  const response = await enterpriseApi.post<DispatchRunResponse>(
    "/dispatch/v1/run",
    {
      date: forDate,
    }
  );
  return response.data;
};

export const runOptimizer = async (forDate?: string) => {
  await enterpriseApi.post("/dispatch/v1/optimizer/run", { date: forDate });
};

export const resetAssignments = async (date?: string) => {
  await enterpriseApi.post("/dispatch/v1/reset", { date });
};

export const getDispatchSettings = async (): Promise<DispatchSettings> => {
  const response = await enterpriseApi.get<DispatchSettings>(
    "/dispatch/v1/settings"
  );
  return response.data;
};

export interface DriverAccountInfo {
  has_driver_account: boolean;
  driver_id?: number;
  driver_type?: "REGULAR" | "EMERGENCY";
  is_active?: boolean;
  is_available?: boolean;
}

export const getMyDriverAccount = async (): Promise<DriverAccountInfo> => {
  console.log("[getMyDriverAccount] Appel de l'endpoint /auth/me/driver-account");
  try {
    const response = await enterpriseApi.get<DriverAccountInfo>(
      "/auth/me/driver-account"
    );
    console.log("[getMyDriverAccount] Réponse reçue:", response.data);
    return response.data;
  } catch (error: any) {
    console.error("[getMyDriverAccount] Erreur:", {
      message: error?.message,
      status: error?.response?.status,
      data: error?.response?.data,
      url: error?.config?.url,
    });
    throw error;
  }
};

export interface SwitchToDriverResponse {
  token: string;
  refresh_token: string;
  user: {
    public_id: string;
    email: string;
    first_name?: string;
    last_name?: string;
  };
  driver: {
    id: number;
    driver_type: "REGULAR" | "EMERGENCY";
  };
}

export const switchToDriverToken = async (): Promise<SwitchToDriverResponse> => {
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'enterpriseDispatch.ts:switchToDriverToken',message:'switchToDriverToken entry',data:{url:'/auth/me/switch-to-driver'},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'I'})}).catch(()=>{});
  // #endregion
  
  const response = await enterpriseApi.post<SwitchToDriverResponse>(
    "/auth/me/switch-to-driver"
  );
  
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'enterpriseDispatch.ts:switchToDriverToken',message:'switchToDriverToken success',data:{hasToken:!!response.data.token,hasRefreshToken:!!response.data.refresh_token,status:response.status},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'I'})}).catch(()=>{});
  // #endregion
  
  return response.data;
};

export const updateDispatchSettings = async (
  payload: DispatchSettingsUpdate
) => {
  const response = await enterpriseApi.put("/dispatch/v1/settings", payload);
  return response.data;
};

type ChatMessagesResponse = {
  messages: DispatchMessage[];
  count: number;
};

const normalizeDispatchMessage = (message: any): DispatchMessage => {
  const createdAt =
    message?.created_at ?? message?.timestamp ?? new Date().toISOString();

  return {
    id: message?.id,
    sender_id: message?.sender_id !== undefined ? message.sender_id : null,
    sender_role: message?.sender_role ?? undefined,
    sender_name: message?.sender_name ?? null,
    content: message?.content ?? "",
    created_at: createdAt,
  };
};

export const getDispatchMessages = async (params?: {
  before?: string;
  limit?: number;
}): Promise<DispatchMessage[]> => {
  const response = await enterpriseApi.get<ChatMessagesResponse>(
    "/dispatch/v1/chat/messages",
    {
      params: {
        before: params?.before,
        limit: params?.limit,
      },
    }
  );
  const rawMessages = response.data?.messages ?? [];
  return rawMessages.map(normalizeDispatchMessage);
};

/**
 * ✅ 3.4.2: Récupère les données du dashboard temps réel dispatch
 * @param date - Date au format YYYY-MM-DD (optionnel, défaut: aujourd'hui)
 */
export interface RealtimeDashboardData {
  date: string;
  timestamp: string;
  quality_metrics: {
    quality_score: number;
    assignment_rate: number;
    on_time_rate: number;
    pooling_rate: number;
    fairness: number;
    avg_delay: number;
  };
  current_delays: Array<{
    assignment_id: number;
    booking_id: number;
    driver_id: number;
    delay_minutes: number;
    status: "late" | "early";
    // ✅ P1-4 Phase 3.3: Utiliser client_name au lieu de customer_name
    client_name: string;
    scheduled_time: string | null;
  }>;
  opportunities: Array<{
    assignment_id: number;
    booking_id: number;
    driver_id: number;
    current_delay_minutes: number;
    severity: "critical" | "high" | "medium" | "low";
    suggestions: Array<{
      action: string;
      priority: string;
      message: string;
      estimated_gain_minutes?: number;
    }>;
    detected_at: string;
    auto_applicable: boolean;
  }>;
  driver_load: Array<{
    driver_id: number;
    name: string;
    bookings_count: number;
    is_emergency: boolean;
  }>;
  stats: {
    total_bookings: number;
    delayed_bookings: number;
    early_bookings: number;
    on_time_bookings: number;
    critical_opportunities: number;
    drivers_active: number;
  };
}

export const fetchRealtimeDashboard = async (
  date?: string
): Promise<RealtimeDashboardData> => {
  const params: { date?: string } = {};
  if (date) {
    params.date = date;
  }
  const response = await enterpriseApi.get<RealtimeDashboardData>(
    "/dispatch/v1/dashboard/realtime",
    { params }
  );
  return response.data;
};

export const sendDispatchMessage = async (content: string) => {
  const response = await enterpriseApi.post<DispatchMessage>(
    "/dispatch/v1/chat/messages",
    { content }
  );
  return normalizeDispatchMessage(response.data);
};

// ✅ Endpoints pour l'édition et la création de courses
export const updateRide = async (
  rideId: string,
  payload: RideEditPayload
): Promise<RideDetail> => {
  const response = await enterpriseApi.put<RideDetail>(
    `/dispatch/v1/rides/${rideId}`,
    payload
  );
  return response.data;
};

export const createRide = async (
  payload: RideCreatePayload
): Promise<RideDetail> => {
  console.log("[createRide] Envoi payload:", JSON.stringify(payload, null, 2));
  try {
    const response = await enterpriseApi.post<{ summary: RideDetail; return_summary?: RideDetail }>(
      "/dispatch/v1/rides",
      payload
    );
    console.log("[createRide] Réponse complète:", response.data);
    // Le backend retourne {summary: RideDetail, return_summary?: RideDetail}, on extrait summary
    const rideDetail = response.data.summary || response.data;
    console.log("[createRide] RideDetail extrait:", rideDetail);
    if (response.data.return_summary) {
      console.log("[createRide] Course retour créée:", response.data.return_summary);
    }
    return rideDetail as RideDetail;
  } catch (error: any) {
    console.error("[createRide] Erreur:", error);
    console.error("[createRide] Erreur response:", error?.response?.data);
    console.error("[createRide] Erreur status:", error?.response?.status);
    throw error;
  }
};

export const searchAddresses = async (
  query: string
): Promise<AddressSuggestion[]> => {
  const response = await enterpriseApi.get<AddressSuggestion[]>(
    "/dispatch/v1/addresses/search",
    {
      params: { q: query },
    }
  );
  return response.data;
};

export const searchClients = async (
  query: string
): Promise<ClientOption[]> => {
  const response = await enterpriseApi.get<ClientOption[]>(
    "/dispatch/v1/clients/search",
    {
      params: { q: query },
    }
  );
  return response.data;
};

export interface AvailableDriver {
  driver_id: string;
  driver_name: string;
  is_emergency: boolean;
  driver_type: string;
}

export const getAvailableDrivers = async (): Promise<AvailableDriver[]> => {
  const response = await enterpriseApi.get<{ drivers: AvailableDriver[] }>(
    "/dispatch/v1/drivers/available"
  );
  return response.data.drivers || [];
};

/**
 * Applique une opportunité d'optimisation en réassignant un chauffeur
 * @param opportunity - L'opportunité à appliquer depuis le dashboard temps réel
 * @param newDriverId - ID du nouveau chauffeur (optionnel, extrait de la suggestion si non fourni)
 */
export const applyOpportunity = async (
  opportunity: {
    assignment_id: number;
    booking_id: number;
    driver_id: number;
    suggestions?: Array<{
      action: string;
      priority: string;
      message: string;
      estimated_gain_minutes?: number;
    }>;
  },
  newDriverId?: string
) => {
  const suggestion = opportunity.suggestions?.[0];
  const targetDriverId =
    newDriverId || String(opportunity.driver_id);

  await reassignRide(String(opportunity.booking_id), {
    driver_id: targetDriverId,
    reason: suggestion?.message || "Application d'opportunité d'optimisation",
    allow_emergency: false,
    respect_preferences: true,
    idempotency_key: `${Date.now()}-${opportunity.assignment_id}`,
  });
};

export interface CreateClientPayload {
  client_type?: "PRIVATE";
  email?: string; // Généré automatiquement si non fourni
  first_name: string;
  last_name: string;
  phone?: string;
  address?: string; // Adresse complète formatée
  birth_date?: string;
  is_institution?: boolean;
  institution_name?: string;
  residence_facility?: string; // Établissement de résidence (EMS, clinique, etc.)
  domicile_address?: string; // Rue seule (sans code postal/ville)
  domicile_zip?: string;
  domicile_city?: string;
  domicile_lat?: number | null;
  domicile_lon?: number | null;
  billing_address?: string; // Adresse complète de facturation
  billing_lat?: number | null;
  billing_lon?: number | null;
  contact_email?: string;
  contact_phone?: string;
  preferential_rate?: number;
}

export const createClient = async (
  payload: CreateClientPayload
): Promise<ClientOption> => {
  // Générer un email interne unique pour le User
  const randomId = Math.random().toString(36).substring(2, 10);
  const timestamp = Date.now().toString(36);
  const email = payload.is_institution
    ? `institution-${randomId}-${timestamp}@internal.atmr.local`
    : `client-${randomId}-${timestamp}@internal.atmr.local`;

  // Construire l'adresse complète comme dans le frontend
  // Format: "Rue, Code postal, Ville"
  let address = payload.address;
  if (!address && payload.domicile_address) {
    const parts = [payload.domicile_address];
    if (payload.domicile_zip) parts.push(payload.domicile_zip);
    if (payload.domicile_city) parts.push(payload.domicile_city);
    address = parts.join(", ");
  }

  // Construire l'adresse de facturation si non fournie
  let billingAddress = payload.billing_address;
  if (!billingAddress) {
    // Utiliser la même adresse que le domicile
    billingAddress = address;
  }

  const fullPayload: any = {
    client_type: payload.client_type || "PRIVATE",
    email,
    first_name: payload.first_name,
    last_name: payload.last_name,
    phone: payload.phone || undefined,
    birth_date: payload.birth_date || undefined,
    is_institution: payload.is_institution || false,
    institution_name: payload.institution_name || undefined,
    // Adresse complète (comme dans le frontend)
    address: address || undefined,
    // Adresse de domicile structurée
    domicile_address: payload.domicile_address || undefined,
    domicile_zip: payload.domicile_zip || undefined,
    domicile_city: payload.domicile_city || undefined,
    // Adresse de facturation
    billing_address: billingAddress || undefined,
    contact_email: payload.contact_email || undefined,
    contact_phone: payload.contact_phone || undefined,
    preferential_rate: payload.preferential_rate || undefined,
  };

  // Ajouter les coordonnées GPS seulement si elles sont définies (pas null)
  if (payload.domicile_lat !== null && payload.domicile_lat !== undefined) {
    fullPayload.domicile_lat = payload.domicile_lat;
  }
  if (payload.domicile_lon !== null && payload.domicile_lon !== undefined) {
    fullPayload.domicile_lon = payload.domicile_lon;
  }
  if (payload.billing_lat !== null && payload.billing_lat !== undefined) {
    fullPayload.billing_lat = payload.billing_lat;
  }
  if (payload.billing_lon !== null && payload.billing_lon !== undefined) {
    fullPayload.billing_lon = payload.billing_lon;
  }

  // Nettoyer le payload : supprimer les valeurs null/undefined/vides
  Object.keys(fullPayload).forEach((key) => {
    if (
      fullPayload[key] === null ||
      fullPayload[key] === undefined ||
      fullPayload[key] === ""
    ) {
      delete fullPayload[key];
    }
  });

  // L'endpoint /companies/me/clients est dans le namespace companies, pas company_mobile
  // Il faut utiliser l'URL complète de l'API standard
  const baseURL = enterpriseApi.defaults.baseURL || "";
  // Remplacer /api/v1/company_mobile par /api/v1 pour accéder à l'API standard
  const standardApiURL = baseURL.replace("/api/v1/company_mobile", "/api/v1");
  
  // Récupérer le token et les headers depuis AsyncStorage
  const token = await AsyncStorage.getItem(ENTERPRISE_TOKEN_KEY);
  const sessionRaw = await AsyncStorage.getItem(ENTERPRISE_SESSION_KEY);
  
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  
  if (sessionRaw) {
    try {
      const session = JSON.parse(sessionRaw);
      if (session?.company?.id) {
        headers["X-Company-ID"] = String(session.company.id);
      }
      if (session?.sessionId) {
        headers["X-Session-ID"] = session.sessionId;
      }
    } catch (e) {
      // Ignore parsing errors
    }
  }

  console.log("[createClient] Payload envoyé:", JSON.stringify(fullPayload, null, 2));
  console.log("[createClient] URL:", `${standardApiURL}/companies/me/clients`);

  const response = await axios.post<{ data: ClientOption }>(
    `${standardApiURL}/companies/me/clients`,
    fullPayload,
    { headers }
  );
  return response.data.data || response.data;
};
