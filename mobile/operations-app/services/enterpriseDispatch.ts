import { getLogger } from "@/utils/logger";
import { enterpriseApi, hasValidToken, retryWithBackoff } from "./enterpriseAuth";
import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";

const log = getLogger("Dispatch");
import { ENTERPRISE_SESSION_KEY } from "./enterpriseAuth";
import * as Sentry from "@sentry/react-native";
import { secureStorage } from "@/services/storage";
import { enterpriseStandardApi } from "@/services/enterpriseStandardApi";
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
  MarkUrgentResponse,
  DispatchMessage,
  RideEditPayload,
  RideCreatePayload,
  AddressSuggestion,
  ClientOption,
} from "@/types/enterpriseDispatch";
import { sendIngestEvent } from "@/src/config/telemetry";

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

const URGENT_DEBUG = false;

export const markRideUrgent = async (
  rideId: string,
  payload: MarkUrgentPayload
): Promise<MarkUrgentResponse> => {
  if (__DEV__ && URGENT_DEBUG) {
    log.debug("mark ride urgent payload", { rideId, payload });
  }
  const { data } = await enterpriseApi.post<MarkUrgentResponse>(
    `/dispatch/v1/rides/${rideId}/urgent`,
    payload
  );
  if (__DEV__ && URGENT_DEBUG) {
    log.debug("mark ride urgent response", data);
  }
  return data;
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
  log.info("calling driver-account endpoint", {});
  try {
    const response = await enterpriseApi.get<DriverAccountInfo>(
      "/auth/me/driver-account"
    );
    log.info("driver-account response received", { data: response.data });
    return response.data;
  } catch (error: any) {
    log.error("get my driver account failed", {
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
  sendIngestEvent({location:'enterpriseDispatch.ts:switchToDriverToken',message:'switchToDriverToken entry',data:{url:'/auth/me/switch-to-driver'},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'I'});
  
  const response = await enterpriseApi.post<SwitchToDriverResponse>(
    "/auth/me/switch-to-driver"
  );
  
  sendIngestEvent({location:'enterpriseDispatch.ts:switchToDriverToken',message:'switchToDriverToken success',data:{hasToken:!!response.data.token,hasRefreshToken:!!response.data.refresh_token,status:response.status},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'I'});
  
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
    image: message?.image ?? null,
    image_url: message?.image_url ?? null,
    pdf: message?.pdf ?? null,
    pdf_url: message?.pdf_url ?? null,
    pdf_filename: message?.pdf_filename ?? null,
    pdf_size: message?.pdf_size ?? null,
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
  // 🔄 Retry avec backoff pour résilience réseau
  return retryWithBackoff(
    async () => {
      // ✅ Guard Pattern : Vérifier qu'on a un token valide avant d'envoyer
      if (!(await hasValidToken())) {
        // ✅ Capturer tentative sans token pour monitoring
        if (!__DEV__) {
          Sentry.captureMessage("updateRide: Token invalide détecté par guard", {
            level: "warning",
            tags: {
              type: "guard_pattern",
              function: "updateRide",
            },
          });
        }
        throw new Error("Aucun token valide. Veuillez vous reconnecter.");
      }

      try {
        const response = await enterpriseApi.put<RideDetail>(
          `/dispatch/v1/rides/${rideId}`,
          payload
        );
        return response.data;
      } catch (error: any) {
        // ✅ Capturer erreurs critiques (401 inattendu après guard)
        if (error?.response?.status === 401 && !__DEV__) {
          Sentry.captureMessage("updateRide: 401 après guard pattern", {
            level: "error",
            tags: {
              type: "unexpected_401",
              function: "updateRide",
            },
            contexts: {
              requestInfo: {
                endpoint: `/dispatch/v1/rides/${rideId}`,
                rideId,
                timestamp: new Date().toISOString(),
              },
            },
          });
        }
        throw error;
      }
    },
    {
      maxRetries: 2,
      baseDelay: 500,
      maxDelay: 5000,
      shouldRetry: (error) => {
        const status = error?.response?.status;
        return !status || status >= 500;
      },
    }
  );
};

export const createRide = async (
  payload: RideCreatePayload
): Promise<RideDetail> => {
  // 🔄 Retry avec backoff pour résilience réseau
  return retryWithBackoff(
    async () => {
      // ✅ Guard Pattern : Vérifier qu'on a un token valide avant d'envoyer
      if (!(await hasValidToken())) {
        // ✅ Capturer tentative sans token pour monitoring
        if (!__DEV__) {
          Sentry.captureMessage("createRide: Token invalide détecté par guard", {
            level: "warning",
            tags: {
              type: "guard_pattern",
              function: "createRide",
            },
          });
        }
        throw new Error("Aucun token valide. Veuillez vous reconnecter.");
      }

      log.info("create ride sending payload", { payload: JSON.stringify(payload, null, 2) });
      try {
        const response = await enterpriseApi.post<{ summary: RideDetail; return_summary?: RideDetail }>(
          "/dispatch/v1/rides",
          payload
        );
        log.info("create ride full response", { data: response.data });
        const rideDetail = response.data.summary || response.data;
        log.info("create ride detail extracted", { rideDetail });
        if (response.data.return_summary) {
          log.info("return ride created", { returnSummary: response.data.return_summary });
        }
        return rideDetail as RideDetail;
      } catch (error: any) {
        log.error("create ride failed", {
          error,
          responseData: error?.response?.data,
          status: error?.response?.status,
        });
        
        // ✅ Capturer erreurs critiques (401 inattendu après guard)
        if (error?.response?.status === 401 && !__DEV__) {
          Sentry.captureMessage("createRide: 401 après guard pattern", {
            level: "error",
            tags: {
              type: "unexpected_401",
              function: "createRide",
            },
            contexts: {
              requestInfo: {
                endpoint: "/dispatch/v1/rides",
                hasPayload: !!payload,
                timestamp: new Date().toISOString(),
              },
            },
          });
        }
        
        throw error;
      }
    },
    {
      maxRetries: 2, // 3 essais au total
      baseDelay: 500, // 500ms de base
      maxDelay: 5000, // Max 5 secondes
      shouldRetry: (error) => {
        const status = error?.response?.status;
        // Retry sur erreurs réseau ou 5xx
        // PAS sur 401 (géré par intercepteur) ou 4xx (erreurs client)
        return (
          !status || // Erreur réseau (pas de status)
          status >= 500 // Erreur serveur
        );
      },
    }
  );
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
  gender: 'male' | 'female'; // Civilité obligatoire
  avs_number?: string;
  phone?: string;
  address?: string;
  birth_date?: string;
  is_institution?: boolean;
  institution_name?: string;
  linked_institution_id?: number | null;
  residence_facility?: string;
  domicile_address?: string;
  domicile_zip?: string;
  domicile_city?: string;
  domicile_lat?: number | null;
  domicile_lon?: number | null;
  billing_address?: string;
  billing_lat?: number | null;
  billing_lon?: number | null;
  contact_email?: string;
  contact_phone?: string;
  preferential_rate?: number;
  door_code?: string;
  floor?: string;
  access_notes?: string;
  gp_name?: string;
  gp_phone?: string;
  default_billed_to_type?: string;
  default_billed_to_contact?: string;
}

export const createClient = async (
  payload: CreateClientPayload
): Promise<ClientOption> => {
  // ✅ Guard Pattern : Vérifier qu'on a un token valide avant d'envoyer
  if (!(await hasValidToken())) {
    // ✅ Capturer tentative sans token pour monitoring
    if (!__DEV__) {
      Sentry.captureMessage("createClient: Token invalide détecté par guard", {
        level: "warning",
        tags: {
          type: "guard_pattern",
          function: "createClient",
        },
      });
    }
    throw new Error("Aucun token valide. Veuillez vous reconnecter.");
  }

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
    // ✅ Gender obligatoire - toujours inclus même si validation frontend a échoué
    gender: payload.gender,
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
    // ✅ Numéro AVS optionnel
    avs_number: payload.avs_number || undefined,
    residence_facility: payload.residence_facility || undefined,
    door_code: payload.door_code || undefined,
    floor: payload.floor || undefined,
    access_notes: payload.access_notes || undefined,
    gp_name: payload.gp_name || undefined,
    gp_phone: payload.gp_phone || undefined,
    default_billed_to_type: payload.default_billed_to_type || undefined,
    default_billed_to_contact: payload.default_billed_to_contact || undefined,
    linked_institution_id: payload.linked_institution_id ?? undefined,
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
  // ✅ EXCEPTION: Ne jamais supprimer gender (obligatoire)
  Object.keys(fullPayload).forEach((key) => {
    if (key === "gender") {
      // ✅ Gender obligatoire - ne jamais supprimer
      return;
    }
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
  
  // Récupérer le token depuis SecureStore (source of truth)
  const token = await secureStorage.getEnterpriseToken();
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

  // ✅ Log détaillé pour diagnostic (toujours activé)
  log.info("create client payload", {
    payload: JSON.stringify(fullPayload, null, 2),
    gender: fullPayload.gender,
    genderType: typeof fullPayload.gender,
    url: `${standardApiURL}/companies/me/clients`,
  });

  if (!fullPayload.gender || (fullPayload.gender !== 'male' && fullPayload.gender !== 'female')) {
    const errorMsg = `Gender invalide: ${fullPayload.gender}. Attendu: 'male' ou 'female'`;
    log.error("create client invalid gender", { gender: fullPayload.gender });
    throw new Error(errorMsg);
  }

  // 🔄 Retry avec backoff pour résilience réseau (prudent : maxRetries=1 pour éviter doublons)
  return retryWithBackoff(
    async () => {
      try {
        const response = await enterpriseStandardApi.post<{ data: ClientOption }>(
          "/companies/me/clients",
          fullPayload,
          { headers }
        );
        return response.data.data || response.data;
      } catch (error: any) {
        // ✅ Capturer erreurs critiques (401 inattendu après guard)
        if (error?.response?.status === 401 && !__DEV__) {
          Sentry.captureMessage("createClient: 401 après guard pattern", {
            level: "error",
            tags: {
              type: "unexpected_401",
              function: "createClient",
            },
            contexts: {
              requestInfo: {
                endpoint: "/companies/me/clients",
                hasPayload: !!fullPayload,
                timestamp: new Date().toISOString(),
              },
            },
          });
        }
        throw error;
      }
    },
    {
      maxRetries: 1, // 2 essais seulement (création unique, éviter doublons)
      baseDelay: 1000, // 1s (plus long pour éviter doublons)
      maxDelay: 3000, // Max 3s
      shouldRetry: (error) => {
        const status = error?.response?.status;
        // Retry UNIQUEMENT sur erreurs réseau (pas de status)
        // PAS sur 4xx/5xx (risque doublon)
        return !status;
      },
    }
  );
};

/** Cliniques disponibles pour hospitalisation (mappings clinique → billing) */
export interface ClinicOption {
  id: number;
  name: string;
}

export const fetchClinicBillingMappings = async (): Promise<ClinicOption[]> => {
  const { data } = await enterpriseStandardApi.get<{ data?: Array<{ clinic_company_id: number; clinic_company_name: string }> }>(
    "/company-settings/billing/clinic-mappings"
  );
  const mappings = data?.data || [];
  const seen = new Set<number>();
  const result: ClinicOption[] = [];
  for (const m of mappings) {
    if (m.clinic_company_id && !seen.has(m.clinic_company_id)) {
      seen.add(m.clinic_company_id);
      result.push({ id: m.clinic_company_id, name: m.clinic_company_name || `Clinique ${m.clinic_company_id}` });
    }
  }
  return result;
};

/** Tiers payeurs / curateurs */
export interface BillingPartyOption {
  id: number;
  display_name: string;
  type: string;
}

export const fetchBillingParties = async (active = true): Promise<BillingPartyOption[]> => {
  const { data } = await enterpriseStandardApi.get<{ data?: BillingPartyOption[] }>(
    "/company-settings/billing/parties",
    { params: { active } }
  );
  return data?.data || [];
};

export const createClientStay = async (
  clientId: number,
  stayData: { company_id: number; start_date: string; end_date?: string | null; notes?: string | null }
) => {
  const { data } = await enterpriseStandardApi.post(`/clients/${clientId}/stays`, stayData);
  return data;
};

export const linkClientBillingParty = async (
  clientId: number | string,
  linkData: {
    billing_party_id: string | number;
    role?: string | null;
    is_default?: boolean;
    contact_name?: string | null;
    contact_email?: string | null;
    contact_phone?: string | null;
  }
) => {
  const { data } = await enterpriseStandardApi.post(`/clients/${clientId}/billing-parties`, linkData);
  return data;
};
