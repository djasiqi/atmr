import { enterpriseApi } from "./enterpriseAuth";
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
} from "@/types/enterpriseDispatch";

export const getDispatchStatus = async (): Promise<DispatchStatus> => {
  const response = await enterpriseApi.get<DispatchStatus>(
    "/dispatch/v1/status"
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

export const sendDispatchMessage = async (content: string) => {
  const response = await enterpriseApi.post<DispatchMessage>(
    "/dispatch/v1/chat/messages",
    { content }
  );
  return normalizeDispatchMessage(response.data);
};
