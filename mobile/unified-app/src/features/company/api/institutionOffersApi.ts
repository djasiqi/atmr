import { AxiosError } from "axios";
import { apiClient } from "../../../core/api/client";
import type { InstitutionOfferErrorCode } from "../utils/institutionOfferResponse";
import { isInstitutionOfferErrorCode } from "../utils/institutionOfferResponse";

export type InstitutionTransportRequestSummary = {
  id?: number;
  public_id?: string;
  institution_name?: string;
  mission_type?: string;
  scheduled_time?: string | null;
  pickup_location?: string | null;
  dropoff_location?: string | null;
  is_round_trip?: boolean;
  return_time?: string | null;
  notes?: string | null;
};

export type InstitutionRequestOffer = {
  id: number;
  status: string;
  mode?: string;
  sent_at?: string | null;
  expires_at?: string | null;
  can_respond?: boolean;
  transport_request?: InstitutionTransportRequestSummary | null;
};

export type InstitutionOffersListResponse = {
  offers: InstitutionRequestOffer[];
  total: number;
};

export type InstitutionOfferAcceptResponse = {
  success: boolean;
  offer_id: number;
  booking_id?: number;
  transport_request_id?: number;
  return_booking_id?: number;
};

export type InstitutionOfferRejectResponse = {
  success: boolean;
  offer_id: number;
  escalated?: boolean;
  next_offer_id?: number | null;
  fallback_broadcast?: boolean;
};

export type InstitutionOfferApiError = {
  message: string;
  code?: InstitutionOfferErrorCode;
  offer_id?: number;
  booking_id?: number;
  transport_request_id?: number;
  status?: number;
};

function readAxiosErrorBody(error: AxiosError<unknown>): Record<string, unknown> | null {
  const data = error.response?.data;
  if (!data || typeof data !== "object") return null;
  return data as Record<string, unknown>;
}

export function parseInstitutionOfferApiError(error: unknown): InstitutionOfferApiError {
  if (error instanceof AxiosError) {
    const body = readAxiosErrorBody(error);
    const codeRaw = body?.code;
    const code = isInstitutionOfferErrorCode(codeRaw) ? codeRaw : undefined;
    const message =
      (typeof body?.error === "string" && body.error.trim()) ||
      (typeof body?.message === "string" && body.message.trim()) ||
      error.message ||
      "Erreur inattendue";
    return {
      message,
      code,
      status: error.response?.status,
      offer_id: typeof body?.offer_id === "number" ? body.offer_id : undefined,
      booking_id: typeof body?.booking_id === "number" ? body.booking_id : undefined,
      transport_request_id:
        typeof body?.transport_request_id === "number"
          ? body.transport_request_id
          : undefined,
    };
  }
  if (error instanceof Error) {
    return { message: error.message };
  }
  return { message: "Erreur inattendue" };
}

export async function fetchInstitutionOffers(
  status?: string
): Promise<InstitutionOffersListResponse> {
  const params: Record<string, string> = {};
  if (status) params.status = status;
  const { data } = await apiClient.get<InstitutionOffersListResponse>(
    "/company/request-offers",
    { params }
  );
  return {
    offers: Array.isArray(data?.offers) ? data.offers : [],
    total: typeof data?.total === "number" ? data.total : 0,
  };
}

export async function fetchInstitutionOfferDetail(
  offerId: number
): Promise<InstitutionRequestOffer> {
  const { data } = await apiClient.get<InstitutionRequestOffer>(
    `/company/request-offers/${offerId}`
  );
  return data;
}

export async function acceptInstitutionOffer(
  offerId: number,
  proposedPickupTime?: string
): Promise<InstitutionOfferAcceptResponse> {
  const body: Record<string, string> = {};
  if (proposedPickupTime) body.proposed_pickup_time = proposedPickupTime;
  const { data } = await apiClient.post<InstitutionOfferAcceptResponse>(
    `/company/request-offers/${offerId}/accept`,
    body
  );
  return data;
}

export async function rejectInstitutionOffer(
  offerId: number,
  reason?: string
): Promise<InstitutionOfferRejectResponse> {
  const { data } = await apiClient.post<InstitutionOfferRejectResponse>(
    `/company/request-offers/${offerId}/reject`,
    reason ? { reason } : {}
  );
  return data;
}
