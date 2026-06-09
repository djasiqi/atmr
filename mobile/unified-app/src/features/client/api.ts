import { AxiosError, isAxiosError } from "axios";
import { apiClient } from "../../core/api/client";
import {
  AddressAutocompleteSuggestion,
  ApiListResponse,
  Booking,
  BookingDraftPayload,
  BookingPreviewResponse,
  ClientApiError,
  ClientProfile,
  CreateClientBookingPayload,
  CreateClientBookingResponse,
  IndicativeFareEstimateResponse,
  SaferpayAssertResponse,
  SaferpayInitializeResponse,
} from "./types";
import { derivePaymentStatusFromBooking } from "./paymentStatus";

type ApiEnvelope<T> = {
  data?: T;
  success?: boolean;
  message?: string;
};

export const SAFERPAY_ASSERT_CLIENT_TIMEOUT_MS = 150000;

function normalizeError(error: unknown): ClientApiError {
  const e = error as AxiosError<{
    error?: string;
    message?: string;
    error_code?: string;
    error_message?: string;
    action_hint?: string;
    retryable?: boolean;
  }>;
  return {
    status: e.response?.status ?? null,
    code: e.response?.data?.error_code ?? e.response?.data?.error ?? "UNKNOWN_ERROR",
    message:
      e.response?.data?.error_message ??
      e.response?.data?.message ??
      e.message ??
      "Erreur inconnue",
    actionHint: e.response?.data?.action_hint,
    retryable: e.response?.data?.retryable,
  };
}

function unwrapData<T>(payload: T | ApiEnvelope<T>): T {
  if (payload && typeof payload === "object" && "data" in (payload as ApiEnvelope<T>)) {
    return ((payload as ApiEnvelope<T>).data ?? payload) as T;
  }
  return payload as T;
}

function normalizeBookings(data: Booking[] | ApiListResponse<Booking>): Booking[] {
  if (Array.isArray(data)) return data;
  return data.items ?? data.results ?? data.data ?? data.bookings ?? [];
}

function normalizeBooking(booking: Booking): Booking {
  const derivedPaymentStatus = derivePaymentStatusFromBooking(booking);
  return {
    ...booking,
    payment_status: derivedPaymentStatus,
    payment_required:
      booking.payment_required ?? derivedPaymentStatus === "required",
    payment_provider: booking.payment_provider ?? booking.online_payment?.provider ?? null,
    payment_reference:
      booking.payment_reference ??
      (booking.online_payment?.payment_id
        ? String(booking.online_payment.payment_id)
        : null),
    payment_amount:
      booking.payment_amount ?? (typeof (booking as { amount?: unknown }).amount === "number"
        ? ((booking as { amount?: number }).amount ?? null)
        : null),
    currency: booking.currency ?? "CHF",
  };
}

export async function getClientMe(): Promise<ClientProfile> {
  try {
    const { data } = await apiClient.get<ClientProfile>("/clients/me");
    return data;
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function deleteClientAccount(): Promise<void> {
  try {
    await apiClient.delete("/clients/me");
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function getClientBookings(params?: {
  limit?: number;
}): Promise<Booking[]> {
  try {
    const { data } = await apiClient.get<Booking[] | ApiListResponse<Booking>>(
      "/clients/me/bookings",
      {
        params: {
          ...(params?.limit ? { limit: params.limit } : {}),
        },
      }
    );
    return normalizeBookings(data).map(normalizeBooking);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function getBookingDetail(bookingId: number): Promise<Booking> {
  try {
    const { data } = await apiClient.get<Booking | ApiEnvelope<Booking>>(`/bookings/${bookingId}`);
    return normalizeBooking(unwrapData(data));
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function initializeSaferpayCheckout(
  bookingId: number,
  returnUrl: string
): Promise<SaferpayInitializeResponse> {
  try {
    const { data } = await apiClient.post<
      SaferpayInitializeResponse | ApiEnvelope<SaferpayInitializeResponse>
    >(`/bookings/${bookingId}/saferpay/initialize`, {
      return_url: returnUrl,
    });
    return unwrapData(data);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function assertSaferpayCheckout(
  bookingId: number,
  paymentId: number
): Promise<SaferpayAssertResponse> {
  try {
    const { data } = await apiClient.post<
      SaferpayAssertResponse | ApiEnvelope<SaferpayAssertResponse>
    >(
      `/bookings/${bookingId}/saferpay/assert`,
      { payment_id: paymentId },
      { timeout: SAFERPAY_ASSERT_CLIENT_TIMEOUT_MS }
    );
    return unwrapData(data);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function autocompleteAddress(
  query: string,
  options?: { lat?: number; lon?: number; limit?: number; country?: "CH" | "FR" }
): Promise<AddressAutocompleteSuggestion[]> {
  if (query.trim().length < 2) {
    return [];
  }
  try {
    const { data } = await apiClient.get<AddressAutocompleteSuggestion[]>(
      "/geocode/autocomplete",
      {
        params: {
          q: query,
          limit: options?.limit ?? 8,
          ...(typeof options?.lat === "number" ? { lat: options.lat } : {}),
          ...(typeof options?.lon === "number" ? { lon: options.lon } : {}),
          ...(options?.country ? { country: options.country } : {}),
        },
      }
    );
    return Array.isArray(data) ? data : [];
  } catch (error) {
    throw normalizeError(error);
  }
}

/** Géocodage inverse (GPS → adresse lisible), même forme que les suggestions. */
export async function reverseGeocodeFromCoordinates(
  lat: number,
  lon: number
): Promise<AddressAutocompleteSuggestion | null> {
  try {
    const { data } = await apiClient.get<AddressAutocompleteSuggestion>(
      "/geocode/reverse",
      {
        params: { lat, lon },
      }
    );
    if (!data || (!(data.address || "").trim() && !(data.label || "").trim())) {
      return null;
    }
    return data;
  } catch {
    return null;
  }
}

/** Résout lat/lon (Google Places) pour une suggestion d’autocomplete n’en ayant pas. */
export async function getGeocodePlaceDetails(
  placeId: string
): Promise<AddressAutocompleteSuggestion | null> {
  const pid = (placeId || "").trim();
  if (!pid) return null;
  try {
    const { data } = await apiClient.get<{
      source?: string;
      place_id?: string;
      address?: string;
      lat?: number | null;
      lon?: number | null;
      lng?: number | null;
      name?: string;
    }>("/geocode/place-details", { params: { place_id: pid } });
    if (!data) return null;
    const line = (data.address ?? data.name ?? "").trim();
    if (!line) {
      if (typeof __DEV__ !== "undefined" && __DEV__) {
        console.warn("[getGeocodePlaceDetails] empty address/name", { place_id: pid });
      }
      return null;
    }
    const latN = data.lat;
    const lonN = data.lon != null ? data.lon : data.lng;
    if (typeof latN !== "number" || !Number.isFinite(latN)) {
      if (typeof __DEV__ !== "undefined" && __DEV__) {
        console.warn("[getGeocodePlaceDetails] invalid lat", { place_id: pid, lat: data.lat });
      }
      return null;
    }
    if (typeof lonN !== "number" || !Number.isFinite(lonN)) {
      if (typeof __DEV__ !== "undefined" && __DEV__) {
        console.warn("[getGeocodePlaceDetails] invalid lon/lng", {
          place_id: pid,
          lon: data.lon,
          lng: data.lng,
        });
      }
      return null;
    }
    return {
      source: data.source,
      place_id: data.place_id ?? pid,
      address: data.address,
      label: data.address ?? data.name ?? line,
      lat: latN,
      lon: lonN,
    };
  } catch (error) {
    if (typeof __DEV__ !== "undefined" && __DEV__) {
      const status = isAxiosError(error) ? error.response?.status : undefined;
      const body = isAxiosError(error) ? error.response?.data : undefined;
      const message = isAxiosError(error)
        ? error.message
        : error instanceof Error
          ? error.message
          : String(error);
      console.warn("[getGeocodePlaceDetails] request failed", {
        place_id: pid,
        status,
        message,
        body,
      });
    }
    return null;
  }
}

export async function previewClientBooking(
  payload: BookingDraftPayload
): Promise<BookingPreviewResponse> {
  try {
    const { data } = await apiClient.post<
      BookingPreviewResponse | ApiEnvelope<BookingPreviewResponse>
    >("/clients/me/bookings/preview", payload);
    return unwrapData(data);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function createClientBooking(
  payload: CreateClientBookingPayload
): Promise<CreateClientBookingResponse> {
  try {
    const { data } = await apiClient.post<
      CreateClientBookingResponse | ApiEnvelope<CreateClientBookingResponse>
    >("/clients/me/bookings", payload);
    return unwrapData(data);
  } catch (error) {
    throw normalizeError(error);
  }
}

export type IndicativeFareEstimateResult =
  | { success: true; data: IndicativeFareEstimateResponse }
  | { success: false; code?: string; status?: number | null };

/**
 * Indicatif non contractuel (calibration serveur). N’alimente pas le `amount` de réservation
 * (celui-ci vient du preview + prévisualisation validée).
 */
export async function postIndicativeFareEstimate(payload: {
  pickup_location: string;
  dropoff_location: string;
}): Promise<IndicativeFareEstimateResult> {
  try {
    const { data } = await apiClient.post<IndicativeFareEstimateResponse>(
      "/clients/me/indicative-fare/estimate",
      payload
    );
    return { success: true, data };
  } catch (error) {
    const e = error as AxiosError<{ error?: string; message?: string }>;
    return {
      success: false,
      status: e.response?.status ?? null,
      code: e.response?.data?.error,
    };
  }
}
