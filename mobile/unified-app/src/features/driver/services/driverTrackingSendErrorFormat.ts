import { isAxiosError } from "axios";

/** Métadonnées sûres pour logs / télémétrie (pas d’URL complète ni d’identifiants). */
export type TrackingSendErrorClass =
  | "circuit_open"
  | "timeout"
  | "network"
  | "http"
  | "canceled"
  | "unknown";

export type TrackingSendErrorMeta = {
  error_class: TrackingSendErrorClass;
  error_message: string;
  http_status: number | null;
  api_error_code: string | null;
  transport_code: string | null;
  retry_after_seconds: number | null;
};

const MAX_MESSAGE_LEN = 240;

function sanitizePublicMessage(raw: string): string {
  return raw
    .replace(/\bhttps?:\/\/[^\s"'<>]+/gi, "[url]")
    .replace(/\b\d{1,3}(?:\.\d{1,3}){3}\b/g, "[ip]")
    .trim()
    .slice(0, MAX_MESSAGE_LEN);
}

function isRecord(e: unknown): e is Record<string, unknown> {
  return typeof e === "object" && e !== null;
}

/**
 * Normalise une erreur de tracking (HTTP, circuit, réseau, timeout) pour observabilité.
 * Les messages sont tronqués et les URL / IPv4 littérales masquées.
 */
export function formatTrackingSendError(error: unknown): TrackingSendErrorMeta {
  const unknownMeta = (msg: string): TrackingSendErrorMeta => ({
    error_class: "unknown",
    error_message: sanitizePublicMessage(msg || "unknown_error"),
    http_status: null,
    api_error_code: null,
    transport_code: null,
    retry_after_seconds: null,
  });

  if (error == null) {
    return unknownMeta("null_error");
  }

  if (isAxiosError(error)) {
    const transportCode = typeof error.code === "string" ? error.code : null;
    const status = error.response?.status ?? null;
    const data = error.response?.data as Record<string, unknown> | undefined;
    const retryAfterRaw = data?.retry_after_seconds;
    const retryAfterSeconds =
      typeof retryAfterRaw === "number" && Number.isFinite(retryAfterRaw)
        ? retryAfterRaw
        : null;
    const apiCode =
      typeof data?.error_code === "string"
        ? data.error_code
        : typeof data?.code === "string"
          ? data.code
          : null;
    const rawMsg =
      (typeof data?.error_message === "string" && data.error_message) ||
      (typeof data?.message === "string" && data.message) ||
      (typeof data?.error === "string" && data.error) ||
      error.message ||
      "axios_error";

    if (error.code === "ERR_CANCELED") {
      return {
        error_class: "canceled",
        error_message: sanitizePublicMessage(String(rawMsg)),
        http_status: status,
        api_error_code: apiCode,
        transport_code: transportCode,
        retry_after_seconds: retryAfterSeconds,
      };
    }

    const lower = String(error.message ?? "").toLowerCase();
    if (error.code === "ECONNABORTED" || lower.includes("timeout")) {
      return {
        error_class: "timeout",
        error_message: sanitizePublicMessage(String(rawMsg)),
        http_status: status,
        api_error_code: apiCode,
        transport_code: transportCode,
        retry_after_seconds: retryAfterSeconds,
      };
    }

    if (error.response == null) {
      return {
        error_class: "network",
        error_message: sanitizePublicMessage(String(rawMsg)),
        http_status: null,
        api_error_code: apiCode,
        transport_code: transportCode,
        retry_after_seconds: retryAfterSeconds,
      };
    }

    return {
      error_class: "http",
      error_message: sanitizePublicMessage(String(rawMsg)),
      http_status: status,
      api_error_code: apiCode,
      transport_code: transportCode,
      retry_after_seconds: retryAfterSeconds,
    };
  }

  /** Erreur normalisée côté `driver/api` (`normalizeError`) ou circuit ouvert. */
  if (isRecord(error) && typeof error.message === "string") {
    const code = typeof error.code === "string" ? error.code : null;
    const status = typeof error.status === "number" ? error.status : null;
    const msg = error.message;
    const retryAfterRaw = error.retry_after_seconds;
    const retryAfterSeconds =
      typeof retryAfterRaw === "number" && Number.isFinite(retryAfterRaw)
        ? retryAfterRaw
        : null;

    if (code === "HTTP_CIRCUIT_BREAKER_OPEN") {
      return {
        error_class: "circuit_open",
        error_message: sanitizePublicMessage(msg),
        http_status: null,
        api_error_code: code,
        transport_code: null,
        retry_after_seconds: null,
      };
    }

    if (status != null) {
      return {
        error_class: "http",
        error_message: sanitizePublicMessage(msg),
        http_status: status,
        api_error_code: code,
        transport_code: null,
        retry_after_seconds: retryAfterSeconds,
      };
    }

    if (code && code !== "UNKNOWN_ERROR") {
      return {
        error_class: "unknown",
        error_message: sanitizePublicMessage(msg),
        http_status: null,
        api_error_code: code,
        transport_code: null,
        retry_after_seconds: retryAfterSeconds,
      };
    }

    return unknownMeta(msg);
  }

  if (error instanceof Error) {
    return unknownMeta(error.message);
  }

  return unknownMeta(String(error));
}
