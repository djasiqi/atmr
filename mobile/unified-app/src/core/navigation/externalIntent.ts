import * as Linking from "expo-linking";
import {
  clearPendingExternalIntentRecord,
  loadPendingExternalIntentRecord,
  savePendingExternalIntentRecord,
} from "../public/pendingExternalIntentStore";

export type ExternalIntentSource = "universal-link" | "deep-link";

export type ExternalIntent =
  | {
      intentId: string;
      type: "activate-email";
      token: string;
      receivedAt: number;
      source: ExternalIntentSource;
    }
  | {
      intentId: string;
      type: "reset-password";
      token: string;
      receivedAt: number;
      source: ExternalIntentSource;
    }
  | {
      intentId: string;
      type: "payment-return";
      bookingId: string;
      paymentId: string;
      outcome: string;
      receivedAt: number;
      source: "deep-link";
    }
  | {
      intentId: string;
      type: "guest-payment-return";
      guestBookingId: string;
      outcome: string;
      receivedAt: number;
      source: "deep-link";
    }
  | {
      intentId: string;
      type: "quick-action";
      missionId: string;
      action: string;
      receivedAt: number;
      source: "deep-link";
    }
  | {
      intentId: string;
      type: "booking-status";
      token: string;
      receivedAt: number;
      source: ExternalIntentSource;
    }
  | {
      intentId: string;
      type: "pre-request-resume";
      draftId: string;
      receivedAt: number;
      source: "deep-link";
    }
  | {
      intentId: string;
      type: "fallback";
      fallbackType: "expired-link" | "invalid-link" | "auth-required" | "resume-later" | "install-app";
      reason?: string;
      next?: string;
      receivedAt: number;
      source: ExternalIntentSource;
    };

const INTENT_TTL_MS = 15 * 60 * 1000;

let pendingExternalIntent: ExternalIntent | null = null;
let lastExpiredIntentAt: number | null = null;

function normalizeQueryParam(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function makeIntentId(seed: string): string {
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}_${seed.slice(0, 12)}`;
}

function inferSource(url: string): ExternalIntentSource {
  return /^https?:\/\//i.test(url) ? "universal-link" : "deep-link";
}

function getIntentPriority(type: ExternalIntent["type"]): number {
  switch (type) {
    case "payment-return":
    case "guest-payment-return":
      return 100;
    case "quick-action":
      return 90;
    case "activate-email":
      return 80;
    case "booking-status":
      return 70;
    case "pre-request-resume":
      return 60;
    default:
      return 50;
  }
}

function parseTokenIntent(url: string): ExternalIntent | null {
  const parsed = Linking.parse(url);
  const host = String((parsed as { hostname?: string }).hostname ?? "").toLowerCase();
  const path = String(parsed.path ?? "").toLowerCase();
  const source = inferSource(url);
  const token = normalizeQueryParam(parsed.queryParams?.token);
  if (!token && (host.includes("activate-account") || path.includes("activate-account"))) {
    return {
      intentId: makeIntentId("invalid-link"),
      type: "fallback",
      fallbackType: "invalid-link",
      reason: "activation_token_missing",
      source,
      receivedAt: Date.now(),
    };
  }
  if (!token && (host.includes("reset-password") || path.includes("reset-password"))) {
    return {
      intentId: makeIntentId("invalid-link"),
      type: "fallback",
      fallbackType: "invalid-link",
      reason: "reset_token_missing",
      source,
      receivedAt: Date.now(),
    };
  }
  if (!token) return null;
  if (host.includes("activate-account") || path.includes("activate-account")) {
    return {
      intentId: makeIntentId("activate-email"),
      type: "activate-email",
      token,
      source,
      receivedAt: Date.now(),
    };
  }
  if (host.includes("reset-password") || path.includes("reset-password")) {
    return {
      intentId: makeIntentId("reset-password"),
      type: "reset-password",
      token,
      source,
      receivedAt: Date.now(),
    };
  }
  return null;
}

function parseQuickActionIntent(url: string): ExternalIntent | null {
  const parsed = Linking.parse(url);
  const host = String((parsed as { hostname?: string }).hostname ?? "").toLowerCase();
  const path = String(parsed.path ?? "").toLowerCase();
  if (!host.includes("quick-action") && !path.includes("quick-action")) return null;
  const missionId = normalizeQueryParam(parsed.queryParams?.missionId ?? parsed.queryParams?.bookingId);
  if (!missionId) {
    return {
      intentId: makeIntentId("invalid-link"),
      type: "fallback",
      fallbackType: "invalid-link",
      reason: "mission_missing",
      source: inferSource(url),
      receivedAt: Date.now(),
    };
  }
  const action = normalizeQueryParam(parsed.queryParams?.action) || "accept";
  return {
    intentId: makeIntentId("quick-action"),
    type: "quick-action",
    missionId,
    action: action.toLowerCase(),
    source: "deep-link",
    receivedAt: Date.now(),
  };
}

function parseGuestPaymentReturnIntent(url: string): ExternalIntent | null {
  const parsed = Linking.parse(url);
  const host = String((parsed as { hostname?: string }).hostname ?? "").toLowerCase();
  const path = String(parsed.path ?? "").toLowerCase();
  if (!host.includes("guest-payment-return") && !path.includes("guest-payment-return")) return null;
  const guestBookingId = normalizeQueryParam(
    parsed.queryParams?.guestBookingId ?? parsed.queryParams?.guest_booking_id
  );
  if (!guestBookingId) {
    return {
      intentId: makeIntentId("invalid-link"),
      type: "fallback",
      fallbackType: "invalid-link",
      reason: "guest_booking_missing",
      source: inferSource(url),
      receivedAt: Date.now(),
    };
  }
  const outcome = normalizeQueryParam(parsed.queryParams?.outcome) || "success";
  return {
    intentId: makeIntentId("guest-payment-return"),
    type: "guest-payment-return",
    guestBookingId,
    outcome,
    source: "deep-link",
    receivedAt: Date.now(),
  };
}

function parsePaymentIntent(url: string): ExternalIntent | null {
  const parsed = Linking.parse(url);
  const host = String((parsed as { hostname?: string }).hostname ?? "").toLowerCase();
  const path = String(parsed.path ?? "").toLowerCase();
  if (host.includes("guest-payment-return") || path.includes("guest-payment-return")) return null;
  if (!host.includes("payment-return") && !path.includes("payment-return")) return null;
  const bookingId = normalizeQueryParam(parsed.queryParams?.bookingId ?? parsed.queryParams?.booking_id);
  if (!bookingId) {
    return {
      intentId: makeIntentId("invalid-link"),
      type: "fallback",
      fallbackType: "invalid-link",
      reason: "booking_missing",
      source: inferSource(url),
      receivedAt: Date.now(),
    };
  }
  const paymentId = normalizeQueryParam(parsed.queryParams?.paymentId ?? parsed.queryParams?.payment_id);
  const outcome = normalizeQueryParam(parsed.queryParams?.outcome);
  return {
    intentId: makeIntentId("payment-return"),
    type: "payment-return",
    bookingId,
    paymentId,
    outcome,
    source: "deep-link",
    receivedAt: Date.now(),
  };
}

function parseBookingStatusIntent(url: string): ExternalIntent | null {
  const parsed = Linking.parse(url);
  const host = String((parsed as { hostname?: string }).hostname ?? "").toLowerCase();
  const path = String(parsed.path ?? "").toLowerCase();
  if (!host.includes("booking-status") && !path.includes("booking-status")) return null;
  const token = normalizeQueryParam(parsed.queryParams?.token);
  if (!token) {
    return {
      intentId: makeIntentId("invalid-link"),
      type: "fallback",
      fallbackType: "invalid-link",
      reason: "booking_status_token_missing",
      source: inferSource(url),
      receivedAt: Date.now(),
    };
  }
  return {
    intentId: makeIntentId("booking-status"),
    type: "booking-status",
    token,
    source: inferSource(url),
    receivedAt: Date.now(),
  };
}

export function parseExternalIntent(url: string | null | undefined): ExternalIntent | null {
  if (!url) return null;
  return (
    parseTokenIntent(url) ??
    parseGuestPaymentReturnIntent(url) ??
    parsePaymentIntent(url) ??
    parseQuickActionIntent(url) ??
    parseBookingStatusIntent(url)
  );
}

export function upsertPendingExternalIntent(intent: ExternalIntent): {
  replacedIntentId: string | null;
} {
  const previous = pendingExternalIntent;
  if (!previous || getIntentPriority(intent.type) >= getIntentPriority(previous.type)) {
    pendingExternalIntent = intent;
  }
  if (pendingExternalIntent) {
    void savePendingExternalIntentRecord({
      intent_id: pendingExternalIntent.intentId,
      intent_type: pendingExternalIntent.type,
      payload: JSON.parse(JSON.stringify(pendingExternalIntent)) as Record<string, unknown>,
      received_at: pendingExternalIntent.receivedAt,
    });
  }
  return { replacedIntentId: previous?.intentId ?? null };
}

export function peekPendingExternalIntent(now: number = Date.now()): ExternalIntent | null {
  if (!pendingExternalIntent) return null;
  if (now - pendingExternalIntent.receivedAt > INTENT_TTL_MS) {
    pendingExternalIntent = null;
    lastExpiredIntentAt = now;
    void clearPendingExternalIntentRecord();
    return null;
  }
  return pendingExternalIntent;
}

export function clearPendingExternalIntent(): void {
  pendingExternalIntent = null;
  void clearPendingExternalIntentRecord();
}

export function getExternalIntentTtlMs(): number {
  return INTENT_TTL_MS;
}

export async function hydratePendingExternalIntent(): Promise<void> {
  if (pendingExternalIntent) return;
  const stored = await loadPendingExternalIntentRecord();
  if (!stored) return;
  const payload = stored.payload as unknown;
  if (!payload || typeof payload !== "object") return;
  const asIntent = payload as ExternalIntent;
  if (!asIntent.type || !asIntent.intentId || !asIntent.receivedAt) return;
  pendingExternalIntent = asIntent;
}

export function consumeExpiredExternalIntentMarker(): boolean {
  if (!lastExpiredIntentAt) return false;
  lastExpiredIntentAt = null;
  return true;
}

export async function queueExternalIntentResume(payload: {
  type: "pre-request-resume";
  draftId: string;
}): Promise<void> {
  upsertPendingExternalIntent({
    intentId: makeIntentId(payload.type),
    type: payload.type,
    draftId: payload.draftId,
    source: "deep-link",
    receivedAt: Date.now(),
  });
}
