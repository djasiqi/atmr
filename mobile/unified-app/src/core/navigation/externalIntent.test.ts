import { describe, expect, it } from "@jest/globals";
import {
  clearPendingExternalIntent,
  consumeExpiredExternalIntentMarker,
  parseExternalIntent,
  peekPendingExternalIntent,
  upsertPendingExternalIntent,
} from "./externalIntent";
import { resolveExternalIntent } from "./externalIntentResolver";

describe("externalIntent parser", () => {
  it("parse activate-account universal link", () => {
    const intent = parseExternalIntent("https://app.lirie.ch/activate-account?token=abc");
    expect(intent?.type).toBe("activate-email");
    expect(intent?.source).toBe("universal-link");
  });

  it("parse reset-password deep link", () => {
    const intent = parseExternalIntent("lirie://reset-password?token=tok123");
    expect(intent?.type).toBe("reset-password");
    expect(intent?.source).toBe("deep-link");
  });

  it("parse payment-return deep link", () => {
    const intent = parseExternalIntent("lirie://payment-return?bookingId=22&paymentId=9&outcome=success");
    expect(intent?.type).toBe("payment-return");
  });

  it("parse guest-payment-return deep link (séparé de payment-return)", () => {
    const intent = parseExternalIntent("lirie://guest-payment-return?guestBookingId=gb-1&outcome=success");
    expect(intent?.type).toBe("guest-payment-return");
    if (intent?.type !== "guest-payment-return") throw new Error("expected guest-payment-return");
    expect(intent.guestBookingId).toBe("gb-1");
  });

  it("guest-payment-return n'est pas confondu avec payment-return", () => {
    const wrong = parseExternalIntent("lirie://guest-payment-return?paymentId=9&outcome=success");
    expect(wrong?.type).toBe("fallback");
  });

  it("parse booking-status deep link", () => {
    const intent = parseExternalIntent("lirie://booking-status?token=abc123");
    expect(intent?.type).toBe("booking-status");
  });

  it("parse invalid payment-return into fallback intent", () => {
    const intent = parseExternalIntent("lirie://payment-return?paymentId=9");
    expect(intent?.type).toBe("fallback");
  });
});

describe("externalIntent store + resolver", () => {
  it("keeps higher-priority intent when concurrent", () => {
    clearPendingExternalIntent();
    const first = parseExternalIntent("lirie://booking-status?token=first");
    const second = parseExternalIntent("lirie://payment-return?bookingId=42&paymentId=9");
    if (!first || !second) throw new Error("intent parse failed");
    upsertPendingExternalIntent(first);
    upsertPendingExternalIntent(second);
    expect(peekPendingExternalIntent()?.intentId).toBe(second.intentId);
    clearPendingExternalIntent();
  });

  it("résout guest-payment-return vers la route publique", () => {
    const intent = parseExternalIntent("lirie://guest-payment-return?guestBookingId=g-99&outcome=fail");
    if (!intent || intent.type !== "guest-payment-return") throw new Error("parse");
    const resolved = resolveExternalIntent(intent, null);
    expect(resolved.route).toBe("/guest-payment-return?guestBookingId=g-99&outcome=fail");
  });

  it("resolves quick-action intent to route", () => {
    const intent = parseExternalIntent("lirie://quick-action?missionId=123&action=accept");
    if (!intent) throw new Error("intent parse failed");
    const resolved = resolveExternalIntent(intent, null);
    expect(resolved.route).toBe(
      "/(public)/fallback/auth-required?next=%2Fquick-action%3FmissionId%3D123%26action%3Daccept"
    );
  });

  it("resolves pre-request-resume to booking form when authenticated", () => {
    clearPendingExternalIntent();
    upsertPendingExternalIntent({
      intentId: "intent_test_resume",
      type: "pre-request-resume",
      draftId: "draft_1",
      source: "deep-link",
      receivedAt: Date.now(),
    });
    const intent = peekPendingExternalIntent();
    if (!intent) throw new Error("missing intent");
    const resolved = resolveExternalIntent(intent, {
      is_authenticated: true,
      active_context_id: "client:self",
    } as any);
    expect(resolved.route).toBe("/(app)/(client)/booking/new?publicDraftId=draft_1");
    clearPendingExternalIntent();
  });

  it("marks expired intent when ttl elapsed", () => {
    clearPendingExternalIntent();
    upsertPendingExternalIntent({
      intentId: "intent_expired_test",
      type: "booking-status",
      token: "t1",
      source: "deep-link",
      receivedAt: Date.now() - 16 * 60 * 1000,
    });
    expect(peekPendingExternalIntent()).toBeNull();
    expect(consumeExpiredExternalIntentMarker()).toBe(true);
    expect(consumeExpiredExternalIntentMarker()).toBe(false);
    clearPendingExternalIntent();
  });
});
