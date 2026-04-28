import type { BootstrapResponse } from "../contracts/auth";
import type { ExternalIntent } from "./externalIntent";

type ResolveExternalIntentResult = {
  route: string | null;
  terminal: boolean;
  retryable: boolean;
};

function buildQuery(params: Record<string, string | undefined>): string {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value && value.trim().length > 0) {
      query.set(key, value);
    }
  });
  const asString = query.toString();
  return asString.length > 0 ? `?${asString}` : "";
}

export function resolveExternalIntent(
  intent: ExternalIntent,
  bootstrap: BootstrapResponse | null
): ResolveExternalIntentResult {
  switch (intent.type) {
    case "activate-email": {
      const route = `/(public)/activate${buildQuery({ token: intent.token })}`;
      return { route, terminal: true, retryable: false };
    }
    case "reset-password": {
      const route = `/(public)/reset-password${buildQuery({ token: intent.token })}`;
      return { route, terminal: true, retryable: false };
    }
    case "payment-return": {
      if (!bootstrap?.is_authenticated) {
        const next = `/payment-return${buildQuery({
          bookingId: intent.bookingId,
          paymentId: intent.paymentId,
          outcome: intent.outcome,
        })}`;
        return {
          route: `/(public)/fallback/auth-required${buildQuery({ next })}`,
          terminal: true,
          retryable: false,
        };
      }
      const route = `/payment-return${buildQuery({
        bookingId: intent.bookingId,
        paymentId: intent.paymentId,
        outcome: intent.outcome,
      })}`;
      return { route, terminal: true, retryable: false };
    }
    case "guest-payment-return": {
      const route = `/guest-payment-return${buildQuery({
        guestBookingId: intent.guestBookingId,
        outcome: intent.outcome,
      })}`;
      return { route, terminal: true, retryable: false };
    }
    case "quick-action": {
      if (!bootstrap?.is_authenticated) {
        const next = `/quick-action${buildQuery({
          missionId: intent.missionId,
          action: intent.action,
        })}`;
        return {
          route: `/(public)/fallback/auth-required${buildQuery({ next })}`,
          terminal: true,
          retryable: false,
        };
      }
      const route = `/quick-action${buildQuery({
        missionId: intent.missionId,
        action: intent.action,
      })}`;
      return { route, terminal: true, retryable: false };
    }
    case "booking-status": {
      const route = `/(public)/booking-status${buildQuery({ token: intent.token })}`;
      return { route, terminal: true, retryable: false };
    }
    case "pre-request-resume": {
      if (!bootstrap?.is_authenticated) {
        return {
          route: `/(public)/fallback/auth-required${buildQuery({
            next: "/(app)/(client)/booking/new",
          })}`,
          terminal: true,
          retryable: false,
        };
      }
      if (!bootstrap.active_context_id) {
        return {
          route: "/(public)/fallback/resume-later",
          terminal: true,
          retryable: false,
        };
      }
      return {
        route: `/(app)/(client)/booking/new${buildQuery({ publicDraftId: intent.draftId })}`,
        terminal: true,
        retryable: false,
      };
    }
    case "fallback": {
      const route = `/(public)/fallback/${intent.fallbackType}${buildQuery({
        reason: intent.reason,
        next: intent.next,
      })}`;
      return { route, terminal: true, retryable: false };
    }
    default:
      return { route: null, terminal: true, retryable: false };
  }
}
