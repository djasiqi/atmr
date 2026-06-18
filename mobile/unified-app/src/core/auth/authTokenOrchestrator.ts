import { refreshAuthTokenNow } from "../api/client";
import { appendSessionJournalEvent } from "../observability/sessionJournal";

let refreshInFlight: Promise<boolean> | null = null;

export async function refreshAuthTokenSingleflight(reason: string): Promise<boolean> {
  if (!refreshInFlight) {
    refreshInFlight = refreshAuthTokenNow()
      .then((ok) => {
        void appendSessionJournalEvent("auth.token.refresh", { reason, success: ok });
        return ok;
      })
      .finally(() => {
        refreshInFlight = null;
      });
  }
  return refreshInFlight;
}

/** @deprecated Utiliser refreshAuthTokenSingleflight */
export const refreshDriverTokenSingleflight = refreshAuthTokenSingleflight;
