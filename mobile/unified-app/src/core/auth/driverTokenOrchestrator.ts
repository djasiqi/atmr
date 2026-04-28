import { refreshAuthTokenNow } from "../api/client";
import { appendSessionJournalEvent } from "../observability/sessionJournal";

let refreshInFlight: Promise<boolean> | null = null;

export async function refreshDriverTokenSingleflight(reason: string): Promise<boolean> {
  if (!refreshInFlight) {
    refreshInFlight = refreshAuthTokenNow()
      .then((ok) => {
        void appendSessionJournalEvent("auth.driver_token.refresh", { reason, success: ok });
        return ok;
      })
      .finally(() => {
        refreshInFlight = null;
      });
  }
  return refreshInFlight;
}
