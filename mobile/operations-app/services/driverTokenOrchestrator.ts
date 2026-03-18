import { runDriverRefreshSingleflight } from "@/services/api";
import { beginRefreshCycle, logAuthEvent } from "@/services/authLogging";
import { secureStorage } from "@/services/storage";
import { buildAuthNamespace } from "@/services/storage/keys";

export type DriverRefreshTriggerSource =
  | "api_401"
  | "socket_connect_error"
  | "socket_unauthorized"
  | "proactive_refresh"
  | "foreground_resync"
  | "boot_restore"
  | "profile_refresh";

/**
 * Orchestrateur unique pour le refresh driver.
 * Toutes les couches (socket, hooks, API) doivent passer par ici pour
 * garder une corrélation homogène des cycles et éviter les races de refresh.
 */
export async function refreshDriverTokenOrchestrated(
  triggerSource: DriverRefreshTriggerSource
): Promise<string> {
  const refreshCycleId = beginRefreshCycle("driver");
  logAuthEvent("AUTH_REFRESH_START", {
    route: "driver",
    trigger_source: triggerSource,
    refresh_cycle_id: refreshCycleId,
  });
  try {
    const userPublicId = (await secureStorage.getUserPublicId()) ?? "unknown";
    const accessToken = await secureStorage.getAccessToken();
    let tenantId: string | number | null = null;
    let sessionId: string | null = null;
    if (accessToken) {
      try {
        const payload = JSON.parse(atob(accessToken.split(".")[1]));
        tenantId = payload?.company_id ?? null;
        sessionId = payload?.session_id ?? null;
      } catch {
        // no-op
      }
    }
    const sessionKey = buildAuthNamespace({
      role: "driver",
      userId: userPublicId,
      tenantId,
      sessionId,
    });
    const token = await runDriverRefreshSingleflight(sessionKey, triggerSource);
    logAuthEvent("AUTH_REFRESH_SUCCESS", {
      route: "driver",
      trigger_source: triggerSource,
      refresh_cycle_id: refreshCycleId,
      outcome: "token_refreshed",
    });
    return token;
  } catch (error: any) {
    logAuthEvent("AUTH_REFRESH_FAIL", {
      route: "driver",
      trigger_source: triggerSource,
      refresh_cycle_id: refreshCycleId,
      status: error?.response?.status ?? "unknown",
      outcome: "failed",
    });
    throw error;
  }
}
