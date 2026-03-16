import { refreshDriverTokenSingleflight } from "@/services/api";
import { beginRefreshCycle, logAuthEvent } from "@/services/authLogging";

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
    const token = await refreshDriverTokenSingleflight();
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
