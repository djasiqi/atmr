/**
 * Orchestrateur unique des cleanups dangereux lors d’un changement de rôle / session.
 * Toute transition driver ↔ entreprise ou fin de session chauffeur doit passer ici (anti-dispersion).
 */
import { Platform } from "react-native";
import { getLogger } from "@/utils/logger";
import { disconnectSocket } from "@/services/socket";
import {
  ensureBackgroundTrackingStopped,
  stopAdaptiveLocationTracking,
} from "@/services/locationTracker";
import { getSyncEngine } from "@/services/syncEngine";
import { MissionStateManager } from "@/services/missionState";
import { setAuthSurfaceRole, type AuthSurfaceRole } from "@/services/authSurface";

const log = getLogger("SessionModeTransition");

export type RunRoleTransitionParams = {
  fromRole: AuthSurfaceRole;
  toRole: AuthSurfaceRole;
  reason: string;
  options?: {
    preserveMissionState?: boolean;
    /** Session chauffeur invalidée sans changement de mode stocké (logout forcé, intercepteurs). */
    forceDriverInfrastructureTeardown?: boolean;
    /** Fin de session entreprise : couper le socket même si le mode stocké ne change pas encore. */
    forceSocketDisconnect?: boolean;
  };
};

function shouldDisconnectSocket(
  from: AuthSurfaceRole,
  to: AuthSurfaceRole,
  opts: RunRoleTransitionParams["options"]
): boolean {
  if (opts?.forceSocketDisconnect) return true;
  if (opts?.forceDriverInfrastructureTeardown) return true;
  return from !== to;
}

function shouldTeardownDriverInfra(
  from: AuthSurfaceRole,
  to: AuthSurfaceRole,
  opts: RunRoleTransitionParams["options"]
): boolean {
  if (opts?.forceDriverInfrastructureTeardown) return true;
  return from === "driver" && to !== "driver";
}

export async function runRoleTransition(params: RunRoleTransitionParams): Promise<void> {
  const { fromRole, toRole, reason, options } = params;
  const preserveMission = options?.preserveMissionState === true;

  log.info("runRoleTransition", {
    fromRole,
    toRole,
    reason,
    preserveMissionState: preserveMission,
    forceDriverInfrastructureTeardown: options?.forceDriverInfrastructureTeardown ?? false,
  });

  if (
    fromRole === toRole &&
    !options?.forceDriverInfrastructureTeardown &&
    !options?.forceSocketDisconnect
  ) {
    setAuthSurfaceRole(toRole);
    return;
  }

  if (shouldTeardownDriverInfra(fromRole, toRole, options)) {
    if (Platform.OS !== "web") {
      try {
        stopAdaptiveLocationTracking();
      } catch (e) {
        log.warn("stopAdaptiveLocationTracking", { error: e });
      }
      try {
        await ensureBackgroundTrackingStopped(
          reason.includes("logout") || reason.includes("driver_logout")
            ? "logout"
            : "role_changed"
        );
      } catch (e) {
        log.warn("ensureBackgroundTrackingStopped", { error: e });
      }
    }
    try {
      getSyncEngine().stop();
    } catch (e) {
      log.warn("syncEngine.stop", { error: e });
    }
    if (!preserveMission) {
      try {
        await MissionStateManager.stopMission();
      } catch (e) {
        log.warn("MissionStateManager.stopMission", { error: e });
      }
    }
  }

  if (shouldDisconnectSocket(fromRole, toRole, options)) {
    try {
      disconnectSocket();
    } catch (e) {
      log.warn("disconnectSocket", { error: e });
    }
  }

  setAuthSurfaceRole(toRole);
}
