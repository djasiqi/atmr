import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { useSession } from "../../core/sessionProvider";
import { setResumeAttemptCorrelationId } from "../../core/api/client";
import { realtimeManager } from "../../core/realtime/realtimeManager";
import { isFeatureEnabled } from "../../core/featureFlags/registry";
import { reconcileDriverMissions } from "./sync";
import { driverOfflineQueue } from "./offlineQueue";
import { invalidateDriverMissionScope } from "./queryKeys";
import { refreshAuthTokenSingleflight } from "../../core/auth/authTokenOrchestrator";
import {
  subscribeDriverForegroundResume,
  tryClaimDriverResumeWork,
} from "./driverForegroundResumeAuthority";

type RuntimeResumeOptions = {
  contextId: string | null;
  enabled: boolean;
  onForegroundResume?: () => Promise<void>;
};

const RESUME_MAX_ATTEMPTS = 2;

export function useDriverRuntimeResume(options: RuntimeResumeOptions) {
  const { contextId, enabled, onForegroundResume } = options;
  const queryClient = useQueryClient();
  const { status, bootstrapSession } = useSession();
  const isResumingRef = useRef(false);
  const resumeAttemptRef = useRef(0);
  const contextIdRef = useRef(contextId);
  const statusRef = useRef(status);
  const bootstrapRef = useRef(bootstrapSession);
  const onResumeRef = useRef(onForegroundResume);
  const queryClientRef = useRef(queryClient);
  contextIdRef.current = contextId;
  statusRef.current = status;
  bootstrapRef.current = bootstrapSession;
  onResumeRef.current = onForegroundResume;
  queryClientRef.current = queryClient;

  useEffect(() => {
    if (!enabled || !contextId) return;
    return subscribeDriverForegroundResume((resumeEpoch) => {
      if (isResumingRef.current) return;
      if (!tryClaimDriverResumeWork("runtime", resumeEpoch)) return;
      const activeContextId = contextIdRef.current;
      if (!activeContextId) return;
      void (async () => {
        isResumingRef.current = true;
        resumeAttemptRef.current += 1;
        const resumeAttemptId = `${activeContextId}:${Date.now()}:${resumeAttemptRef.current}`;
        emitDriverTelemetry("driver.runtime.resume.start", {
          source: "driver.runtime.resume",
          context_id: activeContextId,
          app_state: "active",
          resume_attempt_id: resumeAttemptId,
          resume_epoch: resumeEpoch,
        });
        if (typeof setResumeAttemptCorrelationId === "function") {
          setResumeAttemptCorrelationId(resumeAttemptId);
        }
        try {
          const sessionStatus = statusRef.current;
          if (sessionStatus === "idle" || sessionStatus === "error") {
            await bootstrapRef.current();
          }
          const claimedResync = tryClaimDriverResumeWork("resync", resumeEpoch);
          for (let attempt = 1; attempt <= RESUME_MAX_ATTEMPTS; attempt += 1) {
            try {
              await refreshAuthTokenSingleflight("foreground_resume");
              if (isFeatureEnabled("realtime_auth_reconnect_enabled")) {
                realtimeManager.connect(activeContextId, {
                  enableSocket: isFeatureEnabled("realtime_socket_enabled"),
                });
              }
              if (claimedResync) {
                await reconcileDriverMissions(queryClientRef.current, activeContextId);
              }
              await driverOfflineQueue.flush();
              await onResumeRef.current?.();
              if (claimedResync) {
                invalidateDriverMissionScope(queryClientRef.current, activeContextId);
                emitDriverTelemetry("driver.runtime.resync", {
                  source: "driver.runtime.resume",
                  context_id: activeContextId,
                  trigger: "foreground",
                  resume_attempt_id: resumeAttemptId,
                  resume_epoch: resumeEpoch,
                });
              }
              emitDriverTelemetry("driver.runtime.resume.success", {
                source: "driver.runtime.resume",
                context_id: activeContextId,
                retry_count: attempt - 1,
                resume_attempt_id: resumeAttemptId,
                resume_epoch: resumeEpoch,
              });
              break;
            } catch (error) {
              const reason = error instanceof Error ? error.message : "resume_failed";
              emitDriverTelemetry("driver.runtime.resume.failure", {
                source: "driver.runtime.resume",
                context_id: activeContextId,
                reason,
                retry_count: attempt,
                will_retry: attempt < RESUME_MAX_ATTEMPTS,
                resume_attempt_id: resumeAttemptId,
                resume_epoch: resumeEpoch,
              });
              if (attempt >= RESUME_MAX_ATTEMPTS) {
                break;
              }
            }
          }
        } finally {
          if (typeof setResumeAttemptCorrelationId === "function") {
            setResumeAttemptCorrelationId(null);
          }
          isResumingRef.current = false;
        }
      })();
    });
  }, [enabled, contextId]);
}
