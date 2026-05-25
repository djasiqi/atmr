import { useEffect, useRef } from "react";
import { AppState, AppStateStatus } from "react-native";
import { useQueryClient } from "@tanstack/react-query";
import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { useSession } from "../../core/sessionProvider";
import { setResumeAttemptCorrelationId } from "../../core/api/client";
import { realtimeManager } from "../../core/realtime/realtimeManager";
import { isFeatureEnabled } from "../../core/featureFlags/registry";
import { reconcileDriverMissions } from "./sync";
import { driverOfflineQueue } from "./offlineQueue";
import { invalidateDriverMissionScope } from "./queryKeys";
import { refreshDriverTokenSingleflight } from "../../core/auth/driverTokenOrchestrator";

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

  useEffect(() => {
    if (!enabled || !contextId) return;
    let previousState: AppStateStatus = AppState.currentState;
    const subscription = AppState.addEventListener("change", (nextState) => {
      const resumed = previousState !== "active" && nextState === "active";
      previousState = nextState;
      if (!resumed) return;
      if (isResumingRef.current) return;
      void (async () => {
        isResumingRef.current = true;
        resumeAttemptRef.current += 1;
        const resumeAttemptId = `${contextId}:${Date.now()}:${resumeAttemptRef.current}`;
        emitDriverTelemetry("driver.runtime.resume.start", {
          source: "driver.runtime.resume",
          context_id: contextId,
          app_state: nextState,
          resume_attempt_id: resumeAttemptId,
        });
        if (typeof setResumeAttemptCorrelationId === "function") {
          setResumeAttemptCorrelationId(resumeAttemptId);
        }
        try {
          if (status === "idle" || status === "error") {
            await bootstrapSession();
          }
          for (let attempt = 1; attempt <= RESUME_MAX_ATTEMPTS; attempt += 1) {
            try {
              await refreshDriverTokenSingleflight("foreground_resume");
              if (isFeatureEnabled("realtime_auth_reconnect_enabled")) {
                realtimeManager.connect(contextId, {
                  enableSocket: isFeatureEnabled("realtime_socket_enabled"),
                });
              }
              await reconcileDriverMissions(queryClient, contextId);
              await driverOfflineQueue.flush();
              await onForegroundResume?.();
              invalidateDriverMissionScope(queryClient, contextId);
              emitDriverTelemetry("driver.runtime.resync", {
                source: "driver.runtime.resume",
                context_id: contextId,
                trigger: "foreground",
                resume_attempt_id: resumeAttemptId,
              });
              emitDriverTelemetry("driver.runtime.resume.success", {
                source: "driver.runtime.resume",
                context_id: contextId,
                retry_count: attempt - 1,
                resume_attempt_id: resumeAttemptId,
              });
              break;
            } catch (error) {
              const reason = error instanceof Error ? error.message : "resume_failed";
              emitDriverTelemetry("driver.runtime.resume.failure", {
                source: "driver.runtime.resume",
                context_id: contextId,
                reason,
                retry_count: attempt,
                will_retry: attempt < RESUME_MAX_ATTEMPTS,
                resume_attempt_id: resumeAttemptId,
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
    return () => {
      subscription.remove();
    };
  }, [enabled, contextId, onForegroundResume, queryClient, status, bootstrapSession]);
}
