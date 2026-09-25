import { useEffect, useRef } from "react";
import { AppState, AppStateStatus } from "react-native";
import { useQueryClient } from "@tanstack/react-query";
import { useSession } from "../../core/sessionProvider";
import { setResumeAttemptCorrelationId } from "../../core/api/client";
import { appendSessionJournalEvent } from "../../core/observability/sessionJournal";
import { companyRealtimeBridge } from "./realtime/companyRealtimeBridge";
import { performCompanyRecoveryResync } from "./realtime/useCompanyRecoveryListener";

type CompanyRuntimeResumeOptions = {
  contextId: string | null;
  enabled: boolean;
};

const RESUME_MAX_ATTEMPTS = 2;
const RETRY_DELAY_MS = 250;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function reconnectCompanyBridge(contextId: string) {
  const snap = companyRealtimeBridge.getSnapshot();
  if (snap.status === "idle" || snap.contextId !== contextId) {
    companyRealtimeBridge.connect(contextId);
    return;
  }
  if (snap.status === "failed" || snap.status === "reconnecting" || !snap.connected) {
    companyRealtimeBridge.reconnect();
    return;
  }
  companyRealtimeBridge.connect(contextId);
}

export function useCompanyRuntimeResume(options: CompanyRuntimeResumeOptions) {
  const { contextId, enabled } = options;
  const queryClient = useQueryClient();
  const { status, bootstrapSession, recoverAuthWarm } = useSession();
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
        void appendSessionJournalEvent(
          "session.company.resume.start",
          { resume_attempt_id: resumeAttemptId },
          contextId
        );
        if (typeof setResumeAttemptCorrelationId === "function") {
          setResumeAttemptCorrelationId(resumeAttemptId);
        }
        let succeeded = false;
        let lastFailureReason = "resume_failed";
        try {
          if (status === "idle" || status === "error") {
            await bootstrapSession();
          }
          for (let attempt = 1; attempt <= RESUME_MAX_ATTEMPTS; attempt += 1) {
            try {
              // P0-5 : une seule décision auth (coordinateur), pas refresh nu
              const outcome = await recoverAuthWarm("company_foreground");
              if (outcome === "terminal") {
                lastFailureReason = "auth_terminal";
                break;
              }
              if (outcome === "no_action") {
                lastFailureReason = "recovery_no_action";
                if (attempt < RESUME_MAX_ATTEMPTS) {
                  await sleep(RETRY_DELAY_MS);
                  continue;
                }
                break;
              }
              // recovered | keep_local : sockets après la décision auth
              reconnectCompanyBridge(contextId);
              performCompanyRecoveryResync(queryClient, contextId, "reconnect");
              void appendSessionJournalEvent(
                "session.company.resume.success",
                {
                  retry_count: attempt - 1,
                  resume_attempt_id: resumeAttemptId,
                  recovery_outcome: outcome,
                },
                contextId
              );
              succeeded = true;
              break;
            } catch (error) {
              lastFailureReason = error instanceof Error ? error.message : "resume_failed";
              if (attempt < RESUME_MAX_ATTEMPTS) {
                void appendSessionJournalEvent(
                  "session.company.resume.retry",
                  {
                    attempt,
                    reason: lastFailureReason,
                    resume_attempt_id: resumeAttemptId,
                  },
                  contextId
                );
                await sleep(RETRY_DELAY_MS);
                continue;
              }
            }
          }
          if (!succeeded) {
            void appendSessionJournalEvent(
              "session.company.resume.failed",
              { reason: lastFailureReason, resume_attempt_id: resumeAttemptId },
              contextId
            );
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
  }, [enabled, contextId, queryClient, status, bootstrapSession, recoverAuthWarm]);
}
