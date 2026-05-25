import { getPerfActiveContext } from "./perfActiveContext";
import { recordPerfBucket } from "./perfInstrumentationStore";
import { shouldRecordPerfMetric } from "./perfInstrumentationTier";
import { emitPerfKpi } from "./perfKpi";

export type MessageSendPhase = "optimistic" | "acked" | "displayed" | "error" | "timeout";

export type MessageSendHandle = {
  startedAtMs: number;
  role: string;
  threadId: string;
  clientId: string;
};

const OPTIMISTIC_BUCKETS: Record<MessageSendPhase, string> = {
  optimistic: "input_to_optimistic_ms",
  acked: "input_to_ack_ms",
  displayed: "input_to_display_ms",
  error: "input_to_display_ms",
  timeout: "timeout_ms",
};

export function startMessageSend(params: {
  role: string;
  threadId: string;
  clientId: string;
}): MessageSendHandle {
  return {
    startedAtMs: Date.now(),
    role: params.role,
    threadId: params.threadId,
    clientId: params.clientId,
  };
}

export function endMessageSend(
  handle: MessageSendHandle,
  phase: MessageSendPhase
): void {
  if (!shouldRecordPerfMetric()) return;
  const durationMs = Date.now() - handle.startedAtMs;
  const bucket = OPTIMISTIC_BUCKETS[phase];
  recordPerfBucket("message_send", bucket, durationMs);
  emitPerfKpi("perf.message_send", {
    source: "perf.message.send",
    phase,
    duration_ms: durationMs,
    metric: bucket,
    role: handle.role,
    thread_id: handle.threadId,
    client_id: handle.clientId,
    ...getPerfActiveContext(),
  });
}

export function recordMessageSendRetry(params: {
  role: string;
  threadId: string;
  clientId: string;
}): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("message_send", "retry_count", 0, 1);
  emitPerfKpi("perf.message_send", {
    source: "perf.message.send",
    phase: "retry",
    metric: "retry_count",
    role: params.role,
    thread_id: params.threadId,
    client_id: params.clientId,
    ...getPerfActiveContext(),
  });
}

export type ChatCacheMismatchKind =
  | "unread_drift"
  | "optimistic_payload_diff"
  | "thread_order_drift";

const CRITICAL_ACK_FIELDS = [
  "content",
  "priority",
  "message_type",
  "image_url",
  "pdf_url",
] as const;

export function recordChatCacheMismatch(params: {
  kind: ChatCacheMismatchKind;
  role: string;
  screen?: string;
  details: Record<string, unknown>;
}): void {
  if (!shouldRecordPerfMetric()) return;
  recordPerfBucket("chat_cache_mismatch", params.kind, 0, 1);
  emitPerfKpi("perf.chat_cache_mismatch", {
    source: "perf.chat_cache_mismatch",
    kind: params.kind,
    role: params.role,
    screen: params.screen,
    ...params.details,
    ...getPerfActiveContext(),
  });
}

export function compareOptimisticToServer(
  optimistic: Record<string, unknown>,
  server: Record<string, unknown>
): string[] {
  const diffs: string[] = [];
  for (const field of CRITICAL_ACK_FIELDS) {
    const o = optimistic[field];
    const s = server[field];
    if (o === s) continue;
    if (o == null && s == null) continue;
    if (String(o ?? "") !== String(s ?? "")) {
      diffs.push(field);
    }
  }
  return diffs;
}

export function countThreadOrderDrift(
  localOrder: string[],
  serverOrder: string[],
  threshold = 2
): number {
  const localIndex = new Map<string, number>();
  for (let i = 0; i < localOrder.length; i++) {
    localIndex.set(localOrder[i], i);
  }
  const shared = serverOrder.filter((id) => localIndex.has(id));
  if (shared.length <= 1) return 0;
  const n = shared.length;
  let drift = 0;
  for (let i = 0; i < n; i++) {
    const id = shared[i];
    if (localIndex.get(id) !== i) drift++;
  }
  return drift > threshold ? drift : 0;
}
