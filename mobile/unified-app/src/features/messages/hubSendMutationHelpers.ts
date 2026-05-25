import type { QueryClient } from "@tanstack/react-query";
import {
  appendMessageToThreadCache,
  markMessageFailedInCache,
  OPTIMISTIC_MESSAGE_TIMEOUT_MS,
  patchThreadsOnReceive,
  replaceMessageInThreadCache,
  type HubCacheKeySet,
  type PatchableHubMessage,
} from "./messageHubCachePatch";
import {
  compareOptimisticToServer,
  endMessageSend,
  recordMessageSendRetry,
  recordChatCacheMismatch,
  startMessageSend,
  type MessageSendHandle,
} from "../../core/observability/perfMessageSend";
import { isPerfChatLocalPatchEnabled } from "./chatLocalPatchFlag";

export type OptimisticSendContext = {
  clientId: string;
  optimistic: PatchableHubMessage;
  timeoutTimer: ReturnType<typeof setTimeout> | null;
  perfHandle: MessageSendHandle | null;
};

export function buildOptimisticHubMessage(params: {
  clientId: string;
  threadId: string;
  senderId: string | number | null;
  senderName: string;
  senderRole: string;
  content: string;
  bookingId?: number | null;
  conversationId?: number | null;
  extra?: Partial<PatchableHubMessage>;
}): PatchableHubMessage {
  const now = new Date().toISOString();
  return {
    id: params.clientId,
    _localId: params.clientId,
    sender_id: params.senderId,
    content: params.content,
    sender_role: params.senderRole,
    sender_name: params.senderName,
    timestamp: now,
    optimisticTimestamp: now,
    thread_id: params.threadId,
    booking_id: params.bookingId ?? null,
    conversation_id: params.conversationId ?? null,
    message_type: "text",
    priority: "normal",
    status: "sending",
    ...params.extra,
  };
}

export function scheduleOptimisticSendTimeout(params: {
  qc: QueryClient;
  keys: HubCacheKeySet;
  threadId: string;
  clientId: string;
  conversationId?: string | number | null;
  role: string;
  perfHandle: MessageSendHandle | null;
}): ReturnType<typeof setTimeout> | null {
  if (!isPerfChatLocalPatchEnabled()) return null;
  return setTimeout(() => {
    void markMessageFailedInCache(
      params.qc,
      params.keys,
      params.threadId,
      params.clientId,
      "timeout",
      params.conversationId
    );
    if (params.perfHandle) {
      endMessageSend(params.perfHandle, "timeout");
    }
  }, OPTIMISTIC_MESSAGE_TIMEOUT_MS);
}

export async function applyOptimisticSendMutate(params: {
  qc: QueryClient;
  keys: HubCacheKeySet;
  threadId: string;
  optimistic: PatchableHubMessage;
  conversationId?: string | number | null;
  role: string;
}): Promise<OptimisticSendContext> {
  const clientId =
    params.optimistic._localId ?? String(params.optimistic.id);
  const perfHandle = startMessageSend({
    role: params.role,
    threadId: params.threadId,
    clientId,
  });
  if (isPerfChatLocalPatchEnabled()) {
    // UI already paints optimistic row immediately in screen state; keep metric close to user input.
    endMessageSend(perfHandle, "optimistic");
    await Promise.all([
      appendMessageToThreadCache(
        params.qc,
        params.keys,
        params.threadId,
        params.optimistic,
        params.conversationId
      ),
      patchThreadsOnReceive(
        params.qc,
        params.keys,
        params.threadId,
        { ...params.optimistic, status: "sent" },
        params.threadId
      ),
    ]);
  }
  const timeoutTimer =
    scheduleOptimisticSendTimeout({
    qc: params.qc,
    keys: params.keys,
    threadId: params.threadId,
    clientId,
    conversationId: params.conversationId,
    role: params.role,
    perfHandle,
  });
  return { clientId, optimistic: params.optimistic, timeoutTimer, perfHandle };
}

export function clearOptimisticSendTimer(ctx: OptimisticSendContext | undefined): void {
  if (ctx?.timeoutTimer) clearTimeout(ctx.timeoutTimer);
}

export async function applyOptimisticSendSuccess(params: {
  qc: QueryClient;
  keys: HubCacheKeySet;
  threadId: string;
  context: OptimisticSendContext;
  serverMessage: PatchableHubMessage;
  conversationId?: string | number | null;
  role: string;
}): Promise<void> {
  clearOptimisticSendTimer(params.context);
  if (!isPerfChatLocalPatchEnabled()) return;
  const diffs = compareOptimisticToServer(
    params.context.optimistic as Record<string, unknown>,
    params.serverMessage as Record<string, unknown>
  );
  if (diffs.length > 0) {
    recordChatCacheMismatch({
      kind: "optimistic_payload_diff",
      role: params.role,
      details: { fields: diffs, client_id: params.context.clientId },
    });
  }
  await replaceMessageInThreadCache(
    params.qc,
    params.keys,
    params.threadId,
    params.context.clientId,
    params.serverMessage,
    params.conversationId
  );
  await patchThreadsOnReceive(
    params.qc,
    params.keys,
    params.threadId,
    params.serverMessage,
    params.threadId
  );
  if (params.context.perfHandle) {
    endMessageSend(params.context.perfHandle, "acked");
    endMessageSend(params.context.perfHandle, "displayed");
  }
}

export async function applyOptimisticSendError(params: {
  qc: QueryClient;
  keys: HubCacheKeySet;
  threadId: string;
  context: OptimisticSendContext | undefined;
  reason: string;
  conversationId?: string | number | null;
  role: string;
}): Promise<void> {
  clearOptimisticSendTimer(params.context);
  if (!isPerfChatLocalPatchEnabled() || !params.context) return;
  await markMessageFailedInCache(
    params.qc,
    params.keys,
    params.threadId,
    params.context.clientId,
    params.reason,
    params.conversationId
  );
  if (params.context.perfHandle) {
    endMessageSend(params.context.perfHandle, "error");
  }
}

export function trackMessageSendRetry(params: {
  role: string;
  threadId: string;
  clientId: string;
}): void {
  recordMessageSendRetry(params);
}
