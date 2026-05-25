import type { QueryClient } from "@tanstack/react-query";
import type { HubChatMessage, MessageHubThread } from "../driver/messages/types";

export const OPTIMISTIC_MESSAGE_TIMEOUT_MS = 30_000;

export type HubScopeKey = readonly string[];

export type PatchableHubMessage = HubChatMessage & {
  status?: "sending" | "sent" | "failed";
  failure_reason?: string;
  optimisticTimestamp?: string;
};

export type ThreadsQueryData = {
  threads: MessageHubThread[];
  unread_total?: number;
  fromFallback?: boolean;
};

/** Clés React Query exactes (driver vs company peuvent différer). */
export type HubCacheKeySet = {
  threads: readonly unknown[];
  unread: readonly unknown[];
  messages: (
    threadId: string,
    conversationId?: string | number | null
  ) => readonly unknown[];
};

export function buildDriverHubCacheKeys(
  companyId: number | null,
  myUserId: number | null
): HubCacheKeySet {
  const cid = companyId ?? "none";
  return {
    threads: ["driver", "message-hub", "threads", cid, myUserId],
    unread: ["driver", "message-hub", "unread", cid],
    messages: (threadId, conversationId) => [
      "driver",
      "message-hub",
      "messages",
      cid,
      threadId,
      conversationId ?? "auto",
    ],
  };
}

export function buildCompanyHubCacheKeys(companyId: number | null): HubCacheKeySet {
  const cid = companyId ?? "none";
  return {
    threads: ["company", "message-hub", "threads", cid],
    unread: ["company", "message-hub", "unread", cid],
    messages: (threadId) => ["company", "message-hub", "messages", cid, threadId],
  };
}

function messageLocalId(message: PatchableHubMessage): string | null {
  const local = message._localId ?? (message as { client_id?: string }).client_id;
  return local != null && String(local).length > 0 ? String(local) : null;
}

function messageServerId(message: PatchableHubMessage): string | null {
  if (message.id == null) return null;
  const s = String(message.id);
  if (s.startsWith("local-")) return null;
  return s;
}

export function messageSortKey(message: PatchableHubMessage): number {
  const raw =
    message.timestamp ??
    (message as { created_at?: string }).created_at ??
    message.optimisticTimestamp ??
    "";
  const ms = Date.parse(String(raw));
  return Number.isFinite(ms) ? ms : 0;
}

function dedupeAndSortMessages(messages: PatchableHubMessage[]): PatchableHubMessage[] {
  const byKey = new Map<string, PatchableHubMessage>();
  const order: string[] = [];
  for (const msg of messages) {
    const localId = messageLocalId(msg);
    const serverId = messageServerId(msg);
    const key = localId ?? serverId ?? `anon-${order.length}`;
    if (!byKey.has(key)) order.push(key);
    const existing = byKey.get(key);
    if (!existing) {
      byKey.set(key, msg);
      continue;
    }
    const merged: PatchableHubMessage = {
      ...existing,
      ...msg,
      status: msg.status ?? existing.status,
      failure_reason: msg.failure_reason ?? existing.failure_reason,
    };
    byKey.set(key, merged);
  }
  return order
    .map((k) => byKey.get(k)!)
    .sort((a, b) => {
      const diff = messageSortKey(a) - messageSortKey(b);
      if (diff !== 0) return diff;
      return order.indexOf(
        messageLocalId(a) ?? messageServerId(a) ?? ""
      ) - order.indexOf(messageLocalId(b) ?? messageServerId(b) ?? "");
    });
}

export async function safePatch<T>(
  qc: QueryClient,
  queryKey: readonly unknown[],
  updater: (prev: T | undefined) => T | undefined
): Promise<void> {
  await qc.cancelQueries({ queryKey });
  qc.setQueryData<T>(queryKey, updater);
}

function clampUnread(n: number): number {
  return Math.max(0, Math.floor(n));
}

function previewFromMessage(message: PatchableHubMessage): string {
  const content = message.content?.trim();
  if (content) return content.slice(0, 120);
  if (message.image_url) return "Photo";
  if (message.audio_url) return "Message vocal";
  if (message.pdf_url) return "Document";
  return "";
}

export async function patchUnreadOnRead(
  qc: QueryClient,
  keys: HubCacheKeySet,
  delta: number
): Promise<void> {
  if (delta <= 0) return;
  await safePatch<number>(qc, keys.unread, (prev) =>
    clampUnread((prev ?? 0) - delta)
  );
}

export async function patchUnreadOnReceive(
  qc: QueryClient,
  keys: HubCacheKeySet,
  threadId: string,
  currentOpenThreadId: string | null,
  increment = 1
): Promise<void> {
  if (currentOpenThreadId && threadId === currentOpenThreadId) return;
  const delta = Math.max(1, increment);
  await safePatch<number>(qc, keys.unread, (prev) =>
    clampUnread((prev ?? 0) + delta)
  );
}

export async function patchThreadsOnRead(
  qc: QueryClient,
  keys: HubCacheKeySet,
  threadId: string
): Promise<number> {
  let cleared = 0;
  await safePatch<ThreadsQueryData>(qc, keys.threads, (prev) => {
    if (!prev?.threads) return prev;
    const threads = prev.threads.map((t) => {
      if (t.thread_id !== threadId) return t;
      cleared = t.unread_count ?? 0;
      return {
        ...t,
        unread_count: 0,
      };
    });
    return {
      ...prev,
      threads,
      unread_total: clampUnread((prev.unread_total ?? 0) - cleared),
    };
  });
  return cleared;
}

export async function markTimedOutOptimisticMessages(
  qc: QueryClient,
  keys: HubCacheKeySet,
  nowMs: number,
  activeThreadIds: string[]
): Promise<void> {
  const cutoff = nowMs - OPTIMISTIC_MESSAGE_TIMEOUT_MS;
  for (const threadId of activeThreadIds) {
    const key = keys.messages(threadId, "auto");
    await safePatch<HubChatMessage[]>(qc, key, (prev) => {
      if (!prev) return prev;
      let changed = false;
      const list = (prev as PatchableHubMessage[]).map((m) => {
        if (m.status !== "sending") return m;
        const ts = messageSortKey(m);
        if (ts > cutoff) return m;
        changed = true;
        return {
          ...m,
          status: "failed",
          failure_reason: "timeout",
        };
      });
      return changed ? (list as HubChatMessage[]) : prev;
    });
  }
}

export async function patchThreadsOnReceive(
  qc: QueryClient,
  keys: HubCacheKeySet,
  threadId: string,
  message: PatchableHubMessage,
  currentOpenThreadId: string | null,
  options?: { incrementUnread?: boolean }
): Promise<void> {
  const preview = previewFromMessage(message);
  const at = message.timestamp ?? new Date().toISOString();
  const shouldIncrementUnread = options?.incrementUnread ?? true;
  const bumpUnread =
    shouldIncrementUnread && (!currentOpenThreadId || threadId !== currentOpenThreadId);

  await safePatch<ThreadsQueryData>(qc, keys.threads, (prev) => {
    if (!prev?.threads) return prev;
    const threads = [...prev.threads];
    const stableOrder = new Map<string, number>(
      threads.map((t, index) => [t.thread_id, index])
    );
    const idx = threads.findIndex((t) => t.thread_id === threadId);
    const base: MessageHubThread =
      idx >= 0
        ? { ...threads[idx] }
        : {
            thread_id: threadId,
            section: "dispatch",
            title: threadId,
            unread_count: 0,
            priority: "normal",
          };
    const updated: MessageHubThread = {
      ...base,
      last_message_preview: preview || base.last_message_preview,
      last_message_at: at,
      unread_count: bumpUnread
        ? clampUnread((base.unread_count ?? 0) + 1)
        : base.unread_count ?? 0,
    };
    if (idx >= 0) threads[idx] = updated;
    else threads.push(updated);
    threads.sort((a, b) => {
      const ta = Date.parse(String(a.last_message_at ?? 0));
      const tb = Date.parse(String(b.last_message_at ?? 0));
      const diff = (Number.isFinite(tb) ? tb : 0) - (Number.isFinite(ta) ? ta : 0);
      if (diff !== 0) return diff;
      return (stableOrder.get(a.thread_id) ?? Number.MAX_SAFE_INTEGER) -
        (stableOrder.get(b.thread_id) ?? Number.MAX_SAFE_INTEGER);
    });
    return {
      ...prev,
      threads,
      unread_total: bumpUnread
        ? clampUnread((prev.unread_total ?? 0) + 1)
        : prev.unread_total,
    };
  });
}

export async function appendMessageToThreadCache(
  qc: QueryClient,
  keys: HubCacheKeySet,
  threadId: string,
  message: PatchableHubMessage,
  conversationId?: string | number | null
): Promise<boolean> {
  const key = keys.messages(threadId, conversationId);
  let hadCache = false;
  await safePatch<HubChatMessage[]>(qc, key, (prev) => {
    if (!prev) return prev;
    hadCache = true;
    const list = dedupeAndSortMessages([...(prev as PatchableHubMessage[]), message]);
    return list as HubChatMessage[];
  });
  return hadCache;
}

export async function replaceMessageInThreadCache(
  qc: QueryClient,
  keys: HubCacheKeySet,
  threadId: string,
  clientId: string,
  serverMessage: PatchableHubMessage,
  conversationId?: string | number | null
): Promise<void> {
  const key = keys.messages(threadId, conversationId);
  await safePatch<HubChatMessage[]>(qc, key, (prev) => {
    if (!prev) return prev;
    const list = (prev as PatchableHubMessage[]).map((m) => {
      const lid = messageLocalId(m);
      if (lid && lid === clientId) {
        const next: PatchableHubMessage = {
          ...m,
          ...serverMessage,
          status: "sent",
          failure_reason: undefined,
        };
        if (messageServerId(serverMessage)) {
          delete (next as { _localId?: string })._localId;
        }
        return next;
      }
      return m;
    });
    return dedupeAndSortMessages(list) as HubChatMessage[];
  });
}

export async function markMessageFailedInCache(
  qc: QueryClient,
  keys: HubCacheKeySet,
  threadId: string,
  clientId: string,
  reason: string,
  conversationId?: string | number | null
): Promise<void> {
  const key = keys.messages(threadId, conversationId);
  await safePatch<HubChatMessage[]>(qc, key, (prev) => {
    if (!prev) return prev;
    const list = (prev as PatchableHubMessage[]).map((m) => {
      const lid = messageLocalId(m);
      if (lid && lid === clientId) {
        return {
          ...m,
          status: "failed",
          failure_reason: reason,
        };
      }
      return m;
    });
    return list as HubChatMessage[];
  });
}

export function getThreadUnreadCount(
  qc: QueryClient,
  keys: HubCacheKeySet,
  threadId: string
): number {
  const data = qc.getQueryData<ThreadsQueryData>(keys.threads);
  const thread = data?.threads?.find((t) => t.thread_id === threadId);
  return thread?.unread_count ?? 0;
}

export function payloadToHubMessage(payload: unknown): PatchableHubMessage | null {
  if (!payload || typeof payload !== "object") return null;
  const p = payload as Record<string, unknown>;
  const content =
    typeof p.content === "string"
      ? p.content
      : typeof p.message === "string"
        ? p.message
        : "";
  const id = p.id ?? p.message_id ?? p._localId ?? `rt-${Date.now()}`;
  const timestamp =
    typeof p.timestamp === "string"
      ? p.timestamp
      : typeof p.created_at === "string"
        ? p.created_at
        : new Date().toISOString();
  return {
    id: typeof id === "number" || typeof id === "string" ? id : String(id),
    sender_id: (p.sender_id as HubChatMessage["sender_id"]) ?? null,
    content,
    sender_role: typeof p.sender_role === "string" ? p.sender_role : undefined,
    sender_name: typeof p.sender_name === "string" ? p.sender_name : null,
    timestamp,
    image_url: typeof p.image_url === "string" ? p.image_url : null,
    pdf_url: typeof p.pdf_url === "string" ? p.pdf_url : null,
    pdf_filename: typeof p.pdf_filename === "string" ? p.pdf_filename : null,
    audio_url: typeof p.audio_url === "string" ? p.audio_url : null,
    thread_id: typeof p.thread_id === "string" ? p.thread_id : null,
    booking_id: typeof p.booking_id === "number" ? p.booking_id : null,
    message_type: typeof p.message_type === "string" ? p.message_type : "text",
    priority:
      p.priority === "urgent" || p.priority === "important" ? p.priority : "normal",
    conversation_id:
      typeof p.conversation_id === "number" ? p.conversation_id : null,
    _localId: typeof p._localId === "string" ? p._localId : null,
    status: "sent",
  };
}
