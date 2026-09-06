import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo } from "react";
import { useSession } from "../../../core/sessionProvider";
import { queryCacheOptions, QUERY_STALE_TIME_MS } from "../../../core/queryCachePolicy";
import { contextRealtimeRouter } from "../../../core/realtime/contextRealtimeRouter";
import { normalizeCompanyEventType } from "../../../core/realtime/eventContracts";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { emitPerfKpi } from "../../../core/observability/perfKpi";
import { traceInvalidateQueries } from "../../../core/observability/perfInstrumentation";
import { recordPerfBucket } from "../../../core/observability/perfInstrumentationStore";
import {
  countThreadOrderDrift,
  recordChatCacheMismatch,
} from "../../../core/observability/perfMessageSend";
import { isPerfChatLocalPatchEnabled } from "../../messages/chatLocalPatchFlag";
import {
  appendMessageToThreadCache,
  buildCompanyHubCacheKeys,
  getThreadUnreadCount,
  patchThreadsOnRead,
  patchThreadsOnReceive,
  patchUnreadOnRead,
  patchUnreadOnReceive,
  payloadToHubMessage,
  type PatchableHubMessage,
} from "../../messages/messageHubCachePatch";
import {
  applyOptimisticSendError,
  applyOptimisticSendMutate,
  applyOptimisticSendSuccess,
  buildOptimisticHubMessage,
  trackMessageSendRetry,
} from "../../messages/hubSendMutationHelpers";
import {
  fetchHubUnreadCount,
  fetchMessageHubThreads,
  fetchThreadMessages,
  markThreadRead,
  sendHubMessage,
} from "../../driver/messages/api";
import { mapHubSendError } from "../../messages/mapHubSendError";
import { filterMissionThreadsWithDiscussion } from "../../driver/messages/buildLocalInbox";
import { dedupeMessageHubThreads } from "../../driver/messages/dedupeHubThreads";
import type { HubChatMessage, MessageHubThread, SyncPresenceStatus } from "../../driver/messages/types";
import { useCompanyNumericId } from "./companyId";
import { useCompanyRealtimeStatus } from "../hooks";
import { useCompanySessionNetworkReady } from "../sessionNetworkGate";

const HUB_KEY = ["company", "message-hub"] as const;

function classifyThreadKind(threadId: string | null): "dispatch" | "team" | "mission" | "company_driver" | "support" | "other" {
  if (!threadId) return "other";
  if (threadId === "dispatch") return "dispatch";
  if (threadId === "team") return "team";
  if (threadId === "support") return "support";
  if (threadId.startsWith("mission:")) return "mission";
  if (threadId.startsWith("company_driver:")) return "company_driver";
  return "other";
}

export { HUB_KEY };

export function useCompanyMessageHubThreads(companyId: number | null) {
  const queryClient = useQueryClient();
  const networkReady = useCompanySessionNetworkReady();
  return useQuery({
    queryKey: [...HUB_KEY, "threads", companyId ?? "none"],
    enabled: Boolean(companyId) && networkReady,
    queryFn: async () => {
      const hub = await fetchMessageHubThreads(companyId as number, { source: "hub-only" });
      const threads = filterMissionThreadsWithDiscussion(
        dedupeMessageHubThreads(hub.threads)
      );
      const result = {
        threads,
        unread_total: hub.unread_total,
        fromFallback: false,
      };
      if (isPerfChatLocalPatchEnabled() && companyId != null) {
        const keys = buildCompanyHubCacheKeys(companyId);
        const local = queryClient.getQueryData<{
          threads: MessageHubThread[];
        }>(keys.threads);
        if (local?.threads?.length) {
          const localOrder = local.threads.map((t) => t.thread_id);
          const serverOrder = threads.map((t) => t.thread_id);
          const drift = countThreadOrderDrift(localOrder, serverOrder);
          if (drift > 0) {
            recordChatCacheMismatch({
              kind: "thread_order_drift",
              role: "company",
              details: { drift, local_order: localOrder.slice(0, 8), server_order: serverOrder.slice(0, 8) },
            });
          }
        }
      }
      return result;
    },
    staleTime: 45_000,
    gcTime: 5 * 60_000,
    refetchOnWindowFocus: true,
    refetchInterval: false,
    placeholderData: (previous) => previous,
  });
}

export function useCompanyHubUnreadCount(
  companyId: number | null,
  options?: { enabled?: boolean }
) {
  const realtime = useCompanyRealtimeStatus();
  const queryClient = useQueryClient();
  const networkReady = useCompanySessionNetworkReady();
  return useQuery({
    queryKey: [...HUB_KEY, "unread", companyId ?? "none"],
    queryFn: async () => {
      const server = await fetchHubUnreadCount(companyId as number);
      if (isPerfChatLocalPatchEnabled() && companyId != null) {
        const keys = buildCompanyHubCacheKeys(companyId);
        const local = queryClient.getQueryData<number>(keys.unread);
        if (local != null && local !== server) {
          recordChatCacheMismatch({
            kind: "unread_drift",
            role: "company",
            details: { local, server, delta: Math.abs(local - server) },
          });
        }
      }
      return server;
    },
    enabled: Boolean(companyId) && networkReady && (options?.enabled ?? true),
    refetchInterval:
      realtime.status === "healthy" && realtime.connected ? false : 15_000,
    ...queryCacheOptions("operational"),
    staleTime: queryCacheOptions("operational").staleTime ?? QUERY_STALE_TIME_MS.default,
  });
}

export function useCompanyMessageHubUnreadBadge(options?: { enabled?: boolean }): number {
  const companyId = useCompanyNumericId();
  const hubUnread = useCompanyHubUnreadCount(companyId, options);
  return hubUnread.data ?? 0;
}

export function useCompanyThreadMessages(companyId: number | null, threadId: string | null) {
  const queryClient = useQueryClient();
  const networkReady = useCompanySessionNetworkReady();
  return useQuery({
    queryKey: [...HUB_KEY, "messages", companyId ?? "none", threadId ?? "none"],
    enabled: Boolean(companyId && threadId) && networkReady,
    staleTime: 20_000,
    refetchOnWindowFocus: true,
    placeholderData: (previous) => previous,
    queryFn: async () => {
      const queryStartedAt = Date.now();
      const queryKey: readonly unknown[] = [...HUB_KEY, "messages", companyId ?? "none", threadId ?? "none"];
      const cached = queryClient.getQueryData<unknown[]>(queryKey);
      emitPerfKpi("perf.thread_runtime", {
        source: "company.thread.query",
        phase: "perf.thread.query_start",
        role: "company",
        thread_id: threadId,
      });
      emitPerfKpi("perf.thread_runtime", {
        source: "company.thread.query",
        phase: cached ? "perf.thread.cache_hit" : "perf.thread.cache_miss",
        role: "company",
        thread_id: threadId,
        thread_kind: classifyThreadKind(threadId),
        cached_count: Array.isArray(cached) ? cached.length : 0,
      });
      recordPerfBucket(
        "thread_cache",
        `${cached ? "hit" : "miss"}:${classifyThreadKind(threadId)}`,
        0,
        1
      );
      try {
        const messages = await fetchThreadMessages(companyId as number, threadId as string);
        const queryDurationMs = Date.now() - queryStartedAt;
        recordPerfBucket("thread_runtime", "query_success", queryDurationMs);
        emitPerfKpi("perf.thread_runtime", {
          source: "company.thread.query",
          phase: "perf.thread.query_success",
          role: "company",
          thread_id: threadId,
          path: "hub_thread",
          message_count: messages.length,
          duration_ms: queryDurationMs,
        });
        return messages;
      } catch {
        const queryDurationMs = Date.now() - queryStartedAt;
        recordPerfBucket("thread_runtime", "query_success", queryDurationMs);
        emitPerfKpi("perf.thread_runtime", {
          source: "company.thread.query",
          phase: "perf.thread.query_success",
          role: "company",
          thread_id: threadId,
          path: "hub_thread_error",
          message_count: 0,
          duration_ms: queryDurationMs,
        });
        return [];
      }
    },
  });
}

export function useMarkCompanyThreadRead(companyId: number | null) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (threadId: string) => {
      if (!companyId) return;
      await markThreadRead(companyId, threadId);
    },
    onSuccess: async (_data, threadId) => {
      if (!companyId) return;
      if (isPerfChatLocalPatchEnabled()) {
        const keys = buildCompanyHubCacheKeys(companyId);
        const delta = getThreadUnreadCount(queryClient, keys, threadId);
        await patchThreadsOnRead(queryClient, keys, threadId);
        await patchUnreadOnRead(queryClient, keys, delta);
        return;
      }
      void queryClient.invalidateQueries({
        queryKey: [...HUB_KEY, "threads", companyId],
      });
      void queryClient.invalidateQueries({
        queryKey: [...HUB_KEY, "unread", companyId],
      });
      void queryClient.invalidateQueries({
        queryKey: [...HUB_KEY, "messages", companyId, threadId],
      });
    },
  });
}

export function useCompanySendHubMessage(companyId: number | null, threadId: string) {
  const queryClient = useQueryClient();
  const { bootstrap } = useSession();
  const senderName = useMemo(() => {
    const name = bootstrap?.user?.first_name;
    return typeof name === "string" && name.trim() ? name.trim() : "Exploitation";
  }, [bootstrap?.user?.first_name]);

  return useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      sendHubMessage(companyId as number, threadId, body),
    onMutate: async (body) => {
      if (!companyId || !isPerfChatLocalPatchEnabled()) return undefined;
      const keys = buildCompanyHubCacheKeys(companyId);
      const clientId =
        typeof body._localId === "string"
          ? body._localId
          : `local-${Date.now()}`;
      const optimistic = buildOptimisticHubMessage({
        clientId,
        threadId,
        senderId: bootstrap?.user?.id ?? null,
        senderName,
        senderRole: "COMPANY",
        content:
          typeof body.content === "string" ? body.content : "Message",
        extra: body as Partial<PatchableHubMessage>,
      });
      return applyOptimisticSendMutate({
        qc: queryClient,
        keys,
        threadId,
        optimistic,
        role: "company",
      });
    },
    onSuccess: async (serverMessage, _body, context) => {
      if (!companyId) return;
      if (isPerfChatLocalPatchEnabled() && context) {
        const keys = buildCompanyHubCacheKeys(companyId);
        await applyOptimisticSendSuccess({
          qc: queryClient,
          keys,
          threadId,
          context,
          serverMessage: serverMessage as PatchableHubMessage,
          role: "company",
        });
        return;
      }
      void queryClient.invalidateQueries({
        queryKey: [...HUB_KEY, "messages", companyId, threadId],
        exact: true,
      });
      void queryClient.invalidateQueries({
        queryKey: [...HUB_KEY, "threads", companyId],
        exact: true,
      });
    },
    onError: async (error, body, context) => {
      if (__DEV__) {
        console.warn("[useCompanySendHubMessage]", mapHubSendError(error), error);
      }
      if (!companyId || !isPerfChatLocalPatchEnabled()) return;
      const keys = buildCompanyHubCacheKeys(companyId);
      await applyOptimisticSendError({
        qc: queryClient,
        keys,
        threadId,
        context,
        reason: mapHubSendError(error),
        role: "company",
      });
      if (body && (body as { _retry?: boolean })._retry) {
        const clientId =
          typeof body._localId === "string" ? body._localId : threadId;
        trackMessageSendRetry({ role: "company", threadId, clientId });
      }
    },
  });
}

export { mapHubSendError };

export function useCompanyHubRealtimeSync(
  companyId: number | null,
  options?: { excludeThreadId?: string | null }
) {
  const { activeContext, bootstrap } = useSession();
  const queryClient = useQueryClient();
  const chatSocketEnabled = isFeatureEnabled("company_mobile_chat_socket_enabled");
  const myUserId = useMemo(() => {
    const id = bootstrap?.user?.id;
    if (typeof id === "number" && Number.isFinite(id)) return id;
    if (typeof id === "string") {
      const parsed = Number.parseInt(id, 10);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }, [bootstrap?.user?.id]);
  const excludeThreadId = options?.excludeThreadId ?? null;

  useEffect(() => {
    if (!chatSocketEnabled || !companyId) return;
    if (!activeContext || activeContext.context_type !== "company") return;

    let debounceId: ReturnType<typeof setTimeout> | null = null;
    return contextRealtimeRouter.subscribe(activeContext.context_id, (event) => {
      if (debounceId) clearTimeout(debounceId);
      debounceId = setTimeout(() => {
        debounceId = null;
        const eventType = (event as { event_type?: string } | null)?.event_type;
        if (
          eventType &&
          eventType !== "team_chat_message" &&
          eventType !== "conversation_message"
        ) {
          return;
        }
        const payload = event as { thread_id?: string } | null;
        const threadId =
          typeof payload?.thread_id === "string" ? payload.thread_id : null;
        if (isPerfChatLocalPatchEnabled()) {
          const keys = buildCompanyHubCacheKeys(companyId);
          if (threadId) {
            const message = payloadToHubMessage(payload);
            const senderRaw = (payload as { sender_id?: unknown })?.sender_id;
            const senderId =
              typeof senderRaw === "number"
                ? senderRaw
                : typeof senderRaw === "string"
                  ? Number.parseInt(senderRaw, 10)
                  : null;
            const isOwnEvent = myUserId != null && senderId != null && senderId === myUserId;
            void (async () => {
              if (message) {
                await appendMessageToThreadCache(
                  queryClient,
                  keys,
                  threadId,
                  message
                );
                await patchThreadsOnReceive(
                  queryClient,
                  keys,
                  threadId,
                  message,
                  excludeThreadId,
                  { incrementUnread: !isOwnEvent }
                );
              }
              if (isOwnEvent) return;
              await patchUnreadOnReceive(
                queryClient,
                keys,
                threadId,
                excludeThreadId
              );
            })();
          }
          return;
        }
        void traceInvalidateQueries(
          [...HUB_KEY, "unread", companyId],
          "company_hub_realtime",
          () =>
            queryClient.invalidateQueries({
              queryKey: [...HUB_KEY, "unread", companyId],
              refetchType: "active",
            })
        );
        const onOpenThread = Boolean(threadId && threadId === excludeThreadId);
        if (!onOpenThread) {
          void traceInvalidateQueries(
            [...HUB_KEY, "threads", companyId],
            "company_hub_realtime",
            () =>
              queryClient.invalidateQueries({
                queryKey: [...HUB_KEY, "threads", companyId],
                refetchType: "active",
              })
          );
        }
        if (threadId && !onOpenThread) {
          void traceInvalidateQueries(
            [...HUB_KEY, "messages", companyId, threadId],
            "company_hub_realtime",
            () =>
              queryClient.invalidateQueries({
                queryKey: [...HUB_KEY, "messages", companyId, threadId],
                refetchType: "active",
              })
          );
        }
      }, 1800);
    });
  }, [activeContext, chatSocketEnabled, companyId, excludeThreadId, myUserId, queryClient]);
}

export function useCompanySyncPresenceStatus(httpReachable?: boolean): SyncPresenceStatus {
  const realtime = useCompanyRealtimeStatus();
  const chatSocketEnabled = isFeatureEnabled("company_mobile_chat_socket_enabled");
  if (!chatSocketEnabled) {
    return httpReachable ? "connected" : "offline";
  }
  if (realtime.status === "healthy" && realtime.connected) return "connected";
  if (httpReachable) return "slow";
  if (realtime.status === "connecting" || realtime.status === "reconnecting") return "slow";
  return "offline";
}

export function useCompanyHubEventFilter() {
  return useCallback((event: unknown) => {
    if (!event || typeof event !== "object") return false;
    const eventType = normalizeCompanyEventType((event as { event_type?: unknown }).event_type);
    return eventType === "booking_message_sent" || eventType === "team_chat_message";
  }, []);
}

export function useMarkAllCompanyInboxRead(companyId: number | null) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (threadIds: string[]) => {
      if (!companyId) return;
      await Promise.all(threadIds.map((threadId) => markThreadRead(companyId, threadId)));
    },
    onSuccess: async (_data, threadIds) => {
      if (!companyId) return;
      if (isPerfChatLocalPatchEnabled()) {
        const keys = buildCompanyHubCacheKeys(companyId);
        for (const threadId of threadIds) {
          const delta = getThreadUnreadCount(queryClient, keys, threadId);
          await patchThreadsOnRead(queryClient, keys, threadId);
          await patchUnreadOnRead(queryClient, keys, delta);
        }
        return;
      }
      void queryClient.invalidateQueries({ queryKey: [...HUB_KEY, "threads", companyId] });
      void queryClient.invalidateQueries({ queryKey: [...HUB_KEY, "unread", companyId] });
    },
  });
}

export type { HubChatMessage, MessageHubThread };
