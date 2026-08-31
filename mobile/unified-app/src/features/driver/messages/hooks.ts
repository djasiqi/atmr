import { useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import { useCallback, useMemo , useEffect, useRef, useState } from "react";
import { useSession } from "../../../core/sessionProvider";
import { QUERY_STALE_TIME_MS } from "../../../core/queryStaleTimes";
import { useActiveDriverContextId } from "../hooks";
import { driverQueryKeys } from "../queryKeys";
import { getDriverMissionEta, getDriverProfile , getDriverMessages } from "../api";

import { readDriverProfileCache } from "../services/driverProfileCache";
import {
  ackHubMessage,
  fetchHubColleagues,
  fetchHubUnreadCount,
  fetchConversationMessages,
  fetchMessageHubThreads,
  fetchThreadMessages,
  markThreadRead,
  reportMessageHubEmergency,
  resolveConversationId,
  sendHubMessage,
} from "./api";
import {
  buildLocalInboxThreads,
  filterMissionThreadsWithDiscussion,
  mergeInboxThreads,
} from "./buildLocalInbox";
import { dedupeMessageHubThreads } from "./dedupeHubThreads";
import { applyMobileInboxThreadPolicy } from "./inboxThreadPolicy";
import type { HubChatMessage, MessageHubThread , EmergencyIssueType, SyncPresenceStatus } from "./types";

import { realtimeManager } from "../../../core/realtime/realtimeManager";

import { emitPerfKpi } from "../../../core/observability/perfKpi";
import { traceInvalidateQueries } from "../../../core/observability/perfInstrumentation";
import { recordPerfBucket } from "../../../core/observability/perfInstrumentationStore";
import { recordChatCacheMismatch } from "../../../core/observability/perfMessageSend";
import { isPerfChatLocalPatchEnabled } from "../../messages/chatLocalPatchFlag";
import {
  appendMessageToThreadCache,
  buildDriverHubCacheKeys,
  getThreadUnreadCount,
  patchThreadsOnRead,
  patchThreadsOnReceive,
  patchUnreadOnRead,
  patchUnreadOnReceive,
  payloadToHubMessage,
} from "../../messages/messageHubCachePatch";
import {
  applyOptimisticSendError,
  applyOptimisticSendMutate,
  applyOptimisticSendSuccess,
  buildOptimisticHubMessage,
  trackMessageSendRetry,
} from "../../messages/hubSendMutationHelpers";
import type { PatchableHubMessage } from "../../messages/messageHubCachePatch";

const HUB_KEY = ["driver", "message-hub"] as const;
const THREAD_MESSAGES_STALE_MS = 20_000;
const THREAD_WARM_CACHE_TTL_MS = 60_000;
const THREAD_WARM_CACHE_TTL_COSTLY_MS = 90_000;
const THREAD_WARM_CACHE_MAX = 8;
const THREAD_PREFETCH_TOP_MAX = 5;
const THREAD_PREFETCH_COSTLY_TARGET = 3;
const THREAD_NEIGHBOR_PREFETCH_MAX = 3;
const THREAD_MISSION_BURST_PREFETCH_MAX = 3;
const THREAD_CONVERSATION_ID_TTL_MS = 10 * 60_000;

type ThreadWarmCacheEntry = {
  messages: HubChatMessage[];
  updatedAtMs: number;
  costly: boolean;
};

type ThreadConversationIdCacheEntry = {
  conversationId: number;
  updatedAtMs: number;
};

const threadWarmCache = new Map<string, ThreadWarmCacheEntry>();
const threadWarmLru: string[] = [];
const threadConversationIdCache = new Map<string, ThreadConversationIdCacheEntry>();

function isCostlyThreadId(threadId: string | null): boolean {
  if (!threadId) return false;
  return (
    threadId === "support" ||
    threadId.startsWith("mission:") ||
    threadId.startsWith("company_driver:")
  );
}

function buildDriverThreadMessagesKey(companyId: number | null, threadId: string | null): readonly unknown[] {
  return [...HUB_KEY, "messages", companyId ?? "none", threadId ?? "none"] as const;
}

function buildWarmThreadKey(companyId: number | null, threadId: string | null): string {
  return `${companyId ?? "none"}:${threadId ?? "none"}`;
}

function toNumericId(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function inferConversationIdFromMessages(messages: HubChatMessage[]): number | null {
  for (const message of messages) {
    const conversationId = toNumericId(
      (message as { conversation_id?: unknown }).conversation_id
    );
    if (conversationId != null) return conversationId;
  }
  return null;
}

function readCachedConversationId(companyId: number | null, threadId: string | null): number | null {
  const key = buildWarmThreadKey(companyId, threadId);
  const cached = threadConversationIdCache.get(key);
  if (!cached) return null;
  if (Date.now() - cached.updatedAtMs > THREAD_CONVERSATION_ID_TTL_MS) {
    threadConversationIdCache.delete(key);
    return null;
  }
  return cached.conversationId;
}

function storeCachedConversationId(
  companyId: number | null,
  threadId: string | null,
  conversationId: number | null
): void {
  if (!threadId || conversationId == null) return;
  const key = buildWarmThreadKey(companyId, threadId);
  threadConversationIdCache.set(key, {
    conversationId,
    updatedAtMs: Date.now(),
  });
}

function readWarmThreadMessages(companyId: number | null, threadId: string | null): HubChatMessage[] | null {
  const key = buildWarmThreadKey(companyId, threadId);
  const hit = threadWarmCache.get(key);
  if (!hit) return null;
  const ttlMs = hit.costly ? THREAD_WARM_CACHE_TTL_COSTLY_MS : THREAD_WARM_CACHE_TTL_MS;
  if (Date.now() - hit.updatedAtMs > ttlMs) {
    threadWarmCache.delete(key);
    const idx = threadWarmLru.indexOf(key);
    if (idx >= 0) threadWarmLru.splice(idx, 1);
    return null;
  }
  return hit.messages;
}

function storeWarmThreadMessages(companyId: number | null, threadId: string | null, messages: HubChatMessage[]): void {
  const key = buildWarmThreadKey(companyId, threadId);
  threadWarmCache.set(key, { messages, updatedAtMs: Date.now(), costly: isCostlyThreadId(threadId) });
  const idx = threadWarmLru.indexOf(key);
  if (idx >= 0) threadWarmLru.splice(idx, 1);
  threadWarmLru.unshift(key);
  while (threadWarmLru.length > THREAD_WARM_CACHE_MAX) {
    let evictIndex = -1;
    for (let i = threadWarmLru.length - 1; i >= 0; i -= 1) {
      const candidate = threadWarmLru[i];
      if (!candidate) continue;
      const entry = threadWarmCache.get(candidate);
      if (!entry?.costly) {
        evictIndex = i;
        break;
      }
    }
    if (evictIndex < 0) evictIndex = threadWarmLru.length - 1;
    const [evicted] = threadWarmLru.splice(evictIndex, 1);
    if (evicted) threadWarmCache.delete(evicted);
  }
}

function selectTopPrefetchThreads(threads: MessageHubThread[]): string[] {
  const byId = new Map<string, MessageHubThread>();
  for (const t of threads) byId.set(t.thread_id, t);
  const picked: string[] = [];
  const add = (id: string | null) => {
    if (!id || picked.includes(id) || !byId.has(id)) return;
    picked.push(id);
  };
  add("team");
  add("dispatch");
  add("support");
  let costlyPicked = 0;
  const costlyPriorityThreads = [
    ...threads.filter((t) => t.thread_id.startsWith("company_driver:")),
    ...threads.filter((t) => t.thread_id.startsWith("mission:")),
    ...threads.filter((t) => t.thread_id === "support"),
  ];
  for (const t of costlyPriorityThreads) {
    if (picked.length >= THREAD_PREFETCH_TOP_MAX) break;
    if (
      t.thread_id === "support" ||
      t.thread_id.startsWith("mission:") ||
      t.thread_id.startsWith("company_driver:")
    ) {
      const before = picked.length;
      add(t.thread_id);
      if (picked.length > before) costlyPicked += 1;
      if (costlyPicked >= THREAD_PREFETCH_COSTLY_TARGET) break;
    }
  }
  for (const t of threads) {
    if (picked.length >= THREAD_PREFETCH_TOP_MAX) break;
    add(t.thread_id);
  }
  return picked.slice(0, THREAD_PREFETCH_TOP_MAX);
}

function classifyThreadKind(threadId: string | null): "dispatch" | "team" | "mission" | "company_driver" | "support" | "other" {
  if (!threadId) return "other";
  if (threadId === "dispatch") return "dispatch";
  if (threadId === "team") return "team";
  if (threadId === "support") return "support";
  if (threadId.startsWith("mission:")) return "mission";
  if (threadId.startsWith("company_driver:")) return "company_driver";
  return "other";
}

function pickNeighborPrefetchThreads(
  threads: MessageHubThread[],
  activeThreadId: string,
  limit = THREAD_NEIGHBOR_PREFETCH_MAX
): string[] {
  if (threads.length === 0 || limit <= 0) return [];
  const picked: string[] = [];
  const add = (candidate: string | null) => {
    if (!candidate || candidate === activeThreadId || picked.includes(candidate)) return;
    picked.push(candidate);
  };
  const missionPeers = threads.filter(
    (t) => t.thread_id !== activeThreadId && t.thread_id.startsWith("mission:")
  );
  const companyDriverPeers = threads.filter(
    (t) => t.thread_id !== activeThreadId && t.thread_id.startsWith("company_driver:")
  );
  const supportPeers = threads.filter(
    (t) => t.thread_id !== activeThreadId && t.thread_id === "support"
  );
  const genericCostly = threads.filter(
    (t) =>
      t.thread_id !== activeThreadId &&
      (
        t.thread_id === "support" ||
        t.thread_id.startsWith("mission:") ||
        t.thread_id.startsWith("company_driver:")
      )
  );
  const isMissionActive = activeThreadId.startsWith("mission:");
  const isCompanyDriverActive = activeThreadId.startsWith("company_driver:");
  const isSupportActive = activeThreadId === "support";
  const priority = isMissionActive
    ? [...missionPeers, ...companyDriverPeers, ...supportPeers]
    : isCompanyDriverActive
      ? [...companyDriverPeers, ...supportPeers, ...missionPeers]
      : isSupportActive
        ? [...supportPeers, ...companyDriverPeers, ...missionPeers]
      : genericCostly;
  for (const t of priority) {
    if (picked.length >= limit) break;
    add(t.thread_id);
  }
  const idx = threads.findIndex((t) => t.thread_id === activeThreadId);
  if (picked.length < limit && idx >= 0) {
    add(threads[idx + 1]?.thread_id ?? null);
    if (picked.length < limit) add(threads[idx - 1]?.thread_id ?? null);
    if (picked.length < limit) add(threads[idx + 2]?.thread_id ?? null);
    if (picked.length < limit) add(threads[idx - 2]?.thread_id ?? null);
  }
  if (picked.length < limit) {
    for (const t of threads) {
      if (picked.length >= limit) break;
      add(t.thread_id);
    }
  }
  return picked.slice(0, limit);
}

function pickMissionBurstPrefetchThreads(
  threads: MessageHubThread[],
  activeThreadId: string,
  limit = THREAD_MISSION_BURST_PREFETCH_MAX
): string[] {
  if (!activeThreadId.startsWith("mission:") || threads.length === 0 || limit <= 0) return [];
  const picked: string[] = [];
  for (const t of threads) {
    if (picked.length >= limit) break;
    if (t.thread_id === activeThreadId) continue;
    if (!t.thread_id.startsWith("mission:")) continue;
    picked.push(t.thread_id);
  }
  return picked;
}

function getDriverThreadsSnapshot(
  queryClient: QueryClient,
  companyId: number | null
): MessageHubThread[] {
  if (!companyId) return [];
  const entries = queryClient.getQueriesData<{ threads: MessageHubThread[] }>({
    queryKey: [...HUB_KEY, "threads", companyId],
  });
  for (const [, data] of entries) {
    if (data?.threads?.length) return data.threads;
  }
  return [];
}

function prefetchWarmThreadIfNeeded(
  queryClient: QueryClient,
  companyId: number,
  threadId: string
): void {
  if (readWarmThreadMessages(companyId, threadId)) return;
  void queryClient.prefetchQuery({
    queryKey: buildDriverThreadMessagesKey(companyId, threadId),
    staleTime: THREAD_MESSAGES_STALE_MS,
    queryFn: async () => {
      const warmMessages = await fetchThreadMessages(companyId, threadId).catch(
        () => [] as HubChatMessage[]
      );
      storeWarmThreadMessages(companyId, threadId, warmMessages);
      return warmMessages;
    },
  });
}

function getCachedDriverMissions(
  queryClient: QueryClient,
  driverContextId: string | null
): Parameters<typeof buildLocalInboxThreads>[0] {
  if (!driverContextId) return [];
  return (
    (queryClient.getQueryData(driverQueryKeys.missions(driverContextId)) as
      | Parameters<typeof buildLocalInboxThreads>[0]
      | undefined) ?? []
  );
}

export function invalidateDriverHubScope(
  queryClient: QueryClient,
  companyId: number,
  opts?: { threadId?: string; includeUnread?: boolean; includeMessages?: boolean }
) {
  void queryClient.invalidateQueries({
    queryKey: [...HUB_KEY, "threads", companyId],
  });
  if (opts?.includeUnread !== false) {
    void queryClient.invalidateQueries({
      queryKey: [...HUB_KEY, "unread", companyId],
    });
  }
  if (opts?.includeMessages && opts.threadId) {
    void queryClient.invalidateQueries({
      queryKey: [...HUB_KEY, "messages", companyId, opts.threadId],
    });
  }
}

function legacyMessageMatchesThread(
  message: { thread_id?: string | null; sender_role?: string; receiver_id?: unknown },
  threadId: string
): boolean {
  const tid = message.thread_id;
  if (typeof tid === "string" && tid.length > 0) {
    return tid === threadId;
  }
  if (threadId.startsWith("direct:")) {
    return false;
  }
  if (threadId === "team") {
    return (
      String(message.sender_role ?? "").toUpperCase() === "DRIVER" &&
      message.receiver_id == null
    );
  }
  if (threadId === "dispatch") {
    if (tid === "dispatch") return true;
    if (message.receiver_id != null) return false;
    const role = String(message.sender_role ?? "").toUpperCase();
    return role === "COMPANY" || role === "DRIVER";
  }
  if (threadId === "support") {
    return false;
  }
  if (threadId.startsWith("mission:")) {
    return false;
  }
  return false;
}

function toCompanyId(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function normalizeNumericId(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function useDriverCompanyId(): number | null {
  const { activeContext, bootstrap } = useSession();
  const queryClient = useQueryClient();
  const driverContextId = useActiveDriverContextId();

  const profileCompanyQuery = useQuery({
    queryKey: ["driver", "hub-company-id", driverContextId ?? "none"],
    enabled: activeContext?.context_type === "driver" && Boolean(driverContextId),
    staleTime: 5 * 60_000,
    queryFn: async (): Promise<number | null> => {
      const cached = await readDriverProfileCache({ allowStale: true });
      const fromCache = toCompanyId(cached.profile?.company_id);
      if (fromCache != null) return fromCache;
      const profile = await getDriverProfile();
      return toCompanyId(profile.company_id);
    },
  });

  const driverContextOrgId = useMemo(() => {
    const ctx =
      activeContext?.context_type === "driver"
        ? activeContext
        : bootstrap?.available_contexts?.find((c) => c.context_type === "driver");
    return ctx ? toCompanyId(ctx.organization_id) : null;
  }, [activeContext, bootstrap?.available_contexts]);

  return useMemo(() => {
    const fromProfile = profileCompanyQuery.data;
    if (fromProfile != null) return fromProfile;

    if (activeContext?.context_type === "driver") {
      if (driverContextOrgId != null) return driverContextOrgId;
      const missions =
        (queryClient.getQueryData(driverQueryKeys.missions(activeContext.context_id)) as
          | Record<string, unknown>[]
          | undefined) ?? [];
      const missionCompany = missions
        .map((m) => toCompanyId((m as Record<string, unknown>).company_id))
        .find((v) => v != null);
      return missionCompany ?? null;
    }
    if (activeContext?.context_type === "company") {
      const companyOrg = toCompanyId(activeContext.organization_id);
      if (companyOrg != null) return companyOrg;
    }
    if (driverContextOrgId != null) return driverContextOrgId;
    return null;
  }, [
    activeContext?.context_type,
    activeContext?.context_id,
    activeContext?.organization_id,
    driverContextOrgId,
    profileCompanyQuery.data,
    queryClient,
  ]);
}

export function useMessageHubThreads(companyId: number | null) {
  const queryClient = useQueryClient();
  const driverContextId = useActiveDriverContextId();
  const { bootstrap } = useSession();
  const myUserId = useMemo(() => {
    const id = bootstrap?.user?.id;
    if (typeof id === "number" && Number.isFinite(id)) return id;
    if (typeof id === "string") {
      const parsed = Number.parseInt(id, 10);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }, [bootstrap?.user?.id]);

  const threadsPlaceholder = useMemo(
    () => ({
      threads: buildLocalInboxThreads(
        getCachedDriverMissions(queryClient, driverContextId),
        undefined,
        myUserId
      ),
      unread_total: 0,
    }),
    [queryClient, driverContextId, myUserId]
  );

  const threadsQuery = useQuery({
    queryKey: [...HUB_KEY, "threads", companyId ?? "none", myUserId],
    enabled: Boolean(companyId),
    queryFn: async (): Promise<{
      threads: MessageHubThread[];
      unread_total: number;
      fromFallback?: boolean;
    }> => {
      const missions = getCachedDriverMissions(queryClient, driverContextId);
      const localBase = buildLocalInboxThreads(missions, undefined, myUserId);

      try {
        const hub = await fetchMessageHubThreads(companyId as number);
        const merged = dedupeMessageHubThreads(
          applyMobileInboxThreadPolicy(
            filterMissionThreadsWithDiscussion(mergeInboxThreads(hub.threads, localBase))
          )
        );
        return {
          threads:
            merged.length > 0
              ? merged
              : applyMobileInboxThreadPolicy(filterMissionThreadsWithDiscussion(localBase)),
          unread_total: hub.unread_total,
          fromFallback: false,
        };
      } catch {
        const legacy = await getDriverMessages(companyId as number, { limit: 30 }).catch(() => []);
        const local = buildLocalInboxThreads(missions, legacy, myUserId);
        const unread = legacy.filter(
          (m) => String(m.sender_role ?? "").toUpperCase() !== "DRIVER"
        ).length;
        return {
          threads: applyMobileInboxThreadPolicy(filterMissionThreadsWithDiscussion(local)),
          unread_total: unread,
          fromFallback: true,
        };
      }
    },
    staleTime: 45_000,
    gcTime: 5 * 60_000,
    refetchOnWindowFocus: true,
    refetchInterval: false,
    placeholderData: (previous) => previous ?? threadsPlaceholder,
  });

  const prefetchedSignatureRef = useRef<string>("");
  useEffect(() => {
    if (!companyId) return;
    const threads = threadsQuery.data?.threads ?? [];
    if (threads.length === 0) return;
    const candidates = selectTopPrefetchThreads(threads);
    if (candidates.length === 0) return;
    const signature = `${companyId}:${candidates.join("|")}`;
    if (prefetchedSignatureRef.current === signature) return;
    prefetchedSignatureRef.current = signature;
    for (const candidateThreadId of candidates) {
      void queryClient.prefetchQuery({
        queryKey: buildDriverThreadMessagesKey(companyId, candidateThreadId),
        staleTime: THREAD_MESSAGES_STALE_MS,
        queryFn: async () => {
          const messages = await fetchThreadMessages(companyId, candidateThreadId).catch(
            () => [] as HubChatMessage[]
          );
          storeWarmThreadMessages(companyId, candidateThreadId, messages);
          return messages;
        },
      });
    }
  }, [companyId, queryClient, threadsQuery.data?.threads]);

  return threadsQuery;
}

export function useHubUnreadCount(companyId: number | null) {
  const queryClient = useQueryClient();
  // P1-C3 : hub driver — jamais de polling hors contexte chauffeur actif
  // (sinon poll 15 s permanent en contexte company, invisible pour la garde
  // /driver/me/* car l'endpoint est /messages/<cid>/hub/unread-count).
  const driverContextId = useActiveDriverContextId();
  const { bootstrap } = useSession();
  const myUserId = useMemo(() => {
    const id = bootstrap?.user?.id;
    if (typeof id === "number" && Number.isFinite(id)) return id;
    if (typeof id === "string") {
      const parsed = Number.parseInt(id, 10);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }, [bootstrap?.user?.id]);

  return useQuery({
    queryKey: [...HUB_KEY, "unread", companyId ?? "none"],
    queryFn: async () => {
      const server = await fetchHubUnreadCount(companyId as number);
      if (isPerfChatLocalPatchEnabled() && companyId != null) {
        const keys = buildDriverHubCacheKeys(companyId, myUserId);
        const local = queryClient.getQueryData<number>(keys.unread);
        if (local != null && local !== server) {
          recordChatCacheMismatch({
            kind: "unread_drift",
            role: "driver",
            details: {
              local,
              server,
              delta: Math.abs(local - server),
            },
          });
        }
      }
      return server;
    },
    enabled: Boolean(companyId && driverContextId),
    refetchInterval: () =>
      realtimeManager.isDriverSocketReady() ? false : 15_000,
    staleTime: QUERY_STALE_TIME_MS.default,
  });
}

export function useThreadMessages(
  companyId: number | null,
  threadId: string | null,
  conversationId?: number | null
) {
  const queryClient = useQueryClient();
  return useQuery({
    queryKey: buildDriverThreadMessagesKey(companyId, threadId),
    staleTime: THREAD_MESSAGES_STALE_MS,
    refetchOnWindowFocus: true,
    refetchInterval: false,
    placeholderData: (previous) => previous,
    queryFn: async () => {
      const queryStartedAt = Date.now();
      const queryKey = buildDriverThreadMessagesKey(companyId, threadId);
      const cached = queryClient.getQueryData<unknown[]>(queryKey);
      const warm = readWarmThreadMessages(companyId, threadId);
      const hasCached = Array.isArray(cached) && cached.length > 0;
      const hasWarm = Array.isArray(warm) && warm.length > 0;
      emitPerfKpi("perf.thread_runtime", {
        source: "driver.thread.query",
        phase: "perf.thread.query_start",
        role: "driver",
        thread_id: threadId,
      });
      emitPerfKpi("perf.thread_runtime", {
        source: "driver.thread.query",
        phase: hasCached || hasWarm ? "perf.thread.cache_hit" : "perf.thread.cache_miss",
        role: "driver",
        thread_id: threadId,
        thread_kind: classifyThreadKind(threadId),
        cached_count: hasCached ? cached.length : hasWarm ? warm.length : 0,
      });
      recordPerfBucket(
        "thread_cache",
        `${hasCached || hasWarm ? "hit" : "miss"}:${classifyThreadKind(threadId)}`,
        0,
        1
      );
      if (hasCached && Array.isArray(cached)) {
        const cachedMessages = cached as HubChatMessage[];
        const cachedConversationId = inferConversationIdFromMessages(cachedMessages);
        storeCachedConversationId(companyId, threadId, cachedConversationId);
        const queryDurationMs = Date.now() - queryStartedAt;
        recordPerfBucket("thread_runtime", "query_success", queryDurationMs);
        emitPerfKpi("perf.thread_runtime", {
          source: "driver.thread.query",
          phase: "perf.thread.query_success",
          role: "driver",
          thread_id: threadId,
          path: "query_cache",
          message_count: cachedMessages.length,
          duration_ms: queryDurationMs,
        });
        storeWarmThreadMessages(companyId, threadId, cachedMessages);
        if (companyId && threadId) {
          const threadsSnapshot = getDriverThreadsSnapshot(queryClient, companyId);
          const neighbors = pickNeighborPrefetchThreads(threadsSnapshot, threadId);
          for (const targetId of neighbors) {
            prefetchWarmThreadIfNeeded(queryClient, companyId, targetId);
          }
        }
        return cachedMessages;
      }
      if (hasWarm && warm) {
        const warmConversationId = inferConversationIdFromMessages(warm);
        storeCachedConversationId(companyId, threadId, warmConversationId);
        const queryDurationMs = Date.now() - queryStartedAt;
        recordPerfBucket("thread_runtime", "query_success", queryDurationMs);
        emitPerfKpi("perf.thread_runtime", {
          source: "driver.thread.query",
          phase: "perf.thread.query_success",
          role: "driver",
          thread_id: threadId,
          path: "warm_lru",
          message_count: warm.length,
          duration_ms: queryDurationMs,
        });
        return warm;
      }
      const loadLegacy = async () => {
        const normalizeStartedAt = Date.now();
        const legacy = await getDriverMessages(companyId as number, { limit: 80 }).catch(
          () => [] as Awaited<ReturnType<typeof getDriverMessages>>
        );
        const mapped = legacy.map((m) => ({
          id: m.id,
          sender_id: m.sender_id ?? null,
          receiver_id: m.receiver_id ?? null,
          content: m.content,
          sender_role: m.sender_role,
          sender_name: m.sender_name,
          timestamp: m.timestamp,
          image_url: m.image_url,
          pdf_url: m.pdf_url,
          pdf_filename: m.pdf_filename,
          audio_url: m.audio_url,
          thread_id:
            typeof (m as { thread_id?: string }).thread_id === "string"
              ? (m as { thread_id?: string }).thread_id
              : null,
          conversation_id:
            typeof (m as { conversation_id?: number }).conversation_id === "number"
              ? (m as { conversation_id?: number }).conversation_id
              : null,
          message_type: "text" as const,
          priority: "normal" as const,
        }));
        const filtered = mapped.filter((m) => legacyMessageMatchesThread(m, threadId as string));
        const legacyConversationId = inferConversationIdFromMessages(filtered);
        storeCachedConversationId(companyId, threadId, legacyConversationId);
        emitPerfKpi("perf.thread_runtime", {
          source: "driver.thread.query",
          phase: "perf.thread.normalize_done",
          role: "driver",
          thread_id: threadId,
          path: "legacy",
          row_count: legacy.length,
          message_count: filtered.length,
          duration_ms: Date.now() - normalizeStartedAt,
        });
        const queryDurationMs = Date.now() - queryStartedAt;
        recordPerfBucket("thread_runtime", "query_success", queryDurationMs);
        emitPerfKpi("perf.thread_runtime", {
          source: "driver.thread.query",
          phase: "perf.thread.query_success",
          role: "driver",
          thread_id: threadId,
          path: "legacy",
          message_count: filtered.length,
          duration_ms: queryDurationMs,
        });
        storeWarmThreadMessages(companyId, threadId, filtered);
        return filtered;
      };

      let resolvedConversationId = conversationId ?? null;
      if (resolvedConversationId == null) {
        resolvedConversationId = readCachedConversationId(companyId, threadId);
      }
      if (resolvedConversationId != null) {
        try {
          const viaConversation = await fetchConversationMessages(resolvedConversationId);
          if (viaConversation.length > 0) {
            storeCachedConversationId(companyId, threadId, resolvedConversationId);
            const queryDurationMs = Date.now() - queryStartedAt;
            recordPerfBucket("thread_runtime", "query_success", queryDurationMs);
            emitPerfKpi("perf.thread_runtime", {
              source: "driver.thread.query",
              phase: "perf.thread.query_success",
              role: "driver",
              thread_id: threadId,
              path: "conversation",
              conversation_id: resolvedConversationId,
              message_count: viaConversation.length,
              duration_ms: queryDurationMs,
            });
            storeWarmThreadMessages(companyId, threadId, viaConversation);
            if (companyId && threadId) {
              const threadsSnapshot = getDriverThreadsSnapshot(queryClient, companyId);
              const neighbors = pickNeighborPrefetchThreads(
                threadsSnapshot,
                threadId
              );
              const missionBurst = pickMissionBurstPrefetchThreads(threadsSnapshot, threadId);
              const targets = [...neighbors, ...missionBurst];
              for (const targetId of targets) {
                prefetchWarmThreadIfNeeded(queryClient, companyId, targetId);
              }
            }
            return viaConversation;
          }
        } catch {
          /* fallback hub / legacy */
        }
      }

      try {
        const hub = await fetchThreadMessages(companyId as number, threadId as string);
        if (hub.length > 0) {
          const hubConversationId = inferConversationIdFromMessages(hub);
          storeCachedConversationId(companyId, threadId, hubConversationId);
          const queryDurationMs = Date.now() - queryStartedAt;
          recordPerfBucket("thread_runtime", "query_success", queryDurationMs);
          emitPerfKpi("perf.thread_runtime", {
            source: "driver.thread.query",
            phase: "perf.thread.query_success",
            role: "driver",
            thread_id: threadId,
            path: "hub_thread",
            message_count: hub.length,
            duration_ms: queryDurationMs,
          });
          storeWarmThreadMessages(companyId, threadId, hub);
          if (companyId && threadId) {
            const threadsSnapshot = getDriverThreadsSnapshot(queryClient, companyId);
            const neighbors = pickNeighborPrefetchThreads(
              threadsSnapshot,
              threadId
            );
            const missionBurst = pickMissionBurstPrefetchThreads(threadsSnapshot, threadId);
            const targets = [...neighbors, ...missionBurst];
            for (const targetId of targets) {
              prefetchWarmThreadIfNeeded(queryClient, companyId, targetId);
            }
          }
          return hub;
        }
      } catch {
        /* fallback conversation resolve / legacy */
      }

      const isCanonicalThread =
        threadId === "dispatch" ||
        threadId === "team" ||
        threadId.startsWith("mission:") ||
        threadId.startsWith("company_driver:");
      const shouldResolveCanonicalThread =
        companyId != null &&
        threadId != null &&
        isCanonicalThread &&
        resolvedConversationId == null;
      if (shouldResolveCanonicalThread) {
        const canonical = await resolveConversationId(companyId, threadId).catch(() => null);
        if (canonical != null) {
          resolvedConversationId = canonical;
          storeCachedConversationId(companyId, threadId, canonical);
        }
      }
      if (shouldResolveCanonicalThread && resolvedConversationId != null) {
        try {
          const viaConversation = await fetchConversationMessages(resolvedConversationId);
          if (viaConversation.length > 0) {
            storeCachedConversationId(companyId, threadId, resolvedConversationId);
            const queryDurationMs = Date.now() - queryStartedAt;
            recordPerfBucket("thread_runtime", "query_success", queryDurationMs);
            emitPerfKpi("perf.thread_runtime", {
              source: "driver.thread.query",
              phase: "perf.thread.query_success",
              role: "driver",
              thread_id: threadId,
              path: "conversation",
              conversation_id: resolvedConversationId,
              message_count: viaConversation.length,
              duration_ms: queryDurationMs,
            });
            storeWarmThreadMessages(companyId, threadId, viaConversation);
            if (companyId && threadId) {
              const threadsSnapshot = getDriverThreadsSnapshot(queryClient, companyId);
              const neighbors = pickNeighborPrefetchThreads(
                threadsSnapshot,
                threadId
              );
              const missionBurst = pickMissionBurstPrefetchThreads(threadsSnapshot, threadId);
              const targets = [...neighbors, ...missionBurst];
              for (const targetId of targets) {
                prefetchWarmThreadIfNeeded(queryClient, companyId, targetId);
              }
            }
            return viaConversation;
          }
        } catch {
          /* fallback legacy */
        }
      }

      return loadLegacy();
    },
    enabled: Boolean(companyId && threadId),
  });
}

export function useMarkThreadRead(companyId: number | null) {
  const queryClient = useQueryClient();
  const { bootstrap } = useSession();
  const myUserId = useMemo(() => {
    const id = bootstrap?.user?.id;
    if (typeof id === "number" && Number.isFinite(id)) return id;
    if (typeof id === "string") {
      const parsed = Number.parseInt(id, 10);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }, [bootstrap?.user?.id]);

  return useMutation({
    mutationFn: (threadId: string) => markThreadRead(companyId as number, threadId),
    onSuccess: async (_data, threadId) => {
      if (!companyId) return;
      if (isPerfChatLocalPatchEnabled()) {
        const keys = buildDriverHubCacheKeys(companyId, myUserId);
        const delta = getThreadUnreadCount(queryClient, keys, threadId);
        await patchThreadsOnRead(queryClient, keys, threadId);
        await patchUnreadOnRead(queryClient, keys, delta);
        return;
      }
      invalidateDriverHubScope(queryClient, companyId, {
        threadId,
        includeMessages: true,
      });
    },
  });
}

export function useMarkAllInboxRead(companyId: number | null) {
  const queryClient = useQueryClient();
  const { bootstrap } = useSession();
  const myUserId = useMemo(() => {
    const id = bootstrap?.user?.id;
    if (typeof id === "number" && Number.isFinite(id)) return id;
    if (typeof id === "string") {
      const parsed = Number.parseInt(id, 10);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }, [bootstrap?.user?.id]);
  return useMutation({
    mutationFn: async (threadIds: string[]) => {
      if (!companyId) return;
      await Promise.all(
        threadIds.map((id) => markThreadRead(companyId, id).catch(() => undefined))
      );
    },
    onSuccess: async (_data, threadIds) => {
      if (!companyId) return;
      if (isPerfChatLocalPatchEnabled()) {
        const keys = buildDriverHubCacheKeys(companyId, myUserId);
        for (const threadId of threadIds) {
          const delta = getThreadUnreadCount(queryClient, keys, threadId);
          await patchThreadsOnRead(queryClient, keys, threadId);
          await patchUnreadOnRead(queryClient, keys, delta);
        }
        return;
      }
      invalidateDriverHubScope(queryClient, companyId);
    },
  });
}

export function useAckMessage(companyId: number | null) {
  return useMutation({
    mutationFn: (messageId: number) => ackHubMessage(companyId as number, messageId),
  });
}

export function useReportEmergency(companyId: number | null, threadId?: string | null) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      issue_type: EmergencyIssueType;
      booking_id?: number | null;
      note?: string | null;
    }) => reportMessageHubEmergency(body),
    onSuccess: (result) => {
      const resolvedCompany =
        toCompanyId((result as Record<string, unknown>).company_id) ?? companyId;
      if (!resolvedCompany) return;
      invalidateDriverHubScope(queryClient, resolvedCompany, {
        threadId: threadId ?? undefined,
        includeMessages: Boolean(threadId),
      });
      void queryClient.invalidateQueries({
        queryKey: [...HUB_KEY, "messages"],
      });
    },
  });
}

export { groupThreadsBySection } from "./groupThreads";

export function useSyncPresenceStatus(httpReachable?: boolean): SyncPresenceStatus {
  const [connected, setConnected] = useState(() => realtimeManager.isDriverSocketReady());
  const [connecting, setConnecting] = useState(false);

  useEffect(() => {
    return realtimeManager.subscribe((snapshot) => {
      setConnected(realtimeManager.isDriverSocketReady());
      setConnecting(
        snapshot.desiredTransport === "socket" &&
          !snapshot.authExhausted &&
          !realtimeManager.isDriverSocketReady()
      );
    });
  }, []);

  if (connected) return "connected";
  if (httpReachable) return "connected";
  if (connecting) return "slow";
  return "offline";
}

export function useMissionEtaMinutes(bookingId: number | null | undefined, status?: string | null) {
  // P1-C3 : /driver/me/bookings/eta ne doit jamais poller hors contexte chauffeur.
  const driverContextId = useActiveDriverContextId();
  return useQuery({
    queryKey: [...HUB_KEY, "eta", bookingId ?? "none"],
    queryFn: () => getDriverMissionEta(bookingId as number, { missionStatus: status }),
    enabled: Boolean(bookingId && driverContextId),
    refetchInterval: () =>
      realtimeManager.isDriverSocketReady() ? false : 20_000,
    staleTime: QUERY_STALE_TIME_MS.default,
  });
}

export function useTeamMemberCount(companyId: number | null) {
  return useQuery({
    queryKey: [...HUB_KEY, "team-member-count", companyId ?? "none"],
    queryFn: async () => {
      const { colleagues, team_member_count } = await fetchHubColleagues(companyId as number);
      if (team_member_count != null && team_member_count > 0) {
        return team_member_count;
      }
      return colleagues.length + 1;
    },
    enabled: Boolean(companyId),
    staleTime: 60_000,
    retry: 2,
    placeholderData: (previous) => previous,
  });
}

export function useDriverMessageHubUnreadBadge(): number {
  const companyId = useDriverCompanyId();
  const hubUnread = useHubUnreadCount(companyId);
  return hubUnread.data ?? 0;
}

export function useEnsureDriverSocketForHub() {
  const driverContextId = useActiveDriverContextId();
  useEffect(() => {
    if (driverContextId) realtimeManager.ensureDriverSocket(driverContextId);
  }, [driverContextId]);
}

/** Rafraîchit l’inbox quand un message équipe / conversation arrive en socket. */
export function useHubMessageRealtimeSync(
  companyId: number | null,
  options?: { excludeThreadId?: string | null }
) {
  const queryClient = useQueryClient();
  const { bootstrap } = useSession();
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
    let debounceId: ReturnType<typeof setTimeout> | null = null;
    return realtimeManager.subscribeTeamChatEvents((event) => {
      if (event.type !== "team_chat_message" || !companyId) return;
      if (debounceId) clearTimeout(debounceId);
      debounceId = setTimeout(() => {
        debounceId = null;
        const payload = event.payload as { thread_id?: string } | null;
        const threadId =
          typeof payload?.thread_id === "string" ? payload.thread_id : null;
        if (isPerfChatLocalPatchEnabled()) {
          const keys = buildDriverHubCacheKeys(companyId, myUserId);
          if (threadId) {
            const message = payloadToHubMessage(payload);
            const senderId = normalizeNumericId((payload as { sender_id?: unknown })?.sender_id);
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
          "hub_realtime",
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
            "hub_realtime",
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
            "hub_realtime",
            () =>
              queryClient.invalidateQueries({
                queryKey: [...HUB_KEY, "messages", companyId, threadId],
                refetchType: "active",
              })
          );
        }
      }, 1800);
    });
  }, [companyId, excludeThreadId, myUserId, queryClient]);
}

export function useDriverSendHubMessage(
  companyId: number | null,
  threadId: string,
  options?: {
    senderId?: string | number | null;
    senderName?: string;
    conversationId?: number | null;
  }
) {
  const queryClient = useQueryClient();
  const { bootstrap } = useSession();
  const myUserId = useMemo(() => {
    const id = bootstrap?.user?.id;
    if (typeof id === "number" && Number.isFinite(id)) return id;
    if (typeof id === "string") {
      const parsed = Number.parseInt(id, 10);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }, [bootstrap?.user?.id]);

  return useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      sendHubMessage(companyId as number, threadId, body),
    onMutate: async (body) => {
      if (!companyId || !isPerfChatLocalPatchEnabled()) return undefined;
      const keys = buildDriverHubCacheKeys(companyId, myUserId);
      const clientId =
        typeof body._localId === "string"
          ? body._localId
          : `local-${Date.now()}`;
      const optimistic = buildOptimisticHubMessage({
        clientId,
        threadId,
        senderId: options?.senderId ?? bootstrap?.user?.id ?? null,
        senderName: options?.senderName ?? "Chauffeur",
        senderRole: "DRIVER",
        content:
          typeof body.content === "string" ? body.content : "Message",
        conversationId: options?.conversationId ?? null,
        extra: body as Partial<PatchableHubMessage>,
      });
      const ctx = await applyOptimisticSendMutate({
        qc: queryClient,
        keys,
        threadId,
        optimistic,
        conversationId: options?.conversationId,
        role: "driver",
      });
      return ctx;
    },
    onSuccess: async (serverMessage, _body, context) => {
      if (!companyId) return;
      if (isPerfChatLocalPatchEnabled() && context) {
        const keys = buildDriverHubCacheKeys(companyId, myUserId);
        await applyOptimisticSendSuccess({
          qc: queryClient,
          keys,
          threadId,
          context,
          serverMessage: serverMessage as PatchableHubMessage,
          conversationId: options?.conversationId,
          role: "driver",
        });
        return;
      }
      invalidateDriverHubScope(queryClient, companyId, {
        threadId,
        includeMessages: true,
      });
    },
    onError: async (_error, body, context) => {
      if (!companyId || !isPerfChatLocalPatchEnabled()) return;
      const keys = buildDriverHubCacheKeys(companyId, myUserId);
      await applyOptimisticSendError({
        qc: queryClient,
        keys,
        threadId,
        context,
        reason: "send_error",
        conversationId: options?.conversationId,
        role: "driver",
      });
      if (body && typeof body._retry === "boolean" && body._retry) {
        const clientId =
          typeof body._localId === "string" ? body._localId : threadId;
        trackMessageSendRetry({
          role: "driver",
          threadId,
          clientId,
        });
      }
    },
  });
}

export function useInvalidateMessageHub() {
  const queryClient = useQueryClient();
  const companyId = useDriverCompanyId();
  return useCallback(() => {
    if (companyId) {
      invalidateDriverHubScope(queryClient, companyId);
      return;
    }
    void queryClient.invalidateQueries({ queryKey: HUB_KEY });
  }, [companyId, queryClient]);
}
