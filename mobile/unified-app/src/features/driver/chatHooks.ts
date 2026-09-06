import AsyncStorage from "@react-native-async-storage/async-storage";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo } from "react";
import { useSession } from "../../core/sessionProvider";
import { QUERY_STALE_TIME_MS } from "../../core/queryStaleTimes";
import { realtimeManager } from "../../core/realtime/realtimeManager";
import { useActiveDriverContextId } from "./hooks";
import { getDriverMessages, type DriverChatMessage } from "./api";
import { useDriverSessionNetworkReady } from "./sessionNetworkGate";

const DRIVER_CHAT_KEY = ["driver", "chat"] as const;

function toEpoch(input: string | null | undefined): number {
  if (!input) return 0;
  const value = Date.parse(input);
  return Number.isFinite(value) ? value : 0;
}

function getLastReadKey(contextId: string) {
  return `driver-chat-last-read:${contextId}`;
}

export function useDriverChatMessages(companyId: number | null, contextId: string | null) {
  const networkReady = useDriverSessionNetworkReady();
  return useQuery({
    queryKey: [...DRIVER_CHAT_KEY, contextId ?? "none", companyId ?? "none"],
    queryFn: () => getDriverMessages(companyId as number, { limit: 40 }),
    enabled: networkReady && Boolean(companyId && contextId),
    refetchInterval: () =>
      realtimeManager.isDriverSocketReady() ? false : 10_000,
    staleTime: QUERY_STALE_TIME_MS.default,
  });
}

export function useUnreadMessages(
  companyId: number | null,
  contextId: string | null,
  messages: DriverChatMessage[] | undefined
) {
  const queryClient = useQueryClient();
  const lastReadQuery = useQuery({
    queryKey: [...DRIVER_CHAT_KEY, "last-read", contextId ?? "none"],
    queryFn: async () => {
      if (!contextId) return null;
      return AsyncStorage.getItem(getLastReadKey(contextId));
    },
    enabled: Boolean(contextId),
    staleTime: QUERY_STALE_TIME_MS.default,
  });

  const unreadCount = (() => {
    if (!messages || messages.length === 0) return 0;
    const lastRead = toEpoch(lastReadQuery.data);
    if (!lastRead) return messages.length;
    return messages.filter((message) => toEpoch(message.timestamp) > lastRead).length;
  })();

  const markReadMutation = useMutation({
    mutationFn: async (timestamp: string) => {
      if (!contextId) return;
      await AsyncStorage.setItem(getLastReadKey(contextId), timestamp);
    },
    onSuccess: (_, timestamp) => {
      if (!contextId) return;
      queryClient.setQueryData([...DRIVER_CHAT_KEY, "last-read", contextId], timestamp);
    },
  });

  const markRead = useCallback(
    async (timestamp?: string) => {
      const latest = timestamp ?? messages?.[messages.length - 1]?.timestamp;
      if (!latest) return;
      await markReadMutation.mutateAsync(latest);
    },
    [markReadMutation, messages]
  );

  return useMemo(
    () => ({
      unreadCount,
      /** Dernière horodatage de lecture (AsyncStorage), pour l’ancre de scroll. */
      lastReadAt: lastReadQuery.data ?? null,
      isLoadingLastRead: lastReadQuery.isLoading,
      markRead,
    }),
    [lastReadQuery.data, lastReadQuery.isLoading, markRead, unreadCount]
  );
}

/** Badge chat pour la barre d’onglets driver (même logique que `useUnreadMessages` sur l’accueil). */
export function useDriverChatUnreadCount(): number {
  const { activeContext } = useSession();
  const driverContextId = useActiveDriverContextId();

  const companyId = useMemo((): number | null => {
    const raw = activeContext?.organization_id;
    if (typeof raw === "number" && Number.isFinite(raw)) return raw;
    if (typeof raw === "string") {
      const parsed = Number.parseInt(raw, 10);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }, [activeContext?.organization_id]);

  const messagesQuery = useDriverChatMessages(companyId, driverContextId);
  const { unreadCount } = useUnreadMessages(companyId, driverContextId, messagesQuery.data);
  return unreadCount;
}
