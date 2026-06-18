import { useCallback, useEffect, useMemo, useState } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  QueryClient,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { QUERY_STALE_TIME_MS } from "../../core/queryStaleTimes";
import { useSession } from "../../core/sessionProvider";
import { contextScopedKey } from "../../core/cache/contextCache";
import {
  assignCompanyRide,
  cancelCompanyRide,
  getDispatchStatus,
  markCompanyRideUrgent,
  reassignCompanyRide,
  scheduleCompanyRide,
  getCompanyRideDetail,
  getCompanyClientDetail,
  getCompanyClients,
  getCompanyDispatchDelays,
  getDispatchMissions,
  getDriversLocationsSnapshot,
  getCompanyInvoices,
  getOptimizerStatus,
  getRealtimeDashboard,
  getCompanyDispatchMessages,
  sendCompanyDispatchMessage,
  transferCompanyRide,
} from "./api/companyApi";
import {
  getCompanyInboxNotifications,
  markAllCompanyNotificationsRead,
  markCompanyNotificationRead,
} from "./api/companyInboxApi";
import { companyContextScope, companyQueryKeys } from "./companyQueryKeys";
import { companyRealtimeBridge } from "./realtime/companyRealtimeBridge";
import {
  CompanyRealtimeSnapshot,
  getCompanyRealtimePollingIntervalMs,
} from "./realtime/companyRealtimeState";
import { SharedChatMessage } from "../chat";
import { isFeatureEnabled } from "../../core/featureFlags/registry";
import { resolveMediaUrl } from "../../core/api/mediaUrl";
import { AxiosError } from "axios";
import { traceInvalidateQueries } from "../../core/observability/perfInstrumentation";

const DRIVER_LOCATION_INVALIDATION_BATCH_MS = 500;
const driverLocationInvalidationTimers = new Map<
  string,
  ReturnType<typeof setTimeout>
>();

export function resetDriverLocationInvalidationBatchForTests(): void {
  for (const timer of driverLocationInvalidationTimers.values()) {
    clearTimeout(timer);
  }
  driverLocationInvalidationTimers.clear();
}

function scheduleDriversLocationsInvalidation(
  queryClient: QueryClient,
  contextId: string
): void {
  const existing = driverLocationInvalidationTimers.get(contextId);
  if (existing) clearTimeout(existing);
  const timer = setTimeout(() => {
    driverLocationInvalidationTimers.delete(contextId);
    const key = contextScopedKey(contextId, [
      ...companyQueryKeys.driversLocations(contextId),
    ] as unknown[]);
    void traceInvalidateQueries(key, "driver_location_update_batch", async () => {
      await queryClient.invalidateQueries({ queryKey: key });
    });
  }, DRIVER_LOCATION_INVALIDATION_BATCH_MS);
  driverLocationInvalidationTimers.set(contextId, timer);
}

export type CompanyRealtimeInvalidationEvent =
  | "booking_created"
  | "booking_updated"
  | "booking_cancelled"
  | "driver_location_update"
  | "optimizer_status_changed"
  | "delay_invalidated"
  | "booking_message_sent"
  | "urgent_alert"
  // Phase 2 PR B/C — gate D3.1
  | "dispatch_assignment"
  | "dispatch_run_lifecycle";

type CompanyEventContext = {
  contextId: string;
  missionId?: number;
};

const invalidationDedupMap = new Map<string, number>();
const EVENT_DEDUP_WINDOW_MS = 1_500;

export function resetCompanyInvalidationDedupStateForTests() {
  invalidationDedupMap.clear();
}

function shouldSkipDuplicateInvalidation(
  event: CompanyRealtimeInvalidationEvent,
  context: CompanyEventContext
): boolean {
  const key = `${context.contextId}|${event}|${context.missionId ?? "all"}`;
  const now = Date.now();
  const previous = invalidationDedupMap.get(key);
  invalidationDedupMap.set(key, now);
  if (!previous) return false;
  return now - previous < EVENT_DEDUP_WINDOW_MS;
}

export function useActiveCompanyContextId(): string | null {
  const { activeContext } = useSession();
  if (!activeContext || activeContext.context_type !== "company") return null;
  return activeContext.context_id;
}

export function useCompanyDispatchMissionsQuery(params: { date: string; search?: string; status?: string }) {
  const contextId = useActiveCompanyContextId();
  const search = params.search ?? "";
  const status = params.status ?? "all";
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(
          contextId,
          [...companyQueryKeys.missions(contextId, params.date, search, status)] as unknown[]
        )
      : ["company", "dispatch", "missions", "disabled"],
    queryFn: () =>
      getDispatchMissions({
        contextId: contextId as string,
        date: params.date,
        search,
        status: status === "all" ? undefined : status,
      }),
    enabled: Boolean(contextId),
    staleTime: QUERY_STALE_TIME_MS.companyDetail,
  });
}

/** Retards dispatch : même logique que le web (`delays/live` si dispo, sinon `/delays` + fusion minutes). */
export function useCompanyDispatchDelaysQuery(params: { date: string }) {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(contextId, [
          ...companyQueryKeys.dispatchDelays(contextId, params.date),
        ] as unknown[])
      : ["company", "dispatch", "dispatch-delays", "disabled"],
    queryFn: () =>
      getCompanyDispatchDelays({
        contextId: contextId as string,
        date: params.date,
      }),
    enabled: Boolean(contextId),
    staleTime: 20_000,
  });
}

export function useCompanyInboxQuery() {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(contextId, [...companyQueryKeys.inbox(contextId)] as unknown[])
      : ["company", "inbox", "disabled"],
    queryFn: () => getCompanyInboxNotifications({ limit: 30 }),
    enabled: Boolean(contextId),
    staleTime: QUERY_STALE_TIME_MS.companyList,
  });
}

export function useCompanyInboxReadMutation() {
  const contextId = useActiveCompanyContextId();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (notificationId: number) => markCompanyNotificationRead(notificationId),
    onSuccess: () => {
      if (contextId) {
        void queryClient.invalidateQueries({
          queryKey: contextScopedKey(contextId, [...companyQueryKeys.inbox(contextId)] as unknown[]),
        });
      }
    },
  });
}

export function useCompanyInboxReadAllMutation() {
  const contextId = useActiveCompanyContextId();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => markAllCompanyNotificationsRead(),
    onSuccess: () => {
      if (contextId) {
        void queryClient.invalidateQueries({
          queryKey: contextScopedKey(contextId, [...companyQueryKeys.inbox(contextId)] as unknown[]),
        });
      }
    },
  });
}

export function useCompanyDashboardQuery(date: string) {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(
          contextId,
          [...companyQueryKeys.dashboard(contextId), date] as unknown[]
        )
      : ["company", "dispatch", "dashboard", "disabled"],
    queryFn: () =>
      getRealtimeDashboard({
        contextId: contextId as string,
        date,
      }),
    enabled: Boolean(contextId),
    staleTime: QUERY_STALE_TIME_MS.companyList,
  });
}

export function useCompanyOptimizerStatusQuery() {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(contextId, [...companyQueryKeys.optimizer(contextId)] as unknown[])
      : ["company", "dispatch", "optimizer", "disabled"],
    queryFn: () =>
      getOptimizerStatus({
        contextId: contextId as string,
      }),
    enabled: Boolean(contextId),
    staleTime: QUERY_STALE_TIME_MS.companySlow,
  });
}

export function useCompanyDispatchStatusQuery(date?: string) {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(
          contextId,
          [...companyQueryKeys.root, "dispatch-status", contextId, date ?? "none"] as unknown[]
        )
      : ["company", "dispatch", "dispatch-status", "disabled"],
    queryFn: () =>
      getDispatchStatus({
        contextId: contextId as string,
        date,
      }),
    enabled: Boolean(contextId),
    staleTime: QUERY_STALE_TIME_MS.companyDetail,
  });
}

export function useCompanyDriversLocationsSnapshotQuery() {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(contextId, [...companyQueryKeys.driversLocations(contextId)] as unknown[])
      : ["company", "dispatch", "drivers", "locations", "disabled"],
    queryFn: () =>
      getDriversLocationsSnapshot({
        contextId: contextId as string,
      }),
    enabled: Boolean(contextId),
    staleTime: QUERY_STALE_TIME_MS.companyDetail,
  });
}

export function useCompanyRideDetailsQuery(params: { date: string; rideId: number | null }) {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey:
      contextId && params.rideId != null
        ? contextScopedKey(
            contextId,
            [...companyQueryKeys.rideDetails(contextId, params.rideId)] as unknown[]
          )
        : ["company", "dispatch", "ride-details", "disabled"],
    queryFn: async () => {
      const detailPayload = await getCompanyRideDetail({
        contextId: contextId as string,
        missionId: params.rideId as number,
        date: params.date,
      });
      if (detailPayload && typeof detailPayload === "object") {
        const payload = detailPayload as Record<string, unknown>;
        const directCandidate = payload.summary ?? payload.data ?? payload.item ?? payload.ride ?? payload.mission;
        if (directCandidate && typeof directCandidate === "object") {
          const directMission = directCandidate as Record<string, unknown>;
          const directMissionId = Number.parseInt(
            String(directMission.mission_id ?? directMission.booking_id ?? directMission.id ?? "NaN"),
            10
          );
          if (Number.isFinite(directMissionId) && directMissionId === params.rideId) {
            return directMission;
          }
        }
        const rowsCandidate =
          (Array.isArray(payload.items) && payload.items) ||
          (Array.isArray(payload.missions) && payload.missions) ||
          (Array.isArray(payload.data) && payload.data) ||
          [];
        if (Array.isArray(rowsCandidate)) {
          const match = rowsCandidate.find((entry) => {
            if (!entry || typeof entry !== "object") return false;
            const row = entry as Record<string, unknown>;
            const missionId = Number.parseInt(
              String(row.mission_id ?? row.booking_id ?? row.id ?? "NaN"),
              10
            );
            return Number.isFinite(missionId) && missionId === params.rideId;
          });
          if (match) return match;
        }
      }
      return null;
    },
    enabled: Boolean(contextId) && params.rideId != null,
    staleTime: QUERY_STALE_TIME_MS.companyDetail,
  });
}

export function invalidateCompanyQueriesForEvent(
  queryClient: QueryClient,
  event: CompanyRealtimeInvalidationEvent,
  context: CompanyEventContext
) {
  if (shouldSkipDuplicateInvalidation(event, context)) {
    return;
  }
  const scope = companyContextScope(context.contextId);

  const invalidateMissionsDashboardInbox = () => {
    void queryClient.invalidateQueries({
      queryKey: contextScopedKey(
        context.contextId,
        [...companyQueryKeys.dashboard(context.contextId)] as unknown[]
      ),
      exact: true,
    });
    void queryClient.invalidateQueries({
      queryKey: contextScopedKey(
        context.contextId,
        [...companyQueryKeys.root, "missions", scope] as unknown[]
      ),
      exact: false,
    });
    void queryClient.invalidateQueries({
      queryKey: contextScopedKey(
        context.contextId,
        [...companyQueryKeys.inbox(context.contextId)] as unknown[]
      ),
      exact: false,
    });
  };

  if (event === "booking_created" || event === "urgent_alert") {
    invalidateMissionsDashboardInbox();
    return;
  }

  if (event === "booking_updated") {
    if (typeof context.missionId === "number") {
      void traceInvalidateQueries(
        contextScopedKey(
          context.contextId,
          [...companyQueryKeys.rideDetails(context.contextId, context.missionId)] as unknown[]
        ),
        "booking_updated_mission_detail",
        async () => {
          await queryClient.invalidateQueries({
            queryKey: contextScopedKey(
              context.contextId,
              [...companyQueryKeys.rideDetails(context.contextId, context.missionId)] as unknown[]
            ),
            exact: true,
          });
        }
      );
      void traceInvalidateQueries(
        contextScopedKey(context.contextId, [...companyQueryKeys.dashboard(context.contextId)]),
        "booking_updated_mission_dashboard",
        async () => {
          await queryClient.invalidateQueries({
            queryKey: contextScopedKey(
              context.contextId,
              [...companyQueryKeys.dashboard(context.contextId)] as unknown[]
            ),
            exact: true,
          });
        }
      );
      return;
    }

    void traceInvalidateQueries(
      contextScopedKey(context.contextId, [...companyQueryKeys.dashboard(context.contextId)]),
      "booking_updated",
      async () => {
        await queryClient.invalidateQueries({
          queryKey: contextScopedKey(
            context.contextId,
            [...companyQueryKeys.dashboard(context.contextId)] as unknown[]
          ),
          exact: true,
        });
      }
    );
    void traceInvalidateQueries(
      contextScopedKey(
        context.contextId,
        [...companyQueryKeys.root, "missions", scope] as unknown[]
      ),
      "booking_updated_missions",
      async () => {
        await queryClient.invalidateQueries({
          queryKey: contextScopedKey(
            context.contextId,
            [...companyQueryKeys.root, "missions", scope] as unknown[]
          ),
          exact: false,
        });
      }
    );
    void traceInvalidateQueries(
      contextScopedKey(
        context.contextId,
        [...companyQueryKeys.root, "dispatch-delays", scope] as unknown[]
      ),
      "booking_updated_delays",
      async () => {
        await queryClient.invalidateQueries({
          queryKey: contextScopedKey(
            context.contextId,
            [...companyQueryKeys.root, "dispatch-delays", scope] as unknown[]
          ),
          exact: false,
        });
      }
    );
    void traceInvalidateQueries(
      contextScopedKey(
        context.contextId,
        [...companyQueryKeys.optimizer(context.contextId)] as unknown[]
      ),
      "booking_updated_optimizer",
      async () => {
        await queryClient.invalidateQueries({
          queryKey: contextScopedKey(
            context.contextId,
            [...companyQueryKeys.optimizer(context.contextId)] as unknown[]
          ),
          exact: true,
        });
      }
    );
    return;
  }
  if (event === "booking_cancelled") {
    void queryClient.invalidateQueries({
      queryKey: contextScopedKey(
        context.contextId,
        [...companyQueryKeys.root, "missions", scope] as unknown[]
      ),
      exact: false,
    });
    return;
  }
  if (event === "driver_location_update") {
    scheduleDriversLocationsInvalidation(queryClient, context.contextId);
    return;
  }
  if (event === "optimizer_status_changed") {
    void queryClient.invalidateQueries({
      queryKey: contextScopedKey(
        context.contextId,
        [...companyQueryKeys.optimizer(context.contextId)] as unknown[]
      ),
    });
    return;
  }
  if (event === "delay_invalidated") {
    void traceInvalidateQueries(
      contextScopedKey(
        context.contextId,
        [...companyQueryKeys.dashboard(context.contextId)] as unknown[]
      ),
      "delay_invalidated_dashboard",
      async () => {
        await queryClient.invalidateQueries({
          queryKey: contextScopedKey(
            context.contextId,
            [...companyQueryKeys.dashboard(context.contextId)] as unknown[]
          ),
          exact: true,
        });
      }
    );
    void traceInvalidateQueries(
      contextScopedKey(
        context.contextId,
        [...companyQueryKeys.root, "dispatch-delays", scope] as unknown[]
      ),
      "delay_invalidated_delays",
      async () => {
        await queryClient.invalidateQueries({
          queryKey: contextScopedKey(
            context.contextId,
            [...companyQueryKeys.root, "dispatch-delays", scope] as unknown[]
          ),
          exact: false,
        });
      }
    );
    return;
  }
  if (event === "booking_message_sent") {
    void queryClient.invalidateQueries({
      queryKey: contextScopedKey(
        context.contextId,
        [...companyQueryKeys.root, "chat", context.contextId] as unknown[]
      ),
      exact: false,
    });
    return;
  }
  // Phase 2 PR B/C — gate D3.1 : dispatch_assignment touche dashboard + missions
  // (et ride detail si on connaît le missionId). On ne refetch pas l'optimizer
  // pour rester ciblé et éviter le surfetch identifié dans l'audit.
  if (event === "dispatch_assignment") {
    void traceInvalidateQueries(
      contextScopedKey(
        context.contextId,
        [...companyQueryKeys.dashboard(context.contextId)] as unknown[]
      ),
      "dispatch_assignment_dashboard",
      async () => {
        await queryClient.invalidateQueries({
          queryKey: contextScopedKey(
            context.contextId,
            [...companyQueryKeys.dashboard(context.contextId)] as unknown[]
          ),
          exact: true,
        });
      }
    );
    void traceInvalidateQueries(
      contextScopedKey(
        context.contextId,
        [...companyQueryKeys.root, "missions", scope] as unknown[]
      ),
      "dispatch_assignment_missions",
      async () => {
        await queryClient.invalidateQueries({
          queryKey: contextScopedKey(
            context.contextId,
            [...companyQueryKeys.root, "missions", scope] as unknown[]
          ),
          exact: false,
        });
      }
    );
    if (typeof context.missionId === "number") {
      void traceInvalidateQueries(
        contextScopedKey(
          context.contextId,
          [...companyQueryKeys.rideDetails(context.contextId, context.missionId)] as unknown[]
        ),
        "dispatch_assignment_ride_detail",
        async () => {
          await queryClient.invalidateQueries({
            queryKey: contextScopedKey(
              context.contextId,
              [...companyQueryKeys.rideDetails(context.contextId, context.missionId as number)] as unknown[]
            ),
            exact: true,
          });
        }
      );
    }
    return;
  }
  // Phase 2 PR B/C — gate D3.1 : dispatch_run_lifecycle (started/completed/failed)
  // touche dashboard + missions + dispatch-delays (impact sur l'ETA agrégé).
  if (event === "dispatch_run_lifecycle") {
    void traceInvalidateQueries(
      contextScopedKey(
        context.contextId,
        [...companyQueryKeys.dashboard(context.contextId)] as unknown[]
      ),
      "dispatch_run_lifecycle_dashboard",
      async () => {
        await queryClient.invalidateQueries({
          queryKey: contextScopedKey(
            context.contextId,
            [...companyQueryKeys.dashboard(context.contextId)] as unknown[]
          ),
          exact: true,
        });
      }
    );
    void traceInvalidateQueries(
      contextScopedKey(
        context.contextId,
        [...companyQueryKeys.root, "missions", scope] as unknown[]
      ),
      "dispatch_run_lifecycle_missions",
      async () => {
        await queryClient.invalidateQueries({
          queryKey: contextScopedKey(
            context.contextId,
            [...companyQueryKeys.root, "missions", scope] as unknown[]
          ),
          exact: false,
        });
      }
    );
    void traceInvalidateQueries(
      contextScopedKey(
        context.contextId,
        [...companyQueryKeys.root, "dispatch-delays", scope] as unknown[]
      ),
      "dispatch_run_lifecycle_delays",
      async () => {
        await queryClient.invalidateQueries({
          queryKey: contextScopedKey(
            context.contextId,
            [...companyQueryKeys.root, "dispatch-delays", scope] as unknown[]
          ),
          exact: false,
        });
      }
    );
  }
}

export function useCompanyRealtimeInvalidation() {
  const queryClient = useQueryClient();
  const contextId = useActiveCompanyContextId();

  return useMemo(
    () => ({
      invalidate: (event: CompanyRealtimeInvalidationEvent, missionId?: number) => {
        if (!contextId) return;
        invalidateCompanyQueriesForEvent(queryClient, event, {
          contextId,
          missionId,
        });
      },
    }),
    [contextId, queryClient]
  );
}

export function useCompanyRealtimeStatus() {
  const [snapshot, setSnapshot] = useState<CompanyRealtimeSnapshot>(
    companyRealtimeBridge.getSnapshot()
  );

  useEffect(() => companyRealtimeBridge.subscribe(setSnapshot), []);

  return snapshot;
}

export function useCompanyFallbackPolling(refetch: () => Promise<unknown>) {
  const realtime = useCompanyRealtimeStatus();

  useEffect(() => {
    const intervalMs = getCompanyRealtimePollingIntervalMs(
      realtime.transportStatus,
      realtime.dataFreshness
    );
    if (!intervalMs) return;
    const intervalId = setInterval(() => {
      void refetch();
    }, intervalMs);
    return () => clearInterval(intervalId);
  }, [realtime.dataFreshness, realtime.status, realtime.transportStatus, refetch]);

  return realtime;
}

export function useCompanyRideActions() {
  const contextId = useActiveCompanyContextId();
  const queryClient = useQueryClient();
  const invalidateScope = (missionId?: number) => {
    if (!contextId) return;
    invalidateCompanyQueriesForEvent(queryClient, "booking_updated", { contextId, missionId });
  };
  const optimisticAssignEnabled = isFeatureEnabled("company_mobile_assign_optimistic_enabled");

  const patchMissionsForOptimisticAssign = (
    previousValue: unknown,
    missionId: number,
    driverId: number
  ) => {
    if (!previousValue || typeof previousValue !== "object") return previousValue;
    const payload = previousValue as Record<string, unknown>;
    const missions = Array.isArray(payload.missions) ? payload.missions : null;
    if (!missions) return previousValue;
    const nextMissions = missions.map((entry) => {
      if (!entry || typeof entry !== "object") return entry;
      const row = entry as Record<string, unknown>;
      const rowMissionId = Number(row.mission_id ?? row.booking_id ?? row.id);
      if (!Number.isFinite(rowMissionId) || rowMissionId !== missionId) return entry;
      return {
        ...row,
        driver_id: driverId,
        status: row.status === "pending" ? "assigned" : row.status,
      };
    });
    return { ...payload, missions: nextMissions };
  };

  return {
    assign: useMutation({
      mutationFn: (params: { missionId: number; driverId: number }) =>
        assignCompanyRide({ contextId: contextId as string, ...params }),
      onMutate: async (params) => {
        if (!contextId || !optimisticAssignEnabled) return undefined;
        await queryClient.cancelQueries({
          queryKey: contextScopedKey(contextId, [...companyQueryKeys.root, "missions"] as unknown[]),
          exact: false,
        });
        const previousMissionQueries = queryClient.getQueriesData({
          queryKey: contextScopedKey(contextId, [...companyQueryKeys.root, "missions"] as unknown[]),
          exact: false,
        });
        previousMissionQueries.forEach(([queryKey, currentValue]) => {
          queryClient.setQueryData(queryKey, patchMissionsForOptimisticAssign(currentValue, params.missionId, params.driverId));
        });
        const detailKey = contextScopedKey(
          contextId,
          [...companyQueryKeys.rideDetails(contextId, params.missionId)] as unknown[]
        );
        const previousDetail = queryClient.getQueryData(detailKey);
        if (previousDetail && typeof previousDetail === "object") {
          const detailRow = previousDetail as Record<string, unknown>;
          queryClient.setQueryData(detailKey, {
            ...detailRow,
            driver_id: params.driverId,
            status: detailRow.status === "pending" ? "assigned" : detailRow.status,
          });
        }
        return { previousMissionQueries, previousDetail, missionId: params.missionId };
      },
      onError: (error, _params, context) => {
        const status = (error as AxiosError | undefined)?.response?.status;
        if (!contextId || !context) return;
        if (status === 409 || status === 422) {
          context.previousMissionQueries?.forEach(([queryKey, snapshot]) => {
            queryClient.setQueryData(queryKey, snapshot);
          });
          const detailKey = contextScopedKey(
            contextId,
            [...companyQueryKeys.rideDetails(contextId, context.missionId)] as unknown[]
          );
          if (context.previousDetail !== undefined) {
            queryClient.setQueryData(detailKey, context.previousDetail);
          }
        }
      },
      onSuccess: (_, params) => invalidateScope(params.missionId),
    }),
    reassign: useMutation({
      mutationFn: (params: { missionId: number; driverId: number }) =>
        reassignCompanyRide({ contextId: contextId as string, ...params }),
      onMutate: async (params) => {
        if (!contextId || !optimisticAssignEnabled) return undefined;
        await queryClient.cancelQueries({
          queryKey: contextScopedKey(contextId, [...companyQueryKeys.root, "missions"] as unknown[]),
          exact: false,
        });
        const previousMissionQueries = queryClient.getQueriesData({
          queryKey: contextScopedKey(contextId, [...companyQueryKeys.root, "missions"] as unknown[]),
          exact: false,
        });
        previousMissionQueries.forEach(([queryKey, currentValue]) => {
          queryClient.setQueryData(queryKey, patchMissionsForOptimisticAssign(currentValue, params.missionId, params.driverId));
        });
        const detailKey = contextScopedKey(
          contextId,
          [...companyQueryKeys.rideDetails(contextId, params.missionId)] as unknown[]
        );
        const previousDetail = queryClient.getQueryData(detailKey);
        if (previousDetail && typeof previousDetail === "object") {
          const detailRow = previousDetail as Record<string, unknown>;
          queryClient.setQueryData(detailKey, {
            ...detailRow,
            driver_id: params.driverId,
            status: detailRow.status === "pending" ? "assigned" : detailRow.status,
          });
        }
        return { previousMissionQueries, previousDetail, missionId: params.missionId };
      },
      onError: (error, _params, context) => {
        const status = (error as AxiosError | undefined)?.response?.status;
        if (!contextId || !context) return;
        if (status === 409 || status === 422) {
          context.previousMissionQueries?.forEach(([queryKey, snapshot]) => {
            queryClient.setQueryData(queryKey, snapshot);
          });
          const detailKey = contextScopedKey(
            contextId,
            [...companyQueryKeys.rideDetails(contextId, context.missionId)] as unknown[]
          );
          if (context.previousDetail !== undefined) {
            queryClient.setQueryData(detailKey, context.previousDetail);
          }
        }
      },
      onSuccess: (_, params) => invalidateScope(params.missionId),
    }),
    cancel: useMutation({
      mutationFn: (params: { missionId: number; reasonCode: string; note?: string; reason?: string }) =>
        cancelCompanyRide({ contextId: contextId as string, ...params }),
      onSuccess: (_, params) => invalidateScope(params.missionId),
    }),
    schedule: useMutation({
      mutationFn: (params: {
        missionId: number;
        payload: { pickup_at: string; timezone?: string; force_recompute?: boolean; note?: string | null };
      }) => scheduleCompanyRide({ contextId: contextId as string, ...params }),
      onSuccess: (_, params) => invalidateScope(params.missionId),
    }),
    urgent: useMutation({
      mutationFn: (params: {
        missionId: number;
        payload?: { urgent?: boolean; reason_code?: string | null; note?: string | null; source?: string | null };
      }) =>
        markCompanyRideUrgent({ contextId: contextId as string, ...params }),
      onSuccess: (_, params) => invalidateScope(params.missionId),
    }),
    transfer: useMutation({
      mutationFn: (params: { missionId: number; targetCompanyId: number }) =>
        transferCompanyRide({ contextId: contextId as string, ...params }),
      onSuccess: (_, params) => invalidateScope(params.missionId),
    }),
  };
}

function normalizeReadonlyRows(input: unknown): Record<string, unknown>[] {
  if (!input || typeof input !== "object") return [];
  const payload = input as Record<string, unknown>;
  const candidates = [
    payload.items,
    payload.results,
    payload.data,
    payload.clients,
    payload.invoices,
  ];
  const rows = candidates.find((entry) => Array.isArray(entry));
  return Array.isArray(rows)
    ? rows.filter((entry): entry is Record<string, unknown> => !!entry && typeof entry === "object")
    : [];
}

/** `total` depuis `paginated_response` (`/companies/me/clients`) ou équivalent. */
function extractClientsListTotal(input: unknown): number | null {
  if (!input || typeof input !== "object") return null;
  const p = input as Record<string, unknown>;
  const pag = p.pagination;
  if (pag && typeof pag === "object") {
    const t = (pag as Record<string, unknown>).total;
    if (typeof t === "number" && Number.isFinite(t)) return t;
  }
  if (typeof p.total === "number" && Number.isFinite(p.total)) return p.total;
  return null;
}

export type CompanyClientsReadonlyResult = {
  rows: Record<string, unknown>[];
  total: number | null;
};

export function useCompanyClientsReadonlyQuery(params: { q?: string; page?: number; limit?: number }) {
  const contextId = useActiveCompanyContextId();
  const flagOn = true;
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(
          contextId,
          [...companyQueryKeys.root, "clients-readonly", contextId, params.q ?? "", params.page ?? 1, params.limit ?? 30] as unknown[]
        )
      : ["company", "dispatch", "clients-readonly", "disabled"],
    queryFn: async (): Promise<CompanyClientsReadonlyResult> => {
      const payload = await getCompanyClients({
        contextId: contextId as string,
        q: params.q,
        page: params.page,
        limit: params.limit ?? 30,
      });
      return {
        rows: normalizeReadonlyRows(payload),
        total: extractClientsListTotal(payload),
      };
    },
    enabled: Boolean(contextId) && flagOn,
    staleTime: QUERY_STALE_TIME_MS.companyList,
  });
}

export function useCompanyClientReadonlyDetailQuery(clientId: number | null) {
  const contextId = useActiveCompanyContextId();
  const flagOn = true;
  return useQuery({
    queryKey:
      contextId && clientId
        ? contextScopedKey(
            contextId,
            [...companyQueryKeys.root, "client-readonly-detail", contextId, clientId] as unknown[]
          )
        : ["company", "dispatch", "client-readonly-detail", "disabled"],
    queryFn: async () => {
      const payload = await getCompanyClientDetail({
        contextId: contextId as string,
        clientId: clientId as number,
      });
      if (payload && typeof payload === "object") {
        const candidate = payload as Record<string, unknown>;
        return (candidate.item ??
          candidate.data ??
          candidate.client ??
          candidate.summary ??
          payload) as Record<string, unknown>;
      }
      return null;
    },
    enabled: Boolean(contextId) && flagOn && clientId != null,
    staleTime: QUERY_STALE_TIME_MS.companyList,
  });
}

export function useCompanyInvoicesReadonlyQuery(params: { q?: string; page?: number; limit?: number }) {
  const contextId = useActiveCompanyContextId();
  const flagOn = true;
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(
          contextId,
          [...companyQueryKeys.root, "invoices-readonly", contextId, params.page ?? 1, params.limit ?? 500] as unknown[]
        )
      : ["company", "dispatch", "invoices-readonly", "disabled"],
    queryFn: async () => {
      const payload = await getCompanyInvoices({
        contextId: contextId as string,
        q: params.q,
        page: params.page,
        limit: params.limit ?? 500,
      });
      return normalizeReadonlyRows(payload);
    },
    enabled: Boolean(contextId) && flagOn,
    staleTime: QUERY_STALE_TIME_MS.companyList,
  });
}

function normalizeCompanyChatMessages(input: unknown): SharedChatMessage[] {
  if (!input || typeof input !== "object") return [];
  const payload = input as Record<string, unknown>;
  const buckets = [payload.items, payload.results, payload.data, payload.messages];
  const rows = buckets.find((entry) => Array.isArray(entry));
  if (!Array.isArray(rows)) return [];
  return rows
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null;
      const raw = entry as Record<string, unknown>;
      const timestamp =
        typeof raw.timestamp === "string" && raw.timestamp.length > 0
          ? raw.timestamp
          : new Date().toISOString();
      const id = raw.id ?? `${timestamp}-${Math.random().toString(36).slice(2, 8)}`;
      const imageRaw = raw.image_url ?? raw.image;
      const pdfRaw = raw.pdf_url ?? raw.pdf;
      const audioRaw = raw.audio_url ?? raw.audio;
      const message: SharedChatMessage = {
        id: typeof id === "number" || typeof id === "string" ? id : String(id),
        content: typeof raw.content === "string" ? raw.content : "",
        senderRole: typeof raw.sender_role === "string" ? raw.sender_role : undefined,
        senderName: typeof raw.sender_name === "string" ? raw.sender_name : null,
        timestamp,
        imageUrl: resolveMediaUrl(typeof imageRaw === "string" ? imageRaw : null),
        pdfUrl: resolveMediaUrl(typeof pdfRaw === "string" ? pdfRaw : null),
        pdfFilename: typeof raw.pdf_filename === "string" ? raw.pdf_filename : null,
        audioUrl: resolveMediaUrl(typeof audioRaw === "string" ? audioRaw : null),
      };
      return message;
    })
    .filter((entry): entry is SharedChatMessage => entry !== null)
    .sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
}

export function useCompanyChatMessages(date: string) {
  const contextId = useActiveCompanyContextId();
  const realtime = useCompanyRealtimeStatus();
  const chatSocketEnabled = isFeatureEnabled("company_mobile_chat_socket_enabled");
  return useInfiniteQuery({
    queryKey: contextId
      ? contextScopedKey(
          contextId,
          [...companyQueryKeys.root, "chat", contextId, date] as unknown[]
        )
      : ["company", "dispatch", "chat", "disabled"],
    queryFn: async ({ pageParam }: { pageParam?: string }) => {
      const payload = await getCompanyDispatchMessages({
        contextId: contextId as string,
        date,
        before: pageParam,
        limit: 30,
      });
      const data = payload as Record<string, unknown>;
      const nextBeforeCandidate =
        data.next_before ??
        data.next_cursor ??
        (typeof data.before === "string" ? data.before : null) ??
        null;
      return {
        messages: normalizeCompanyChatMessages(payload),
        nextBefore:
          typeof nextBeforeCandidate === "string" && nextBeforeCandidate.trim().length > 0
            ? nextBeforeCandidate
            : null,
      };
    },
    enabled: Boolean(contextId),
    getNextPageParam: (lastPage) => lastPage.nextBefore ?? undefined,
    initialPageParam: undefined as string | undefined,
    refetchInterval:
      chatSocketEnabled || realtime.status === "healthy" ? false : 6_000,
    maxPages: 5,
  });
}

export function useCompanySendChatMessage() {
  const contextId = useActiveCompanyContextId();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: { content: string; missionId?: number }) =>
      sendCompanyDispatchMessage({
        contextId: contextId as string,
        content: params.content,
        missionId: params.missionId,
      }),
    onSuccess: () => {
      if (!contextId) return;
      void queryClient.invalidateQueries({
        queryKey: contextScopedKey(
          contextId,
          [...companyQueryKeys.root, "chat"] as unknown[]
        ),
        exact: false,
      });
    },
  });
}

const companyDispatchChatLastReadKey = (contextId: string) => `company-dispatch-chat-last-read:${contextId}`;

function toEpoch(s: string | null | undefined): number {
  if (!s) return 0;
  const v = Date.parse(s);
  return Number.isFinite(v) ? v : 0;
}

/** Message entrant côté entreprise (hors rôle compagnie). */
function isCompanyChatInbound(message: SharedChatMessage): boolean {
  return message.senderRole?.toUpperCase() !== "COMPANY";
}

const companyChatLastReadSubKey = (contextId: string) => [...companyQueryKeys.root, "last-read", contextId] as const;

/**
 * Compteur de messages reçus non lus (onglet + badge) d’après l’horodatage
 * de dernière lecture stocké localement (**AsyncStorage** par contexte).
 *
 * **Écart produit / backend** : il n’existe pas aujourd’hui d’endpoint dispatch « read receipt »
 * côté API unifiée ; la « lecture » n’est pas synchronisée entre appareils ni avec le serveur.
 * `markAsRead` met à jour le cache React Query + AsyncStorage (`company-dispatch-chat-last-read:{contextId}`).
 * Pour des accusés serveur ou une reprise multi-device, prévoir p.ex. `PATCH …/chat/read` avec
 * `last_read_message_id` ou équivalent, + invalidation du badge.
 */
export function useCompanyChatUnread() {
  const contextId = useActiveCompanyContextId();
  const day = useMemo(() => new Date().toISOString().slice(0, 10), []);
  const queryClient = useQueryClient();
  const messagesQuery = useCompanyChatMessages(day);

  const lastReadQ = useQuery({
    queryKey: contextId
      ? contextScopedKey(
          contextId,
          companyChatLastReadSubKey(contextId) as unknown as unknown[]
        )
      : ["company", "chat", "last-read", "off"],
    queryFn: async () => {
      if (!contextId) return null;
      return AsyncStorage.getItem(companyDispatchChatLastReadKey(contextId));
    },
    enabled: Boolean(contextId),
    staleTime: QUERY_STALE_TIME_MS.default,
  });

  const lastReadEpoch = toEpoch(lastReadQ.data);

  const unreadCount = useMemo(() => {
    const pages = messagesQuery.data?.pages ?? [];
    const incoming = pages
      .flatMap((p) => p.messages)
      .filter((m) => isCompanyChatInbound(m) && m.content?.trim() !== "");
    if (incoming.length === 0) {
      return 0;
    }
    if (lastReadEpoch === 0) {
      return incoming.length;
    }
    return incoming.filter((m) => toEpoch(m.timestamp) > lastReadEpoch).length;
  }, [messagesQuery.data?.pages, lastReadEpoch]);

  const markAsRead = useCallback(
    async (latestViewedTimestamp?: string | null) => {
      if (!contextId) return;
      let t = (latestViewedTimestamp ?? "").trim() || null;
      if (!t) {
        const all = (messagesQuery.data?.pages ?? []).flatMap((p) => p.messages);
        if (all.length === 0) return;
        t = all.reduce((a, b) => (toEpoch(b.timestamp) > toEpoch(a.timestamp) ? b : a)).timestamp;
      }
      await AsyncStorage.setItem(companyDispatchChatLastReadKey(contextId), t);
      queryClient.setQueryData(
        contextScopedKey(
          contextId,
          companyChatLastReadSubKey(contextId) as unknown as unknown[]
        ),
        t
      );
    },
    [contextId, messagesQuery.data?.pages, queryClient]
  );

  return {
    unreadCount,
    isLoading: messagesQuery.isLoading || lastReadQ.isLoading,
    markAsRead,
    lastReadAt: lastReadQ.data ?? null,
  };
}
