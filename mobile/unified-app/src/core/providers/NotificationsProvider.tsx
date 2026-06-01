import { AppState, Platform } from "react-native";
import { useRouter } from "expo-router";
import { useCallback, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { emitDriverTelemetry } from "../observability/driverTelemetry";
import { isFeatureEnabled } from "../featureFlags/registry";
import { PropsWithChildren, useEffect } from "../reactCompat";
import { useSession } from "../sessionProvider";
import { CompanyNotificationsBridge } from "./CompanyNotificationsBridge";
import { DriverNotificationsBridge } from "./DriverNotificationsBridge";
import { DriverPushPayload, handleDriverPushQuickAction } from "../../features/driver/push";
import { ensureDriverNotificationChannels } from "../../features/driver/notificationChannels";
import { ensureDriverNotificationActions } from "../../features/driver/notificationActions";
import { ensureDriverNotificationGrouping } from "../../features/driver/notificationGrouping";
import {
  handleSilentPushPayload,
  isSilentPayload,
  shouldSuppressVisualPush,
} from "../../features/driver/silentNotifications";
import {
  configureDriverRealtimeSync,
  requestChatRefresh,
  requestMissionRefresh,
} from "../../features/driver/driverRealtimeSync";
import {
  disposeDriverFirebaseMessaging,
  initDriverFirebaseMessaging,
} from "../../features/driver/firebaseMessaging";
import { resolveDriverDeepLink } from "../navigation/deepLinkHandler";
import { configureMissionBarIOS } from "../../features/driver/missionBarIOS";
import { registerMissionBarBackgroundHandlers } from "../../features/driver/missionBarBackground";
import { loadNotifee } from "../../features/driver/notifeeCompat";
import { resolveDriverNotificationContract } from "../../features/driver/notificationChannels";
import { appendSessionJournalEvent } from "../observability/sessionJournal";
import { shouldIgnoreNotification } from "../notifications/shouldIgnoreNotification";
import { getExpoNotificationsModule } from "../notifications/expoNotificationsCompat";
import {
  clearPushPermissionDenied,
  setPushPermissionDenied,
} from "../notifications/pushPermissionState";
import {
  buildNotificationDedupKey,
  markNotificationHandled,
} from "../notifications/notificationDedupStore";
import {
  computeNotificationAgeMs,
  emitNotificationDedupHit,
  emitNotificationDuplicateDropped,
  emitNotificationFiltered,
  emitNotificationNavigation,
  emitNotificationProcessingMs,
  emitNotificationReceived,
  emitNotificationSoundPlayed,
  emitNotificationSuppressed,
} from "../notifications/notificationTelemetry";
import {
  recordContextSwitchDurationMs,
  recordContextSwitchTotal,
  recordNotificationDuplicateDroppedTotal,
  recordNotificationDeduplicatedTotal,
  recordNotificationFilteredTotal,
  recordNotificationNavigationTotal,
  recordNotificationProcessingMs,
  recordNotificationReceivedTotal,
} from "../observability/runtimeStabilityMetrics";
import {
  resolveForegroundPresentation,
  shouldSkipNavigationForActiveScreen,
} from "../notifications/notificationPolicy";

function extractThreadId(data: Record<string, unknown>): string | null {
  const raw =
    data.thread_id ??
    data.threadId ??
    (typeof data.thread === "object" && data.thread != null
      ? (data.thread as Record<string, unknown>).id
      : null);
  if (typeof raw === "string" && raw.length > 0) return raw;
  if (typeof raw === "number" && Number.isFinite(raw)) return String(raw);
  return null;
}

function extractRawType(data: unknown): string | null {
  if (!data || typeof data !== "object") return null;
  const value = data as Record<string, unknown>;
  return typeof value.type === "string" ? value.type : null;
}

function extractEventId(data: unknown): string | null {
  if (!data || typeof data !== "object") return null;
  const value = data as Record<string, unknown>;
  return typeof value.event_id === "string" ? value.event_id : null;
}

function shouldDedupSkip(data: unknown, notificationId?: string | null): boolean {
  if (!data || typeof data !== "object") return false;
  const record = data as Record<string, unknown>;
  const missionIdRaw = record.mission_id ?? record.missionId ?? record.booking_id;
  const missionId = Number(missionIdRaw);
  const key = buildNotificationDedupKey({
    eventId: extractEventId(data),
    notificationId: notificationId ?? null,
    missionId: Number.isFinite(missionId) ? missionId : null,
    type: extractRawType(data),
  });
  if (markNotificationHandled(key)) {
    recordNotificationDeduplicatedTotal({ dedup_key: key });
    recordNotificationDuplicateDroppedTotal({
      dedup_key: key,
      notification_id: notificationId ?? null,
    });
    emitNotificationDedupHit({
      dedup_key: key,
      notification_id: notificationId ?? null,
    });
    emitNotificationDuplicateDropped({
      dedup_key: key,
      notification_id: notificationId ?? null,
    });
    return true;
  }
  return false;
}

export function NotificationsProvider({ children }: PropsWithChildren) {
  const Notifications = getExpoNotificationsModule();
  const router = useRouter();
  const queryClient = useQueryClient();
  const { status, activeContext, bootstrap } = useSession();
  const pendingPayloadRef = useRef<DriverPushPayload | null>(null);
  const pendingPayloadTimedOutRef = useRef(false);
  const pendingPayloadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const initialResponseConsumedRef = useRef(false);
  const filterContextRef = useRef({
    contextType: null as string | null,
    userId: null as string | number | null,
    companyId: null as string | number | null,
  });
  const lastStableFilterContextRef = useRef({
    contextType: null as string | null,
    userId: null as string | number | null,
    companyId: null as string | number | null,
  });
  const sessionRef = useRef({
    status,
    contextType: activeContext?.context_type ?? null,
  });
  const previousContextKeyRef = useRef<string | null>(null);
  const contextSwitchStartedAtRef = useRef<number | null>(null);
  const DEEPLINK_BOOTSTRAP_TIMEOUT_MS = Number(
    process.env.EXPO_PUBLIC_DRIVER_DEEPLINK_BOOTSTRAP_TIMEOUT_MS ?? "8000"
  );

  filterContextRef.current = {
    contextType: activeContext?.context_type ?? null,
    userId: bootstrap?.user?.id ?? null,
    companyId: activeContext?.organization_id ?? null,
  };
  if (activeContext?.context_type) {
    lastStableFilterContextRef.current = {
      contextType: activeContext.context_type,
      userId: bootstrap?.user?.id ?? null,
      companyId: activeContext.organization_id ?? null,
    };
  }
  sessionRef.current = {
    status,
    contextType: activeContext?.context_type ?? null,
  };

  const resolveFilterContext = useCallback(() => {
    const live = filterContextRef.current;
    if (live.contextType) return live;
    return lastStableFilterContextRef.current;
  }, []);

  const shouldApplyNotificationFilter = useCallback(() => {
    const session = sessionRef.current;
    const filterContext = resolveFilterContext();
    return session.status === "ready" && Boolean(filterContext.contextType);
  }, [resolveFilterContext]);

  useEffect(() => {
    const nextKey =
      activeContext?.context_id != null
        ? `${activeContext.context_type ?? "?"}:${activeContext.context_id}`
        : null;
    if (previousContextKeyRef.current === nextKey) return;
    if (contextSwitchStartedAtRef.current != null) {
      const durationMs = Date.now() - contextSwitchStartedAtRef.current;
      recordContextSwitchDurationMs(durationMs, {
        from_context: previousContextKeyRef.current,
        to_context: nextKey,
      });
    }
    recordContextSwitchTotal({
      from_context: previousContextKeyRef.current,
      to_context: nextKey,
    });
    previousContextKeyRef.current = nextKey;
    contextSwitchStartedAtRef.current = Date.now();
  }, [activeContext?.context_id, activeContext?.context_type]);

  useEffect(() => {
    configureDriverRealtimeSync({
      queryClient,
      getContextId: () =>
        activeContext?.context_type === "driver" ? activeContext.context_id : null,
    });
    return () => {
      configureDriverRealtimeSync(null);
    };
  }, [activeContext?.context_id, activeContext?.context_type, queryClient]);

  const evaluateNotificationFilter = useCallback(
    (input: unknown) => shouldIgnoreNotification(input, resolveFilterContext()),
    [resolveFilterContext]
  );

  const normalizePushType = useCallback((raw: string): DriverPushPayload["type"] | null => {
    if (raw === "mission_assigned" || raw === "booking_assigned") return "mission_assigned";
    if (raw === "mission_updated" || raw === "booking_updated") return "mission_updated";
    if (raw === "mission_cancelled" || raw === "booking_cancelled") return "mission_cancelled";
    if (raw === "mission_reassigned" || raw === "booking_reassigned") return "mission_reassigned";
    if (raw === "chat_message" || raw === "message" || raw === "team_chat_message") return "chat_message";
    if (raw === "reminder_action") return "reminder_action";
    if (raw === "informative") return "informative";
    if (raw === "delay") return null;
    return null;
  }, []);

  const parseDeepLinkTarget = useCallback((deepLink: string | undefined) => {
    return resolveDriverDeepLink(deepLink ?? null);
  }, []);

  const normalizeQuickAction = useCallback(
    (value: unknown): DriverPushPayload["action"] | undefined => {
      if (typeof value !== "string") return undefined;
      const normalized = value.toLowerCase();
      if (normalized.includes("decline")) return "reject";
      if (normalized.includes("accept")) return "accept";
      if (normalized.includes("reject")) return "reject";
      if (normalized.includes("start")) return "start";
      if (normalized.includes("complete")) return "complete";
      return undefined;
    },
    []
  );

  const parsePayload = useCallback(
    (input: unknown, actionIdentifier?: string): DriverPushPayload | null => {
      if (!input || typeof input !== "object") return null;
      const value = input as Record<string, unknown>;
      const rawType = String(value.type ?? "mission_updated");
      const type = normalizePushType(rawType);
      if (!type) return null;
      const missionIdRaw =
        value.mission_id ?? value.missionId ?? value.booking_id ?? value.bookingId;
      const missionId = Number(missionIdRaw);
      const deepLink =
        typeof value.deepLink === "string"
          ? value.deepLink
          : typeof value.deep_link === "string"
            ? value.deep_link
            : undefined;
      const deepLinkTarget = parseDeepLinkTarget(deepLink);
      const resolvedMissionId = Number.isFinite(missionId) ? missionId : deepLinkTarget?.missionId ?? null;
      const requiresMissionId = type !== "chat_message" && type !== "informative";
      if (requiresMissionId && !Number.isFinite(resolvedMissionId)) return null;
      const schema =
        value.mission_id != null || value.missionId != null
          ? "mission_v2"
          : value.booking_id != null || value.bookingId != null
            ? "booking_v1"
            : "unknown";
      const threadId = extractThreadId(value);
      return {
        mission_id: Number.isFinite(resolvedMissionId) ? (resolvedMissionId as number) : null,
        type,
        event_id: typeof value.event_id === "string" ? value.event_id : undefined,
        action:
          normalizeQuickAction(value.action) ?? normalizeQuickAction(actionIdentifier),
        deep_link: deepLink,
        payload_schema: schema,
        thread_id: threadId ?? undefined,
      };
    },
    [normalizePushType, parseDeepLinkTarget, normalizeQuickAction]
  );

  const triggerDriverResync = useCallback(
    async (reason: "push_update" | "push_route_fallback", missionId: number | null) => {
      if (status !== "ready" || activeContext?.context_type !== "driver") return;
      try {
        emitDriverTelemetry("driver.runtime.resync", {
          source: "core.notifications.provider",
          trigger: reason,
          context_id: activeContext.context_id,
          mission_id: missionId,
        });
        requestMissionRefresh(reason, missionId);
      } catch (error) {
        emitDriverTelemetry("driver.runtime.resync", {
          source: "core.notifications.provider",
          trigger: reason,
          context_id: activeContext.context_id,
          mission_id: missionId,
          reason: error instanceof Error ? error.message : "push_resync_failed",
        });
      }
    },
    [activeContext, status]
  );

  const routePayload = useCallback(
    async (payload: DriverPushPayload) => {
      void appendSessionJournalEvent(
        "push.route.start",
        {
          mission_id: payload.mission_id,
          type: payload.type,
          action: payload.action ?? null,
        },
        activeContext?.context_id ?? null
      );
      if (payload.action && Number.isFinite(payload.mission_id)) {
        emitDriverTelemetry("push.quick_action.dispatch", {
          source: "core.notifications.provider",
          mission_id: payload.mission_id,
          action: payload.action,
        });
      }
      await handleDriverPushQuickAction(payload);
      if (payload.action && Number.isFinite(payload.mission_id)) {
        emitDriverTelemetry("push.quick_action.success", {
          source: "core.notifications.provider",
          mission_id: payload.mission_id,
          action: payload.action,
        });
      }
      if (
        payload.mission_id != null &&
        (payload.type === "mission_updated" ||
          payload.type === "mission_reassigned" ||
          payload.type === "mission_cancelled")
      ) {
        await triggerDriverResync("push_update", payload.mission_id ?? null);
      }
      if (payload.thread_id) {
        requestChatRefresh(payload.thread_id, "push_chat");
      }

      if (shouldSkipNavigationForActiveScreen({ payload, threadId: payload.thread_id })) {
        recordNotificationNavigationTotal({ skipped: true, reason: "active_screen" });
        emitNotificationNavigation({
          mission_id: payload.mission_id,
          skipped: true,
          reason: "active_screen",
        });
        return;
      }

      if (payload.type === "chat_message") {
        const deepLinkTarget = parseDeepLinkTarget(payload.deep_link);
        const route = deepLinkTarget?.route
          ? deepLinkTarget.route
          : payload.thread_id
            ? `/(app)/(driver)/messages/${encodeURIComponent(payload.thread_id)}`
            : "/(app)/(driver)/chat";
        recordNotificationNavigationTotal({ route });
        emitNotificationNavigation({
          mission_id: payload.mission_id,
          route,
        });
        router.push(route as any);
        return;
      }

      if (payload.type === "mission_assigned") {
        const route = "/(app)/(driver)";
        recordNotificationNavigationTotal({ route });
        emitNotificationNavigation({ mission_id: payload.mission_id, route });
        router.push(route as any);
        return;
      }

      if (payload.type === "mission_reassigned") {
        const route = "/(app)/(driver)/trips";
        recordNotificationNavigationTotal({ route });
        emitNotificationNavigation({ mission_id: payload.mission_id, route });
        router.push(route as any);
        return;
      }

      if (payload.type === "informative") {
        const deepLinkTarget = parseDeepLinkTarget(payload.deep_link);
        if (deepLinkTarget) {
          recordNotificationNavigationTotal({ route: deepLinkTarget.route });
          emitNotificationNavigation({
            mission_id: payload.mission_id,
            route: deepLinkTarget.route,
          });
          router.push(deepLinkTarget.route as any);
        }
        return;
      }

      const deepLinkTarget = parseDeepLinkTarget(payload.deep_link);
      if (deepLinkTarget?.route) {
        recordNotificationNavigationTotal({ route: deepLinkTarget.route });
        emitNotificationNavigation({
          mission_id: payload.mission_id,
          route: deepLinkTarget.route,
        });
        router.push(deepLinkTarget.route as any);
      } else if (
        payload.type === "mission_cancelled" ||
        payload.type === "reminder_action" ||
        (payload.action && Number.isFinite(payload.mission_id))
      ) {
        if (!Number.isFinite(payload.mission_id)) return;
        const route = `/(app)/(driver)/missions/${payload.mission_id}`;
        recordNotificationNavigationTotal({ route });
        emitNotificationNavigation({ mission_id: payload.mission_id, route });
        router.push(route as any);
      }
    },
    [activeContext?.context_id, parseDeepLinkTarget, router, triggerDriverResync]
  );

  const queuePendingPayload = useCallback(
    (payload: DriverPushPayload) => {
      pendingPayloadRef.current = payload;
      pendingPayloadTimedOutRef.current = false;
      if (pendingPayloadTimerRef.current) {
        clearTimeout(pendingPayloadTimerRef.current);
        pendingPayloadTimerRef.current = null;
      }
      pendingPayloadTimerRef.current = setTimeout(() => {
        pendingPayloadTimedOutRef.current = true;
        emitDriverTelemetry("push.notification.route_timeout", {
          source: "core.notifications.provider",
          mission_id: payload.mission_id,
          timeout_ms: DEEPLINK_BOOTSTRAP_TIMEOUT_MS,
          reason: "route_timeout",
        });
        if (status === "ready" && activeContext?.context_type === "driver") {
          if (payload.type !== "chat_message") {
            void triggerDriverResync("push_route_fallback", payload.mission_id);
            router.push("/(app)/(driver)/missions" as any);
          } else {
            const route = payload.thread_id
              ? `/(app)/(driver)/messages/${encodeURIComponent(payload.thread_id)}`
              : "/(app)/(driver)/chat";
            router.push(route as any);
          }
        }
      }, DEEPLINK_BOOTSTRAP_TIMEOUT_MS);
    },
    [DEEPLINK_BOOTSTRAP_TIMEOUT_MS, activeContext?.context_type, router, status, triggerDriverResync]
  );

  const clearPendingPayload = useCallback(() => {
    pendingPayloadRef.current = null;
    pendingPayloadTimedOutRef.current = false;
    if (pendingPayloadTimerRef.current) {
      clearTimeout(pendingPayloadTimerRef.current);
      pendingPayloadTimerRef.current = null;
    }
  }, []);

  const navigateFromPayload = useCallback(
    async (payload: DriverPushPayload | null) => {
      if (!payload) return;
      if (status !== "ready" || activeContext?.context_type !== "driver") {
        queuePendingPayload(payload);
        return;
      }
      try {
        await routePayload(payload);
      } catch (error) {
        if (payload.action) {
          emitDriverTelemetry("push.quick_action.failure", {
            source: "core.notifications.provider",
            mission_id: payload.mission_id,
            action: payload.action,
            reason: error instanceof Error ? error.message : "quick_action_failed",
          });
        }
        emitDriverTelemetry("push.notification.route_failed", {
          source: "core.notifications.provider",
          mission_id: payload.mission_id,
          payload_schema: payload.payload_schema ?? "unknown",
          reason: error instanceof Error ? error.message : "route_failed",
        });
      }
    },
    [activeContext?.context_type, queuePendingPayload, routePayload, status]
  );

  const processSilentPayload = useCallback(
    async (data: unknown) => {
      await handleSilentPushPayload(data, async (missionId) => {
        await triggerDriverResync("push_update", missionId);
      });
    },
    [triggerDriverResync]
  );

  const displayDriverPushNotification = useCallback(
    async (payload: Record<string, unknown>) => {
      if (isSilentPayload(payload)) return;
      if (shouldSuppressVisualPush(payload)) return;

      const rawType = extractRawType(payload);
      const contract = resolveDriverNotificationContract(rawType);
      const title =
        typeof payload.title === "string"
          ? payload.title
          : typeof (payload as { notification?: { title?: string } }).notification?.title === "string"
            ? (payload as { notification?: { title?: string } }).notification?.title
            : "Liri";
      const body =
        typeof payload.body === "string"
          ? payload.body
          : typeof (payload as { notification?: { body?: string } }).notification?.body === "string"
            ? (payload as { notification?: { body?: string } }).notification?.body
            : "Mise à jour mission";
      const channelId =
        typeof payload.channelId === "string" && payload.channelId.length > 0
          ? payload.channelId
          : contract.channelId;

      if (Platform.OS === "android" && AppState.currentState !== "active") {
        const mod = await loadNotifee();
        if (mod) {
          const { default: notifee, AndroidImportance } = mod;
          await notifee.createChannel({
            id: channelId,
            name: "Missions",
            importance: AndroidImportance.HIGH,
          });
          await notifee.displayNotification({
            title,
            body,
            data: payload as Record<string, string>,
            android: {
              channelId,
              pressAction: { id: "default" },
            },
          });
          return;
        }
      }

      if (!Notifications) return;
      await Notifications.scheduleNotificationAsync({
        content: {
          title,
          body,
          data: payload as Record<string, unknown>,
        },
        trigger: null,
      });
    },
    [Notifications]
  );

  const clearForegroundNotifications = useCallback(async () => {
    if (Platform.OS === "web") return;
    if (!Notifications) return;

    if (typeof Notifications.dismissAllNotificationsAsync === "function") {
      await Notifications.dismissAllNotificationsAsync().catch(() => undefined);
    }

    const mod = await loadNotifee();
    if (!mod) return;
    const notifee = mod.default as {
      cancelAllNotifications?: () => Promise<void>;
      cancelDisplayedNotifications?: () => Promise<void>;
    };
    if (typeof notifee.cancelDisplayedNotifications === "function") {
      await notifee.cancelDisplayedNotifications().catch(() => undefined);
    }
    if (typeof notifee.cancelAllNotifications === "function") {
      await notifee.cancelAllNotifications().catch(() => undefined);
    }
  }, [Notifications]);

  useEffect(() => {
    if (!Notifications) return;
    Notifications.setNotificationHandler({
      handleNotification: async (notification) => {
        const pipelineStartedAt = Date.now();
        const data = notification.request.content.data;
        const notificationId = notification.request.identifier;
        const receivedAt = Date.now();
        const notificationAgeMs = computeNotificationAgeMs(data, receivedAt);
        recordNotificationReceivedTotal({
          stage: "foreground_handler",
          notification_id: notificationId,
        });

        if (shouldDedupSkip(data, notificationId)) {
          emitNotificationProcessingMs(Date.now() - pipelineStartedAt, {
            stage: "foreground_handler",
            outcome: "dedup_skip",
          });
          recordNotificationProcessingMs(Date.now() - pipelineStartedAt, {
            stage: "foreground_handler",
            outcome: "dedup_skip",
          });
          return {
            shouldShowBanner: false,
            shouldShowList: false,
            shouldPlaySound: false,
            shouldSetBadge: false,
          };
        }

        if (isSilentPayload(data)) {
          emitNotificationSuppressed("silent", {
            stage: "foreground_handler",
            notification_id: notificationId,
          });
          void processSilentPayload(data);
          return {
            shouldShowBanner: false,
            shouldShowList: false,
            shouldPlaySound: false,
            shouldSetBadge: false,
          };
        }

        if (!shouldApplyNotificationFilter()) {
          return {
            shouldShowBanner: false,
            shouldShowList: false,
            shouldPlaySound: false,
            shouldSetBadge: false,
          };
        }

        const filter = evaluateNotificationFilter(data);
        if (filter.ignore) {
          recordNotificationFilteredTotal({ reason: filter.reason ?? "filtered" });
          emitNotificationFiltered(filter.reason ?? "filtered", {
            stage: "foreground_handler",
            notification_id: notificationId,
            notification_age_ms: notificationAgeMs,
          });
          emitDriverTelemetry("push.notification.ignored", {
            source: "core.notifications.provider",
            stage: "foreground_handler",
            reason: filter.reason ?? "ignored",
          });
          return {
            shouldShowBanner: false,
            shouldShowList: false,
            shouldPlaySound: false,
            shouldSetBadge: false,
          };
        }

        const parsed = parsePayload(data);
        const threadId = data && typeof data === "object" ? extractThreadId(data as Record<string, unknown>) : null;
        const presentation = resolveForegroundPresentation({
          rawType: extractRawType(data),
          payload: parsed,
          threadId,
          silent: false,
        });

        if (presentation.suppressReason === "active_screen") {
          emitNotificationSuppressed("active_screen", {
            stage: "foreground_handler",
            notification_id: notificationId,
            mission_id: parsed?.mission_id,
          });
          if (parsed) {
            void triggerDriverResync("push_update", parsed.mission_id);
          }
        }

        if (!presentation.shouldShowBanner && presentation.suppressReason === "silent") {
          void processSilentPayload(data);
        }

        if (presentation.shouldPlaySound) {
          emitNotificationSoundPlayed({
            notification_id: notificationId,
            type: parsed?.type ?? extractRawType(data),
          });
        }

        const processingMs = Date.now() - pipelineStartedAt;
        emitNotificationProcessingMs(processingMs, {
          stage: "foreground_handler",
          outcome: presentation.shouldShowBanner ? "shown" : "suppressed",
        });
        recordNotificationProcessingMs(processingMs, {
          stage: "foreground_handler",
          outcome: presentation.shouldShowBanner ? "shown" : "suppressed",
        });

        return {
          shouldShowBanner: presentation.shouldShowBanner,
          shouldShowList: presentation.shouldShowList,
          shouldPlaySound: presentation.shouldPlaySound,
          shouldSetBadge: presentation.shouldSetBadge,
        };
      },
    });
  }, [
    Notifications,
    evaluateNotificationFilter,
    parsePayload,
    processSilentPayload,
    shouldApplyNotificationFilter,
    triggerDriverResync,
  ]);

  useEffect(() => {
    if (Platform.OS === "web") return;
    if (activeContext?.context_type !== "driver") return;
    if (status !== "ready") return;

    if (AppState.currentState === "active") {
      void clearForegroundNotifications();
    }

    const subscription = AppState.addEventListener("change", (nextState) => {
      if (nextState !== "active") return;
      void clearForegroundNotifications();
    });

    return () => {
      subscription.remove();
    };
  }, [activeContext?.context_type, clearForegroundNotifications, status]);

  useEffect(() => {
    if (status !== "ready" || activeContext?.context_type !== "driver") return;
    const pending = pendingPayloadRef.current;
    if (!pending) return;
    if (pendingPayloadTimedOutRef.current) {
      clearPendingPayload();
      void triggerDriverResync("push_route_fallback", pending.mission_id);
      router.push("/(app)/(driver)/missions" as any);
      return;
    }
    clearPendingPayload();
    void navigateFromPayload(pending);
  }, [
    activeContext?.context_type,
    clearPendingPayload,
    navigateFromPayload,
    router,
    status,
    triggerDriverResync,
  ]);

  useEffect(() => {
    if (!isFeatureEnabled("driver_push_enabled")) return;
    if (!Notifications) return;
    const isWeb = Platform.OS === "web";
    if (isWeb) return;
    if (isFeatureEnabled("driver_notification_actions_enabled")) {
      void ensureDriverNotificationChannels().catch((err) => {
        console.error("[Android] Notification channels setup failed:", err);
        emitDriverTelemetry("push.channels.setup_failed", {
          source: "core.notifications.provider",
          error: err instanceof Error ? err.message : "unknown",
        });
      });
      void ensureDriverNotificationActions().catch((err) => {
        console.error("[Android] Notification actions setup failed:", err);
        emitDriverTelemetry("push.actions.setup_failed", {
          source: "core.notifications.provider",
          error: err instanceof Error ? err.message : "unknown",
        });
      });
      void ensureDriverNotificationGrouping().catch((err) => {
        console.error("[Android] Notification grouping setup failed:", err);
        emitDriverTelemetry("push.grouping.setup_failed", {
          source: "core.notifications.provider",
          error: err instanceof Error ? err.message : "unknown",
        });
      });
      void configureMissionBarIOS().catch((err) => {
        console.error("[iOS] Mission bar setup failed:", err);
        emitDriverTelemetry("push.mission_bar.setup_failed", {
          source: "core.notifications.provider",
          error: err instanceof Error ? err.message : "unknown",
        });
      });
      registerMissionBarBackgroundHandlers();
    }
    void Notifications.requestPermissionsAsync()
      .then((perm) => {
        if (!perm.granted) {
          setPushPermissionDenied(true);
          emitDriverTelemetry("push.token.permission_denied", {
            source: "core.notifications.provider",
          });
          return;
        }
        clearPushPermissionDenied();
      })
      .catch((err) => {
        console.error("[Notifications] requestPermissionsAsync failed:", err);
        emitDriverTelemetry("push.permission.request_failed", {
          source: "core.notifications.provider",
          error: err instanceof Error ? err.message : "unknown",
        });
      });
    const received = Notifications.addNotificationReceivedListener((notification) => {
      const data = notification.request.content.data;
      const notificationId = notification.request.identifier;
      if (shouldDedupSkip(data, notificationId)) return;

      if (shouldApplyNotificationFilter()) {
        const filter = evaluateNotificationFilter(data);
        if (filter.ignore) {
          recordNotificationFilteredTotal({ reason: filter.reason ?? "filtered" });
          emitNotificationFiltered(filter.reason ?? "filtered", {
            stage: "received_listener",
            notification_id: notificationId,
          });
          emitDriverTelemetry("push.notification.ignored", {
            source: "core.notifications.provider",
            stage: "received_listener",
            notification_id: notificationId,
            reason: filter.reason ?? "ignored",
          });
          return;
        }
      }
      void processSilentPayload(data);
      recordNotificationReceivedTotal({
        stage: "received_listener",
        notification_id: notificationId,
      });
      emitNotificationReceived({
        app_state: AppState.currentState,
        notification_id: notificationId,
        notification_age_ms: computeNotificationAgeMs(data),
      });
    });
    const opened = Notifications.addNotificationResponseReceivedListener((response) => {
      const data = response.notification.request.content.data;
      const notificationId = response.notification.request.identifier;
      if (shouldDedupSkip(data, notificationId)) return;

      if (shouldApplyNotificationFilter()) {
        const filter = evaluateNotificationFilter(data);
        if (filter.ignore) {
          recordNotificationFilteredTotal({ reason: filter.reason ?? "filtered" });
          emitNotificationFiltered(filter.reason ?? "filtered", {
            stage: "response_listener",
            notification_id: notificationId,
          });
          emitDriverTelemetry("push.notification.ignored", {
            source: "core.notifications.provider",
            stage: "response_listener",
            notification_id: notificationId,
            reason: filter.reason ?? "ignored",
          });
          return;
        }
      }
      const payload = parsePayload(data, response.actionIdentifier);
      if (isSilentPayload(data)) {
        void processSilentPayload(data);
        return;
      }
      emitDriverTelemetry("push.notification.opened", {
        source: "core.notifications.provider",
        notification_id: notificationId,
        payload_schema: payload?.payload_schema ?? "unknown",
      });
      void navigateFromPayload(payload);
    });
    let tokenRefresh: { remove: () => void } | null = null;
    tokenRefresh = Notifications.addPushTokenListener((event) => {
      emitDriverTelemetry("push.token.refresh", {
        source: "core.notifications.provider",
        token_type: event.type,
      });
    });
    return () => {
      received.remove();
      opened.remove();
      tokenRefresh?.remove();
    };
  }, [
    Notifications,
    evaluateNotificationFilter,
    navigateFromPayload,
    parsePayload,
    processSilentPayload,
    shouldApplyNotificationFilter,
  ]);

  useEffect(() => {
    if (!isFeatureEnabled("driver_fcm_native_enabled")) return;
    if (activeContext?.context_type !== "driver") return;
    void initDriverFirebaseMessaging(async (payload) => {
      if (shouldDedupSkip(payload)) return;

      if (isSilentPayload(payload)) {
        await processSilentPayload(payload);
        return;
      }
      if (shouldApplyNotificationFilter()) {
        const filter = evaluateNotificationFilter(payload);
        if (filter.ignore) {
          recordNotificationFilteredTotal({ reason: filter.reason ?? "filtered" });
          emitNotificationFiltered(filter.reason ?? "filtered", { stage: "fcm_listener" });
          emitDriverTelemetry("push.notification.ignored", {
            source: "core.notifications.provider",
            stage: "fcm_listener",
            reason: filter.reason ?? "ignored",
          });
          return;
        }
      }
      const parsed = parsePayload(payload);
      if (parsed) {
        await triggerDriverResync("push_update", parsed.mission_id);
        if (parsed.thread_id) {
          requestChatRefresh(parsed.thread_id, "push_fcm");
        }
      }
      await displayDriverPushNotification(payload).catch((err) => {
        console.error("[FCM] display notification failed:", err);
        emitDriverTelemetry("push.display.schedule_failed", {
          source: "core.notifications.provider",
          error: err instanceof Error ? err.message : "unknown",
        });
      });
    });
    return () => {
      disposeDriverFirebaseMessaging();
    };
  }, [
    Notifications,
    activeContext?.context_type,
    evaluateNotificationFilter,
    parsePayload,
    processSilentPayload,
    shouldApplyNotificationFilter,
    displayDriverPushNotification,
    triggerDriverResync,
  ]);

  useEffect(() => {
    if (!isFeatureEnabled("driver_push_enabled")) return;
    if (!Notifications) return;
    if (Platform.OS === "web") return;
    if (status !== "ready") return;
    if (activeContext?.context_type !== "driver") return;
    if (initialResponseConsumedRef.current) return;
    initialResponseConsumedRef.current = true;

    void (async () => {
      const initialResponse = await Notifications.getLastNotificationResponseAsync().catch(() => null);
      if (!initialResponse?.notification) return;
      const data = initialResponse.notification.request.content.data;
      const notificationId = initialResponse.notification.request.identifier;
      if (shouldDedupSkip(data, notificationId)) return;

      if (shouldApplyNotificationFilter()) {
        const filter = evaluateNotificationFilter(data);
        if (filter.ignore) {
          recordNotificationFilteredTotal({ reason: filter.reason ?? "filtered" });
          emitNotificationFiltered(filter.reason ?? "filtered", {
            stage: "initial_response",
            notification_id: notificationId,
          });
          emitDriverTelemetry("push.notification.ignored", {
            source: "core.notifications.provider",
            stage: "initial_response",
            notification_id: notificationId,
            reason: filter.reason ?? "ignored",
          });
          return;
        }
      }
      const payload = parsePayload(data, initialResponse.actionIdentifier);
      if (isSilentPayload(data)) {
        await processSilentPayload(data);
        return;
      }
      await navigateFromPayload(payload);
    })();
  }, [
    Notifications,
    status,
    activeContext?.context_type,
    evaluateNotificationFilter,
    navigateFromPayload,
    parsePayload,
    processSilentPayload,
    shouldApplyNotificationFilter,
  ]);

  useEffect(() => {
    return () => {
      if (pendingPayloadTimerRef.current) {
        clearTimeout(pendingPayloadTimerRef.current);
        pendingPayloadTimerRef.current = null;
      }
    };
  }, []);

  const isCompanyContext = activeContext?.context_type === "company";
  const isDriverContext = activeContext?.context_type === "driver";

  return (
    <>
      {isDriverContext ? <DriverNotificationsBridge /> : null}
      {isCompanyContext ? (
        <CompanyNotificationsBridge>{children}</CompanyNotificationsBridge>
      ) : (
        children
      )}
    </>
  );
}
