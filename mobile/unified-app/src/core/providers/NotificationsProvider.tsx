import { AppState, Platform } from "react-native";
import { useRouter } from "expo-router";
import { useCallback, useRef } from "react";
import { emitDriverTelemetry } from "../observability/driverTelemetry";
import { isFeatureEnabled } from "../featureFlags/registry";
import { PropsWithChildren, useEffect } from "../reactCompat";
import { useSession } from "../sessionProvider";
import { registerDriverPushToken } from "../../features/driver/api/driverHttp";
import { DriverPushPayload, handleDriverPushQuickAction } from "../../features/driver/push";
import { ensureDriverNotificationChannels } from "../../features/driver/notificationChannels";
import { ensureDriverNotificationActions } from "../../features/driver/notificationActions";
import { ensureDriverNotificationGrouping } from "../../features/driver/notificationGrouping";
import { handleSilentPushPayload, isSilentPayload } from "../../features/driver/silentNotifications";
import {
  disposeDriverFirebaseMessaging,
  initDriverFirebaseMessaging,
} from "../../features/driver/firebaseMessaging";
import { resolveDriverDeepLink } from "../navigation/deepLinkHandler";
import { configureMissionBarIOS } from "../../features/driver/missionBarIOS";
import { registerMissionBarBackgroundHandlers } from "../../features/driver/missionBarBackground";
import { appendSessionJournalEvent } from "../observability/sessionJournal";
import { shouldIgnoreNotification } from "../notifications/shouldIgnoreNotification";
import { getExpoNotificationsModule } from "../notifications/expoNotificationsCompat";

export function NotificationsProvider({ children }: PropsWithChildren) {
  const Notifications = getExpoNotificationsModule();
  const router = useRouter();
  const { status, activeContext, bootstrap } = useSession();
  const pendingPayloadRef = useRef<DriverPushPayload | null>(null);
  const pendingPayloadTimedOutRef = useRef(false);
  const pendingPayloadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const DEEPLINK_BOOTSTRAP_TIMEOUT_MS = Number(
    process.env.EXPO_PUBLIC_DRIVER_DEEPLINK_BOOTSTRAP_TIMEOUT_MS ?? "8000"
  );

  const evaluateNotificationFilter = useCallback(
    (input: unknown) =>
      shouldIgnoreNotification(input, {
        contextType: activeContext?.context_type ?? null,
        userId: bootstrap?.user?.id ?? null,
        companyId: activeContext?.organization_id ?? null,
      }),
    [activeContext?.context_type, activeContext?.organization_id, bootstrap?.user?.id]
  );

  const normalizePushType = useCallback((raw: string): DriverPushPayload["type"] | null => {
    if (raw === "mission_assigned" || raw === "booking_assigned") return "mission_assigned";
    if (raw === "mission_updated" || raw === "booking_updated") return "mission_updated";
    if (raw === "mission_cancelled" || raw === "booking_cancelled") return "mission_cancelled";
    if (raw === "mission_reassigned" || raw === "booking_reassigned") return "mission_reassigned";
    if (raw === "reminder_action") return "reminder_action";
    if (raw === "informative" || raw === "delay") return "informative";
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

  const parsePayload = useCallback((input: unknown, actionIdentifier?: string): DriverPushPayload | null => {
    if (!input || typeof input !== "object") return null;
    const value = input as Record<string, unknown>;
    const missionIdRaw =
      value.mission_id ??
      value.missionId ??
      value.booking_id ??
      value.bookingId;
    const missionId = Number(missionIdRaw);
    const deepLink =
      typeof value.deepLink === "string"
        ? value.deepLink
        : typeof value.deep_link === "string"
          ? value.deep_link
          : undefined;
    const deepLinkTarget = parseDeepLinkTarget(deepLink);
    const resolvedMissionId = Number.isFinite(missionId) ? missionId : deepLinkTarget?.missionId ?? null;
    if (!Number.isFinite(resolvedMissionId)) return null;
    const rawType = String(value.type ?? "mission_updated");
    const type = normalizePushType(rawType);
    if (!type) return null;
    const schema =
      value.mission_id != null || value.missionId != null
        ? "mission_v2"
        : value.booking_id != null || value.bookingId != null
          ? "booking_v1"
          : "unknown";
    return {
      mission_id: resolvedMissionId as number,
      type,
      event_id: typeof value.event_id === "string" ? value.event_id : undefined,
      action:
        normalizeQuickAction(value.action) ??
        normalizeQuickAction(actionIdentifier),
      deep_link: deepLink,
      payload_schema: schema,
    };
  }, [normalizePushType, parseDeepLinkTarget, normalizeQuickAction]);

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

  const routePayload = useCallback(async (payload: DriverPushPayload) => {
    void appendSessionJournalEvent("push.route.start", {
      mission_id: payload.mission_id,
      type: payload.type,
      action: payload.action ?? null,
    }, activeContext?.context_id ?? null);
    if (payload.action) {
      emitDriverTelemetry("push.quick_action.dispatch", {
        source: "core.notifications.provider",
        mission_id: payload.mission_id,
        action: payload.action,
      });
    }
    await handleDriverPushQuickAction(payload);
    if (payload.action) {
      emitDriverTelemetry("push.quick_action.success", {
        source: "core.notifications.provider",
        mission_id: payload.mission_id,
        action: payload.action,
      });
    }
    if (
      payload.type === "mission_updated" ||
      payload.type === "mission_reassigned" ||
      payload.type === "mission_cancelled"
    ) {
      await triggerDriverResync("push_update", payload.mission_id);
    }
    if (payload.type === "informative") {
      const deepLinkTarget = parseDeepLinkTarget(payload.deep_link);
      if (deepLinkTarget) {
        router.push(deepLinkTarget.route as any);
      }
      return;
    }
    const deepLinkTarget = parseDeepLinkTarget(payload.deep_link);
    if (deepLinkTarget?.route) {
      router.push(deepLinkTarget.route as any);
    } else {
      router.push(`/(app)/(driver)/missions/${payload.mission_id}` as any);
    }
  }, [activeContext?.context_id, parseDeepLinkTarget, router, triggerDriverResync]);

  const queuePendingPayload = useCallback((payload: DriverPushPayload) => {
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
        void triggerDriverResync("push_route_fallback", payload.mission_id);
        router.push("/(app)/(driver)/missions" as any);
      }
    }, DEEPLINK_BOOTSTRAP_TIMEOUT_MS);
  }, [DEEPLINK_BOOTSTRAP_TIMEOUT_MS, activeContext?.context_type, router, status, triggerDriverResync]);

  const clearPendingPayload = useCallback(() => {
    pendingPayloadRef.current = null;
    pendingPayloadTimedOutRef.current = false;
    if (pendingPayloadTimerRef.current) {
      clearTimeout(pendingPayloadTimerRef.current);
      pendingPayloadTimerRef.current = null;
    }
  }, []);

  const navigateFromPayload = useCallback(async (payload: DriverPushPayload | null) => {
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
  }, [activeContext?.context_type, queuePendingPayload, routePayload, status]);

  useEffect(() => {
    if (!Notifications) return;
    Notifications.setNotificationHandler({
      handleNotification: async (notification) => {
        const filter = evaluateNotificationFilter(notification.request.content.data);
        if (filter.ignore) {
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
        return {
          shouldShowBanner: true,
          shouldShowList: true,
          shouldPlaySound: false,
          shouldSetBadge: false,
        };
      },
    });
  }, [Notifications, evaluateNotificationFilter]);

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
      void ensureDriverNotificationChannels().catch(() => undefined);
      void ensureDriverNotificationActions().catch(() => undefined);
      void ensureDriverNotificationGrouping().catch(() => undefined);
      void configureMissionBarIOS().catch(() => undefined);
      registerMissionBarBackgroundHandlers();
    }
    void Notifications.requestPermissionsAsync().catch(() => undefined);
    const received = Notifications.addNotificationReceivedListener((notification) => {
      const data = notification.request.content.data;
      const filter = evaluateNotificationFilter(data);
      if (filter.ignore) {
        emitDriverTelemetry("push.notification.ignored", {
          source: "core.notifications.provider",
          stage: "received_listener",
          notification_id: notification.request.identifier,
          reason: filter.reason ?? "ignored",
        });
        return;
      }
      void handleSilentPushPayload(data, async (missionId) => {
        await triggerDriverResync("push_update", missionId);
      });
      emitDriverTelemetry("push.notification.received", {
        source: "core.notifications.provider",
        app_state: AppState.currentState,
        notification_id: notification.request.identifier,
      });
    });
    const opened = Notifications.addNotificationResponseReceivedListener((response) => {
      const data = response.notification.request.content.data;
      const filter = evaluateNotificationFilter(data);
      if (filter.ignore) {
        emitDriverTelemetry("push.notification.ignored", {
          source: "core.notifications.provider",
          stage: "response_listener",
          notification_id: response.notification.request.identifier,
          reason: filter.reason ?? "ignored",
        });
        return;
      }
      const payload = parsePayload(data, response.actionIdentifier);
      if (isSilentPayload(data)) {
        void handleSilentPushPayload(data, async (missionId) => {
          await triggerDriverResync("push_update", missionId);
        });
        return;
      }
      emitDriverTelemetry("push.notification.opened", {
        source: "core.notifications.provider",
        notification_id: response.notification.request.identifier,
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
    triggerDriverResync,
  ]);

  useEffect(() => {
    if (!isFeatureEnabled("driver_fcm_native_enabled")) return;
    void initDriverFirebaseMessaging(async (payload) => {
      if (isSilentPayload(payload)) {
        await handleSilentPushPayload(payload, async (missionId) => {
          await triggerDriverResync("push_update", missionId);
        });
        return;
      }
      const filter = evaluateNotificationFilter(payload);
      if (filter.ignore) {
        emitDriverTelemetry("push.notification.ignored", {
          source: "core.notifications.provider",
          stage: "fcm_listener",
          reason: filter.reason ?? "ignored",
        });
        return;
      }
      const parsed = parsePayload(payload);
      await navigateFromPayload(parsed);
    });
    return () => {
      disposeDriverFirebaseMessaging();
    };
  }, [evaluateNotificationFilter, navigateFromPayload, parsePayload, triggerDriverResync]);

  useEffect(() => {
    if (!isFeatureEnabled("driver_push_enabled")) return;
    if (!Notifications) return;
    if (Platform.OS === "web") return;
    if (status !== "ready") return;
    if (activeContext?.context_type !== "driver") return;
    const driverId = Number(bootstrap?.user?.id);
    if (!Number.isFinite(driverId)) return;
    void (async () => {
      const token = await Notifications.getExpoPushTokenAsync().catch(() => null);
      if (!token?.data) return;
      await registerDriverPushToken({
        token: token.data,
        driverId,
        platform: Platform.OS === "ios" ? "ios" : "android",
        provider: "expo",
      }).catch(() => undefined);
      emitDriverTelemetry("push.token.registered", {
        source: "core.notifications.provider",
        driver_id: String(driverId),
      });
      const initialResponse = await Notifications.getLastNotificationResponseAsync().catch(() => null);
      if (initialResponse?.notification) {
        const filter = evaluateNotificationFilter(initialResponse.notification.request.content.data);
        if (filter.ignore) {
          emitDriverTelemetry("push.notification.ignored", {
            source: "core.notifications.provider",
            stage: "initial_response",
            notification_id: initialResponse.notification.request.identifier,
            reason: filter.reason ?? "ignored",
          });
          return;
        }
        const payload = parsePayload(
          initialResponse.notification.request.content.data,
          initialResponse.actionIdentifier
        );
        await navigateFromPayload(payload);
      }
    })();
  }, [
    Notifications,
    status,
    activeContext?.context_type,
    bootstrap?.user?.id,
    navigateFromPayload,
    parsePayload,
    evaluateNotificationFilter,
  ]);

  useEffect(() => {
    return () => {
      if (pendingPayloadTimerRef.current) {
        clearTimeout(pendingPayloadTimerRef.current);
        pendingPayloadTimerRef.current = null;
      }
    };
  }, []);

  return children;
}
