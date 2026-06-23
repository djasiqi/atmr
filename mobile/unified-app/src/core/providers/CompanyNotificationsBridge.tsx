import { AppState, Platform } from "react-native";
import { useRouter } from "expo-router";
import { useCallback, useMemo, useRef } from "react";
import NetInfo from "@react-native-community/netinfo";
import { useQueryClient } from "@tanstack/react-query";

import { isFeatureEnabled } from "../featureFlags/registry";
import { useEffect } from "../reactCompat";
import { useSession } from "../sessionProvider";
import { emitDriverTelemetry } from "../observability/driverTelemetry";
import { getExpoNotificationsModule } from "../notifications/expoNotificationsCompat";
import {
  extractEventIdFromData,
  shouldSkipCrossChannelEvent,
} from "../notifications/crossChannelDedup";
import { shouldIgnoreNotification } from "../notifications/shouldIgnoreNotification";
import { useRegisterPushTokenEffect } from "../notifications/registerPushToken";
import {
  initDriverFirebaseMessaging,
  disposeDriverFirebaseMessaging,
} from "../../features/driver/firebaseMessaging";
import { registerCompanyPushToken } from "../../features/company/api/companyPushApi";
import { reportCompanyPushTelemetry } from "../../features/company/api/companyPushTelemetryApi";
import {
  markOfferPushOpened,
  navigateFromCompanyPush,
  parseCompanyPushPayload,
  resolveCompanyPushTitleBody,
  type CompanyPushPayload,
} from "../../features/company/push/companyPush";
import {
  consumePendingCompanyPushPress,
  registerCompanyNotifeeForegroundPressHandler,
} from "../../features/company/push/companyNotifeePress";
import { invalidateInstitutionOfferQueries } from "../../features/company/realtime/useInstitutionOffersRealtimeListener";

type Props = {
  children?: React.ReactNode;
};

export function CompanyNotificationsBridge({ children }: Props) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { status, activeContext } = useSession();
  const Notifications = getExpoNotificationsModule();
  const initialResponseConsumedRef = useRef(false);

  const companyContextId = useMemo(() => {
    if (activeContext?.context_type !== "company") return null;
    return activeContext.context_id ?? null;
  }, [activeContext?.context_id, activeContext?.context_type]);

  const companyId = useMemo(() => {
    if (activeContext?.context_type !== "company") return null;
    const id = Number(activeContext.organization_id);
    return Number.isFinite(id) ? id : null;
  }, [activeContext?.context_type, activeContext?.organization_id]);

  const refreshInstitutionOffers = useCallback(
    (offerId?: number) => {
      if (!companyContextId) return;
      void invalidateInstitutionOfferQueries(queryClient, companyContextId, offerId);
    },
    [companyContextId, queryClient]
  );

  const pushEnabled =
    isFeatureEnabled("company_push_enabled") &&
    status === "ready" &&
    activeContext?.context_type === "company" &&
    companyId != null;

  const filterContext = useMemo(
    () => ({
      contextType: activeContext?.context_type,
      userId: null,
      companyId,
    }),
    [activeContext?.context_type, companyId]
  );

  const registerCallbacks = useMemo(
    () => ({
      registerExpo: async (input: {
        token: string;
        deviceId: string;
        platform: "ios" | "android";
      }) => {
        if (companyId == null) return;
        await registerCompanyPushToken({
          token: input.token,
          companyId,
          deviceId: input.deviceId,
          platform: input.platform,
          provider: "expo",
        });
      },
      registerFcm: async (input: {
        token: string;
        deviceId: string;
        platform: "ios" | "android";
      }) => {
        if (companyId == null) return;
        await registerCompanyPushToken({
          token: input.token,
          companyId,
          deviceId: input.deviceId,
          platform: input.platform,
          provider: "fcm",
        });
      },
    }),
    [companyId]
  );

  useEffect(() => {
    if (Platform.OS === "web") return;
    emitDriverTelemetry("push.company.register_gate", {
      source: "company.notifications.bridge",
      context_type: activeContext?.context_type ?? null,
      status,
      company_id: companyId,
      company_push_enabled: isFeatureEnabled("company_push_enabled"),
      push_enabled: pushEnabled,
    });
  }, [activeContext?.context_type, companyId, pushEnabled, status]);

  useRegisterPushTokenEffect({
    enabled: pushEnabled,
    fcmEnabled: isFeatureEnabled("driver_fcm_native_enabled"),
    callbacks: registerCallbacks,
    telemetrySource: "company.notifications.bridge",
  });

  const shouldProcessPayload = useCallback(
    (data: Record<string, unknown>, notificationId?: string | null): boolean => {
      const filter = shouldIgnoreNotification(data, filterContext);
      if (filter.ignore) {
        emitDriverTelemetry("push.notification.ignored", {
          source: "company.notifications.bridge",
          reason: filter.reason ?? "ignored",
          notification_id: notificationId ?? null,
        });
        return false;
      }
      if (
        shouldSkipCrossChannelEvent({
          eventId: extractEventIdFromData(data),
          notificationId,
          bookingId: Number(data.booking_id ?? data.bookingId) || null,
          type: typeof data.type === "string" ? data.type : null,
        })
      ) {
        return false;
      }
      return true;
    },
    [filterContext]
  );

  const recordOpenedTelemetry = useCallback(
    async (payload: CompanyPushPayload) => {
      if (payload.type !== "new_request") return;
      if (payload.offer_id != null) {
        markOfferPushOpened(payload.offer_id);
      }
      await reportCompanyPushTelemetry({
        event: "company_push.new_request.opened",
        offerId: payload.offer_id,
        requestId: payload.request_id,
      });
    },
    []
  );

  const navigateFromPush = useCallback(
    async (data: Record<string, unknown>, options?: { fromUserTap?: boolean }) => {
      if (!shouldProcessPayload(data)) return;
      const payload = parseCompanyPushPayload(data);
      if (!payload) return;

      if (options?.fromUserTap) {
        const net = await NetInfo.fetch().catch(() => null);
        const online =
          Boolean(net?.isConnected) && net?.isInternetReachable !== false;
        if (!online) {
          await reportCompanyPushTelemetry({
            event: "company_push.new_request.tap_without_network",
            offerId: payload.offer_id,
            requestId: payload.request_id,
          });
        }
        if (payload.type === "new_request") {
          await recordOpenedTelemetry(payload);
        }
      }

      if (payload.type === "new_request" || payload.type === "request_updated") {
        refreshInstitutionOffers(payload.offer_id);
      }

      navigateFromCompanyPush(router, payload);
    },
    [recordOpenedTelemetry, refreshInstitutionOffers, router, shouldProcessPayload]
  );

  const showForegroundNotification = useCallback(
    async (data: Record<string, unknown>) => {
      if (!Notifications || AppState.currentState !== "active") return;
      if (!shouldProcessPayload(data)) return;
      const payload = parseCompanyPushPayload(data);
      if (
        payload?.type === "new_request" ||
        payload?.type === "request_updated"
      ) {
        refreshInstitutionOffers(payload.offer_id);
      }
      const { title, body } = resolveCompanyPushTitleBody(data);
      await Notifications.scheduleNotificationAsync({
        content: {
          title,
          body,
          data,
        },
        trigger: null,
      }).catch(() => undefined);
    },
    [Notifications, refreshInstitutionOffers, shouldProcessPayload]
  );

  useEffect(() => {
    if (!pushEnabled || Platform.OS === "web") return;

    void initDriverFirebaseMessaging(async (payload, meta) => {
      if (!payload || typeof payload !== "object") return;
      const data = payload as Record<string, unknown>;
      if (meta?.source === "foreground") {
        await showForegroundNotification(data);
        return;
      }
      // Background / headless : pas de navigation automatique.
      // La notif locale (displayLocalDriverPush) + tap utilisateur
      // déclenchent navigateFromPush via Notifee / Expo / cold start.
    });

    return () => {
      disposeDriverFirebaseMessaging();
    };
  }, [navigateFromPush, pushEnabled, showForegroundNotification]);

  useEffect(() => {
    if (!pushEnabled || Platform.OS === "web") return;

    let disposeNotifeePress: (() => void) | null = null;
    void (async () => {
      disposeNotifeePress = await registerCompanyNotifeeForegroundPressHandler((data) => {
        void navigateFromPush(data, { fromUserTap: true });
      });
    })();

    return () => {
      disposeNotifeePress?.();
      disposeNotifeePress = null;
    };
  }, [navigateFromPush, pushEnabled]);

  useEffect(() => {
    if (!pushEnabled || !Notifications) return;

    const received = Notifications.addNotificationReceivedListener((event) => {
      const data = event.request.content.data;
      if (!data || typeof data !== "object") return;
      void showForegroundNotification(data as Record<string, unknown>);
    });

    const opened = Notifications.addNotificationResponseReceivedListener((response) => {
      const data = response.notification.request.content.data;
      if (!data || typeof data !== "object") return;
      void navigateFromPush(data as Record<string, unknown>, { fromUserTap: true });
    });

    return () => {
      received.remove();
      opened.remove();
    };
  }, [Notifications, navigateFromPush, pushEnabled, showForegroundNotification]);

  useEffect(() => {
    if (!pushEnabled) return;
    if (initialResponseConsumedRef.current) return;
    initialResponseConsumedRef.current = true;

    void (async () => {
      const pending = await consumePendingCompanyPushPress();
      if (pending) {
        await navigateFromPush(pending, { fromUserTap: true });
        return;
      }

      if (!Notifications) return;
      const initialResponse = await Notifications.getLastNotificationResponseAsync().catch(
        () => null
      );
      if (!initialResponse?.notification) return;
      const data = initialResponse.notification.request.content.data;
      if (!data || typeof data !== "object") return;
      await navigateFromPush(data as Record<string, unknown>, { fromUserTap: true });
    })();
  }, [Notifications, navigateFromPush, pushEnabled]);

  return children ?? null;
}
