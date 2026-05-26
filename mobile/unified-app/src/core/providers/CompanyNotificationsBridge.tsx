import { AppState, Platform } from "react-native";
import { useRouter } from "expo-router";
import { useCallback, useMemo } from "react";

import { isFeatureEnabled } from "../featureFlags/registry";
import { useEffect } from "../reactCompat";
import { useSession } from "../sessionProvider";
import { getExpoNotificationsModule } from "../notifications/expoNotificationsCompat";
import {
  extractEventIdFromData,
  shouldSkipCrossChannelEvent,
} from "../notifications/crossChannelDedup";
import { useRegisterPushTokenEffect } from "../notifications/registerPushToken";
import {
  initDriverFirebaseMessaging,
  disposeDriverFirebaseMessaging,
} from "../../features/driver/firebaseMessaging";
import { registerCompanyPushToken } from "../../features/company/api/companyPushApi";
import {
  navigateFromCompanyPush,
  parseCompanyPushPayload,
} from "../../features/company/push/companyPush";

type Props = {
  children?: React.ReactNode;
};

export function CompanyNotificationsBridge({ children }: Props) {
  const router = useRouter();
  const { status, activeContext } = useSession();
  const Notifications = getExpoNotificationsModule();

  const companyId = useMemo(() => {
    if (activeContext?.context_type !== "company") return null;
    const id = Number(activeContext.organization_id);
    return Number.isFinite(id) ? id : null;
  }, [activeContext?.context_type, activeContext?.organization_id]);

  const pushEnabled =
    isFeatureEnabled("company_push_enabled") &&
    status === "ready" &&
    activeContext?.context_type === "company" &&
    companyId != null;

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

  useRegisterPushTokenEffect({
    enabled: pushEnabled,
    fcmEnabled: isFeatureEnabled("driver_fcm_native_enabled"),
    callbacks: registerCallbacks,
    telemetrySource: "company.notifications.bridge",
  });

  const handlePushData = useCallback(
    async (data: Record<string, unknown>) => {
      if (
        shouldSkipCrossChannelEvent({
          eventId: extractEventIdFromData(data),
          bookingId: Number(data.booking_id ?? data.bookingId) || null,
          type: typeof data.type === "string" ? data.type : null,
        })
      ) {
        return;
      }
      const payload = parseCompanyPushPayload(data);
      if (!payload) return;

      if (AppState.currentState === "active" && Notifications) {
        await Notifications.scheduleNotificationAsync({
          content: {
            title: "Liri Entreprise",
            body: "Nouvelle activité sur vos courses",
            data,
          },
          trigger: null,
        }).catch(() => undefined);
      }
      navigateFromCompanyPush(router, payload);
    },
    [Notifications, router]
  );

  useEffect(() => {
    if (!pushEnabled || Platform.OS === "web") return;

    void initDriverFirebaseMessaging(async (payload) => {
      if (!payload || typeof payload !== "object") return;
      await handlePushData(payload as Record<string, unknown>);
    });

    return () => {
      disposeDriverFirebaseMessaging();
    };
  }, [handlePushData, pushEnabled]);

  useEffect(() => {
    if (!pushEnabled || !Notifications) return;

    const received = Notifications.addNotificationReceivedListener((event) => {
      const data = event.request.content.data;
      if (!data || typeof data !== "object") return;
      void handlePushData(data as Record<string, unknown>);
    });

    const opened = Notifications.addNotificationResponseReceivedListener((response) => {
      const data = response.notification.request.content.data;
      if (!data || typeof data !== "object") return;
      void handlePushData(data as Record<string, unknown>);
    });

    return () => {
      received.remove();
      opened.remove();
    };
  }, [Notifications, handlePushData, pushEnabled]);

  return children ?? null;
}
