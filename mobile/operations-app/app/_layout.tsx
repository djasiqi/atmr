// app/_layout.tsx
import * as Sentry from "@sentry/react-native";
import React, { useEffect, useRef } from "react";
import { Slot, useSegments, useRouter } from "expo-router";
import { AuthProvider, useAuth } from "@/hooks/useAuth";
import {
  configureNotifications,
  initNotifications,
} from "@/services/notification";
import { Platform } from "react-native";
import Constants from "expo-constants";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { registerPushToken } from "@/services/api"; // si l’alias '@' n'est pas configuré: ../services/api

Sentry.init({
  dsn: "https://500ea836dce2e802b27109d857cb3534@o4509736814772224.ingest.de.sentry.io/4509736867201104",
  sendDefaultPii: true,
  tracesSampleRate: 1.0,
  profilesSampleRate: 1.0,
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,
  integrations: [Sentry.mobileReplayIntegration()],
});

export default function RootLayout() {
  return (
    <AuthProvider>
      <RootNav />
    </AuthProvider>
  );
}

function RootNav() {
  const {
    mode,
    isAuthenticated,
    isDriverAuthenticated,
    isEnterpriseAuthenticated,
    loading,
    driver,
  } = useAuth();
  const userId = driver?.id ?? null;

  const segments = useSegments();
  const router = useRouter();
  const registeringRef = useRef(false);

  // 🔐 Redirections selon l’état d’auth
  useEffect(() => {
    if (loading) return;
    const firstSegment = (segments[0] as string | undefined) ?? "";
    const isDriverAuthGroup = firstSegment === "(auth)";
    const isEnterpriseAuthGroup = firstSegment === "(enterprise-auth)";
    const isEnterpriseGroup = firstSegment === "(enterprise)";

    if (mode === "enterprise") {
      if (!isEnterpriseAuthenticated) {
        if (
          !isEnterpriseAuthGroup &&
          firstSegment !== "" &&
          firstSegment !== "index"
        ) {
          router.replace("/(enterprise-auth)/login" as any);
        }
      } else {
        if (
          isEnterpriseAuthGroup ||
          firstSegment === "(auth)" ||
          firstSegment === "" ||
          firstSegment === "index"
        ) {
          router.replace("/(enterprise)/dashboard" as any);
        }
      }
      if (firstSegment === "(tabs)" || firstSegment === "(dashboard)") {
        router.replace("/(enterprise)/dashboard" as any);
      }
    } else {
      if (!isDriverAuthenticated) {
        if (
          !isDriverAuthGroup &&
          firstSegment !== "" &&
          firstSegment !== "index"
        ) {
          router.replace("/(auth)/login");
        }
      } else if (
        isDriverAuthGroup ||
        firstSegment === "" ||
        firstSegment === "index" ||
        isEnterpriseAuthGroup ||
        isEnterpriseGroup
      ) {
        router.replace("/(tabs)/mission");
      }
    }
  }, [
    isDriverAuthenticated,
    isEnterpriseAuthenticated,
    loading,
    mode,
    router,
    segments,
  ]);

  // 🔔 Config + enregistrement push (quand prêt)
  useEffect(() => {
    if (loading || !isDriverAuthenticated || !driver) return;
    const currentUserId = driver.id;

    // Expo Go n’embarque pas google-services.json → skip pour éviter l’erreur Firebase
    if (Constants.appOwnership === "expo") {
      console.warn(
        "Skip FCM in Expo Go. Use a Development Build to test push."
      );
      return;
    }

    // éviter les doubles exécutions
    if (registeringRef.current) return;
    registeringRef.current = true;

    let cancelled = false;

    (async () => {
      try {
        console.log("🔔 Initialisation des notifications…");
        await configureNotifications();

        // Récupération token (device ou expo) avec peu de retries pour éviter le spam
        const tokens = await initNotifications({
          withExpoToken: true,
          maxRetries: 2,
        });

        if (cancelled) return;

        const device = (tokens as any)?.device ?? null;
        const expo = (tokens as any)?.expo ?? null;
        const tokenToUse = device || expo;

        if (!tokenToUse) {
          console.warn("⚠️ Aucun token push disponible (APK sans Firebase ?)");
          return;
        }

        // Empêcher les re-posts si inchangé (mémo par utilisateur si dispo)
        const key = currentUserId
          ? `push_token_${currentUserId}`
          : "push_token_default";
        const last = await AsyncStorage.getItem(key);
        if (last === tokenToUse) {
          console.log("🔔 Token inchangé, on ne ré-enregistre pas.");
          return;
        }

        await registerPushToken({
          token: tokenToUse,
          driverId: currentUserId,
        });
        await AsyncStorage.setItem(key, tokenToUse);
        console.log("✅ Push token enregistré côté backend");
      } catch (e: any) {
        console.warn(
          "❌ Enregistrement des notifications:",
          e?.message || String(e)
        );
      } finally {
        registeringRef.current = false;
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [driver, isDriverAuthenticated, loading]);

  return <Slot />;
}
