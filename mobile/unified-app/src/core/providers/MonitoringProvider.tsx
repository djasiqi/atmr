import { useEffect } from "../reactCompat";
import type { PropsWithChildren } from "../reactCompat";
import { setDriverTelemetrySink } from "../observability/driverTelemetry";
import { sendIngestEvent } from "../observability/ingestAdapter";
import * as Sentry from "@sentry/react-native";
import * as Updates from "expo-updates";
import Constants from "expo-constants";

function applyFleetMapSentryContext(): void {
  try {
    Sentry.setTag("expo_update_id", Updates.updateId ?? "embedded");
    Sentry.setTag(
      "is_embedded_launch",
      String(Updates.isEmbeddedLaunch ?? true)
    );
    Sentry.setTag(
      "runtime_version",
      Updates.runtimeVersion ?? Constants.expoConfig?.runtimeVersion ?? "unknown"
    );
  } catch {
    // Best effort — monitoring ne doit pas bloquer le démarrage
  }
}

export function MonitoringProvider({ children }: PropsWithChildren) {
  useEffect(() => {
    const dsn = process.env.EXPO_PUBLIC_SENTRY_DSN;
    if (typeof dsn === "string" && dsn.length > 0) {
      Sentry.init({
        dsn,
        enableNative: true,
        enableNativeNagger: false,
        tracesSampleRate: 0.2,
        environment: process.env.EXPO_PUBLIC_APP_ENV ?? "development",
        // expo-updates peut rejeter "Failed to load all assets" en arrière-plan
        // (réseau instable, bascule Wi‑Fi/4G) alors que l'app continue sur le bundle embarqué.
        ignoreErrors: ["Failed to load all assets"],
      });
      applyFleetMapSentryContext();
    }

    setDriverTelemetrySink((event, payload) => {
      sendIngestEvent(event, payload as Record<string, unknown>);
      if (__DEV__) {
        console.info(`[driver-telemetry] ${event}`, payload);
      }
    });

    return () => {
      setDriverTelemetrySink(null);
    };
  }, []);

  return children;
}
