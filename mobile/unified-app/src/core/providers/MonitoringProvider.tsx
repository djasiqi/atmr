import { useEffect } from "../reactCompat";
import type { PropsWithChildren } from "../reactCompat";
import { setDriverTelemetrySink } from "../observability/driverTelemetry";
import { sendIngestEvent } from "../observability/ingestAdapter";
import * as Sentry from "@sentry/react-native";

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
      });
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
