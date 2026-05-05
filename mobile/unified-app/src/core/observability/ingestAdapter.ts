import { Platform } from "react-native";
import Constants from "expo-constants";

const INGEST_PORT = 7242;
const MAX_FAILURES = 3;
const WEB_INGEST_ENABLED = process.env.EXPO_PUBLIC_ENABLE_WEB_INGEST === "1";

const DISABLED_BY_ENV =
  process.env.EXPO_PUBLIC_DISABLE_INGEST === "1" ||
  (Platform.OS === "web" && !WEB_INGEST_ENABLED);

function resolveBaseUrl(): string {
  if (Platform.OS === "android") {
    const onEmulator =
      !Constants.isDevice ||
      Constants.deviceName?.toLowerCase().includes("emulator") ||
      Constants.deviceName?.toLowerCase().includes("sdk");
    return `http://${onEmulator ? "10.0.2.2" : "localhost"}:${INGEST_PORT}`;
  }
  return `http://localhost:${INGEST_PORT}`;
}

let failures = 0;
let suspended = DISABLED_BY_ENV || !__DEV__;

/** `AbortSignal.timeout` n’existe pas sous Hermes / certains navigateurs. */
function abortAfterMs(ms: number): { signal: AbortSignal; clear: () => void } {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  return {
    signal: controller.signal,
    clear: () => clearTimeout(id),
  };
}

export function sendIngestEvent(
  eventName: string,
  payload: Record<string, unknown>
): void {
  if (suspended) return;

  const body = JSON.stringify({
    event: eventName,
    ts: new Date().toISOString(),
    ...payload,
  });

  const { signal, clear } = abortAfterMs(3000);

  fetch(`${resolveBaseUrl()}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal,
  })
    .finally(clear)
    .then(() => {
      failures = 0;
    })
    .catch(() => {
      failures += 1;
      if (failures >= MAX_FAILURES) {
        suspended = true;
      }
    });
}

export function resetIngestAdapterForTests(): void {
  failures = 0;
  suspended = DISABLED_BY_ENV || !__DEV__;
}
