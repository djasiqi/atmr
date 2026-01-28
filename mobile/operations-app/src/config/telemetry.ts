/**
 * Config centralisée pour l’ingest de debug / télémétrie (Cursor / agent).
 * En dev : désactivable via EXPO_PUBLIC_DISABLE_INGEST=true pour éviter
 * ERR_CONNECTION_REFUSED quand aucun service n’écoute sur le port ingest.
 * Anti-spam : log une seule fois si ingest down, puis désactivation après N échecs.
 */
import { Platform } from "react-native";
import Constants from "expo-constants";

const INGEST_DEFAULT_PORT = "7242";
const INGEST_DEFAULT_ENDPOINT_ID =
  "5d8025f1-2a4d-4796-97fe-faa80ad8db74";
const MAX_CONSECUTIVE_FAILURES = 3;

function getEnv(name: string): string | undefined {
  const v = process.env[name as keyof typeof process.env];
  if (v != null && String(v).trim() !== "") return String(v).trim();
  const extra = (Constants.expoConfig?.extra as Record<string, unknown>) || {};
  const ev = extra[name];
  return ev != null ? String(ev).trim() : undefined;
}

/** true = ne jamais envoyer vers l’ingest. En dev, désactivé par défaut (pas de service sur 7242). */
export const isIngestDisabledByEnv = (): boolean => {
  const v = getEnv("EXPO_PUBLIC_DISABLE_INGEST")?.toLowerCase();
  if (v === "true") return true;
  if (v === "false") return false;
  if (__DEV__) return true;
  return false;
};

/** URL de base (sans /ingest/<id>) ou URL complète. Si complète, doit contenir /ingest/. */
function getIngestBaseFromEnv(): string | undefined {
  const url = getEnv("EXPO_PUBLIC_INGEST_URL");
  if (!url) return undefined;
  if (url.includes("/ingest/")) return url;
  const id = getEnv("EXPO_PUBLIC_INGEST_ENDPOINT_ID") || INGEST_DEFAULT_ENDPOINT_ID;
  const base = url.replace(/\/$/, "");
  return `${base}/ingest/${id}`;
}

/**
 * URL complète vers l’endpoint ingest (POST).
 * - En dev : web = localhost, Android emulator = 10.0.2.2, iOS = localhost.
 * - En prod / device réel : EXPO_PUBLIC_INGEST_URL requis (ex: IP LAN ou domaine).
 */
export function getIngestUrl(): string {
  const fromEnv = getIngestBaseFromEnv();
  if (fromEnv) return fromEnv;

  const port = getEnv("EXPO_PUBLIC_INGEST_PORT") || INGEST_DEFAULT_PORT;
  const id = getEnv("EXPO_PUBLIC_INGEST_ENDPOINT_ID") || INGEST_DEFAULT_ENDPOINT_ID;

  if (Platform.OS === "android") {
    const host = (Constants as any)?.expoConfig?.hostUri?.split(":")[0]
      || (Constants as any)?.manifest?.debuggerHost?.split(":")[0];
    if (host === "10.0.2.2" || host === "localhost" || host === "127.0.0.1") {
      return `http://10.0.2.2:${port}/ingest/${id}`;
    }
    if (host && host !== "localhost") {
      return `http://${host}:${port}/ingest/${id}`;
    }
    return `http://10.0.2.2:${port}/ingest/${id}`;
  }

  if (Platform.OS === "web" && typeof window !== "undefined") {
    const host = window.location?.hostname || "localhost";
    return `http://${host}:${port}/ingest/${id}`;
  }

  return `http://localhost:${port}/ingest/${id}`;
}

/** true si l’ingest est activé (env non désactivé et pas encore coupé après échecs). */
export function isTelemetryEnabled(): boolean {
  if (isIngestDisabledByEnv()) return false;
  return !ingestState.disabledAfterFailures;
}

// État module : anti-spam et désactivation après N échecs
const ingestState = {
  consecutiveFailures: 0,
  disabledAfterFailures: false,
  logOnceDone: false,
};

function logOnce(message: string): void {
  if (ingestState.logOnceDone) return;
  ingestState.logOnceDone = true;
  if (__DEV__) {
    console.warn("[ingest]", message);
  }
}

/**
 * Envoie un payload JSON vers l’endpoint ingest (fire-and-forget).
 * Ne lance pas, ne spamme pas : après MAX_CONSECUTIVE_FAILURES échecs, désactive
 * et ne fait plus rien. En dev si EXPO_PUBLIC_DISABLE_INGEST=true, no-op.
 */
export function sendIngestEvent(payload: Record<string, unknown>): void {
  if (isIngestDisabledByEnv() || ingestState.disabledAfterFailures) return;

  const url = getIngestUrl();
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).catch(() => {
    ingestState.consecutiveFailures += 1;
    if (ingestState.consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
      ingestState.disabledAfterFailures = true;
      logOnce(
        `Ingest unavailable (connection refused or network error). Disabled after ${MAX_CONSECUTIVE_FAILURES} failures. Set EXPO_PUBLIC_DISABLE_INGEST=true to silence.`
      );
    }
  });
}
