/**
 * P2.2 — Logs structurés auth/socket SRE-grade.
 *
 * - session_id (UUID au boot)
 * - refresh_cycle_id (par cycle refresh)
 * - dedupe anti-spam (5s par event|route|outcome|status)
 * - Contexte auto (logContext + network)
 * - Jamais de secrets/PII
 */

import { getLogger } from "@/utils/logger";
import { getLogContextSnapshot } from "./logContext";
import { getNetworkStateSnapshot } from "./networkState";

const log = getLogger("AuthLog");

/** Champs interdits (jamais logger de secrets). */
const FORBIDDEN_KEYS = ["token", "password", "refresh_token", "authorization", "cookie"];

/** session_id — UUID v4 au boot */
let sessionId: string | null = null;
function getSessionId(): string {
  if (!sessionId) {
    try {
      const Crypto = require("expo-crypto");
      sessionId = typeof Crypto.randomUUID === "function" ? Crypto.randomUUID() : `sess-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    } catch {
      sessionId = `sess-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    }
  }
  return sessionId!;
}

/** refresh_cycle_id — généré à chaque AUTH_REFRESH_START */
let currentRefreshCycleId: string | null = null;

/**
 * Démarre un cycle refresh, retourne l'id à propager.
 * Appelé uniquement à chaque tentative réelle (pas à chaque 401 en queue).
 *
 * @param route - "driver" | "enterprise"
 * @returns refresh_cycle_id à propager aux events AUTH_REFRESH_*
 */
export function beginRefreshCycle(route: string): string {
  try {
    const Crypto = require("expo-crypto");
    currentRefreshCycleId = typeof Crypto.randomUUID === "function" ? Crypto.randomUUID() : `rc-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  } catch {
    currentRefreshCycleId = `rc-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }
  return currentRefreshCycleId!;
}

export function getCurrentRefreshCycleId(): string | null {
  return currentRefreshCycleId;
}

/** Dedupe anti-spam — key -> lastTs. Inclut role pour socket (driver/enterprise distincts). */
const DEDUPE_WINDOW_MS = 5000;
const MAX_DEDUPE_ENTRIES = 200;
const dedupeMap = new Map<string, number>();

function shouldDedupe(event: string, payload: Record<string, unknown>): boolean {
  if (event === "LOGOUT_TRANSITION") return false;
  const route = String(payload.route ?? "");
  const outcome = String(payload.outcome ?? "");
  const status = String(payload.status ?? "");
  const role = String(payload.role ?? "");
  const key = `${event}|${route}|${outcome}|${status}|${role}`;
  const now = Date.now();
  const last = dedupeMap.get(key);
  if (last != null && now - last < DEDUPE_WINDOW_MS) return true;
  dedupeMap.set(key, now);
  // Éviction : supprimer les entrées anciennes si la map dépasse le seuil
  if (dedupeMap.size > MAX_DEDUPE_ENTRIES) {
    const expiry = now - DEDUPE_WINDOW_MS * 2;
    for (const [k, ts] of dedupeMap) {
      if (ts < expiry) dedupeMap.delete(k);
    }
  }
  return false;
}

function sanitize(payload: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(payload)) {
    const keyLower = k.toLowerCase();
    if (FORBIDDEN_KEYS.some((fk) => keyLower.includes(fk))) continue;
    out[k] = typeof v === "string" && v.length > 200 ? v.slice(0, 200) + "…" : v;
  }
  return out;
}

/**
 * Log structuré auth/socket (JSON, sans secrets).
 * Auto-injecte: session_id, refresh_cycle_id (si cycle en cours), logContext, network.
 *
 * Triggers refresh (optionnel): api_401 | proactive | socket_auth | boot
 */
export function logAuthEvent(
  event: string,
  payload: Record<string, unknown> = {}
): void {
  try {
    if (shouldDedupe(event, payload)) return;

    const ctx = getLogContextSnapshot();
    const network = getNetworkStateSnapshot();
    const base: Record<string, unknown> = {
      event,
      ts: Date.now(),
      session_id: getSessionId(),
      ...(currentRefreshCycleId && ["AUTH_REFRESH_START", "AUTH_REFRESH_SUCCESS", "AUTH_REFRESH_FAIL", "AUTH_401_HANDLING"].includes(event)
        ? { refresh_cycle_id: currentRefreshCycleId }
        : {}),
      ...ctx,
      ...(network ? { network } : {}),
      ...payload,
    };
    const safe = sanitize(base);
    const line = JSON.stringify(safe);
    if (__DEV__) {
      log.debug("auth event", { line });
    } else {
      log.info("auth event", { line });
    }
  } catch {
    // Fire-and-forget : ne jamais faire échouer l'appelant
  }
}

/** Alias pour événements socket. */
export const logSocketEvent = logAuthEvent;
