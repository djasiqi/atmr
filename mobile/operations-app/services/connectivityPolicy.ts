/**
 * connectivityPolicy — Décide du mode réseau (normal / dégradé / ultra-éco) à partir des signaux.
 * Plan migration 2G/3G — Phase 1.
 * Source unique pour les décisions réseau ; supporte le mode forcé diagnostic (QA).
 */

import AsyncStorage from "@react-native-async-storage/async-storage";
import { AppState, InteractionManager, type AppStateStatus } from "react-native";
import { getLogger } from "@/utils/logger";
import {
  getNetworkStateSnapshot,
  subscribeToNetworkState,
} from "./networkState";

const log = getLogger("ConnectivityPolicy");

const FORCED_MODE_KEY = "@atmr:forced_network_mode";

export type NetworkMode = "normal" | "degraded" | "ultra_eco";

let cachedForcedMode: NetworkMode | null = null;
let appState: AppStateStatus = "active";
let appStateUnsub: (() => void) | null = null;
const listeners = new Set<(mode: NetworkMode) => void>();

function notifyListeners(mode: NetworkMode): void {
  listeners.forEach((fn) => {
    try {
      fn(mode);
    } catch (e) {
      log.warn("connectivityPolicy listener error", { error: e });
    }
  });
}

/**
 * Charge le mode forcé depuis AsyncStorage (diagnostic QA).
 */
async function loadForcedMode(): Promise<NetworkMode | null> {
  try {
    const raw = await AsyncStorage.getItem(FORCED_MODE_KEY);
    if (raw === "normal" || raw === "degraded" || raw === "ultra_eco") {
      return raw;
    }
  } catch (e) {
    log.warn("load forced mode failed", { error: e });
  }
  return null;
}

/**
 * Retourne le mode forcé (sync, depuis le cache).
 */
function getForcedModeSync(): NetworkMode | null {
  return cachedForcedMode;
}

/**
 * Définit le mode forcé (diagnostic QA). null = désactiver.
 */
export async function setForcedMode(mode: NetworkMode | null): Promise<void> {
  if (mode === null) {
    await AsyncStorage.removeItem(FORCED_MODE_KEY);
    cachedForcedMode = null;
  } else {
    await AsyncStorage.setItem(FORCED_MODE_KEY, mode);
    cachedForcedMode = mode;
  }
  const current = getMode();
  notifyListeners(current);
  log.info("forced mode updated", { mode, current });
}

/**
 * Dérive le mode à partir des signaux réseau et AppState.
 */
function deriveMode(): NetworkMode {
  const net = getNetworkStateSnapshot();
  const isOnline =
    net?.isConnected === true && net?.isInternetReachable !== false;
  const netType = (net?.type as string) ?? "unknown";
  const isForeground = appState === "active";

  if (!isOnline) {
    return "ultra_eco";
  }

  // 2G ou connexion très limitée
  if (netType === "none" || netType === "unknown") {
    return "ultra_eco";
  }

  // WiFi ou 4G = normal si foreground
  if (netType === "wifi" || netType === "cellular") {
    if (isForeground) {
      return "normal";
    }
    return "degraded";
  }

  // Par défaut
  return isForeground ? "normal" : "degraded";
}

/**
 * Retourne le mode réseau actuel.
 * Si un mode forcé est défini (diagnostic QA), il override la dérivation.
 */
export function getMode(): NetworkMode {
  const forced = getForcedModeSync();
  if (forced !== null) {
    return forced;
  }
  return deriveMode();
}

/**
 * S'abonner aux changements de mode.
 * Retourne une fonction de désabonnement.
 */
export function subscribeToMode(onChange: (mode: NetworkMode) => void): () => void {
  listeners.add(onChange);
  onChange(getMode());
  return () => {
    listeners.delete(onChange);
  };
}

/**
 * Initialise la policy (écoute networkState et AppState).
 * Appelé par syncEngine.start() ou au boot.
 */
export function initConnectivityPolicy(): void {
  if (appStateUnsub) return;

  // Charger le mode forcé au démarrage
  loadForcedMode().then((m) => {
    cachedForcedMode = m;
    if (m) log.info("forced mode loaded", { mode: m });
  });

  subscribeToNetworkState(() => {
    const mode = getMode();
    notifyListeners(mode);
  });

  const appStateSub = AppState.addEventListener("change", (next: AppStateStatus) => {
    appState = next;
    // Déferrer notifyListeners en background pour éviter ANR (transition prioritaire).
    if (next === "background" || next === "inactive") {
      InteractionManager.runAfterInteractions(() => {
        const m = getMode();
        notifyListeners(m);
      });
    } else {
      const mode = getMode();
      notifyListeners(mode);
    }
  });
  appStateUnsub = () => appStateSub.remove();

  log.info("connectivityPolicy initialized");
}

/**
 * Arrête la policy (cleanup).
 */
export function stopConnectivityPolicy(): void {
  appStateUnsub?.();
  appStateUnsub = null;
  listeners.clear();
  cachedForcedMode = null;
  log.info("connectivityPolicy stopped");
}
