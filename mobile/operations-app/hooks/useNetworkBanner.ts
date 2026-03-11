/**
 * P1.C — Hook pour bannière "Hors ligne".
 * Event-driven : NetInfo.addEventListener + AppState (pas de poll).
 * Perf : online stable → aucune boucle, offline → updates via NetInfo.
 */

import { useEffect, useState } from "react";
import { AppState, AppStateStatus, InteractionManager } from "react-native";
import {
  getNetworkStateSnapshot,
  subscribeToNetworkState,
} from "@/services/networkState";

function computeIsOffline(snapshot: Record<string, unknown> | null): boolean {
  if (!snapshot) return false;
  const isConnected = snapshot.isConnected;
  const isInternetReachable = snapshot.isInternetReachable;

  // isConnected === false → offline certain
  if (isConnected === false) return true;
  // isInternetReachable === false → offline certain
  if (isInternetReachable === false) return true;
  // isInternetReachable === null → incertain (démarrage), ne pas afficher
  if (isInternetReachable === null) return false;
  // isConnected === null → incertain, ne pas afficher
  if (isConnected === null) return false;
  return false;
}

export function useNetworkBanner(): boolean {
  const [isOffline, setIsOffline] = useState(false);

  useEffect(() => {
    const check = () => {
      setIsOffline(computeIsOffline(getNetworkStateSnapshot()));
    };

    check();
    const unsub = subscribeToNetworkState(check);
    const appSub = AppState.addEventListener("change", (state: AppStateStatus) => {
      if (state === "active") {
        InteractionManager.runAfterInteractions(() => {
          setTimeout(check, 0);
        });
      }
    });

    return () => {
      unsub();
      appSub.remove();
    };
  }, []);

  return isOffline;
}
