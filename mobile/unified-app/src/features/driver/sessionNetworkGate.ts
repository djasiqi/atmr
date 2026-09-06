import { useEffect, useState } from "react";
import {
  isDriverSessionNetworkReady,
  subscribeDriverSessionNetworkReady,
} from "../../core/network/driverSessionNetworkGate";

/**
 * DRIVER-RUNTIME-01B — le ready UI (snapshot local) n’ouvre pas le réseau.
 * Seule la barrière partagée après SESSION_READY bootstrap autorise les GET.
 */
export function isDriverNetworkSessionReady(
  _status?: string | null
): boolean {
  return isDriverSessionNetworkReady();
}

/** S’abonne au flag réseau pour re-rendre quand le bootstrap ouvre la barrière. */
export function useDriverSessionNetworkReady(): boolean {
  const [ready, setReady] = useState(() => isDriverSessionNetworkReady());
  useEffect(() => {
    setReady(isDriverSessionNetworkReady());
    return subscribeDriverSessionNetworkReady(() => {
      setReady(isDriverSessionNetworkReady());
    });
  }, []);
  return ready;
}
