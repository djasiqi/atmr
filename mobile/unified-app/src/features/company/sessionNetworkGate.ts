import { useEffect, useState } from "react";
import {
  isCompanySessionNetworkReady,
  subscribeCompanySessionNetworkReady,
} from "../../core/network/companySessionNetworkGate";

/**
 * COMPANY-AUTH-GATE-01 — le ready UI (snapshot local) n’ouvre pas le réseau.
 * Seule la barrière partagée après SESSION_READY bootstrap autorise les GET.
 */
export function useCompanySessionNetworkReady(): boolean {
  const [ready, setReady] = useState(() => isCompanySessionNetworkReady());
  useEffect(() => {
    setReady(isCompanySessionNetworkReady());
    return subscribeCompanySessionNetworkReady(() => {
      setReady(isCompanySessionNetworkReady());
    });
  }, []);
  return ready;
}
