import { useEffect, useRef } from "react";
import { endPageLoad, startPageLoad } from "./perfKpi";

/**
 * Mesure le temps jusqu'à données utilisables (pattern inbox / thread / hub).
 */
export function usePerfScreenReady(screen: string, source: string, ready: boolean): void {
  const startedRef = useRef(false);
  useEffect(() => {
    if (startedRef.current) return;
    startPageLoad(screen);
    startedRef.current = true;
  }, [screen]);
  useEffect(() => {
    if (!ready || !startedRef.current) return;
    endPageLoad(screen, source);
  }, [ready, screen, source]);
}
