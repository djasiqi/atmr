import { createContext, useContext, useEffect, useState } from "react";
import { InteractionManager } from "react-native";

/** Filet si `runAfterInteractions` ne se résout pas (animations / splash). */
export const COMPANY_BACKGROUND_BOOT_FALLBACK_MS = 800;

export type CompanyColdStartPhase = "critical" | "background";

export const CompanyColdStartPhaseContext = createContext<CompanyColdStartPhase>("critical");

export function useCompanyColdStartPhase(): CompanyColdStartPhase {
  return useContext(CompanyColdStartPhaseContext);
}

export function useCompanyBackgroundBootReady(): boolean {
  return useCompanyColdStartPhase() === "background";
}

/**
 * Passe en lane background après le premier écran utile (interactions + 1 frame).
 * Ne bloque jamais le shell : le Cockpit peut déjà s’afficher.
 */
export function useCompanyColdStartPhaseState(): CompanyColdStartPhase {
  const [phase, setPhase] = useState<CompanyColdStartPhase>("critical");

  useEffect(() => {
    let cancelled = false;
    const handle = InteractionManager.runAfterInteractions(() => {
      requestAnimationFrame(() => {
        if (!cancelled) setPhase("background");
      });
    });
    const fallback = setTimeout(() => {
      if (!cancelled) setPhase("background");
    }, COMPANY_BACKGROUND_BOOT_FALLBACK_MS);
    return () => {
      cancelled = true;
      handle.cancel();
      clearTimeout(fallback);
    };
  }, []);

  return phase;
}
