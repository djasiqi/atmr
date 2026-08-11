import { useEffect, useMemo, useState } from "react";
import {
  BUSINESS_TIME_ZONE,
  getBusinessTimeParts,
  PRESENCE_WINDOW_END_HOUR,
  PRESENCE_WINDOW_START_HOUR,
  zonedWallClockToUtcDate,
} from "./businessTime";

/**
 * Fenêtre horaire présence GPS — Europe/Zurich (P0-F TIME).
 *
 * Règle métier :
 *  - [07:00 ; 19:00[ Europe/Zurich : présence FG + BG autorisée (si dispo + disclosure)
 *  - Hors plage sans mission → OFF (FG et BG)
 *  - Mission active (isTrackingActiveStatus) → tracking indépendant de la fenêtre
 *
 * Indépendant du timezone du téléphone.
 */

export type TrackingWindowConfig = {
  startHour: number;
  endHour: number;
};

const FROZEN_CONFIG: TrackingWindowConfig = {
  startHour: PRESENCE_WINDOW_START_HOUR,
  endHour: PRESENCE_WINDOW_END_HOUR,
};

/** Runtime lookup (Jest-safe). */
function readExpoPublicEnv(name: string): string | undefined {
  return process.env[name];
}

function warnIfEnvDiverges(): void {
  const startRaw = readExpoPublicEnv("EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_START_HOUR");
  const endRaw = readExpoPublicEnv("EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_END_HOUR");
  if (startRaw == null && endRaw == null) return;
  const start = startRaw != null ? Number(startRaw) : PRESENCE_WINDOW_START_HOUR;
  const end = endRaw != null ? Number(endRaw) : PRESENCE_WINDOW_END_HOUR;
  if (start === PRESENCE_WINDOW_START_HOUR && end === PRESENCE_WINDOW_END_HOUR) {
    return;
  }
  // Pas de divergence silencieuse : log explicite ; bornes figées restent 7/19.
  console.error(
    `[trackingWindow] EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_* ignorees (P0-F TIME). ` +
      `demande=${start}-${end} force=${PRESENCE_WINDOW_START_HOUR}-${PRESENCE_WINDOW_END_HOUR} tz=${BUSINESS_TIME_ZONE}`
  );
}

/**
 * Config figée 07–19 Europe/Zurich.
 * Les variables d'env ne peuvent plus modifier silencieusement la règle.
 */
export function getTrackingWindowConfig(): TrackingWindowConfig {
  warnIfEnvDiverges();
  return { ...FROZEN_CONFIG };
}

/**
 * Vrai si l'instant est dans `[startHour ; endHour[` en Europe/Zurich.
 */
export function isWithinTrackingWindow(
  now: Date = new Date(),
  config: TrackingWindowConfig = getTrackingWindowConfig()
): boolean {
  const { hour } = getBusinessTimeParts(now);
  return hour >= config.startHour && hour < config.endHour;
}

/**
 * Instant absolu UTC du prochain bord 07:00 / 19:00 Europe/Zurich.
 */
export function getNextTrackingWindowEdge(
  now: Date = new Date(),
  config: TrackingWindowConfig = getTrackingWindowConfig()
): { at: Date; type: "open" | "close" } {
  const parts = getBusinessTimeParts(now);
  const isOpen = parts.hour >= config.startHour && parts.hour < config.endHour;

  if (isOpen) {
    const closeToday = zonedWallClockToUtcDate(
      parts.year,
      parts.month,
      parts.day,
      config.endHour,
      0,
      0
    );
    if (closeToday.getTime() > now.getTime()) {
      return { at: closeToday, type: "close" };
    }
    // Exactement à/après 19:00 murale : prochain open = demain 07:00 Zurich
    const probe = new Date(
      Date.UTC(parts.year, parts.month - 1, parts.day, 12, 0, 0) + 24 * 60 * 60 * 1000
    );
    const np = getBusinessTimeParts(probe);
    return {
      at: zonedWallClockToUtcDate(np.year, np.month, np.day, config.startHour, 0, 0),
      type: "open",
    };
  }

  // Fermé : prochain open = aujourd'hui 07:00 si avant 07:00, sinon demain 07:00
  if (parts.hour < config.startHour) {
    const openToday = zonedWallClockToUtcDate(
      parts.year,
      parts.month,
      parts.day,
      config.startHour,
      0,
      0
    );
    if (openToday.getTime() > now.getTime()) {
      return { at: openToday, type: "open" };
    }
  }

  const probe = new Date(
    Date.UTC(parts.year, parts.month - 1, parts.day, 12, 0, 0) + 24 * 60 * 60 * 1000
  );
  const np = getBusinessTimeParts(probe);
  return {
    at: zonedWallClockToUtcDate(np.year, np.month, np.day, config.startHour, 0, 0),
    type: "open",
  };
}

/**
 * Délai ms avant le prochain edge (min 60s pour éviter setTimeout(0)).
 */
export function getMsUntilNextWindowEdge(
  now: Date = new Date(),
  config: TrackingWindowConfig = getTrackingWindowConfig()
): number {
  const { at } = getNextTrackingWindowEdge(now, config);
  return Math.max(60_000, at.getTime() - now.getTime());
}

export type TrackingWindowState = {
  isOpen: boolean;
  nextEdgeAt: Date;
  nextEdgeType: "open" | "close";
  config: TrackingWindowConfig;
};

/**
 * Hook React : état fenêtre + timer jusqu'au prochain bord (accélérateur FG).
 * La garantie BG repose sur le re-check natif (TIME-3C/3D), pas sur ce timer.
 */
export function useTrackingWindowState(
  config: TrackingWindowConfig = getTrackingWindowConfig()
): TrackingWindowState {
  const stableConfig = useMemo(
    () => ({ startHour: config.startHour, endHour: config.endHour }),
    [config.startHour, config.endHour]
  );

  const computeState = (): TrackingWindowState => {
    const now = new Date();
    const edge = getNextTrackingWindowEdge(now, stableConfig);
    return {
      isOpen: isWithinTrackingWindow(now, stableConfig),
      nextEdgeAt: edge.at,
      nextEdgeType: edge.type,
      config: stableConfig,
    };
  };

  const [state, setState] = useState<TrackingWindowState>(computeState);

  useEffect(() => {
    let cancelled = false;
    let timeoutRef: ReturnType<typeof setTimeout> | null = null;
    const schedule = () => {
      if (cancelled) return;
      const now = new Date();
      const ms = getMsUntilNextWindowEdge(now, stableConfig);
      timeoutRef = setTimeout(() => {
        if (cancelled) return;
        setState(computeState());
        schedule();
      }, ms);
    };
    setState(computeState());
    schedule();
    return () => {
      cancelled = true;
      if (timeoutRef) clearTimeout(timeoutRef);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stableConfig.startHour, stableConfig.endHour]);

  return state;
}
