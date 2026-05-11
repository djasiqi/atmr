import { useEffect, useMemo, useState } from "react";

/**
 * Fenêtre horaire « heures de travail » du chauffeur (heure locale du téléphone).
 *
 * Règle métier validée par les opérations :
 *  - Pendant 07h00 → 19h00 : tracking actif en mode présence (sans mission requise).
 *  - En dehors de cette plage : tracking actif uniquement si une mission éligible
 *    est en cours (ASSIGNED / EN_ROUTE / ON_BOARD / ARRIVED). Une mission qui
 *    démarre à 19h30 doit donc continuer à émettre des coordonnées jusqu'à sa
 *    fin, peu importe l'heure.
 *
 * Cette logique vit côté client : c'est l'app driver qui décide d'allumer ou
 * non son foreground service, l'envoi des points GPS, etc. Le backend ne fait
 * que recevoir ce qu'il reçoit.
 */

const DEFAULT_WORK_START_HOUR = 7;
const DEFAULT_WORK_END_HOUR = 19;

function clampHour(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_WORK_START_HOUR;
  const int = Math.floor(value);
  if (int < 0) return 0;
  if (int > 24) return 24;
  return int;
}

function readHourFromEnvVar(raw: string | undefined, fallback: number): number {
  if (raw == null) return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return fallback;
  return clampHour(parsed);
}

/** Runtime lookup (Jest-safe) : évite l’inline Babel des `process.env.EXPO_PUBLIC_*` qui figerait les valeurs au build. */
function readExpoPublicEnv(name: string): string | undefined {
  return process.env[name];
}

export type TrackingWindowConfig = {
  startHour: number;
  endHour: number;
};

export function getTrackingWindowConfig(): TrackingWindowConfig {
  const startHour = readHourFromEnvVar(
    readExpoPublicEnv("EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_START_HOUR"),
    DEFAULT_WORK_START_HOUR
  );
  const endHour = readHourFromEnvVar(
    readExpoPublicEnv("EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_END_HOUR"),
    DEFAULT_WORK_END_HOUR
  );
  if (endHour <= startHour) {
    return { startHour: DEFAULT_WORK_START_HOUR, endHour: DEFAULT_WORK_END_HOUR };
  }
  return { startHour, endHour };
}

/**
 * Vrai si l'instant donné se trouve dans la plage `[startHour ; endHour[`
 * exprimée en heure locale du téléphone (le chauffeur peut être à Genève ou
 * en mission longue distance).
 */
export function isWithinTrackingWindow(
  now: Date = new Date(),
  config: TrackingWindowConfig = getTrackingWindowConfig()
): boolean {
  const hour = now.getHours();
  return hour >= config.startHour && hour < config.endHour;
}

/**
 * Renvoie l'instant du prochain bord de fenêtre (ouverture ou fermeture)
 * relatif à `now`. Utile pour planifier un timer qui re-déclenche la
 * réconciliation tracking pile à 07h00 et 19h00.
 */
export function getNextTrackingWindowEdge(
  now: Date = new Date(),
  config: TrackingWindowConfig = getTrackingWindowConfig()
): { at: Date; type: "open" | "close" } {
  const isOpen = isWithinTrackingWindow(now, config);
  const next = new Date(now);
  next.setSeconds(0, 0);
  next.setMinutes(0);
  if (isOpen) {
    next.setHours(config.endHour, 0, 0, 0);
    if (next.getTime() <= now.getTime()) {
      next.setDate(next.getDate() + 1);
    }
    return { at: next, type: "close" };
  }
  next.setHours(config.startHour, 0, 0, 0);
  if (next.getTime() <= now.getTime()) {
    next.setDate(next.getDate() + 1);
  }
  return { at: next, type: "open" };
}

/**
 * Délai en millisecondes avant le prochain edge. Borné à 1 minute minimum
 * pour éviter les `setTimeout(0)` qui boucleraient (cas d'horloge gelée /
 * `now` égal à l'edge à la milliseconde près).
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
 * Hook React qui expose l'état courant de la fenêtre et se ré-évalue
 * automatiquement au prochain bord (07h00, 19h00…). On ne sample pas toutes
 * les minutes : on programme un seul `setTimeout` jusqu'au prochain changement
 * d'état pour économiser CPU/batterie en background.
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
    const schedule = () => {
      if (cancelled) return;
      const now = new Date();
      const ms = getMsUntilNextWindowEdge(now, stableConfig);
      const timeout = setTimeout(() => {
        if (cancelled) return;
        setState(computeState());
        schedule();
      }, ms);
      timeoutRef = timeout;
    };
    let timeoutRef: ReturnType<typeof setTimeout> | null = null;
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
