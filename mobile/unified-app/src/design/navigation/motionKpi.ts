import { emitPerfKpi } from "../../core/observability/perfKpi";
import { MotionDebug } from "./navigationMotion";

export type MotionLayer = "tab" | "stack" | "data" | "modal" | "inbox_indicator";

export type MotionAnimationMeta = {
  layer: MotionLayer;
  kind: string;
  duration_expected_ms: number;
  screen?: string;
  source?: string;
};

/**
 * Démarre le chronomètre d'une animation (dev uniquement si MotionDebug.enabled).
 * Retourne `end()` à appeler à la fin de l'animation.
 */
export function startMotionAnimation(meta: MotionAnimationMeta): () => void {
  if (!MotionDebug.enabled) {
    return () => undefined;
  }
  const startedAt = Date.now();
  const source = meta.source ?? `motion.${meta.layer}`;
  return () => {
    const duration_real_ms = Date.now() - startedAt;
    emitPerfKpi("perf.motion.animation", {
      source,
      layer: meta.layer,
      kind: meta.kind,
      duration_expected_ms: meta.duration_expected_ms,
      duration_real_ms,
      screen: meta.screen,
      duration_ratio: duration_real_ms / Math.max(meta.duration_expected_ms, 1),
    });
  };
}
