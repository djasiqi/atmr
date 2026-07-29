import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type FloatingBarMetrics = {
  /** Hauteur mesurée (ou fallback) de la pilule. */
  innerHeight: number;
  /** Padding bas sous la pilule (safe area / confort). */
  bottomPadding: number;
  /** Réserve totale pour le contenu scrollé : innerHeight + bottomPadding. */
  clearance: number;
};

export type FloatingBarPresetKind = "client" | "company" | "driver";

/** Hauteurs de repli déterministes avant le premier onLayout. */
export const FLOATING_BAR_FALLBACK_INNER: Record<FloatingBarPresetKind, number> = {
  client: 64,
  company: 56,
  driver: 56,
};

/**
 * Clearance de repli avant mesure réelle.
 * Pure / testable — ne dépend pas de Yoga.
 */
export function computeFloatingBarFallbackClearance(
  innerMinHeight: number,
  bottomPadding: number
): number {
  const inner = Number.isFinite(innerMinHeight) ? Math.max(0, innerMinHeight) : 0;
  const pad = Number.isFinite(bottomPadding) ? Math.max(0, bottomPadding) : 0;
  return inner + pad;
}

export function computeFloatingBarMetrics(
  innerHeight: number,
  bottomPadding: number
): FloatingBarMetrics {
  const bottom = Number.isFinite(bottomPadding) ? Math.max(0, bottomPadding) : 0;
  const inner = Number.isFinite(innerHeight) ? Math.max(0, innerHeight) : 0;
  return {
    innerHeight: inner,
    bottomPadding: bottom,
    clearance: computeFloatingBarFallbackClearance(inner, bottom),
  };
}

type FloatingBarMetricsContextValue = {
  metrics: FloatingBarMetrics;
  reportInnerHeight: (height: number) => void;
};

const FloatingBarMetricsContext = createContext<FloatingBarMetricsContextValue | null>(null);

export type FloatingBarMetricsProviderProps = {
  children: ReactNode;
  preset: FloatingBarPresetKind;
  bottomPadding: number;
  /** Surcharge du fallback inner (ex. large text). */
  fallbackInnerHeight?: number;
};

export function FloatingBarMetricsProvider({
  children,
  preset,
  bottomPadding,
  fallbackInnerHeight,
}: FloatingBarMetricsProviderProps) {
  const fallbackInner = fallbackInnerHeight ?? FLOATING_BAR_FALLBACK_INNER[preset];
  const [measuredInner, setMeasuredInner] = useState<number | null>(null);

  const reportInnerHeight = useCallback((height: number) => {
    if (!Number.isFinite(height) || height <= 0) return;
    setMeasuredInner((prev) => (prev != null && Math.abs(prev - height) < 0.5 ? prev : height));
  }, []);

  const metrics = useMemo(
    () => computeFloatingBarMetrics(measuredInner ?? fallbackInner, bottomPadding),
    [measuredInner, fallbackInner, bottomPadding]
  );

  const value = useMemo(
    () => ({
      metrics,
      reportInnerHeight,
    }),
    [metrics, reportInnerHeight]
  );

  return (
    <FloatingBarMetricsContext.Provider value={value}>{children}</FloatingBarMetricsContext.Provider>
  );
}

/**
 * Métriques de la barre flottante courante.
 * Hors provider : retourne un fallback déterministe.
 */
export function useFloatingBarMetrics(
  fallbackPreset: FloatingBarPresetKind = "company",
  fallbackBottomPadding = 0
): FloatingBarMetrics {
  const ctx = useContext(FloatingBarMetricsContext);
  return useMemo(() => {
    if (ctx) return ctx.metrics;
    return computeFloatingBarMetrics(
      FLOATING_BAR_FALLBACK_INNER[fallbackPreset],
      fallbackBottomPadding
    );
  }, [ctx, fallbackPreset, fallbackBottomPadding]);
}

export function useFloatingBarMetricsReporter(): {
  reportInnerHeight: (height: number) => void;
} | null {
  const ctx = useContext(FloatingBarMetricsContext);
  if (!ctx) return null;
  return { reportInnerHeight: ctx.reportInnerHeight };
}

/** Clearance contenu = métriques partagées ou fallback pur. */
export function useFloatingBarClearance(
  fallbackPreset: FloatingBarPresetKind = "company",
  fallbackBottomPadding = 0
): number {
  return useFloatingBarMetrics(fallbackPreset, fallbackBottomPadding).clearance;
}
