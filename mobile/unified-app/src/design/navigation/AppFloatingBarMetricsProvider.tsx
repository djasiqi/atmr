import type { ReactNode } from "react";
import {
  FloatingBarMetricsProvider,
  FLOATING_BAR_FALLBACK_INNER,
  type FloatingBarPresetKind,
} from "./floatingBarMetrics";
import {
  computeClientFloatingBottomPad,
  computeCompanyFloatingBottomPad,
} from "./BaseFloatingBar";
import { useAppViewport } from "../responsive/useAppViewport";
import { useAccessibilityScale } from "../responsive/useAccessibilityScale";

export type AppFloatingBarMetricsProviderProps = {
  children: ReactNode;
  preset: FloatingBarPresetKind;
};

/**
 * Provider layout-level : clearance partagée entre barre flottante (onLayout)
 * et écrans (padding scroll).
 */
export function AppFloatingBarMetricsProvider({
  children,
  preset,
}: AppFloatingBarMetricsProviderProps) {
  const { bottomInset } = useAppViewport();
  const { isLargeText } = useAccessibilityScale();
  const bottomPadding =
    preset === "client"
      ? computeClientFloatingBottomPad(bottomInset)
      : computeCompanyFloatingBottomPad(bottomInset);

  const fallbackInner =
    preset === "client"
      ? isLargeText
        ? 72
        : FLOATING_BAR_FALLBACK_INNER.client
      : isLargeText
        ? 62
        : FLOATING_BAR_FALLBACK_INNER[preset];

  return (
    <FloatingBarMetricsProvider
      preset={preset}
      bottomPadding={bottomPadding}
      fallbackInnerHeight={fallbackInner}
    >
      {children}
    </FloatingBarMetricsProvider>
  );
}
