import type { ReactNode } from "react";
import { Platform, View, type LayoutChangeEvent } from "react-native";
import { borderMuted } from "../responsive/colors";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";
import {
  computeFloatingBarFallbackClearance,
  useFloatingBarMetricsReporter,
} from "./floatingBarMetrics";

export type FloatingBarPreset = "client" | "company";

const PRESETS: Record<
  FloatingBarPreset,
  {
    barBg: string;
    border: string;
    webShadow: string;
    nativeShadow: {
      shadowColor: string;
      shadowOpacity: number;
      shadowOffset: { width: number; height: number };
      shadowRadius: number;
      elevation: number;
    };
  }
> = {
  client: {
    barBg: "rgba(255, 255, 255, 0.94)",
    border: borderMuted,
    webShadow: "0 8px 28px rgba(15, 23, 42, 0.1)",
    nativeShadow: {
      shadowColor: "#0f172a",
      shadowOpacity: 0.12,
      shadowOffset: { width: 0, height: 6 },
      shadowRadius: 16,
      elevation: 8,
    },
  },
  company: {
    barBg: "#FFFFFF",
    border: "rgba(148, 163, 184, 0.22)",
    webShadow: "0 8px 28px rgba(15, 23, 42, 0.1)",
    nativeShadow: {
      shadowColor: "#0F172A",
      shadowOpacity: 0.12,
      shadowOffset: { width: 0, height: 6 },
      shadowRadius: 18,
      elevation: 10,
    },
  },
};

export type BaseFloatingBarProps = {
  children: ReactNode;
  /**
   * @deprecated La hauteur est désormais dérivée du flux (pilule + paddingBottom).
   * Conservé pour compatibilité d’appel ; ignoré si fourni.
   */
  containerHeight?: number;
  /** `paddingBottom` du conteneur aligné en bas (safe area / confort). */
  paddingBottom: number;
  /** Largeur max de la pilule (`Math.min(cap, usableWidth - 2 * horizontalPadding)`). */
  maxBarWidth: number;
  horizontalPadding: number;
  preset: FloatingBarPreset;
  /** `minHeight` de la pilule (hors grande police). */
  minInnerHeight: number;
  /** Si la police accessibilité est grande, hauteur mini relevée. */
  minInnerHeightLargeText?: number;
  isLargeText: boolean;
};

/**
 * Coque commune des barres d’onglets flottantes (pilule en flux + ombre + bordure).
 * La pilule n’est plus en `position: absolute` afin de contribuer à la hauteur du parent.
 */
export function BaseFloatingBar({
  children,
  paddingBottom,
  maxBarWidth,
  horizontalPadding,
  preset,
  minInnerHeight,
  minInnerHeightLargeText,
  isLargeText,
}: BaseFloatingBarProps) {
  const p = PRESETS[preset];
  const innerMin =
    isLargeText && minInnerHeightLargeText != null ? minInnerHeightLargeText : minInnerHeight;
  const t = useResponsiveTokens();
  const reporter = useFloatingBarMetricsReporter();
  /** Pilule client : ~6px à l’échelle 1 (entre spacingXs et spacingSm). */
  const FLOATING_BAR_CLIENT_PAD_X = t.spacingSm - 2;
  const FLOATING_BAR_CLIENT_PAD_Y = t.spacingSm - 2;
  const FLOATING_BAR_COMPANY_PAD_X = t.spacingXs;
  const FLOATING_BAR_COMPANY_PAD_Y = t.spacingXs;

  const onPillLayout = (e: LayoutChangeEvent) => {
    reporter?.reportInnerHeight(e.nativeEvent.layout.height);
  };

  return (
    <View
      style={{
        backgroundColor: "transparent",
        pointerEvents: "box-none",
        paddingBottom,
        alignItems: "center",
        width: "100%",
      }}
      pointerEvents="box-none"
    >
      <View
        onLayout={onPillLayout}
        style={[
          {
            maxWidth: maxBarWidth,
            width: "100%",
            marginHorizontal: horizontalPadding,
            minHeight: innerMin,
            flexDirection: "row",
            alignItems: "center",
            paddingHorizontal: preset === "client" ? FLOATING_BAR_CLIENT_PAD_X : FLOATING_BAR_COMPANY_PAD_X,
            paddingVertical: preset === "client" ? FLOATING_BAR_CLIENT_PAD_Y : FLOATING_BAR_COMPANY_PAD_Y,
            backgroundColor: p.barBg,
            borderWidth: 1,
            borderColor: p.border,
            borderRadius: 9999,
          },
          Platform.select({
            web: { boxShadow: p.webShadow } as object,
            default: p.nativeShadow,
          }) as object,
        ]}
      >
        {children}
      </View>
    </View>
  );
}

/** Padding bas sous la pilule — barre client. */
export function computeClientFloatingBottomPad(bottomInset: number): number {
  return Math.max(16, bottomInset + 8);
}

/** Padding bas sous la pilule — barre entreprise / chauffeur. */
export function computeCompanyFloatingBottomPad(bottomInset: number): number {
  return Math.max(12, bottomInset + 4);
}

/**
 * Clearance de repli (avant onLayout) pour un pied de page fixe au-dessus de la pilule.
 * @deprecated Préférer `useFloatingBarClearance` / métriques mesurées.
 */
export function computeFloatingTabBarClearance(bottomInset: number): number {
  return computeFloatingBarFallbackClearance(56, computeCompanyFloatingBottomPad(bottomInset));
}

/**
 * Clearance composeur chat — fallback déterministe.
 * @deprecated Préférer `useFloatingBarClearance`.
 */
export function computeFloatingTabComposerClearance(bottomInset: number): number {
  return computeFloatingBarFallbackClearance(56, computeCompanyFloatingBottomPad(bottomInset));
}
