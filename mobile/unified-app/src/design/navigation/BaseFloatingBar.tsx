import type { ReactNode } from "react";
import { Platform, View } from "react-native";
import { borderMuted, surfaceCard } from "../responsive/colors";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";

/** Contour pilule entreprise : teinte marque très légère. */
const COMPANY_BAR_BORDER = "rgba(0, 121, 107, 0.08)";

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
  /** Hauteur totale de la zone réservée (pilule + padding bas). */
  containerHeight: number;
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
 * Coque commune des barres d’onglets flottantes (pilule + ombre + bordure).
 */
export function BaseFloatingBar({
  children,
  containerHeight,
  paddingBottom,
  maxBarWidth,
  horizontalPadding,
  preset,
  minInnerHeight,
  minInnerHeightLargeText,
  isLargeText,
}: BaseFloatingBarProps) {
  const p = PRESETS[preset];
  const innerMin = isLargeText && minInnerHeightLargeText != null ? minInnerHeightLargeText : minInnerHeight;
  const t = useResponsiveTokens();
  /** Pilule client : ~6px à l’échelle 1 (entre spacingXs et spacingSm). */
  const FLOATING_BAR_CLIENT_PAD_X = t.spacingSm - 2;
  const FLOATING_BAR_CLIENT_PAD_Y = t.spacingSm - 2;
  const FLOATING_BAR_COMPANY_PAD_X = t.spacingXs;
  const FLOATING_BAR_COMPANY_PAD_Y = t.spacingXs;

  return (
    <View
      style={{ height: containerHeight, backgroundColor: "transparent", pointerEvents: "box-none" }}
    >
      <View
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: 0,
          alignItems: "center",
          paddingBottom,
          pointerEvents: "box-none",
        }}
      >
        <View
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
              web: { boxShadow: p.webShadow },
              default: p.nativeShadow,
            }),
          ]}
        >
          {children}
        </View>
      </View>
    </View>
  );
}

/** Padding bas sous la pilule — barre client. */
export function computeClientFloatingBottomPad(bottomInset: number): number {
  return Math.max(16, bottomInset + 8);
}

/** Padding bas sous la pilule — barre entreprise. */
export function computeCompanyFloatingBottomPad(bottomInset: number): number {
  return Math.max(12, bottomInset + 4);
}

/**
 * Réserve verticale pour un pied de page fixe (ex. composeur de fil) afin qu’il
 * reste au-dessus de la pilule d’onglets flottante (contenu pleine hauteur + tab bar overlay).
 */
export function computeFloatingTabBarClearance(bottomInset: number): number {
  return 64 + computeCompanyFloatingBottomPad(bottomInset);
}

/** Hauteur pilule + padding bas du slot — juste au-dessus du menu visible. */
const FLOATING_TAB_PILL_HEIGHT = 56;

export function computeFloatingTabComposerClearance(bottomInset: number): number {
  return FLOATING_TAB_PILL_HEIGHT + computeCompanyFloatingBottomPad(bottomInset);
}
