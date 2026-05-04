import { View, type StyleProp, type ViewStyle } from "react-native";
import {
  borderMuted,
  brandTextMuted,
  semanticDanger,
  semanticInfo,
  semanticRoute,
  semanticSuccess,
  semanticWarning,
  surfaceMuted,
} from "../responsive/colors";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";
import { AppText } from "./AppText";

export type ClientBookingStatusPillColors = {
  backgroundColor: string;
  borderColor: string;
  color: string;
};

const BASE_PILL: ClientBookingStatusPillColors = {
  backgroundColor: surfaceMuted,
  borderColor: borderMuted,
  color: brandTextMuted,
};

/**
 * Couleurs pastille statut réservation (client home / liste web).
 * Exportée pour les écrans qui appliquent un layout custom tout en gardant la charte.
 */
export function getClientBookingStatusPillColors(
  status: string | null | undefined
): ClientBookingStatusPillColors {
  const s = String(status ?? "")
    .trim()
    .toLowerCase();

  const variants: Record<string, ClientBookingStatusPillColors> = {
    pending: {
      backgroundColor: semanticWarning.bg,
      borderColor: semanticWarning.border,
      color: semanticWarning.fg,
    },
    requested: {
      backgroundColor: semanticInfo.bg,
      borderColor: semanticInfo.border,
      color: semanticInfo.fg,
    },
    confirmed: {
      backgroundColor: semanticSuccess.bg,
      borderColor: semanticSuccess.border,
      color: semanticSuccess.fg,
    },
    assigned: BASE_PILL,
    accepted: BASE_PILL,
    en_route: {
      backgroundColor: semanticRoute.bg,
      borderColor: semanticRoute.border,
      color: semanticRoute.fg,
    },
    on_route: {
      backgroundColor: semanticRoute.bg,
      borderColor: semanticRoute.border,
      color: semanticRoute.fg,
    },
    in_progress: {
      backgroundColor: semanticWarning.bg,
      borderColor: semanticWarning.border,
      color: semanticWarning.fg,
    },
    completed: {
      backgroundColor: semanticSuccess.bg,
      borderColor: semanticSuccess.border,
      color: semanticSuccess.fg,
    },
    cancelled: {
      backgroundColor: semanticDanger.bg,
      borderColor: semanticDanger.border,
      color: semanticDanger.fg,
    },
    canceled: {
      backgroundColor: semanticDanger.bg,
      borderColor: semanticDanger.border,
      color: semanticDanger.fg,
    },
  };

  return variants[s] ?? BASE_PILL;
}

export type AppStatusBadgeProps = {
  status: string | null | undefined;
  /** Texte affiché (ex. libellé FR). */
  label: string;
  style?: StyleProp<ViewStyle>;
};

export function AppStatusBadge({ status, label, style }: AppStatusBadgeProps) {
  const pill = getClientBookingStatusPillColors(status);
  const t = useResponsiveTokens();
  /** Pastille statut : léger resserrement vs multiples stricts de spacing (charte home client). */
  const BADGE_PAD_X = t.spacingSm + 2;
  const BADGE_PAD_Y = t.spacingXs + 1;

  return (
    <View
      style={[
        {
          alignSelf: "flex-end",
          paddingHorizontal: BADGE_PAD_X,
          paddingVertical: BADGE_PAD_Y,
          borderRadius: 999,
          borderWidth: 1,
          backgroundColor: pill.backgroundColor,
          borderColor: pill.borderColor,
        },
        style,
      ]}
    >
      <AppText variant="caption" style={{ color: pill.color, fontWeight: "600" }}>
        {label}
      </AppText>
    </View>
  );
}
