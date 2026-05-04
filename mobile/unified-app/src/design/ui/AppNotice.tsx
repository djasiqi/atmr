import type { ReactNode } from "react";
import { Pressable, View, type StyleProp, type ViewStyle } from "react-native";
import {
  brandPrimary,
  semanticDanger,
  semanticInfo,
  semanticWarning,
} from "../responsive/colors";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";
import { AppText } from "./AppText";

export type AppNoticeVariant = "danger" | "warning" | "info";

export type AppNoticeProps = {
  variant: AppNoticeVariant;
  /** Titre optionnel (ex. « Mise à jour obligatoire »). */
  title?: string;
  children?: ReactNode;
  ctaLabel?: string;
  onCtaPress?: () => void;
  style?: StyleProp<ViewStyle>;
};

export function AppNotice({
  variant,
  title,
  children,
  ctaLabel,
  onCtaPress,
  style,
}: AppNoticeProps) {
  const t = useResponsiveTokens();

  const palette =
    variant === "danger"
      ? {
          border: semanticDanger.border,
          bg: semanticDanger.bg,
          titleFg: semanticDanger.fg,
          bodyFg: semanticDanger.fg,
          ctaBg: semanticDanger.ctaBg,
          ctaFg: semanticDanger.ctaFg,
        }
      : variant === "warning"
        ? {
            border: semanticWarning.border,
            bg: semanticWarning.bg,
            titleFg: semanticWarning.fg,
            bodyFg: semanticWarning.fg,
            ctaBg: semanticWarning.ctaBg,
            ctaFg: semanticWarning.ctaFg,
          }
        : {
            border: semanticInfo.border,
            bg: semanticInfo.bg,
            titleFg: semanticInfo.fg,
            bodyFg: semanticInfo.fg,
            ctaBg: brandPrimary,
            ctaFg: "#FFFFFF",
          };

  return (
    <View
      style={[
        {
          borderRadius: t.radiusLg,
          borderWidth: 1,
          padding: t.spacingMd,
          gap: t.spacingSm,
          borderColor: palette.border,
          backgroundColor: palette.bg,
        },
        style,
      ]}
    >
      {title ? (
        <AppText variant="body" style={{ color: palette.titleFg, fontWeight: "700" }}>
          {title}
        </AppText>
      ) : null}
      {children ? (
        typeof children === "string" ? (
          <AppText variant="body" style={{ color: palette.bodyFg }}>
            {children}
          </AppText>
        ) : (
          children
        )
      ) : null}
      {ctaLabel && onCtaPress ? (
        <Pressable
          onPress={onCtaPress}
          style={{
            alignItems: "center",
            justifyContent: "center",
            minHeight: t.minTouchHeight,
            borderRadius: t.radiusMd,
            backgroundColor: palette.ctaBg,
          }}
        >
          <AppText variant="body" style={{ color: palette.ctaFg, fontWeight: "700" }}>
            {ctaLabel}
          </AppText>
        </Pressable>
      ) : null}
    </View>
  );
}
