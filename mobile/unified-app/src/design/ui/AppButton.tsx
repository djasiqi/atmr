import type { ReactNode } from "react";
import {
  ActivityIndicator,
  Pressable,
  type PressableProps,
  StyleSheet,
  Text,
  View,
} from "react-native";
import {
  brandPrimary,
  brandPrimaryDisabled,
  brandText,
  brandTextMuted,
} from "../responsive/brand";
import { CONTENT_FONT_CAP } from "../responsive/fontScaleCaps";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";

export type AppButtonVariant = "primary" | "secondary" | "ghost" | "danger";

export type AppButtonProps = Omit<PressableProps, "children"> & {
  title: string;
  variant?: AppButtonVariant;
  loading?: boolean;
  leftIcon?: ReactNode;
};

const DANGER_BG = "#DC2626";
const DANGER_BORDER = "#B91C1C";
const GHOST_BORDER = "rgba(145, 165, 157, 0.5)";

export function AppButton({
  title,
  variant = "primary",
  loading = false,
  disabled,
  leftIcon,
  style,
  ...rest
}: AppButtonProps) {
  const t = useResponsiveTokens();
  const isDisabled = disabled === true || loading;
  const height = t.formButtonMinHeight;

  const labelColor = (): string => {
    if (isDisabled) {
      if (variant === "primary" || variant === "danger") return "#fff";
      if (variant === "ghost") return brandPrimaryDisabled;
      return brandTextMuted;
    }
    switch (variant) {
      case "primary":
      case "danger":
        return "#fff";
      case "secondary":
        return brandText;
      case "ghost":
        return brandPrimary;
      default:
        return "#fff";
    }
  };

  return (
    <Pressable
      accessibilityRole="button"
      disabled={isDisabled}
      {...rest}
      style={(state) => {
        const pressed = state.pressed;
        const p = pressPalette(variant, isDisabled, pressed);
        return [
          styles.row,
          {
            minHeight: height,
            paddingHorizontal: t.spacingMd,
            paddingVertical: Math.max(10, Math.round(t.spacingSm * 0.75)),
            borderRadius: t.radiusMd,
            borderWidth: variant === "ghost" ? 0 : 1,
            backgroundColor: p.bg,
            borderColor: p.border,
            opacity: p.opacity,
          },
          typeof style === "function" ? style(state) : style,
        ];
      }}
    >
      {loading ? (
        <ActivityIndicator
          color={variant === "secondary" ? brandPrimary : variant === "ghost" ? brandPrimary : "#fff"}
        />
      ) : (
        <View style={[styles.inner, { gap: t.spacingSm }]}>
          {leftIcon ? <View style={styles.icon}>{leftIcon}</View> : null}
          <Text
            maxFontSizeMultiplier={CONTENT_FONT_CAP}
            style={{
              color: labelColor(),
              fontWeight: "600",
              fontSize: t.buttonFontSize,
              lineHeight: Math.round(t.buttonFontSize * t.bodyLineHeightRatio),
              flexShrink: 1,
              textAlign: "center",
            }}
          >
            {title}
          </Text>
        </View>
      )}
    </Pressable>
  );
}

function pressPalette(
  variant: AppButtonVariant,
  isDisabled: boolean,
  pressed: boolean
): { bg: string; border: string; opacity: number } {
  if (isDisabled) {
    switch (variant) {
      case "primary":
        return { bg: brandPrimaryDisabled, border: brandPrimaryDisabled, opacity: 1 };
      case "danger":
        return { bg: "#FCA5A5", border: "#FCA5A5", opacity: 1 };
      case "secondary":
        return { bg: "#F1F5F4", border: "rgba(145, 165, 157, 0.45)", opacity: 1 };
      case "ghost":
        return { bg: "transparent", border: "transparent", opacity: 0.55 };
      default:
        return { bg: brandPrimaryDisabled, border: brandPrimaryDisabled, opacity: 1 };
    }
  }
  if (pressed) {
    switch (variant) {
      case "primary":
        return { bg: "#00695C", border: "#00695C", opacity: 1 };
      case "danger":
        return { bg: DANGER_BORDER, border: DANGER_BORDER, opacity: 1 };
      case "secondary":
        return { bg: "#E8F0EE", border: "rgba(145, 165, 157, 0.55)", opacity: 1 };
      case "ghost":
        return { bg: "rgba(0, 121, 107, 0.08)", border: GHOST_BORDER, opacity: 1 };
      default:
        return { bg: "#00695C", border: "#00695C", opacity: 1 };
    }
  }
  switch (variant) {
    case "primary":
      return { bg: brandPrimary, border: brandPrimary, opacity: 1 };
    case "danger":
      return { bg: DANGER_BG, border: DANGER_BORDER, opacity: 1 };
    case "secondary":
      return { bg: "#fff", border: "rgba(145, 165, 157, 0.55)", opacity: 1 };
    case "ghost":
      return { bg: "transparent", border: GHOST_BORDER, opacity: 1 };
    default:
      return { bg: brandPrimary, border: brandPrimary, opacity: 1 };
  }
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
  },
  inner: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 1,
    flexWrap: "wrap",
  },
  icon: {
    flexShrink: 0,
  },
});
