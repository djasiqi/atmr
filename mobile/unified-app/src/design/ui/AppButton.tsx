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

const MAX_FONT_MULTIPLIER = 1.35;

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
            maxFontSizeMultiplier={MAX_FONT_MULTIPLIER}
            style={{
              color: labelColor(),
              fontWeight: "600",
              fontSize: t.buttonFontSize,
              lineHeight: Math.round(t.buttonFontSize * 1.25),
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
      case "secondary":
        return { bg: "#f1f5f4", border: GHOST_BORDER, opacity: 0.85 };
      case "ghost":
        return { bg: "transparent", border: "transparent", opacity: 0.6 };
      case "danger":
        return { bg: "#fca5a5", border: "#fca5a5", opacity: 1 };
      default:
        return { bg: brandPrimaryDisabled, border: brandPrimaryDisabled, opacity: 1 };
    }
  }
  switch (variant) {
    case "primary":
      return {
        bg: brandPrimary,
        border: brandPrimary,
        opacity: pressed ? 0.92 : 1,
      };
    case "secondary":
      return {
        bg: pressed ? "#f8fafc" : "#fff",
        border: GHOST_BORDER,
        opacity: 1,
      };
    case "ghost":
      return {
        bg: pressed ? "rgba(10, 143, 122, 0.1)" : "transparent",
        border: "transparent",
        opacity: 1,
      };
    case "danger":
      return {
        bg: DANGER_BG,
        border: DANGER_BORDER,
        opacity: pressed ? 0.92 : 1,
      };
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
  },
  icon: {
    marginRight: 2,
  },
});
