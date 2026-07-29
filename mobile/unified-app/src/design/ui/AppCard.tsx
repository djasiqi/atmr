import type { PropsWithChildren } from "react";
import { Platform, Pressable, StyleSheet, View, type ViewProps } from "react-native";
import { brandSurfaceSoft } from "../responsive/brand";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";

const UI_BORDER_SOFT = "rgba(145, 165, 157, 0.38)";

export type AppCardVariant = "surface" | "elevated" | "compact" | "interactive";

export type AppCardProps = PropsWithChildren<
  ViewProps & {
    variant?: AppCardVariant;
    /** Requis pour `variant="interactive"` ; sinon ignoré. */
    onPress?: () => void;
  }
>;

export function AppCard({
  variant = "surface",
  onPress,
  children,
  style,
  ...rest
}: AppCardProps) {
  const t = useResponsiveTokens();
  const padding =
    variant === "compact"
      ? Math.max(t.spacingSm, Math.round(t.cardPadding * 0.65))
      : t.cardPadding;

  const base = {
    borderRadius: t.radiusMd,
    padding,
  };

  const elevated = variant === "elevated";
  const softSurface =
    variant === "surface" || variant === "compact" || variant === "interactive";

  const surfaceStyles = elevated
    ? {
        backgroundColor: "#fff",
        borderWidth: 0,
        ...Platform.select({
          ios: {
            shadowColor: "#000",
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.08,
            shadowRadius: 10,
          },
          android: { elevation: 3 },
          default: {},
        }),
      }
    : {
        backgroundColor: softSurface ? brandSurfaceSoft : "#fff",
        borderWidth: 1,
        borderColor: UI_BORDER_SOFT,
      };

  const merged = [base, surfaceStyles, style];

  if (variant === "interactive" && onPress) {
    return (
      <Pressable
        accessibilityRole="button"
        onPress={onPress}
        style={(state) => [
          merged,
          state.pressed ? styles.pressed : null,
        ]}
        {...rest}
      >
        {children}
      </Pressable>
    );
  }

  return (
    <View style={merged} {...rest}>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  pressed: {
    opacity: 0.92,
  },
});
