import { View } from "react-native";
import { brandPrimary } from "../responsive/brand";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";
import { AppText } from "./AppText";

export type AppBadgeTone = "neutral" | "info" | "success";

export type AppBadgeProps = {
  label: string;
  tone?: AppBadgeTone;
};

export function AppBadge({ label, tone = "neutral" }: AppBadgeProps) {
  const t = useResponsiveTokens();

  const palette =
    tone === "success"
      ? { bg: "rgba(0, 121, 107, 0.12)", fg: brandPrimary }
      : tone === "info"
        ? { bg: "rgba(59, 130, 246, 0.12)", fg: "#1d4ed8" }
        : { bg: "rgba(100, 116, 139, 0.14)", fg: "#475569" };

  return (
    <View
      style={{
        alignSelf: "flex-start",
        paddingHorizontal: t.spacingSm,
        paddingVertical: t.spacingXs,
        borderRadius: t.radiusSm,
        backgroundColor: palette.bg,
      }}
    >
      <AppText variant="caption" style={{ color: palette.fg, fontWeight: "600" }}>
        {label}
      </AppText>
    </View>
  );
}
