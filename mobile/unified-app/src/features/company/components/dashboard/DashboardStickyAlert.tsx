import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import type { DashboardDelayedMissionCard } from "../../dashboard/companyDashboardViewModel";
import { D } from "../../theme/companyDashboardTokens";
import { M } from "./dashboardMobileTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

export type DashboardAlertLine = {
  id: string;
  text: string;
  severity: "error" | "warning";
};

type Props = {
  delayedMission: DashboardDelayedMissionCard | null;
  alertLines: DashboardAlertLine[];
  errorText?: string | null;
  onPressDelayed?: () => void;
  onPressAlerts?: () => void;
};

export function DashboardStickyAlert({
  delayedMission,
  alertLines,
  errorText,
  onPressDelayed,
  onPressAlerts,
}: Props) {
  const topAlert = alertLines[0];
  const hasError = Boolean(errorText?.trim());
  const visible = delayedMission != null || topAlert != null || hasError;

  if (!visible) return null;

  const isUrgent = delayedMission != null || topAlert?.severity === "error" || hasError;
  const title = delayedMission
    ? `Retard ${delayedMission.delayMinutes} min · ${delayedMission.clientName}`
    : hasError
      ? errorText!
      : topAlert!.text;

  const subtitle = delayedMission
    ? `${delayedMission.scheduledTimeLabel} · ${delayedMission.routeLabel}`
    : alertLines.length > 1
      ? `+${alertLines.length - 1} autre${alertLines.length > 2 ? "s" : ""} alerte${alertLines.length > 2 ? "s" : ""}`
      : topAlert && !delayedMission
        ? "Appuyer pour voir le détail"
        : undefined;

  const onPress = delayedMission ? onPressDelayed : onPressAlerts;

  return (
    <Pressable
      onPress={onPress}
      disabled={!onPress}
      style={({ pressed }) => [
        s.wrap,
        isUrgent ? s.wrapDanger : s.wrapWarn,
        pressed && onPress && s.pressed,
      ]}
      accessibilityRole={onPress ? "button" : "text"}
      accessibilityLabel={title}
    >
      <View style={[s.accent, { backgroundColor: isUrgent ? D.danger : "#F59E0B" }]} />
      <View style={s.iconWell}>
        <Ionicons
          name={isUrgent ? "warning" : "information-circle"}
          size={18}
          color={isUrgent ? D.danger : "#B45309"}
        />
      </View>
      <View style={s.textCol}>
        <AppText variant="label" style={[s.title, isUrgent && s.titleDanger]} numberOfLines={2}>
          {title}
        </AppText>
        {subtitle ? (
          <AppText variant="caption" style={s.subtitle} numberOfLines={1}>
            {subtitle}
          </AppText>
        ) : null}
      </View>
      {onPress ? (
        <Ionicons name="chevron-forward" size={18} color={isUrgent ? D.danger : "#B45309"} />
      ) : null}
    </Pressable>
  );
}

const s = StyleSheet.create({
  wrap: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginHorizontal: M.padH,
    marginBottom: 8,
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: M.radiusRow,
    overflow: "hidden",
  },
  wrapDanger: {
    backgroundColor: "rgba(254, 242, 242, 0.95)",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: D.dangerBorder,
  },
  wrapWarn: {
    backgroundColor: "rgba(255, 251, 235, 0.95)",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(245, 158, 11, 0.35)",
  },
  pressed: { opacity: 0.92 },
  accent: {
    position: "absolute",
    left: 0,
    top: 0,
    bottom: 0,
    width: 3,
  },
  iconWell: {
    width: M.iconSm,
    height: M.iconSm,
    borderRadius: 10,
    backgroundColor: "rgba(255,255,255,0.7)",
    alignItems: "center",
    justifyContent: "center",
  },
  textCol: { flex: 1, minWidth: 0, gap: 1 },
  title: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "700",
    color: D.text,
    lineHeight: 17,
  },
  titleDanger: { color: "#991B1B" },
  subtitle: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "500",
    color: D.textSecondary,
    lineHeight: 14,
  },
});
