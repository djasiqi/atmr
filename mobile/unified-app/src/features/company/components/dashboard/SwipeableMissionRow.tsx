import { Platform, Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Swipeable } from "react-native-gesture-handler";
import { AppText } from "../../../../design/ui/AppText";
import type { DashboardCompactMissionRow } from "../../dashboard/companyDashboardViewModel";
import { D } from "../../theme/companyDashboardTokens";
import { M } from "./dashboardMobileTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  row: DashboardCompactMissionRow;
  showSeparator?: boolean;
  onPress?: () => void;
  onSwipePrimary?: () => void;
  onSwipeSecondary?: () => void;
};

function MissionRowContent({
  row,
  showSeparator,
  onPress,
}: {
  row: DashboardCompactMissionRow;
  showSeparator?: boolean;
  onPress?: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      disabled={!onPress}
      style={({ pressed }) => [s.row, showSeparator && s.sep, pressed && onPress && s.pressed]}
      accessibilityRole={onPress ? "button" : "text"}
      accessibilityLabel={`${row.timeLabel} ${row.clientName}, ${row.status.label}`}
    >
      <View style={[s.bar, { backgroundColor: row.status.barColor }]} />
      <View style={s.body}>
        <View style={s.topLine}>
          <AppText style={s.time}>{row.timeLabel}</AppText>
          <View style={[s.badge, { backgroundColor: `${row.status.barColor}16` }]}>
            <AppText style={[s.badgeText, { color: row.status.barColor }]}>{row.status.label}</AppText>
          </View>
        </View>
        <AppText style={s.client} numberOfLines={1}>
          {row.clientName}
        </AppText>
        <AppText style={s.route} numberOfLines={1}>
          {row.routeLabel}
          {row.etaLabel ? ` · ETA ${row.etaLabel}` : ""}
        </AppText>
      </View>
      <Ionicons name="chevron-forward" size={16} color={D.textMuted} style={s.chev} />
    </Pressable>
  );
}

function SwipeActions({
  row,
  onPrimary,
  onSecondary,
}: {
  row: DashboardCompactMissionRow;
  onPrimary?: () => void;
  onSecondary?: () => void;
}) {
  const secondary =
    row.status.tone === "delayed"
      ? { label: "Urgences", icon: "warning-outline" as const, bg: D.danger }
      : row.status.tone === "in_progress"
        ? { label: "Suivre", icon: "navigate-outline" as const, bg: D.inProgress }
        : { label: "Assigner", icon: "person-add-outline" as const, bg: "#F59E0B" };

  return (
    <View style={s.actions}>
      {onSecondary ? (
        <Pressable
          onPress={onSecondary}
          style={({ pressed }) => [s.actionBtn, { backgroundColor: secondary.bg }, pressed && s.actionPressed]}
          accessibilityRole="button"
          accessibilityLabel={secondary.label}
        >
          <Ionicons name={secondary.icon} size={18} color="#fff" />
        </Pressable>
      ) : null}
      {onPrimary ? (
        <Pressable
          onPress={onPrimary}
          style={({ pressed }) => [s.actionBtn, s.actionPrimary, pressed && s.actionPressed]}
          accessibilityRole="button"
          accessibilityLabel="Ouvrir la course"
        >
          <Ionicons name="open-outline" size={18} color="#fff" />
        </Pressable>
      ) : null}
    </View>
  );
}

export function SwipeableMissionRow({
  row,
  showSeparator,
  onPress,
  onSwipePrimary,
  onSwipeSecondary,
}: Props) {
  const enableSwipe = Platform.OS !== "web" && (onSwipePrimary != null || onSwipeSecondary != null);

  if (!enableSwipe) {
    return <MissionRowContent row={row} showSeparator={showSeparator} onPress={onPress} />;
  }

  return (
    <Swipeable
      overshootRight={false}
      friction={2}
      rightThreshold={32}
      renderRightActions={() => (
        <SwipeActions row={row} onPrimary={onSwipePrimary ?? onPress} onSecondary={onSwipeSecondary} />
      )}
    >
      <View style={s.swipeBg}>
        <MissionRowContent row={row} showSeparator={showSeparator} onPress={onPress} />
      </View>
    </Swipeable>
  );
}

const s = StyleSheet.create({
  swipeBg: { backgroundColor: D.cardBg },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: M.padH,
    paddingVertical: M.padRow + 1,
    minHeight: 56,
  },
  sep: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: M.hairline,
  },
  pressed: { backgroundColor: "rgba(148, 163, 184, 0.06)" },
  bar: {
    width: M.barW,
    alignSelf: "stretch",
    borderRadius: 2,
    minHeight: 40,
  },
  body: { flex: 1, minWidth: 0, gap: 1 },
  topLine: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 6,
  },
  time: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "800",
    color: D.text,
    letterSpacing: -0.2,
  },
  client: {
    fontSize: FONT_SIZE.px14,
    fontWeight: "700",
    color: D.text,
    letterSpacing: -0.2,
  },
  route: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "500",
    color: D.textSecondary,
    lineHeight: 14,
  },
  badge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 6,
  },
  badgeText: {
    fontSize: FONT_SIZE.px9,
    fontWeight: "800",
    letterSpacing: 0.3,
    textTransform: "uppercase",
  },
  chev: { opacity: 0.45, marginLeft: 2 },
  actions: {
    flexDirection: "row",
    alignItems: "stretch",
    marginVertical: 1,
  },
  actionBtn: {
    width: 52,
    alignItems: "center",
    justifyContent: "center",
  },
  actionPrimary: {
    backgroundColor: D.brandDark,
  },
  actionPressed: { opacity: 0.88 },
});
