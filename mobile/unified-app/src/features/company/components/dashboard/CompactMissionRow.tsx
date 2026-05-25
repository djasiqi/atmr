import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import type { DashboardCompactMissionRow } from "../../dashboard/companyDashboardViewModel";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  row: DashboardCompactMissionRow;
  onPress?: () => void;
  onAction?: () => void;
  showSeparator?: boolean;
};

export function CompactMissionRow({ row, onPress, onAction, showSeparator }: Props) {
  return (
    <Pressable
      onPress={onPress}
      disabled={!onPress}
      style={({ pressed }) => [s.wrap, showSeparator && s.sep, pressed && onPress && { opacity: 0.9 }]}
      accessibilityRole={onPress ? "button" : "text"}
      accessibilityLabel={`Course ${row.timeLabel} ${row.clientName}`}
    >
      <View style={[s.bar, { backgroundColor: row.status.barColor }]} />
      <View style={s.body}>
        <View style={s.top}>
          <AppText variant="label" style={s.time}>
            {row.timeLabel}
          </AppText>
          {row.etaLabel ? (
            <AppText variant="caption" style={s.eta}>
              ETA {row.etaLabel}
            </AppText>
          ) : null}
        </View>
        <AppText variant="body" style={s.client} numberOfLines={1}>
          {row.clientName}
        </AppText>
        <AppText variant="caption" style={s.route} numberOfLines={1}>
          {row.routeLabel}
        </AppText>
        <View style={[s.badge, { backgroundColor: `${row.status.barColor}18` }]}>
          <AppText variant="caption" style={[s.badgeText, { color: row.status.barColor }]}>
            {row.status.label}
          </AppText>
        </View>
      </View>
      <Pressable
        onPress={onAction ?? onPress}
        style={({ pressed: p }) => [s.actionBtn, p && { opacity: 0.85 }]}
        accessibilityRole="button"
        accessibilityLabel="Ouvrir la course"
        hitSlop={8}
      >
        <Ionicons name="chevron-forward" size={20} color="#64748B" />
      </Pressable>
    </Pressable>
  );
}

const s = StyleSheet.create({
  wrap: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 10,
  },
  sep: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "rgba(148, 163, 184, 0.28)",
  },
  bar: { width: 4, alignSelf: "stretch", borderRadius: 4, minHeight: 52 },
  body: { flex: 1, minWidth: 0, gap: 3 },
  top: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 8 },
  time: { color: "#0F172A", fontWeight: "800" },
  eta: { color: "#64748B", fontWeight: "600" },
  client: { color: "#0F172A", fontWeight: "700" },
  route: { color: "#64748B" },
  badge: {
    alignSelf: "flex-start",
    marginTop: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  badgeText: { fontWeight: "700", fontSize: FONT_SIZE.px11 },
  actionBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(148, 163, 184, 0.12)",
  },
});
