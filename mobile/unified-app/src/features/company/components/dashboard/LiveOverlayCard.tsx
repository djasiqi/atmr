import { Platform, Pressable, StyleSheet, View } from "react-native";
import { BlurView } from "expo-blur";
import { AppText } from "../../../../design/ui/AppText";
import type { DashboardLiveOverlay } from "../../dashboard/companyDashboardViewModel";
import { D } from "../../theme/companyDashboardTokens";
import { FLEET_UI, fleetGlassPanel } from "../maps/fleetMapUiTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  overlay: DashboardLiveOverlay;
  onPressUrgent?: () => void;
};

export function LiveOverlayCard({ overlay, onPressUrgent }: Props) {
  const showUrgent = overlay.delayedMissions > 0 && onPressUrgent != null;

  const content = (
    <View style={s.inner}>
      <RowStat value={overlay.activeDrivers} label="actifs" dotColor={D.brand} />
      <View style={s.sep} />
      <RowStat value={overlay.missionsInProgress} label="en cours" dotColor={D.inProgress} />
      <View style={s.sep} />
      <RowStat
        value={overlay.delayedMissions}
        label="urgence"
        dotColor={overlay.delayedMissions > 0 ? D.danger : D.textMuted}
      />
      {showUrgent ? (
        <Pressable
          onPress={onPressUrgent}
          style={({ pressed }) => [s.chevronBtn, pressed && { opacity: 0.75 }]}
          accessibilityRole="button"
          accessibilityLabel="Voir les urgences"
          hitSlop={8}
        >
          <AppText style={s.chevronText}>›</AppText>
        </Pressable>
      ) : null}
    </View>
  );

  return (
    <View
      style={s.wrap}
      pointerEvents="box-none"
      accessibilityLabel="Activité en direct"
    >
      {Platform.OS === "ios" || Platform.OS === "android" ? (
        <BlurView intensity={48} tint="light" style={s.blurShell}>
          {content}
        </BlurView>
      ) : (
        <View style={[fleetGlassPanel(), s.glassWeb]}>{content}</View>
      )}
    </View>
  );
}

function RowStat({
  value,
  label,
  dotColor,
}: {
  value: number;
  label: string;
  dotColor: string;
}) {
  return (
    <View style={s.stat} accessibilityLabel={`${value} ${label}`}>
      <View style={[s.dot, { backgroundColor: dotColor }]} />
      <AppText style={s.statValue}>{value}</AppText>
      <AppText style={s.statLabel} numberOfLines={1}>{label}</AppText>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: {
    position: "absolute",
    left: FLEET_UI.overlayLeft,
    bottom: FLEET_UI.overlayBottom,
    alignSelf: "flex-start",
    zIndex: 14,
  },
  blurShell: {
    borderRadius: FLEET_UI.overlayRadius,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: FLEET_UI.glassBorder,
    ...FLEET_UI.shadow,
  },
  glassWeb: {
    paddingHorizontal: 0,
    paddingVertical: 0,
  },
  inner: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 8,
    gap: 8,
  },
  stat: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  statValue: {
    color: D.text,
    fontSize: FONT_SIZE.px13,
    fontWeight: "700",
    lineHeight: 16,
  },
  statLabel: {
    color: D.textSecondary,
    fontSize: FONT_SIZE.px12,
    fontWeight: "500",
    lineHeight: 16,
  },
  sep: {
    width: StyleSheet.hairlineWidth,
    height: 14,
    backgroundColor: "rgba(148, 163, 184, 0.45)",
  },
  chevronBtn: {
    marginLeft: 2,
    justifyContent: "center",
    alignItems: "center",
  },
  chevronText: {
    color: D.danger,
    fontSize: FONT_SIZE.px18,
    fontWeight: "700",
    lineHeight: 20,
  },
});
