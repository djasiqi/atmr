import { Modal, Platform, Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { AppText } from "../../../../design/ui/AppText";
import { computeCompanyFloatingBottomPad } from "../../../../design/navigation/BaseFloatingBar";
import { fleetGlassPanel } from "../maps/fleetMapUiTokens";
import { DashboardOpsFeed, type DashboardOpsFeedProps } from "./DashboardOpsFeed";
import { FLEET_COCKPIT } from "./companyFleetCockpitLayout";

type Props = {
  visible: boolean;
  onClose: () => void;
} & DashboardOpsFeedProps;

/** Panneau opérations (feed) — glisse au-dessus de la carte sans quitter le cockpit. */
export function DashboardOpsSheet({ visible, onClose, ...feed }: Props) {
  const insets = useSafeAreaInsets();
  const bottomPad =
    FLEET_COCKPIT.tabBarHeight + computeCompanyFloatingBottomPad(insets.bottom) + 8;

  if (!visible) return null;

  return (
    <Modal visible transparent animationType="slide" onRequestClose={onClose}>
      <View style={s.root}>
        <Pressable style={s.backdrop} onPress={onClose} accessibilityLabel="Fermer" />
        <View style={[s.sheet, fleetGlassPanel(), { paddingBottom: bottomPad }]}>
          <View style={s.handleRow}>
            <View style={s.handle} />
            <Pressable
              onPress={onClose}
              hitSlop={10}
              style={({ pressed }) => [s.closeBtn, pressed && { opacity: 0.85 }]}
              accessibilityRole="button"
              accessibilityLabel="Fermer le panneau opérations"
            >
              <Ionicons name="chevron-down" size={22} color="#64748B" />
            </Pressable>
          </View>
          <AppText variant="sectionTitle" style={s.title}>
            Opérations
          </AppText>
          <View style={s.feedWrap}>
            <DashboardOpsFeed {...feed} />
          </View>
        </View>
      </View>
    </Modal>
  );
}

const s = StyleSheet.create({
  root: {
    flex: 1,
    justifyContent: "flex-end",
    ...Platform.select({
      web: { position: "fixed" as const, inset: 0, zIndex: 80 },
      default: {},
    }),
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(15, 23, 42, 0.28)",
  },
  sheet: {
    maxHeight: "72%",
    borderBottomLeftRadius: 0,
    borderBottomRightRadius: 0,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    overflow: "hidden",
  },
  handleRow: {
    alignItems: "center",
    paddingTop: 10,
    paddingBottom: 4,
  },
  handle: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: "rgba(148, 163, 184, 0.45)",
  },
  closeBtn: {
    position: "absolute",
    right: 12,
    top: 6,
    padding: 4,
  },
  title: {
    paddingHorizontal: FLEET_COCKPIT.sideGutter,
    paddingBottom: 6,
  },
  feedWrap: {
    paddingHorizontal: 4,
  },
});
