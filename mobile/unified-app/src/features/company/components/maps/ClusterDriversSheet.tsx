import { Modal, Pressable, ScrollView, StyleSheet, View } from "react-native";
import { useBottomSheetLayout } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import type { FleetDriverMapItem } from "./fleetMapTypes";
import { FLEET_STATUS_THEME, FLEET_MAP_COLORS } from "./mapStatusTheme";
import { resolveDriverDisplayName } from "../../utils/companyDriverMapStatus";

type Props = {
  visible: boolean;
  drivers: FleetDriverMapItem[];
  onClose: () => void;
  onSelectDriver: (driver: FleetDriverMapItem) => void;
};

export function ClusterDriversSheet({ visible, drivers, onClose, onSelectDriver }: Props) {
  const sheet = useBottomSheetLayout({
    maxHeightRatio: 0.45,
    maxHeightCap: 360,
    reservedChromeHeight: 56,
  });
  if (drivers.length === 0) return null;

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={s.backdrop}>
        <Pressable style={s.backdropTap} onPress={onClose} />
        <View style={[s.card, { maxHeight: sheet.cardMaxHeight, paddingBottom: sheet.paddingBottom }]}>
          <AppText variant="body" style={s.title}>
            {drivers.length} chauffeurs à proximité
          </AppText>
          <ScrollView style={[s.list, { maxHeight: sheet.scrollMaxHeight }]} keyboardShouldPersistTaps="handled">
            {drivers.map((d) => {
              const theme = FLEET_STATUS_THEME[d.enrichment.operationalStatus];
              return (
                <Pressable
                  key={d.driver_id}
                  onPress={() => onSelectDriver(d)}
                  style={({ pressed }) => [s.row, pressed && { opacity: 0.85 }]}
                >
                  <View style={[s.dot, { backgroundColor: theme.fill }]} />
                  <View style={s.rowText}>
                    <AppText variant="body" style={s.name} numberOfLines={1}>
                      {resolveDriverDisplayName(d)}
                    </AppText>
                    <AppText variant="caption" style={s.sub}>
                      {theme.label}
                    </AppText>
                  </View>
                </Pressable>
              );
            })}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

const s = StyleSheet.create({
  backdrop: { flex: 1, justifyContent: "flex-end" },
  backdropTap: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(15, 23, 42, 0.35)" },
  card: {
    backgroundColor: "#fff",
    borderTopLeftRadius: FLEET_MAP_COLORS.sheetRadius,
    borderTopRightRadius: FLEET_MAP_COLORS.sheetRadius,
    paddingHorizontal: 16,
    paddingTop: 16,
  },
  title: { fontWeight: "700", color: FLEET_MAP_COLORS.text, marginBottom: 8 },
  list: {},
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: FLEET_MAP_COLORS.fabBorder,
  },
  dot: { width: 10, height: 10, borderRadius: 5 },
  rowText: { flex: 1, minWidth: 0 },
  name: { color: FLEET_MAP_COLORS.text, fontWeight: "600" },
  sub: { color: FLEET_MAP_COLORS.textMuted },
});
