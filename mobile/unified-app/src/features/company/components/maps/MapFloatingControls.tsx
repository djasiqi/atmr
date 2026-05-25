import { Platform, Pressable, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { FLEET_MAP_COLORS } from "./mapStatusTheme";
import { FLEET_UI, fleetGlassPanel } from "./fleetMapUiTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  activeFilterCount: number;
  searchActive?: boolean;
  onRecenter: () => void;
  onOpenSearch: () => void;
  onOpenFilters: () => void;
  onOpenLayers: () => void;
  stackTop?: number;
  isFullscreen?: boolean;
  onToggleFullscreen?: () => void;
};

export function MapFloatingControls({
  activeFilterCount,
  searchActive = false,
  onRecenter,
  onOpenSearch,
  onOpenFilters,
  onOpenLayers,
  stackTop,
  isFullscreen = false,
  onToggleFullscreen,
}: Props) {
  return (
    <View style={[s.col, stackTop != null ? { top: stackTop } : null]} pointerEvents="box-none">
      {typeof onToggleFullscreen === "function" ? (
        <Fab
          icon={isFullscreen ? "contract-outline" : "expand-outline"}
          label={isFullscreen ? "Quitter le plein écran" : "Carte en plein écran"}
          onPress={onToggleFullscreen}
        />
      ) : null}
      <Fab icon="layers-outline" label="Couches carte" onPress={onOpenLayers} />
      <Fab
        icon="search-outline"
        label="Rechercher un chauffeur"
        onPress={onOpenSearch}
        active={searchActive}
      />
      <Fab
        icon="funnel-outline"
        label="Filtres chauffeurs"
        onPress={onOpenFilters}
        badge={activeFilterCount > 0 ? activeFilterCount : undefined}
      />
      <Fab icon="locate-outline" label="Recentrer la carte" onPress={onRecenter} />
    </View>
  );
}

function Fab({
  icon,
  label,
  onPress,
  badge,
  active = false,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  onPress: () => void;
  badge?: number;
  active?: boolean;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        fleetGlassPanel(s.fab),
        pressed && s.fabPressed,
      ]}
      android_ripple={
        Platform.OS === "android"
          ? { color: "rgba(15, 23, 42, 0.08)", borderless: false }
          : undefined
      }
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ selected: active }}
      pointerEvents="auto"
    >
      <Ionicons name={icon} size={18} color={FLEET_MAP_COLORS.text} />
      {badge != null && badge > 0 ? (
        <View style={s.badge}>
          <Text style={s.badgeText}>{badge > 9 ? "9+" : badge}</Text>
        </View>
      ) : null}
    </Pressable>
  );
}

const s = StyleSheet.create({
  col: {
    position: "absolute",
    top: FLEET_UI.fabStackTop,
    right: FLEET_UI.fabRight,
    gap: FLEET_UI.fabGap,
    zIndex: 40,
  },
  fab: {
    width: FLEET_UI.fabSize,
    height: FLEET_UI.fabSize,
    borderRadius: FLEET_UI.fabSize / 2,
    alignItems: "center",
    justifyContent: "center",
  },
  fabPressed: { opacity: 0.9, transform: [{ scale: 0.97 }] },
  badge: {
    position: "absolute",
    top: 0,
    right: 0,
    minWidth: 15,
    height: 15,
    borderRadius: 8,
    paddingHorizontal: 3,
    backgroundColor: FLEET_MAP_COLORS.brand,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1.5,
    borderColor: "#fff",
  },
  badgeText: {
    color: "#fff",
    fontSize: FONT_SIZE.px9,
    fontWeight: "800",
  },
});
