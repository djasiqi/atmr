import { LinearGradient } from "expo-linear-gradient";
import { Pressable, ScrollView, StyleSheet, View } from "react-native";
import { useAppViewport, useBottomSheetLayout } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import { AppModal } from "../../../../design/ui/AppModal";
import type { FleetMapLayerType, FleetMapLayersState } from "./fleetMapTypes";
import { DEFAULT_FLEET_MAP_LAYERS, DEFAULT_FLEET_MAP_MISSION_LAYERS } from "./fleetMapTypes";
import { FLEET_MAP_COLORS } from "./mapStatusTheme";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

const MAP_TYPES: { key: FleetMapLayerType; label: string }[] = [
  { key: "standard", label: "Plan" },
  { key: "satellite", label: "Satellite" },
  { key: "terrain", label: "Relief" },
];

type Props = {
  visible: boolean;
  layers: FleetMapLayersState;
  onChange: (next: FleetMapLayersState) => void;
  onClose: () => void;
};

type ToggleRowProps = {
  title: string;
  subtitle: string;
  checked: boolean;
  onToggle: () => void;
};

function ToggleRow({ title, subtitle, checked, onToggle }: ToggleRowProps) {
  return (
    <View style={s.toggleRow}>
      <View style={s.toggleMain}>
        <AppText variant="body" style={s.toggleTitle}>{title}</AppText>
        <AppText variant="caption" style={s.toggleSubtitle}>{subtitle}</AppText>
      </View>
      <Pressable
        onPress={onToggle}
        style={({ pressed }) => [s.switch, checked && s.switchOn, pressed && s.switchPressed]}
        accessibilityRole="switch"
        accessibilityState={{ checked }}
      >
        <View style={[s.switchThumb, checked && s.switchThumbOn]} />
      </Pressable>
    </View>
  );
}

function normalizeMissionLayers(layers: FleetMapLayersState): FleetMapLayersState["mission"] {
  return {
    ...DEFAULT_FLEET_MAP_MISSION_LAYERS,
    ...(layers.mission ?? DEFAULT_FLEET_MAP_MISSION_LAYERS),
  };
}

function MapTypePreview({ type, compact }: { type: FleetMapLayerType; compact: boolean }) {
  return (
    <View style={[s.mapTypeThumb, compact && s.mapTypeThumbCompact, type === "satellite" && s.thumbSatellite, type === "terrain" && s.thumbTerrain]}>
      {type === "standard" ? (
        <>
          <View style={[s.road, s.roadPrimary]} />
          <View style={[s.road, s.roadSecondary]} />
          <View style={[s.road, s.roadTertiary]} />
          <View style={[s.mapPatch, s.mapPatchPlan]} />
        </>
      ) : null}
      {type === "satellite" ? (
        <>
          <View style={[s.mapPatch, s.mapPatchSatelliteA]} />
          <View style={[s.mapPatch, s.mapPatchSatelliteB]} />
          <View style={[s.mapPatch, s.mapPatchSatelliteC]} />
        </>
      ) : null}
      {type === "terrain" ? (
        <>
          <View style={[s.reliefLine, { top: 10 }]} />
          <View style={[s.reliefLine, { top: 20 }]} />
          <View style={[s.reliefLine, { top: 30 }]} />
          <View style={[s.reliefLine, { top: 40 }]} />
          <View style={[s.mapPatch, s.mapPatchTerrain]} />
        </>
      ) : null}
      <LinearGradient
        colors={["rgba(15,23,42,0.04)", "rgba(15,23,42,0.18)"]}
        style={StyleSheet.absoluteFillObject}
      />
    </View>
  );
}

export function MapLayersSheet({ visible, layers, onChange, onClose }: Props) {
  const { width } = useAppViewport();
  const sheet = useBottomSheetLayout({ reservedChromeHeight: 64 });
  const mission = normalizeMissionLayers(layers);
  const isWide = width >= 760;
  const isCompactMobile = width < 430;

  const activeLabels = [
    layers.mapType === "standard" ? "Plan" : layers.mapType === "satellite" ? "Satellite" : "Relief",
    mission.missionRoutes ? "Trajets mission" : null,
    mission.compactRoutes ? "Mode compact" : null,
    mission.focusActive ? "Focus mission active" : null,
    mission.autoSimplify ? "Simplification auto" : null,
    layers.traffic ? "Traffic en temps reel" : null,
    layers.heatmapMode === "delays" ? "Retards (heatmap)" : null,
  ].filter((v): v is string => Boolean(v));

  const applyMapType = (mapType: FleetMapLayerType) => onChange({ ...layers, mapType });
  const reset = () => onChange(DEFAULT_FLEET_MAP_LAYERS);

  return (
    <AppModal
      visible={visible}
      onClose={onClose}
      variant="bottomSheet"
      backdropOpacity={0.34}
      screen="company.map_layers"
    >
      <View
        style={[
          s.card,
          isCompactMobile && s.cardCompact,
          { maxHeight: sheet.cardMaxHeight, paddingBottom: sheet.paddingBottom },
        ]}
      >
          <View style={[s.headerTop, isCompactMobile && s.headerTopCompact]}>
            <View style={s.headerLeft}>
              <View>
                <AppText variant="body" style={s.title}>Vue Carte</AppText>
                <AppText variant="caption" style={s.subtitle}>Affichage operationnel</AppText>
              </View>
            </View>
          </View>

          <ScrollView
            style={[s.scroll, { maxHeight: sheet.scrollMaxHeight }]}
            contentContainerStyle={s.scrollContent}
            keyboardShouldPersistTaps="handled"
          >
            <View style={[s.layout, isWide && s.layoutWide]}>
              <View style={s.mainCol}>
                <View style={[s.panel, isCompactMobile && s.panelCompact]}>
                  <AppText variant="caption" style={s.panelTitle}>Style de carte</AppText>
                  <View style={s.mapTypeRow}>
                    {MAP_TYPES.map((opt) => {
                      const active = layers.mapType === opt.key;
                      return (
                        <Pressable
                          key={opt.key}
                          onPress={() => applyMapType(opt.key)}
                          style={({ pressed }) => [s.mapTypeCard, isCompactMobile && s.mapTypeCardCompact, active && s.mapTypeCardActive, pressed && s.btnPressed]}
                          accessibilityRole="radio"
                          accessibilityState={{ selected: active }}
                        >
                          <MapTypePreview type={opt.key} compact={isCompactMobile} />
                          <AppText variant="body" style={[s.mapTypeLabel, active && s.mapTypeLabelActive]}>{opt.label}</AppText>
                        </Pressable>
                      );
                    })}
                  </View>
                </View>

                <View style={[s.panel, isCompactMobile && s.panelCompact]}>
                  <AppText variant="caption" style={s.panelTitle}>Couches en temps reel</AppText>
                  <ToggleRow
                    title="Trajets mission"
                    subtitle="Routes, pickups, destinations et ETA"
                    checked={mission.missionRoutes}
                    onToggle={() => onChange({ ...layers, mission: { ...mission, missionRoutes: !mission.missionRoutes } })}
                  />
                  <ToggleRow
                    title="Traffic en temps reel"
                    subtitle="Conditions de circulation"
                    checked={layers.traffic}
                    onToggle={() => onChange({ ...layers, traffic: !layers.traffic })}
                  />
                  <ToggleRow
                    title="Retards (heatmap)"
                    subtitle="Zones rouges selon les chauffeurs en retard"
                    checked={layers.heatmapMode === "delays"}
                    onToggle={() => onChange({ ...layers, heatmapMode: layers.heatmapMode === "delays" ? "off" : "delays" })}
                  />
                </View>

                <View style={[s.panel, isCompactMobile && s.panelCompact]}>
                  <AppText variant="caption" style={s.panelTitle}>Optimisation visuelle</AppText>
                  <ToggleRow
                    title="Mode compact (anti-spaghetti)"
                    subtitle="Limite les routes secondaires"
                    checked={mission.compactRoutes}
                    onToggle={() => onChange({ ...layers, mission: { ...mission, compactRoutes: !mission.compactRoutes } })}
                  />
                  <ToggleRow
                    title="Focus mission active"
                    subtitle="Met en avant la mission selectionnee"
                    checked={mission.focusActive}
                    onToggle={() => onChange({ ...layers, mission: { ...mission, focusActive: !mission.focusActive } })}
                  />
                  <ToggleRow
                    title="Simplification automatique"
                    subtitle="Reduit la densite sous forte charge"
                    checked={mission.autoSimplify}
                    onToggle={() => onChange({ ...layers, mission: { ...mission, autoSimplify: !mission.autoSimplify } })}
                  />
                </View>
              </View>

              {!isCompactMobile ? (
              <View style={s.summaryCol}>
                <View style={s.summaryCard}>
                  <AppText variant="caption" style={s.panelTitle}>Affichage actuel</AppText>
                  {activeLabels.map((label) => (
                    <View key={label} style={s.summaryItem}>
                      <View style={s.summaryDot} />
                      <AppText variant="caption" style={s.summaryText}>{label}</AppText>
                    </View>
                  ))}
                </View>
              </View>
              ) : null}
            </View>
          </ScrollView>

        <View style={[s.footerRow, isCompactMobile && s.footerRowCompact]}>
          <Pressable onPress={reset} style={({ pressed }) => [s.footerGhost, isCompactMobile && s.footerBtnCompact, pressed && s.btnPressed]}>
            <AppText variant="body" style={s.footerGhostText}>Reinitialiser</AppText>
          </Pressable>
          <Pressable onPress={onClose} style={({ pressed }) => [s.footerPrimary, isCompactMobile && s.footerBtnCompact, pressed && s.btnPressed]}>
            <AppText variant="body" style={s.footerPrimaryText}>Fermer</AppText>
          </Pressable>
        </View>
      </View>
    </AppModal>
  );
}

const s = StyleSheet.create({
  card: {
    backgroundColor: "#fff",
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingHorizontal: 16,
    paddingTop: 14,
  },
  cardCompact: {
    paddingHorizontal: 12,
    paddingTop: 12,
  },
  headerTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 10,
    marginBottom: 8,
  },
  headerTopCompact: {
    marginBottom: 8,
  },
  headerLeft: { flexDirection: "row", alignItems: "center", flex: 1 },
  title: {
    color: FLEET_MAP_COLORS.text,
    fontSize: FONT_SIZE.px18,
    lineHeight: 22,
    fontWeight: "600",
    marginBottom: 0,
  },
  subtitle: {
    color: FLEET_MAP_COLORS.textMuted,
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    fontWeight: "600",
    marginTop: 8,
  },
  scroll: {},
  scrollContent: { paddingBottom: 12 },
  layout: { gap: 10 },
  layoutWide: { flexDirection: "row", alignItems: "flex-start" },
  mainCol: { flex: 1, gap: 12 },
  summaryCol: { width: "100%" },
  panel: {
    backgroundColor: "#fff",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(226,232,240,0.9)",
    padding: 10,
    gap: 8,
  },
  panelCompact: {
    padding: 10,
    gap: 8,
    borderRadius: 14,
  },
  panelTitle: {
    color: FLEET_MAP_COLORS.textMuted,
    fontWeight: "600",
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
  },
  mapTypeRow: { flexDirection: "row", gap: 8 },
  mapTypeCard: {
    flex: 1,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(226,232,240,0.95)",
    padding: 5,
    backgroundColor: "#fff",
    gap: 4,
  },
  mapTypeCardCompact: {
    padding: 5,
    gap: 4,
  },
  mapTypeCardActive: {
    borderColor: "rgba(0,121,107,0.45)",
    backgroundColor: "rgba(0,121,107,0.06)",
  },
  mapTypeThumb: {
    height: 46,
    borderRadius: 10,
    overflow: "hidden",
    justifyContent: "flex-end",
    alignItems: "flex-end",
    padding: 6,
    backgroundColor: "#E8F2F8",
  },
  thumbSatellite: {
    backgroundColor: "#2D3F4A",
  },
  thumbTerrain: {
    backgroundColor: "#E7EFDD",
  },
  mapTypeThumbCompact: {
    height: 36,
    borderRadius: 8,
  },
  road: {
    position: "absolute",
    borderRadius: 3,
    backgroundColor: "rgba(255,255,255,0.96)",
  },
  roadPrimary: {
    left: -8,
    right: -8,
    top: 22,
    height: 9,
    transform: [{ rotate: "8deg" }],
  },
  roadSecondary: {
    width: 70,
    height: 7,
    top: 8,
    right: -10,
    transform: [{ rotate: "-20deg" }],
  },
  roadTertiary: {
    width: 56,
    height: 6,
    bottom: 6,
    left: -12,
    transform: [{ rotate: "-8deg" }],
  },
  mapPatch: {
    position: "absolute",
    borderRadius: 8,
  },
  mapPatchPlan: {
    width: 26,
    height: 18,
    right: 6,
    bottom: 8,
    backgroundColor: "rgba(0, 121, 107, 0.30)",
  },
  mapPatchSatelliteA: {
    width: 42,
    height: 18,
    top: 8,
    left: 6,
    backgroundColor: "rgba(78, 109, 66, 0.48)",
  },
  mapPatchSatelliteB: {
    width: 34,
    height: 16,
    bottom: 10,
    right: 10,
    backgroundColor: "rgba(95, 74, 54, 0.46)",
  },
  mapPatchSatelliteC: {
    width: 16,
    height: 12,
    top: 24,
    right: 30,
    backgroundColor: "rgba(166, 140, 106, 0.42)",
  },
  reliefLine: {
    position: "absolute",
    left: -10,
    right: -6,
    height: 3,
    borderRadius: 3,
    backgroundColor: "rgba(147, 176, 129, 0.5)",
    transform: [{ rotate: "-7deg" }],
  },
  mapPatchTerrain: {
    width: 28,
    height: 16,
    right: 8,
    bottom: 9,
    backgroundColor: "rgba(116, 146, 86, 0.35)",
  },
  mapTypeLabel: { textAlign: "center", fontWeight: "600", color: FLEET_MAP_COLORS.text },
  mapTypeLabelActive: { color: "#00796B" },
  toggleRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingVertical: 4,
  },
  toggleMain: { flex: 1, minWidth: 0 },
  toggleTitle: { color: FLEET_MAP_COLORS.text, fontWeight: "600" },
  toggleSubtitle: { color: FLEET_MAP_COLORS.textMuted, marginTop: 0 },
  switch: {
    width: 42,
    height: 25,
    borderRadius: 13,
    backgroundColor: "rgba(203,213,225,0.9)",
    padding: 3,
    justifyContent: "center",
  },
  switchOn: {
    backgroundColor: FLEET_MAP_COLORS.brand,
  },
  switchPressed: { opacity: 0.9 },
  switchThumb: {
    width: 19,
    height: 19,
    borderRadius: 10,
    backgroundColor: "#fff",
  },
  switchThumbOn: {
    alignSelf: "flex-end",
  },
  summaryCard: {
    backgroundColor: "#fff",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(226,232,240,0.9)",
    padding: 10,
    gap: 6,
  },
  summaryItem: { flexDirection: "row", alignItems: "center", gap: 8 },
  summaryDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: "rgba(148,163,184,0.9)",
  },
  summaryText: { color: FLEET_MAP_COLORS.text, fontWeight: "500" },
  footerRow: { flexDirection: "row", gap: 10, marginTop: 10 },
  footerRowCompact: { gap: 8, marginTop: 8 },
  footerGhost: {
    flex: 1,
    minHeight: 44,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.30)",
    backgroundColor: "#fff",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
  },
  footerGhostText: { color: "#475569", fontSize: FONT_SIZE.px12, lineHeight: 16, fontWeight: "500" },
  footerPrimary: {
    flex: 1.15,
    minHeight: 44,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(0,121,107,0.28)",
    backgroundColor: "rgba(0,121,107,0.08)",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
  },
  footerBtnCompact: {
    minHeight: 44,
  },
  footerPrimaryText: { color: "#00796B", fontSize: FONT_SIZE.px12, lineHeight: 16, fontWeight: "600" },
  btnPressed: { opacity: 0.9 },
});
