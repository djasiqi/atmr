import { useState } from "react";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

import { Pressable, ScrollView, StyleSheet, View } from "react-native";

import { Ionicons } from "@expo/vector-icons";

import { AppText } from "../../../../design/ui/AppText";

import { FLEET_MISSION_LIFECYCLE_LEGEND } from "./fleetMapMissionPolicies";
import { FLEET_MAP_LEGEND_ITEMS, FLEET_MAP_COLORS } from "./mapStatusTheme";

import { FleetLegendMarkerSwatch } from "./FleetDriverMarkerVisual";

import { FLEET_UI, fleetGlassPanel } from "./fleetMapUiTokens";



/** Légende repliable — icônes identiques aux marqueurs carte. */

export function MapLegend() {

  const [open, setOpen] = useState(false);



  return (

    <View style={s.wrap} pointerEvents="box-none">

      <Pressable

        onPress={() => setOpen((v) => !v)}

        style={({ pressed }) => [fleetGlassPanel(s.toggle), pressed && { opacity: 0.9 }]}

        accessibilityRole="button"

        accessibilityLabel={open ? "Masquer la légende" : "Afficher la légende"}

        accessibilityState={{ expanded: open }}

      >

        <Ionicons

          name={open ? "chevron-up" : "information-circle-outline"}

          size={17}

          color={FLEET_MAP_COLORS.text}

        />

        {!open ? (

          <AppText variant="caption" style={s.toggleLabel}>

            Légende

          </AppText>

        ) : null}

      </Pressable>



      {open ? (

        <ScrollView

          horizontal

          showsHorizontalScrollIndicator={false}

          style={s.scroll}

          contentContainerStyle={[fleetGlassPanel(s.capsule), s.row]}

          keyboardShouldPersistTaps="handled"

        >

          {FLEET_MISSION_LIFECYCLE_LEGEND.map((item) => (
            <View key={`mission-${item.phase}`} style={s.chip}>
              <View
                style={[
                  s.routeSwatch,
                  {
                    backgroundColor: item.color,
                    opacity: item.dash ? 0.65 : 1,
                    borderStyle: item.dash ? "dashed" : "solid",
                  },
                ]}
              />
              <AppText variant="caption" style={s.label} numberOfLines={1}>
                {item.label}
              </AppText>
            </View>
          ))}
          {FLEET_MAP_LEGEND_ITEMS.map((item) => (

            <View key={item.status} style={s.chip}>

              {item.status === "cluster" ? (

                <View style={[s.clusterDot, { backgroundColor: item.color }]}>

                  <AppText variant="caption" style={s.clusterNum}>

                    3

                  </AppText>

                </View>

              ) : (

                <FleetLegendMarkerSwatch color={item.color} variant={item.variant} />

              )}

              <AppText variant="caption" style={s.label} numberOfLines={1}>

                {item.label}

              </AppText>

            </View>

          ))}

        </ScrollView>

      ) : null}

    </View>

  );

}



const s = StyleSheet.create({

  wrap: {

    position: "absolute",

    top: FLEET_UI.legendTop,

    left: FLEET_UI.legendLeft,

    maxWidth: "72%",

    zIndex: 13,

    gap: 6,

  },

  toggle: {

    flexDirection: "row",

    alignItems: "center",

    alignSelf: "flex-start",

    gap: 5,

    paddingHorizontal: 10,

    paddingVertical: 7,

    borderRadius: 20,

  },

  toggleLabel: {

    color: FLEET_MAP_COLORS.textMuted,

    fontSize: FONT_SIZE.px11,

    fontWeight: "600",

  },

  scroll: { maxHeight: 40 },

  capsule: {

    flexDirection: "row",

    paddingHorizontal: 6,

    paddingVertical: 4,

    borderRadius: 18,

  },

  row: { flexDirection: "row", alignItems: "center", gap: 2 },

  chip: {

    flexDirection: "row",

    alignItems: "center",

    gap: 5,

    paddingHorizontal: 7,

    paddingVertical: 3,

  },

  routeSwatch: {
    width: 18,
    height: 4,
    borderRadius: 2,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.45)",
  },

  label: {

    color: FLEET_MAP_COLORS.textMuted,

    fontSize: FONT_SIZE.px9,

    fontWeight: "600",

  },

  clusterDot: {

    width: 16,

    height: 16,

    borderRadius: 8,

    borderWidth: 1.5,

    borderColor: "#fff",

    alignItems: "center",

    justifyContent: "center",

  },

  clusterNum: {

    color: "#fff",

    fontSize: FONT_SIZE.px7,

    fontWeight: "800",

    lineHeight: 8,

  },

});


