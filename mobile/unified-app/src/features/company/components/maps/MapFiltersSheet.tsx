import { Pressable, StyleSheet, TextInput, View } from "react-native";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

import { AppText } from "../../../../design/ui/AppText";

import { EnterpriseBottomSheet } from "../EnterpriseBottomSheet";

import type {

  FleetDriverMapItem,

  FleetMapFiltersState,

  FleetMapStatusFilter,

  FleetVehicleFilter,

} from "./fleetMapTypes";

import { FLEET_MAP_COLORS } from "./mapStatusTheme";

import { listFleetDriverOptions } from "./fleetMapLogic";



const STATUS_OPTIONS: { key: FleetMapStatusFilter; label: string }[] = [

  { key: "all", label: "Tous" },

  { key: "available", label: "Disponible" },

  { key: "on_mission", label: "En mission" },

  { key: "break", label: "En pause" },

  { key: "delayed", label: "En retard" },

  { key: "urgent", label: "Urgences" },

];



const VEHICLE_OPTIONS: { key: FleetVehicleFilter; label: string }[] = [

  { key: "all", label: "Tous" },

  { key: "berline", label: "Berline" },

  { key: "van", label: "Van" },

  { key: "vsl", label: "VSL" },

  { key: "urgence", label: "Urgence" },

];



type Props = {

  visible: boolean;

  filters: FleetMapFiltersState;

  drivers: FleetDriverMapItem[];

  onChange: (next: FleetMapFiltersState) => void;

  onClose: () => void;

};



export function MapFiltersSheet({ visible, filters, drivers, onChange, onClose }: Props) {

  const driverOptions = listFleetDriverOptions(drivers);



  return (

    <EnterpriseBottomSheet

      visible={visible}

      onClose={onClose}

      title="Filtres carte"

      subtitle="Affinez les chauffeurs visibles"

    >

      <AppText variant="caption" style={s.section}>

        Statut

      </AppText>

      <View style={s.chips}>

        {STATUS_OPTIONS.map((opt) => {

          const active = filters.status === opt.key;

          return (

            <Pressable

              key={opt.key}

              onPress={() => onChange({ ...filters, status: opt.key })}

              style={[s.chip, active && s.chipActive]}

              accessibilityRole="button"

              accessibilityState={{ selected: active }}

            >

              <AppText variant="caption" style={[s.chipText, active && s.chipTextActive]}>

                {opt.label}

              </AppText>

            </Pressable>

          );

        })}

      </View>



      <AppText variant="caption" style={s.section}>

        Type de véhicule

      </AppText>

      <View style={s.chips}>

        {VEHICLE_OPTIONS.map((opt) => {

          const active = filters.vehicleType === opt.key;

          return (

            <Pressable

              key={opt.key}

              onPress={() => onChange({ ...filters, vehicleType: opt.key })}

              style={[s.chip, active && s.chipActive]}

              accessibilityRole="button"

              accessibilityState={{ selected: active }}

            >

              <AppText variant="caption" style={[s.chipText, active && s.chipTextActive]}>

                {opt.label}

              </AppText>

            </Pressable>

          );

        })}

      </View>



      <AppText variant="caption" style={s.section}>

        Chauffeur

      </AppText>

      <TextInput

        value={filters.driverSearch}

        onChangeText={(driverSearch) => onChange({ ...filters, driverSearch })}

        placeholder="Rechercher un chauffeur…"

        placeholderTextColor={FLEET_MAP_COLORS.textMuted}

        style={s.search}

        autoCapitalize="words"

        autoCorrect={false}

        clearButtonMode="while-editing"

        accessibilityLabel="Rechercher un chauffeur"

      />



      <AppText variant="caption" style={s.section}>

        Mission

      </AppText>

      <ToggleRow

        label="Avec mission en cours"

        active={filters.withMissionOnly}

        onPress={() =>

          onChange({

            ...filters,

            withMissionOnly: !filters.withMissionOnly,

            withoutMissionOnly: false,

          })

        }

      />

      <ToggleRow

        label="Sans mission"

        active={filters.withoutMissionOnly}

        onPress={() =>

          onChange({

            ...filters,

            withoutMissionOnly: !filters.withoutMissionOnly,

            withMissionOnly: false,

          })

        }

      />



      {driverOptions.length > 0 ? (

        <>

          <AppText variant="caption" style={s.section}>

            Sélection rapide

          </AppText>

          <Pressable

            onPress={() => onChange({ ...filters, driverId: null })}

            style={[s.driverRow, filters.driverId == null && s.driverRowActive]}

          >

            <AppText variant="body" style={s.driverLabel}>

              Tous les chauffeurs

            </AppText>

          </Pressable>

          {driverOptions.slice(0, 8).map((d) => {

            const active = filters.driverId === d.id;

            return (

              <Pressable

                key={d.id}

                onPress={() => onChange({ ...filters, driverId: active ? null : d.id })}

                style={[s.driverRow, active && s.driverRowActive]}

              >

                <AppText variant="body" style={s.driverLabel} numberOfLines={1}>

                  {d.label}

                </AppText>

              </Pressable>

            );

          })}

        </>

      ) : null}



      <Pressable

        onPress={onClose}

        style={({ pressed }) => [s.apply, pressed && { opacity: 0.9 }]}

        accessibilityRole="button"

        accessibilityLabel="Appliquer les filtres"

      >

        <AppText variant="label" style={s.applyText}>

          Appliquer

        </AppText>

      </Pressable>

    </EnterpriseBottomSheet>

  );

}



function ToggleRow({

  label,

  active,

  onPress,

}: {

  label: string;

  active: boolean;

  onPress: () => void;

}) {

  return (

    <Pressable onPress={onPress} style={s.toggleRow} accessibilityRole="switch" accessibilityState={{ checked: active }}>

      <AppText variant="body" style={s.toggleLabel}>

        {label}

      </AppText>

      <View style={[s.toggle, active && s.toggleOn]} />

    </Pressable>

  );

}



const s = StyleSheet.create({

  section: {

    color: FLEET_MAP_COLORS.textMuted,

    fontWeight: "700",

    textTransform: "uppercase",

    letterSpacing: 0.5,

    fontSize: FONT_SIZE.px10,

    marginTop: 8,

  },

  chips: { flexDirection: "row", flexWrap: "wrap", gap: 8 },

  chip: {

    paddingHorizontal: 12,

    paddingVertical: 8,

    borderRadius: 20,

    borderWidth: 1,

    borderColor: FLEET_MAP_COLORS.fabBorder,

    backgroundColor: "#fff",

  },

  chipActive: {

    borderColor: FLEET_MAP_COLORS.brand,

    backgroundColor: "rgba(0, 121, 107, 0.12)",

  },

  chipText: { color: FLEET_MAP_COLORS.text, fontWeight: "600" },

  chipTextActive: { color: FLEET_MAP_COLORS.brand },

  search: {

    borderWidth: 1,

    borderColor: FLEET_MAP_COLORS.fabBorder,

    borderRadius: 12,

    paddingHorizontal: 12,

    paddingVertical: 10,

    fontSize: FONT_SIZE.px14,

    color: FLEET_MAP_COLORS.text,

    backgroundColor: "#fff",

  },

  toggleRow: {

    flexDirection: "row",

    alignItems: "center",

    justifyContent: "space-between",

    paddingVertical: 10,

    borderBottomWidth: StyleSheet.hairlineWidth,

    borderBottomColor: FLEET_MAP_COLORS.fabBorder,

  },

  toggleLabel: { color: FLEET_MAP_COLORS.text },

  toggle: {

    width: 44,

    height: 26,

    borderRadius: 13,

    backgroundColor: "#E2E8F0",

  },

  toggleOn: { backgroundColor: FLEET_MAP_COLORS.brand },

  driverRow: {

    paddingVertical: 10,

    paddingHorizontal: 10,

    borderRadius: 10,

    borderWidth: 1,

    borderColor: "transparent",

  },

  driverRowActive: {

    borderColor: FLEET_MAP_COLORS.brand,

    backgroundColor: "rgba(0, 121, 107, 0.08)",

  },

  driverLabel: { color: FLEET_MAP_COLORS.text, fontWeight: "500" },

  apply: {

    marginTop: 16,

    minHeight: 48,

    borderRadius: 14,

    backgroundColor: FLEET_MAP_COLORS.brand,

    alignItems: "center",

    justifyContent: "center",

  },

  applyText: { color: "#fff", fontWeight: "700" },

});


