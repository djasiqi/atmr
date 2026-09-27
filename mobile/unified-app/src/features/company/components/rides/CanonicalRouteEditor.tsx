import { useState } from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { E } from "../../theme/enterpriseOpsTheme";
import {
  addDestination,
  destinationIndexes,
  formatStepClock,
  hasReturn,
  moveDestination,
  removeDestination,
  setRoundTrip,
  setStepDateTime,
  departureClockVisible,
  snapClockToMissionDate,
  type CanonicalRouteDraft,
} from "../../utils/canonicalRouteBuilder";
import { AddressFieldTrigger } from "./AddressPickerSheet";
import { TimeDatePicker } from "./TimeDatePicker";

type CanonicalRouteEditorProps = {
  draft: CanonicalRouteDraft;
  onChange: (draft: CanonicalRouteDraft) => void;
  onPickAddress: (index: number) => void;
  onClearAddress: (index: number) => void;
};

type ClockTarget = {
  index: number;
  field: "arrivalAt" | "departureAt";
  otherDay: boolean;
  signal: number;
};

function stepTitle(draft: CanonicalRouteDraft, index: number): string {
  const step = draft.routeSteps[index];
  if (!step || step.kind === "pickup") return "Départ";
  if (step.kind === "return") return "Retour";
  const ordinal = destinationIndexes(draft.routeSteps).indexOf(index);
  return `Destination ${ordinal + 1}`;
}

export function CanonicalRouteEditor({
  draft,
  onChange,
  onPickAddress,
  onClearAddress,
}: CanonicalRouteEditorProps) {
  const [menuIndex, setMenuIndex] = useState<number | null>(null);
  const [otherDayOpen, setOtherDayOpen] = useState<Record<string, boolean>>({});
  const [clockTarget, setClockTarget] = useState<ClockTarget | null>(null);
  const destinations = destinationIndexes(draft.routeSteps);
  const roundTrip = hasReturn(draft.routeSteps);

  const openClock = (index: number, field: "arrivalAt" | "departureAt") => {
    const key = `${index}:${field}`;
    const current = draft.routeSteps[index]?.[field] ?? null;
    setClockTarget({
      index,
      field,
      otherDay: Boolean(otherDayOpen[key]) || Boolean(current && current.slice(0, 10) !== draft.missionDate),
      signal: (clockTarget?.signal ?? 0) + 1,
    });
  };

  const toggleOtherDay = (index: number, field: "arrivalAt" | "departureAt") => {
    const key = `${index}:${field}`;
    const opening = !otherDayOpen[key];
    setOtherDayOpen((current) => ({ ...current, [key]: opening }));
    if (!opening) {
      const currentValue = draft.routeSteps[index]?.[field] ?? null;
      onChange(setStepDateTime(draft, index, field, snapClockToMissionDate(draft.missionDate, currentValue)));
      return;
    }
    setClockTarget({
      index,
      field,
      otherDay: true,
      signal: (clockTarget?.signal ?? 0) + 1,
    });
  };

  const commitClock = (value: string) => {
    if (!clockTarget) return;
    const next = clockTarget.otherDay ? value : snapClockToMissionDate(draft.missionDate, value);
    onChange(setStepDateTime(draft, clockTarget.index, clockTarget.field, next));
  };

  return (
    <View style={styles.wrap}>
      {draft.routeSteps.map((step, index) => {
        const showDeparture = departureClockVisible(draft.routeSteps, index);
        const destinationOrdinal = destinations.indexOf(index);
        const menuOpen = menuIndex === index;
        return (
          <View key={`${step.kind}-${index}`} style={styles.stepRow}>
            <View style={styles.rail}>
              <View style={styles.dot} />
              {index < draft.routeSteps.length - 1 ? <View style={styles.line} /> : null}
            </View>
            <View style={styles.stepBody}>
              <View style={styles.titleRow}>
                <AppText variant="label" style={styles.stepTitle}>
                  {stepTitle(draft, index)}
                </AppText>
                {step.kind === "destination" ? (
                  <Pressable
                    onPress={() => setMenuIndex(menuOpen ? null : index)}
                    accessibilityRole="button"
                    accessibilityLabel={`Actions ${stepTitle(draft, index)}`}
                    hitSlop={8}
                  >
                    <Ionicons name="ellipsis-vertical" size={16} color={E.TEXT_SEC} />
                  </Pressable>
                ) : null}
              </View>
              {step.kind === "return" ? (
                <AppText variant="body" style={styles.returnLocation}>
                  {step.location.trim() || "Même adresse que le départ"}
                </AppText>
              ) : (
                <AddressFieldTrigger
                  value={step.location}
                  placeholder={step.kind === "pickup" ? "Adresse de départ…" : "Adresse de destination…"}
                  required
                  onPress={() => onPickAddress(index)}
                  onClear={() => onClearAddress(index)}
                  leftSlot={
                    <Ionicons
                      name={step.kind === "pickup" ? "navigate-outline" : "location-outline"}
                      size={16}
                      color={E.TEXT_SEC}
                    />
                  }
                  footer={
                    showDeparture ? (
                      <ClockRow
                        embedded
                        required={step.kind === "pickup"}
                        label="Départ"
                        showLabel={false}
                        value={formatStepClock(
                          draft.missionDate,
                          step.departureAt,
                          Boolean(otherDayOpen[`${index}:departureAt`]),
                        )}
                        otherDay={Boolean(otherDayOpen[`${index}:departureAt`])}
                        onPress={() => openClock(index, "departureAt")}
                        onToggleOtherDay={() => toggleOtherDay(index, "departureAt")}
                      />
                    ) : undefined
                  }
                />
              )}
              {menuOpen && step.kind === "destination" ? (
                <View style={styles.menu}>
                  <MenuAction
                    label="Monter"
                    disabled={destinationOrdinal <= 0}
                    onPress={() => {
                      onChange(moveDestination(draft, destinationOrdinal, -1));
                      setMenuIndex(null);
                    }}
                  />
                  <MenuAction
                    label="Descendre"
                    disabled={destinationOrdinal < 0 || destinationOrdinal >= destinations.length - 1}
                    onPress={() => {
                      onChange(moveDestination(draft, destinationOrdinal, 1));
                      setMenuIndex(null);
                    }}
                  />
                  <MenuAction
                    label="Supprimer"
                    disabled={destinations.length <= 1}
                    onPress={() => {
                      onChange(removeDestination(draft, destinationOrdinal));
                      setMenuIndex(null);
                    }}
                  />
                </View>
              ) : null}
            </View>
          </View>
        );
      })}

      <View style={styles.routeActions}>
        <Pressable
          onPress={() => onChange(addDestination(draft))}
          style={styles.addBtn}
          accessibilityRole="button"
          accessibilityLabel="Ajouter une destination"
        >
          <Ionicons name="add" size={18} color={E.BRAND} />
          <AppText variant="label" style={styles.addLabel}>
            Ajouter une destination
          </AppText>
        </Pressable>

        <Pressable
          onPress={() => onChange(setRoundTrip(draft, !roundTrip))}
          style={[styles.roundBtn, roundTrip ? styles.roundBtnOn : styles.roundBtnOff]}
          accessibilityRole="button"
          accessibilityState={{ selected: roundTrip }}
          accessibilityLabel="Aller-retour"
        >
          <Ionicons name="repeat-outline" size={16} color={roundTrip ? "#0F766E" : "#475569"} />
          <AppText variant="label" style={roundTrip ? styles.roundLabelOn : styles.roundLabelOff}>
            A/R
          </AppText>
        </Pressable>
      </View>

      <TimeDatePicker
        standaloneEditor
        openEditorSignal={clockTarget?.signal ?? 0}
        timeOnly={!clockTarget?.otherDay}
        value={clockValue(draft, clockTarget)}
        onChange={commitClock}
        onEditorConfirm={commitClock}
        modalTitle={clockTarget?.field === "arrivalAt" ? "Arrivée" : "Départ"}
        label=""
        emptyPreviewReferenceIso={`${draft.missionDate}T08:00:00`}
      />
    </View>
  );
}

function clockValue(draft: CanonicalRouteDraft, target: ClockTarget | null): string {
  if (!target) return `${draft.missionDate}T08:00:00`;
  const current = draft.routeSteps[target.index]?.[target.field];
  if (current) return current;
  return `${draft.missionDate}T08:00:00`;
}

function ClockRow({
  label,
  showLabel,
  value,
  otherDay,
  embedded = false,
  required = false,
  onPress,
  onToggleOtherDay,
}: {
  label: string;
  /** Légende utile seulement quand arrivée et départ sont tous les deux affichés. */
  showLabel: boolean;
  value: string;
  otherDay: boolean;
  embedded?: boolean;
  /** Seul le premier départ est exigé. */
  required?: boolean;
  onPress: () => void;
  onToggleOtherDay: () => void;
}) {
  const filled = value.length > 0;
  const shown = filled ? value : "Choisir l'heure";
  return (
    <View style={styles.clockBlock}>
      {showLabel ? (
        <AppText variant="caption" style={styles.clockCaption}>
          {label}
        </AppText>
      ) : null}
      <View style={embedded ? styles.clockFieldEmbedded : styles.clockField}>
        <Pressable
          onPress={onPress}
          style={styles.clockMain}
          accessibilityRole="button"
          accessibilityLabel={filled ? `${label} ${shown}` : `${label}, choisir l'heure`}
        >
          <Ionicons
            name={otherDay ? "calendar-outline" : "time-outline"}
            size={16}
            color={E.TEXT_SEC}
          />
          <AppText
            variant="body"
            numberOfLines={1}
            style={filled ? styles.clockValue : styles.clockEmpty}
          >
            {shown}
          </AppText>
          {filled || !required ? null : (
            <AppText variant="label" accessibilityLabel="Champ obligatoire" style={styles.clockRequired}>
              *
            </AppText>
          )}
        </Pressable>
        <Pressable
          onPress={onToggleOtherDay}
          style={[styles.dayChip, otherDay ? styles.dayChipOn : styles.dayChipOff]}
          accessibilityRole="button"
          accessibilityState={{ selected: otherDay }}
          accessibilityLabel={otherDay ? "Autre jour, activé" : "Même jour que la course"}
        >
          <AppText variant="caption" style={otherDay ? styles.dayChipTextOn : styles.dayChipTextOff}>
            {otherDay ? "Autre jour" : "Même jour"}
          </AppText>
        </Pressable>
      </View>
    </View>
  );
}

function MenuAction({
  label,
  disabled,
  onPress,
}: {
  label: string;
  disabled: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable onPress={onPress} disabled={disabled} accessibilityRole="button" accessibilityState={{ disabled }}>
      <AppText variant="label" style={disabled ? styles.menuDisabled : styles.menuLabel}>
        {label}
      </AppText>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  wrap: { gap: 4 },
  stepRow: { flexDirection: "row", gap: 10 },
  rail: { width: 16, alignItems: "center" },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: E.BRAND,
    marginTop: 4,
  },
  line: { width: 2, flex: 1, backgroundColor: "rgba(15, 23, 42, 0.12)", marginTop: 4 },
  stepBody: { flex: 1, gap: 6, paddingBottom: 12 },
  titleRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  stepTitle: { color: E.TEXT },
  returnLocation: { color: E.TEXT },
  menu: { flexDirection: "row", gap: 16, paddingVertical: 4 },
  menuLabel: { color: E.BRAND },
  menuDisabled: { color: "rgba(15, 23, 42, 0.28)" },
  clockBlock: { gap: 4 },
  clockCaption: { color: E.TEXT_SEC },
  clockField: {
    flexDirection: "row",
    alignItems: "center",
    minHeight: 44,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.38)",
    backgroundColor: "#FFFFFF",
    paddingLeft: 10,
    paddingRight: 6,
  },
  clockFieldEmbedded: {
    flexDirection: "row",
    alignItems: "center",
    height: 35,
    paddingRight: 0,
  },
  clockMain: { flex: 1, flexDirection: "row", alignItems: "center", gap: 8, minWidth: 0 },
  clockValue: { flex: 1, color: E.TEXT },
  clockEmpty: { flex: 1, color: "#475569", fontSize: 13, lineHeight: 16 },
  clockRequired: { color: "#DC2626", fontWeight: "700", width: 16, textAlign: "center" },
  dayChip: {
    height: 24,
    borderRadius: 8,
    paddingHorizontal: 8,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "transparent",
  },
  dayChipOff: { backgroundColor: "#F1F5F9" },
  dayChipOn: { backgroundColor: "#F0FDFA", borderColor: "#0F766E" },
  dayChipTextOff: { color: "#475569", fontSize: 12, fontWeight: "600" },
  dayChipTextOn: { color: "#0F766E", fontSize: 12, fontWeight: "600" },
  routeActions: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  addBtn: { flexDirection: "row", alignItems: "center", gap: 6, paddingVertical: 8, flexShrink: 1 },
  addLabel: { color: E.BRAND },
  roundBtn: {
    flexShrink: 0,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    height: 30,
    minWidth: 64,
    paddingHorizontal: 14,
    borderRadius: 8,
    borderWidth: 1,
  },
  roundBtnOn: { backgroundColor: "#F0FDFA", borderColor: "#0F766E" },
  roundBtnOff: { backgroundColor: "#F1F5F9", borderColor: "transparent" },
  roundLabelOn: { color: "#0F766E", fontSize: 13, fontWeight: "700" },
  roundLabelOff: { color: "#475569", fontSize: 13, fontWeight: "700" },
});
