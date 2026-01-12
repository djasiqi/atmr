import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  TextInput,
  ScrollView,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

const palette = {
  primary: "#0A7F59",
  primaryLight: "#E8F5F1",
  text: "#15362B",
  textSecondary: "#5F7369",
  border: "rgba(15,54,43,0.12)",
  background: "#FFFFFF",
  warning: "#FFA500",
  error: "#EF4444",
};

interface RecurrenceSelectorProps {
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  recurrenceType: "daily" | "weekly" | "custom";
  onRecurrenceTypeChange: (type: "daily" | "weekly" | "custom") => void;
  recurrenceDays: number[];
  onRecurrenceDaysChange: (days: number[]) => void;
  occurrences: number;
  onOccurrencesChange: (occurrences: number) => void;
  endDate: string;
  onEndDateChange: (date: string) => void;
}

const DAYS_OF_WEEK = [
  { label: "Lun", value: 0 },
  { label: "Mar", value: 1 },
  { label: "Mer", value: 2 },
  { label: "Jeu", value: 3 },
  { label: "Ven", value: 4 },
  { label: "Sam", value: 5 },
  { label: "Dim", value: 6 },
];

export const RecurrenceSelector: React.FC<RecurrenceSelectorProps> = ({
  enabled,
  onEnabledChange,
  recurrenceType,
  onRecurrenceTypeChange,
  recurrenceDays,
  onRecurrenceDaysChange,
  occurrences,
  onOccurrencesChange,
  endDate,
  onEndDateChange,
}) => {
  const toggleDay = (day: number) => {
    if (recurrenceDays.includes(day)) {
      onRecurrenceDaysChange(recurrenceDays.filter((d) => d !== day));
    } else {
      onRecurrenceDaysChange([...recurrenceDays, day].sort());
    }
  };

  const calculateTotalRides = () => {
    if (recurrenceType === "custom" && recurrenceDays.length > 0) {
      return occurrences * recurrenceDays.length;
    }
    return occurrences;
  };

  return (
    <View style={styles.container}>
      {/* Toggle principal */}
      <TouchableOpacity
        style={styles.toggleContainer}
        onPress={() => onEnabledChange(!enabled)}
        activeOpacity={0.7}
      >
        <View style={styles.toggleLeft}>
          <Ionicons
            name="repeat"
            size={20}
            color={enabled ? palette.primary : palette.textSecondary}
          />
          <Text style={[styles.toggleText, enabled && styles.toggleTextActive]}>
            Réservation récurrente
          </Text>
        </View>
        <View
          style={[
            styles.toggle,
            enabled ? styles.toggleActive : styles.toggleInactive,
          ]}
        >
          <View
            style={[
              styles.toggleButton,
              enabled ? styles.toggleButtonActive : styles.toggleButtonInactive,
            ]}
          />
        </View>
      </TouchableOpacity>

      {/* Configuration de la récurrence */}
      {enabled && (
        <View style={styles.config}>
          {/* Type de récurrence */}
          <View style={styles.section}>
            <Text style={styles.label}>Type de récurrence</Text>
            <View style={styles.typeButtons}>
              <TouchableOpacity
                style={[
                  styles.typeButton,
                  recurrenceType === "daily" && styles.typeButtonActive,
                ]}
                onPress={() => onRecurrenceTypeChange("daily")}
                activeOpacity={0.7}
              >
                <Ionicons
                  name="calendar"
                  size={16}
                  color={
                    recurrenceType === "daily" ? "#FFFFFF" : palette.textSecondary
                  }
                />
                <Text
                  style={[
                    styles.typeButtonText,
                    recurrenceType === "daily" && styles.typeButtonTextActive,
                  ]}
                >
                  Quotidien
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[
                  styles.typeButton,
                  recurrenceType === "weekly" && styles.typeButtonActive,
                ]}
                onPress={() => onRecurrenceTypeChange("weekly")}
                activeOpacity={0.7}
              >
                <Ionicons
                  name="calendar-outline"
                  size={16}
                  color={
                    recurrenceType === "weekly" ? "#FFFFFF" : palette.textSecondary
                  }
                />
                <Text
                  style={[
                    styles.typeButtonText,
                    recurrenceType === "weekly" && styles.typeButtonTextActive,
                  ]}
                >
                  Hebdomadaire
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[
                  styles.typeButton,
                  recurrenceType === "custom" && styles.typeButtonActive,
                ]}
                onPress={() => onRecurrenceTypeChange("custom")}
                activeOpacity={0.7}
              >
                <Ionicons
                  name="settings-outline"
                  size={16}
                  color={
                    recurrenceType === "custom" ? "#FFFFFF" : palette.textSecondary
                  }
                />
                <Text
                  style={[
                    styles.typeButtonText,
                    recurrenceType === "custom" && styles.typeButtonTextActive,
                  ]}
                >
                  Personnalisé
                </Text>
              </TouchableOpacity>
            </View>
          </View>

          {/* Sélection des jours (mode custom) */}
          {recurrenceType === "custom" && (
            <View style={styles.section}>
              <Text style={styles.label}>
                Sélectionner les jours ({recurrenceDays.length} jour
                {recurrenceDays.length !== 1 ? "s" : ""})
              </Text>
              <View style={styles.daysContainer}>
                {DAYS_OF_WEEK.map((day) => (
                  <TouchableOpacity
                    key={day.value}
                    style={[
                      styles.dayButton,
                      recurrenceDays.includes(day.value) && styles.dayButtonActive,
                    ]}
                    onPress={() => toggleDay(day.value)}
                    activeOpacity={0.7}
                  >
                    <Text
                      style={[
                        styles.dayButtonText,
                        recurrenceDays.includes(day.value) &&
                          styles.dayButtonTextActive,
                      ]}
                    >
                      {day.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
              {recurrenceType === "custom" && recurrenceDays.length === 0 && (
                <View style={styles.warningBox}>
                  <Ionicons
                    name="warning-outline"
                    size={16}
                    color={palette.warning}
                  />
                  <Text style={styles.warningText}>
                    Veuillez sélectionner au moins un jour
                  </Text>
                </View>
              )}
            </View>
          )}

          {/* Nombre de répétitions */}
          <View style={styles.section}>
            <Text style={styles.label}>Nombre de répétitions</Text>
            <View style={styles.inputRow}>
              <TouchableOpacity
                style={styles.counterButton}
                onPress={() => onOccurrencesChange(Math.max(1, occurrences - 1))}
                disabled={occurrences <= 1}
                activeOpacity={0.7}
              >
                <Ionicons
                  name="remove"
                  size={20}
                  color={occurrences <= 1 ? palette.border : palette.primary}
                />
              </TouchableOpacity>
              <TextInput
                style={styles.counterInput}
                value={String(occurrences)}
                onChangeText={(text) => {
                  const num = parseInt(text) || 1;
                  onOccurrencesChange(Math.max(1, Math.min(52, num)));
                }}
                keyboardType="number-pad"
                maxLength={2}
              />
              <TouchableOpacity
                style={styles.counterButton}
                onPress={() => onOccurrencesChange(Math.min(52, occurrences + 1))}
                disabled={occurrences >= 52}
                activeOpacity={0.7}
              >
                <Ionicons
                  name="add"
                  size={20}
                  color={occurrences >= 52 ? palette.border : palette.primary}
                />
              </TouchableOpacity>
            </View>
            <View style={styles.infoBox}>
              <Ionicons
                name="information-circle-outline"
                size={16}
                color={palette.primary}
              />
              <Text style={styles.infoText}>
                {recurrenceType === "custom" && recurrenceDays.length > 0
                  ? `Créera ${occurrences} × ${recurrenceDays.length} jour${
                      recurrenceDays.length > 1 ? "s" : ""
                    } = ${calculateTotalRides()} réservation${
                      calculateTotalRides() > 1 ? "s" : ""
                    }`
                  : recurrenceType === "weekly"
                  ? `Créera ${occurrences} réservation${
                      occurrences > 1 ? "s" : ""
                    } (une par semaine)`
                  : `Créera ${occurrences} réservation${
                      occurrences > 1 ? "s" : ""
                    } (une par jour)`}
              </Text>
            </View>
          </View>

          {/* Date de fin (optionnel) */}
          <View style={styles.section}>
            <Text style={styles.label}>Jusqu'au (optionnel)</Text>
            <TextInput
              style={styles.dateInput}
              value={endDate}
              onChangeText={onEndDateChange}
              placeholder="AAAA-MM-JJ"
              placeholderTextColor={palette.textSecondary}
            />
            <Text style={styles.hint}>
              Si vide, la récurrence s'arrêtera après {occurrences} répétition
              {occurrences > 1 ? "s" : ""}
            </Text>
          </View>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginTop: 16,
  },
  toggleContainer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: palette.primaryLight,
    borderRadius: 8,
  },
  toggleLeft: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  toggleText: {
    fontSize: 15,
    fontWeight: "500",
    color: palette.textSecondary,
  },
  toggleTextActive: {
    color: palette.primary,
  },
  toggle: {
    width: 48,
    height: 28,
    borderRadius: 14,
    padding: 2,
    justifyContent: "center",
  },
  toggleActive: {
    backgroundColor: palette.primary,
    alignItems: "flex-end",
  },
  toggleInactive: {
    backgroundColor: "#D1D5DB",
    alignItems: "flex-start",
  },
  toggleButton: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: "#FFFFFF",
  },
  toggleButtonActive: {},
  toggleButtonInactive: {},
  config: {
    marginTop: 12,
    padding: 16,
    backgroundColor: palette.background,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: palette.border,
  },
  section: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: "600",
    color: palette.text,
    marginBottom: 8,
  },
  typeButtons: {
    flexDirection: "row",
    gap: 8,
  },
  typeButton: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: palette.border,
    backgroundColor: palette.background,
  },
  typeButtonActive: {
    backgroundColor: palette.primary,
    borderColor: palette.primary,
  },
  typeButtonText: {
    fontSize: 13,
    fontWeight: "500",
    color: palette.textSecondary,
  },
  typeButtonTextActive: {
    color: "#FFFFFF",
  },
  daysContainer: {
    flexDirection: "row",
    gap: 8,
    flexWrap: "wrap",
  },
  dayButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: palette.border,
    backgroundColor: palette.background,
    alignItems: "center",
    justifyContent: "center",
  },
  dayButtonActive: {
    backgroundColor: palette.primary,
    borderColor: palette.primary,
  },
  dayButtonText: {
    fontSize: 13,
    fontWeight: "600",
    color: palette.textSecondary,
  },
  dayButtonTextActive: {
    color: "#FFFFFF",
  },
  warningBox: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginTop: 8,
    padding: 10,
    backgroundColor: "#FFF3CD",
    borderRadius: 6,
    borderWidth: 1,
    borderColor: palette.warning,
  },
  warningText: {
    fontSize: 13,
    color: "#856404",
    flex: 1,
  },
  inputRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  counterButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: palette.border,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: palette.background,
  },
  counterInput: {
    flex: 1,
    height: 50,
    borderWidth: 1,
    borderColor: palette.border,
    borderRadius: 8,
    paddingHorizontal: 16,
    fontSize: 16,
    color: palette.text,
    textAlign: "center",
    fontWeight: "600",
    backgroundColor: palette.background,
  },
  infoBox: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginTop: 8,
    padding: 10,
    backgroundColor: palette.primaryLight,
    borderRadius: 6,
  },
  infoText: {
    fontSize: 13,
    color: palette.primary,
    flex: 1,
  },
  dateInput: {
    height: 50,
    borderWidth: 1,
    borderColor: palette.border,
    borderRadius: 8,
    paddingHorizontal: 16,
    fontSize: 15,
    color: palette.text,
    backgroundColor: palette.background,
  },
  hint: {
    fontSize: 12,
    color: palette.textSecondary,
    marginTop: 6,
  },
});
