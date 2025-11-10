import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Platform,
  Alert,
} from "react-native";
import dayjs from "dayjs";
import { Ionicons } from "@expo/vector-icons";

import { useEnterpriseContext } from "@/context/EnterpriseContext";
import { useAuth } from "@/hooks/useAuth";
import { switchDispatchMode } from "@/services/enterpriseDispatch";
import { ModeSelectionModal } from "@/components/modals/ModeSelectionModal";
import { DateSelectionModal } from "@/components/modals/DateSelectionModal";

const headerPalette = {
  background: "#07130E",
  border: "rgba(42,121,86,0.32)",
  dateBackground: "rgba(18,66,48,0.6)",
  dateText: "#E6F7EF",
  dateIcon: "#87D9AE",
  modeBackground: "rgba(19,71,52,0.78)",
  modeText: "#F4FFFA",
  autoButton: "#1EB980",
  autoButtonPaused: "#F5B83D",
  autoText: "#052015",
};

const MODES: Array<{
  value: "manual" | "semi_auto" | "fully_auto";
  label: string;
  subtitle: string;
  locked?: boolean;
}> = [
  {
    value: "manual",
    label: "Manuel",
    subtitle: "Assignations 100% manuelles, aucune automatisation.",
  },
  {
    value: "semi_auto",
    label: "Semi-auto",
    subtitle: "Optimisation assistée, validation opérateur requise.",
  },
  {
    value: "fully_auto",
    label: "Auto",
    subtitle: "Mode autonome complet — bientôt disponible.",
    locked: true,
  },
];

const modeLabel = (value: string | undefined) => {
  return MODES.find((m) => m.value === value)?.label ?? "—";
};

export const EnterpriseHeader: React.FC = () => {
  const {
    selectedDate,
    setSelectedDate,
    mode,
    setMode,
    autoPaused,
    setAutoPaused,
  } = useEnterpriseContext();
  const { enterpriseSession, refreshEnterprise } = useAuth();
  const [modeModalVisible, setModeModalVisible] = useState(false);
  const [dateModalVisible, setDateModalVisible] = useState(false);
  const [dateInitialized, setDateInitialized] = useState(false);

  useEffect(() => {
    if (!enterpriseSession?.company?.dispatchMode) return;
    const current = enterpriseSession.company.dispatchMode as
      | "manual"
      | "semi_auto"
      | "fully_auto";
    setMode(current);
  }, [enterpriseSession?.company?.dispatchMode, setMode]);

  const formattedDate = useMemo(() => {
    return dayjs(selectedDate).format("ddd D MMM");
  }, [selectedDate]);

  useEffect(() => {
    if (!dateInitialized) {
      setSelectedDate(dayjs().format("YYYY-MM-DD"));
      setDateInitialized(true);
    }
  }, [dateInitialized, setSelectedDate]);

  const handleCycleDate = useCallback(() => {
    setDateModalVisible(true);
  }, []);

  const applyMode = useCallback(
    async (target: "manual" | "semi_auto" | "fully_auto") => {
      if (target === mode) return;
      const locked = MODES.find((m) => m.value === target)?.locked;
      if (locked) {
        Alert.alert(
          "Mode indisponible",
          "🚧 Le mode « Totalement Automatique » est encore en développement. Nous te préviendrons dès qu'il sera activable."
        );
        return;
      }
      try {
        await switchDispatchMode(
          target,
          `Changement de mode depuis l'en-tête (${mode} → ${target})`
        );
        setMode(target);
        await refreshEnterprise();
      } catch (error) {
        console.warn("Impossible de changer de mode", error);
        Alert.alert(
          "Changement impossible",
          "La modification du mode dispatch a échoué."
        );
      }
    },
    [mode, refreshEnterprise, setMode]
  );

  const handleModeSheet = useCallback(() => {
    setModeModalVisible(true);
  }, []);

  const handleToggleAuto = useCallback(() => {
    // TODO: appeler endpoint pause/reprise quand disponible
    setAutoPaused(!autoPaused);
  }, [autoPaused, setAutoPaused]);

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.dateButton} onPress={handleCycleDate}>
        <Ionicons
          name="calendar-outline"
          size={18}
          color={headerPalette.dateIcon}
        />
        <Text style={styles.dateText}>{formattedDate}</Text>
        <Ionicons name="chevron-down" size={16} color={headerPalette.dateIcon} />
      </TouchableOpacity>

      <View style={styles.rightGroup}>
        <TouchableOpacity style={styles.modeBadge} onPress={handleModeSheet}>
          <View style={[styles.modeDot, MODE_DOT[mode]]} />
          <Text style={styles.modeText}>{modeLabel(mode)}</Text>
        </TouchableOpacity>

        {mode === "fully_auto" ? (
          <TouchableOpacity
            style={[styles.autoButton, autoPaused && styles.autoButtonPaused]}
            onPress={handleToggleAuto}
          >
            <Ionicons
              name={autoPaused ? "play" : "pause"}
              size={16}
              color={headerPalette.autoText}
              style={{ marginRight: 4 }}
            />
            <Text style={styles.autoButtonText}>
              {autoPaused ? "Relancer" : "Pause"}
            </Text>
          </TouchableOpacity>
        ) : null}
      </View>

      <ModeSelectionModal
        visible={modeModalVisible}
        onClose={() => setModeModalVisible(false)}
        currentMode={mode}
        modes={MODES}
        onSelectMode={async (target) => {
          setModeModalVisible(false);
          await applyMode(target);
        }}
      />

      <DateSelectionModal
        visible={dateModalVisible}
        onClose={() => setDateModalVisible(false)}
        selectedDate={selectedDate}
        onSelectDate={(iso) => {
          setDateModalVisible(false);
          setSelectedDate(iso);
        }}
      />
    </View>
  );
};

const MODE_DOT: Record<string, { backgroundColor: string }> = {
  manual: { backgroundColor: "#49D6B3" },
  semi_auto: { backgroundColor: "#FDD66B" },
  fully_auto: { backgroundColor: "#7FF6A8" },
};

const styles = StyleSheet.create({
  container: {
    paddingTop: Platform.select({ ios: 52, android: 32, default: 24 }),
    paddingBottom: 12,
    paddingHorizontal: 20,
    backgroundColor: headerPalette.background,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: headerPalette.border,
  },
  rightGroup: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  dateButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    backgroundColor: headerPalette.dateBackground,
  },
  dateText: {
    color: headerPalette.dateText,
    fontWeight: "600",
    fontSize: 14,
    textTransform: "capitalize",
  },
  modeBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 16,
    backgroundColor: headerPalette.modeBackground,
  },
  modeDot: {
    width: 10,
    height: 10,
    borderRadius: 999,
  },
  modeText: {
    color: headerPalette.modeText,
    fontWeight: "600",
    fontSize: 14,
  },
  autoButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 16,
    backgroundColor: headerPalette.autoButton,
  },
  autoButtonPaused: {
    backgroundColor: headerPalette.autoButtonPaused,
  },
  autoButtonIcon: {
    marginTop: 1,
  },
  autoButtonText: {
    color: headerPalette.autoText,
    fontWeight: "700",
    fontSize: 13,
  },
});

