import React from "react";
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

type DispatchModeOption = {
  value: "manual" | "semi_auto" | "fully_auto";
  label: string;
  subtitle: string;
  locked?: boolean;
};

type ModeSelectionModalProps = {
  visible: boolean;
  onClose: () => void;
  currentMode: "manual" | "semi_auto" | "fully_auto";
  modes: DispatchModeOption[];
  onSelectMode: (target: "manual" | "semi_auto" | "fully_auto") => void;
};

export const ModeSelectionModal: React.FC<ModeSelectionModalProps> = ({
  visible,
  onClose,
  currentMode,
  modes,
  onSelectMode,
}) => {
  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <View style={styles.backdrop}>
        <View style={styles.card}>
          <View style={styles.header}>
            <Text style={styles.title}>Choisis un mode</Text>
            <Text style={styles.subtitle}>
              Sélectionne le mode adapté à ta situation. Tu peux revenir en
              arrière à tout moment.
            </Text>
          </View>

          <ScrollView
            style={styles.list}
            contentContainerStyle={{ gap: 12, paddingBottom: 12 }}
            showsVerticalScrollIndicator={false}
          >
            {modes.map((mode) => {
              const isActive = mode.value === currentMode;
              return (
                <TouchableOpacity
                  key={mode.value}
                  style={[
                    styles.option,
                    isActive && styles.optionActive,
                    mode.locked && styles.optionLocked,
                  ]}
                  activeOpacity={mode.locked ? 1 : 0.9}
                  onPress={() => {
                    if (!mode.locked) {
                      onSelectMode(mode.value);
                    }
                  }}
                >
                  <View style={styles.optionHeader}>
                    <View
                      style={[
                        styles.modeIcon,
                        modeIconBackground[mode.value],
                        mode.locked && styles.modeIconLocked,
                      ]}
                    >
                      <Ionicons
                        name={modeIconName[mode.value]}
                        size={18}
                        color="#f8faff"
                      />
                    </View>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.optionLabel}>{mode.label}</Text>
                      <Text style={styles.optionDescription}>
                        {mode.subtitle}
                      </Text>
                    </View>
                    {mode.locked ? (
                      <View style={styles.lockBadge}>
                        <Ionicons
                          name="lock-closed"
                          size={16}
                          color={modalPalette.lockBadgeText}
                        />
                        <Text style={styles.lockLabel}>Bientôt</Text>
                      </View>
                    ) : isActive ? (
                      <Ionicons
                        name="checkmark-circle"
                        size={22}
                        color={modalPalette.check}
                      />
                    ) : (
                      <Ionicons
                        name="chevron-forward"
                        size={20}
                        color={modalPalette.chevron}
                      />
                    )}
                  </View>
                </TouchableOpacity>
              );
            })}
          </ScrollView>

          <TouchableOpacity style={styles.cancelButton} onPress={onClose}>
            <Text style={styles.cancelLabel}>Annuler</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
};

const modeIconName: Record<
  "manual" | "semi_auto" | "fully_auto",
  keyof typeof Ionicons.glyphMap
> = {
  manual: "hand-left-outline",
  semi_auto: "flash-outline",
  fully_auto: "rocket-outline",
};

const modalPalette = {
  backdrop: "rgba(5,22,16,0.8)",
  cardBackground: "#08211A",
  cardBorder: "rgba(46,128,94,0.4)",
  title: "#E6F2EA",
  subtitle: "rgba(184,214,198,0.76)",
  optionBackground: "rgba(10,34,26,0.92)",
  optionBorder: "rgba(59,143,105,0.26)",
  optionActiveBackground: "rgba(16,58,44,0.95)",
  optionActiveBorder: "rgba(78,214,160,0.55)",
  optionText: "#F4FFFA",
  optionDescription: "rgba(184,214,198,0.76)",
  cancelBackground: "rgba(255,255,255,0.06)",
  cancelText: "rgba(214,236,224,0.92)",
  manualIcon: "rgba(78,214,160,0.16)",
  semiIcon: "rgba(236,196,94,0.16)",
  autoIcon: "rgba(126,246,168,0.16)",
  iconBorder: "rgba(62,155,116,0.36)",
  iconLockedBorder: "rgba(246,193,88,0.45)",
  chevron: "rgba(180,218,201,0.7)",
  check: "#4ADE80",
  lockBadgeBg: "rgba(246,193,88,0.18)",
  lockBadgeBorder: "rgba(246,193,88,0.45)",
  lockBadgeText: "#F6C158",
};

const modeIconBackground: Record<
  "manual" | "semi_auto" | "fully_auto",
  { backgroundColor: string }
> = {
  manual: { backgroundColor: modalPalette.manualIcon },
  semi_auto: { backgroundColor: modalPalette.semiIcon },
  fully_auto: { backgroundColor: modalPalette.autoIcon },
};

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: modalPalette.backdrop,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 20,
  },
  card: {
    width: "100%",
    maxWidth: 380,
    backgroundColor: modalPalette.cardBackground,
    borderRadius: 24,
    padding: 20,
    borderWidth: 1,
    borderColor: modalPalette.cardBorder,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.35,
    shadowRadius: 24,
    elevation: 12,
  },
  header: {
    marginBottom: 20,
  },
  title: {
    color: modalPalette.title,
    fontSize: 20,
    fontWeight: "700",
  },
  subtitle: {
    marginTop: 6,
    color: modalPalette.subtitle,
    fontSize: 13,
    lineHeight: 18,
  },
  list: {
    maxHeight: 280,
  },
  option: {
    borderRadius: 18,
    backgroundColor: modalPalette.optionBackground,
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: modalPalette.optionBorder,
  },
  optionActive: {
    borderColor: modalPalette.optionActiveBorder,
    backgroundColor: modalPalette.optionActiveBackground,
  },
  optionLocked: {
    opacity: 0.65,
  },
  optionHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 14,
  },
  optionLabel: {
    color: modalPalette.optionText,
    fontSize: 16,
    fontWeight: "700",
  },
  optionDescription: {
    marginTop: 2,
    color: modalPalette.optionDescription,
    fontSize: 13,
    lineHeight: 18,
  },
  modeIcon: {
    width: 42,
    height: 42,
    borderRadius: 21,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: modalPalette.iconBorder,
  },
  modeIconLocked: {
    borderColor: modalPalette.iconLockedBorder,
  },
  lockBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    backgroundColor: modalPalette.lockBadgeBg,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: modalPalette.lockBadgeBorder,
  },
  lockLabel: {
    color: modalPalette.lockBadgeText,
    fontWeight: "600",
    fontSize: 12,
    textTransform: "uppercase",
    letterSpacing: 0.7,
  },
  cancelButton: {
    marginTop: 16,
    alignSelf: "stretch",
    paddingVertical: 12,
    borderRadius: 16,
    backgroundColor: modalPalette.cancelBackground,
    alignItems: "center",
  },
  cancelLabel: {
    color: modalPalette.cancelText,
    fontSize: 15,
    fontWeight: "600",
  },
});

export default ModeSelectionModal;

