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
                        color={
                          mode.value === "manual"
                            ? "#0A7F59"
                            : mode.value === "semi_auto"
                              ? "#F59E0B"
                              : "#10B981"
                        }
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

// ✅ Palette professionnelle claire cohérente avec le dashboard
const modalPalette = {
  backdrop: "rgba(5,22,16,0.82)",
  cardBackground: "#FFFFFF",
  cardBorder: "rgba(15,54,43,0.08)",
  title: "#15362B",
  subtitle: "#5F7369",
  optionBackground: "#FFFFFF",
  optionBorder: "rgba(15,54,43,0.08)",
  optionActiveBackground: "rgba(10,127,89,0.06)",
  optionActiveBorder: "#0A7F59",
  optionText: "#15362B",
  optionDescription: "#5F7369",
  cancelBackground: "#F5F7F6",
  cancelText: "#5F7369",
  manualIcon: "rgba(10,127,89,0.12)",
  semiIcon: "rgba(245,158,11,0.12)",
  autoIcon: "rgba(16,185,129,0.12)",
  iconBorder: "rgba(10,127,89,0.2)",
  iconLockedBorder: "rgba(245,158,11,0.3)",
  chevron: "#91A59D",
  check: "#0A7F59",
  lockBadgeBg: "rgba(245,158,11,0.12)",
  lockBadgeBorder: "rgba(245,158,11,0.25)",
  lockBadgeText: "#F59E0B",
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
    padding: 24,
    borderWidth: 1,
    borderColor: modalPalette.cardBorder,
    shadowColor: "rgba(15,54,43,0.12)",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 1,
    shadowRadius: 24,
    elevation: 8,
  },
  header: {
    marginBottom: 20,
  },
  title: {
    color: modalPalette.title,
    fontSize: 22,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  subtitle: {
    marginTop: 8,
    color: modalPalette.subtitle,
    fontSize: 14,
    lineHeight: 20,
  },
  list: {
    maxHeight: 280,
  },
  option: {
    borderRadius: 18,
    backgroundColor: modalPalette.optionBackground,
    paddingVertical: 16,
    paddingHorizontal: 18,
    borderWidth: 1,
    borderColor: modalPalette.optionBorder,
    shadowColor: "rgba(15,54,43,0.04)",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 1,
    shadowRadius: 4,
    elevation: 1,
  },
  optionActive: {
    borderColor: modalPalette.optionActiveBorder,
    borderWidth: 2,
    backgroundColor: modalPalette.optionActiveBackground,
    shadowColor: modalPalette.optionActiveBorder,
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 3,
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
    letterSpacing: 0.2,
  },
  optionDescription: {
    marginTop: 4,
    color: modalPalette.optionDescription,
    fontSize: 13,
    lineHeight: 20,
    fontWeight: "500",
  },
  modeIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1.5,
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
    marginTop: 20,
    alignSelf: "stretch",
    paddingVertical: 14,
    borderRadius: 18,
    backgroundColor: modalPalette.cancelBackground,
    borderWidth: 1,
    borderColor: modalPalette.optionBorder,
    alignItems: "center",
  },
  cancelLabel: {
    color: modalPalette.cancelText,
    fontSize: 15,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
});

export default ModeSelectionModal;

