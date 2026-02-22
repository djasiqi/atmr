import React from "react";
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Platform,
  Pressable,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

const BRAND = "#00796B";
const TEXT = "#1E293B";
const TEXT_SEC = "#64748B";
const TEXT_MUTED = "#94A3B8";
const BORDER = "rgba(0,121,107,0.08)";
const CARD = "#FFFFFF";

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

const MODE_ICON: Record<string, keyof typeof Ionicons.glyphMap> = {
  manual: "hand-left-outline",
  semi_auto: "flash-outline",
  fully_auto: "rocket-outline",
};

const MODE_COLOR: Record<string, string> = {
  manual: BRAND,
  semi_auto: "#F59E0B",
  fully_auto: "#6366F1",
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
      animationType="slide"
      onRequestClose={onClose}
    >
      <Pressable style={s.overlay} onPress={onClose}>
        <Pressable style={s.sheet} onPress={(e) => e.stopPropagation()}>
          <View style={s.handle} />

          <View style={s.header}>
            <View style={s.headerIconWrap}>
              <Ionicons name="options-outline" size={18} color={BRAND} />
            </View>
            <View style={s.headerText}>
              <Text style={s.title}>Mode de dispatch</Text>
              <Text style={s.subtitle}>Choisissez votre mode d'assignation</Text>
            </View>
          </View>

          <View style={s.list}>
            {modes.map((mode) => {
              const isActive = mode.value === currentMode;
              const isLocked = !!mode.locked;
              const color = MODE_COLOR[mode.value] ?? BRAND;

              return (
                <TouchableOpacity
                  key={mode.value}
                  style={[
                    s.row,
                    isActive && s.rowActive,
                    isLocked && s.rowLocked,
                  ]}
                  activeOpacity={isLocked ? 1 : 0.7}
                  onPress={() => {
                    if (!isLocked) onSelectMode(mode.value);
                  }}
                >
                  <View
                    style={[
                      s.modeIcon,
                      { backgroundColor: `${color}14` },
                      isActive && { backgroundColor: color },
                    ]}
                  >
                    <Ionicons
                      name={MODE_ICON[mode.value] ?? "settings-outline"}
                      size={18}
                      color={isActive ? "#FFFFFF" : color}
                    />
                  </View>

                  <View style={s.rowInfo}>
                    <View style={s.rowLabelRow}>
                      <Text
                        style={[
                          s.rowLabel,
                          isActive && s.rowLabelActive,
                          isLocked && s.rowLabelLocked,
                        ]}
                      >
                        {mode.label}
                      </Text>
                      {isLocked && (
                        <View style={s.lockTag}>
                          <Ionicons name="lock-closed" size={10} color="#F59E0B" />
                          <Text style={s.lockTagText}>Bientôt</Text>
                        </View>
                      )}
                    </View>
                    <Text
                      style={[s.rowSub, isLocked && s.rowSubLocked]}
                      numberOfLines={2}
                    >
                      {mode.subtitle}
                    </Text>
                  </View>

                  {isActive && !isLocked && (
                    <Ionicons name="checkmark-circle" size={20} color={BRAND} />
                  )}
                </TouchableOpacity>
              );
            })}
          </View>

          <TouchableOpacity style={s.closeBtn} onPress={onClose} activeOpacity={0.7}>
            <Text style={s.closeBtnText}>Fermer</Text>
          </TouchableOpacity>
        </Pressable>
      </Pressable>
    </Modal>
  );
};

const sheetShadow =
  Platform.OS === "web"
    ? { boxShadow: "0 -4px 24px rgba(0,0,0,0.12)" }
    : {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: -4 },
        shadowOpacity: 0.1,
        shadowRadius: 16,
        elevation: 12,
      };

const s = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: "rgba(30,41,59,0.45)",
    justifyContent: "flex-end",
  },
  sheet: {
    backgroundColor: CARD,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: Platform.OS === "ios" ? 34 : 24,
    ...sheetShadow,
  },
  handle: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: "rgba(0,0,0,0.1)",
    alignSelf: "center",
    marginBottom: 16,
  },

  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginBottom: 18,
  },
  headerIconWrap: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  headerText: {
    flex: 1,
  },
  title: {
    color: TEXT,
    fontSize: 17,
    fontWeight: "700",
  },
  subtitle: {
    color: TEXT_MUTED,
    fontSize: 13,
    marginTop: 2,
  },

  list: {
    gap: 8,
  },

  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 14,
    backgroundColor: CARD,
    borderWidth: 1,
    borderColor: BORDER,
  },
  rowActive: {
    backgroundColor: "rgba(0,121,107,0.05)",
    borderColor: BRAND,
    borderWidth: 1.5,
  },
  rowLocked: {
    opacity: 0.55,
  },

  modeIcon: {
    width: 40,
    height: 40,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
  },

  rowInfo: {
    flex: 1,
    gap: 2,
  },
  rowLabelRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  rowLabel: {
    color: TEXT,
    fontSize: 15,
    fontWeight: "600",
  },
  rowLabelActive: {
    color: BRAND,
    fontWeight: "700",
  },
  rowLabelLocked: {
    color: TEXT_MUTED,
  },
  rowSub: {
    color: TEXT_SEC,
    fontSize: 12,
    lineHeight: 16,
  },
  rowSubLocked: {
    color: TEXT_MUTED,
  },

  lockTag: {
    flexDirection: "row",
    alignItems: "center",
    gap: 3,
    backgroundColor: "rgba(245,158,11,0.1)",
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: 6,
  },
  lockTagText: {
    color: "#F59E0B",
    fontSize: 10,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },

  closeBtn: {
    marginTop: 16,
    alignSelf: "stretch",
    paddingVertical: 13,
    borderRadius: 14,
    backgroundColor: "#f4f7fc",
    borderWidth: 1,
    borderColor: BORDER,
    alignItems: "center",
  },
  closeBtnText: {
    color: TEXT_SEC,
    fontSize: 14,
    fontWeight: "700",
  },
});

export default ModeSelectionModal;
