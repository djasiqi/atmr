import React, { useMemo } from "react";
import {
  Modal,
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
} from "react-native";
import dayjs from "dayjs";
import { Ionicons } from "@expo/vector-icons";

type DateSelectionModalProps = {
  visible: boolean;
  onClose: () => void;
  selectedDate: string;
  onSelectDate: (isoDate: string) => void;
  rangeDays?: number;
};

export const DateSelectionModal: React.FC<DateSelectionModalProps> = ({
  visible,
  onClose,
  selectedDate,
  onSelectDate,
  rangeDays = 6,
}) => {
  const dates = useMemo(() => {
    const base = dayjs();
    return Array.from({ length: rangeDays + 1 }).map((_, idx) => {
      const date = base.add(idx, "day");
      return {
        iso: date.format("YYYY-MM-DD"),
        label: date.format("dddd D MMMM"),
        shortLabel: date.format("ddd D"),
        isToday: idx === 0,
        isTomorrow: idx === 1,
      };
    });
  }, [rangeDays]);

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
            <Text style={styles.title}>Sélectionne une date</Text>
            <Text style={styles.subtitle}>
              Visualise les trajets planifiés pour aujourd'hui, demain ou les
              prochains jours.
            </Text>
          </View>

          <ScrollView
            style={styles.list}
            contentContainerStyle={{ gap: 12, paddingBottom: 12 }}
            showsVerticalScrollIndicator={false}
          >
            {dates.map((item) => {
              const isActive = item.iso === selectedDate;
              return (
                <TouchableOpacity
                  key={item.iso}
                  style={[
                    styles.dateOption,
                    isActive && styles.dateOptionActive,
                  ]}
                  onPress={() => {
                    onSelectDate(item.iso);
                  }}
                  activeOpacity={0.9}
                >
                  <View style={styles.optionLeft}>
                    <View style={styles.iconCircle}>
                      <Ionicons
                        name={
                          item.isToday
                            ? "sunny-outline"
                            : item.isTomorrow
                              ? "partly-sunny-outline"
                              : "calendar-outline"
                        }
                        size={18}
                        color={modalPalette.iconColor}
                      />
                    </View>

                    <View style={{ flex: 1 }}>
                      <Text style={styles.optionLabel}>
                        {item.label.charAt(0).toUpperCase() + item.label.slice(1)}
                      </Text>
                      <Text style={styles.optionDescription}>
                        {item.isToday
                          ? "Aujourd'hui"
                          : item.isTomorrow
                            ? "Demain"
                            : ""}
                      </Text>
                    </View>
                  </View>

                  {isActive ? (
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
                </TouchableOpacity>
              );
            })}
          </ScrollView>

          <TouchableOpacity style={styles.cancelButton} onPress={onClose}>
            <Text style={styles.cancelLabel}>Fermer</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
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
  optionLabel: "#15362B",
  optionDescription: "#5F7369",
  iconCircle: "rgba(10,127,89,0.12)",
  iconBorder: "rgba(10,127,89,0.2)",
  check: "#0A7F59",
  chevron: "#91A59D",
  cancelBackground: "#F5F7F6",
  cancelText: "#5F7369",
  iconColor: "#0A7F59",
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
    maxHeight: 320,
  },
  dateOption: {
    borderRadius: 18,
    backgroundColor: modalPalette.optionBackground,
    paddingVertical: 16,
    paddingHorizontal: 18,
    borderWidth: 1,
    borderColor: modalPalette.optionBorder,
    flexDirection: "row",
    alignItems: "center",
    gap: 16,
  },
  dateOptionActive: {
    borderColor: modalPalette.optionActiveBorder,
    borderWidth: 2,
    backgroundColor: modalPalette.optionActiveBackground,
  },
  optionLeft: {
    flexDirection: "row",
    alignItems: "center",
    gap: 14,
    flex: 1,
  },
  optionLabel: {
    color: modalPalette.optionLabel,
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  optionDescription: {
    marginTop: 4,
    color: modalPalette.optionDescription,
    fontSize: 13,
    fontWeight: "500",
  },
  iconCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: modalPalette.iconCircle,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1.5,
    borderColor: modalPalette.iconBorder,
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

export default DateSelectionModal;

