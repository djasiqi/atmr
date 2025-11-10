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

const modalPalette = {
  backdrop: "rgba(5,22,16,0.82)",
  cardBackground: "#08211A",
  cardBorder: "rgba(46,128,94,0.4)",
  title: "#E6F2EA",
  subtitle: "rgba(184,214,198,0.76)",
  optionBackground: "rgba(10,34,26,0.9)",
  optionBorder: "rgba(59,143,105,0.24)",
  optionActiveBackground: "rgba(16,58,44,0.95)",
  optionActiveBorder: "rgba(78,214,160,0.55)",
  optionLabel: "#F4FFFA",
  optionDescription: "rgba(184,214,198,0.7)",
  iconCircle: "rgba(60,148,109,0.24)",
  iconBorder: "rgba(62,155,116,0.36)",
  check: "#4ADE80",
  chevron: "rgba(180,218,201,0.7)",
  cancelBackground: "rgba(255,255,255,0.06)",
  cancelText: "rgba(214,236,224,0.92)",
  iconColor: "#F4FFFA",
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
    maxHeight: 320,
  },
  dateOption: {
    borderRadius: 18,
    backgroundColor: modalPalette.optionBackground,
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: modalPalette.optionBorder,
    flexDirection: "row",
    alignItems: "center",
    gap: 16,
  },
  dateOptionActive: {
    borderColor: modalPalette.optionActiveBorder,
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
  },
  optionDescription: {
    marginTop: 2,
    color: modalPalette.optionDescription,
    fontSize: 13,
  },
  iconCircle: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: modalPalette.iconCircle,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: modalPalette.iconBorder,
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

export default DateSelectionModal;

