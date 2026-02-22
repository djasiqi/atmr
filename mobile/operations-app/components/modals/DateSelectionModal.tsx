import React, { useMemo } from "react";
import {
  Modal,
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Platform,
  Pressable,
} from "react-native";
import dayjs from "dayjs";
import { Ionicons } from "@expo/vector-icons";

const BRAND = "#00796B";
const TEXT = "#1E293B";
const TEXT_SEC = "#64748B";
const TEXT_MUTED = "#94A3B8";
const BORDER = "rgba(0,121,107,0.08)";
const CARD = "#FFFFFF";

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
        dayName: date.format("dddd"),
        dayNumber: date.format("D"),
        monthName: date.format("MMMM"),
        isToday: idx === 0,
        isTomorrow: idx === 1,
      };
    });
  }, [rangeDays]);

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
              <Ionicons name="calendar-outline" size={18} color={BRAND} />
            </View>
            <View style={s.headerText}>
              <Text style={s.title}>Choisir une date</Text>
              <Text style={s.subtitle}>Courses des 7 prochains jours</Text>
            </View>
          </View>

          <ScrollView
            style={s.list}
            contentContainerStyle={s.listContent}
            showsVerticalScrollIndicator={false}
          >
            {dates.map((item) => {
              const isActive = item.iso === selectedDate;
              const tag = item.isToday
                ? "Aujourd\u2019hui"
                : item.isTomorrow
                  ? "Demain"
                  : null;

              return (
                <TouchableOpacity
                  key={item.iso}
                  style={[s.row, isActive && s.rowActive]}
                  onPress={() => onSelectDate(item.iso)}
                  activeOpacity={0.7}
                >
                  <View style={[s.dayCircle, isActive && s.dayCircleActive]}>
                    <Text style={[s.dayNumber, isActive && s.dayNumberActive]}>
                      {item.dayNumber}
                    </Text>
                  </View>

                  <View style={s.rowInfo}>
                    <Text style={[s.rowLabel, isActive && s.rowLabelActive]}>
                      {item.dayName.charAt(0).toUpperCase() + item.dayName.slice(1)}
                    </Text>
                    <Text style={s.rowMonth}>
                      {item.monthName.charAt(0).toUpperCase() + item.monthName.slice(1)}
                      {tag ? ` \u00b7 ${tag}` : ""}
                    </Text>
                  </View>

                  {isActive && (
                    <Ionicons name="checkmark-circle" size={20} color={BRAND} />
                  )}
                </TouchableOpacity>
              );
            })}
          </ScrollView>

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
    maxHeight: 340,
  },
  listContent: {
    gap: 6,
    paddingBottom: 4,
  },

  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 10,
    paddingHorizontal: 12,
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

  dayCircle: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: "#f4f7fc",
    alignItems: "center",
    justifyContent: "center",
  },
  dayCircleActive: {
    backgroundColor: BRAND,
  },
  dayNumber: {
    color: TEXT,
    fontSize: 16,
    fontWeight: "700",
  },
  dayNumberActive: {
    color: "#FFFFFF",
  },

  rowInfo: {
    flex: 1,
  },
  rowLabel: {
    color: TEXT,
    fontSize: 15,
    fontWeight: "600",
  },
  rowLabelActive: {
    color: BRAND,
  },
  rowMonth: {
    color: TEXT_SEC,
    fontSize: 12,
    marginTop: 1,
  },

  closeBtn: {
    marginTop: 14,
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

export default DateSelectionModal;
