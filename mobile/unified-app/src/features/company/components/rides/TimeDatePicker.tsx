import { useMemo, useState } from "react";
import {
  Modal,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
  type ViewStyle,
} from "react-native";
import DateTimePicker, { type DateTimePickerEvent } from "@react-native-community/datetimepicker";
import { Ionicons } from "@expo/vector-icons";
import { useResponsiveTokens } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import { E } from "../../theme/enterpriseOpsTheme";
import { normalizeScheduledTimeIso } from "../../useRideForms";

const SWISS_TZ = "Europe/Zurich";
const ROW_RADIUS = 12;
const ICON_COLOR = E.TEXT_SEC;
const WEEKDAY_LABELS = ["Lu", "Ma", "Me", "Je", "Ve", "Sa", "Di"] as const;
const TIME_PRESETS = [6, 8, 10, 12, 14, 16, 18] as const;

const styles = StyleSheet.create({
  label: {
    fontSize: 13,
    fontWeight: "600" as const,
    color: E.TEXT,
    marginBottom: 4,
  },
  row: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    minHeight: 50,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.38)",
    borderRadius: ROW_RADIUS,
    paddingHorizontal: 14,
    backgroundColor: "#fff",
  },
  rowIdle: {
    backgroundColor: "#FFFFFF",
  },
  rowEmpty: {
    backgroundColor: "#FCFDFC",
    borderColor: "rgba(145, 165, 157, 0.3)",
  },
  rowHovered: {
    borderColor: "rgba(0, 121, 107, 0.34)",
    backgroundColor: "#F9FCFB",
  },
  rowFocused: {
    borderColor: E.BRAND,
    shadowColor: E.BRAND,
    shadowOpacity: 0.16,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 2 },
    elevation: 1,
  },
  rowPressed: {
    backgroundColor: "#F4FAF8",
  },
  leadingIconWrap: {
    width: 24,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    marginRight: 10,
  },
  rowMuted: {
    flex: 1,
    marginRight: 10,
  },
  rowValue: {
    fontSize: 14,
    fontWeight: "600" as const,
    lineHeight: 20,
  },
  rowValueDefined: {
    color: E.TEXT,
  },
  rowValueUndefined: {
    color: E.TEXT_MUTED,
    fontWeight: "500" as const,
  },
  trailingIconWrap: {
    width: 20,
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  chipRow: {
    flexDirection: "row" as const,
    flexWrap: "wrap" as const,
    gap: 8,
    marginTop: 10,
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    minHeight: 44,
    justifyContent: "center" as const,
    backgroundColor: "rgba(0, 121, 107, 0.06)",
    borderColor: "rgba(0, 121, 107, 0.2)",
  },
  chipText: {
    fontSize: 13,
    fontWeight: "600" as const,
    color: E.BRAND,
  },
  webEditorCard: {
    marginTop: 10,
    padding: 12,
    borderRadius: ROW_RADIUS,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.42)",
    backgroundColor: "#FBFDFC",
    gap: 10,
  },
  webInlineGrid: {
    flexDirection: "row" as const,
    gap: 8,
    flexWrap: "wrap" as const,
  },
  webPanel: {
    flex: 1,
    minWidth: 244,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.3)",
    backgroundColor: "#FFFFFF",
    padding: 10,
    gap: 8,
  },
  webPanelHeaderRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "space-between" as const,
  },
  webPanelTitle: {
    fontSize: 13,
    fontWeight: "700" as const,
    color: E.TEXT,
    textTransform: "capitalize" as const,
  },
  webPanelNavButton: {
    width: 30,
    height: 30,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: "rgba(100, 116, 139, 0.24)",
    alignItems: "center" as const,
    justifyContent: "center" as const,
    backgroundColor: "#F9FAFB",
  },
  webWeekdayRow: {
    flexDirection: "row" as const,
    gap: 3,
  },
  webWeekdayCell: {
    flex: 1,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    paddingVertical: 4,
  },
  webWeekdayText: {
    fontSize: 11,
    fontWeight: "700" as const,
    color: E.TEXT_SEC,
  },
  webDaysGrid: {
    flexDirection: "row" as const,
    flexWrap: "wrap" as const,
    gap: 3,
  },
  webDayCell: {
    width: "13.2%",
    minHeight: 32,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "transparent",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  webDayCellOutsideMonth: {
    opacity: 0.42,
  },
  webDayCellToday: {
    borderColor: "rgba(0, 121, 107, 0.36)",
    backgroundColor: "rgba(0, 121, 107, 0.08)",
  },
  webDayCellSelected: {
    borderColor: E.BRAND,
    backgroundColor: "rgba(0, 121, 107, 0.18)",
  },
  webDayText: {
    fontSize: 12,
    fontWeight: "600" as const,
    color: E.TEXT,
  },
  webDayTextSelected: {
    color: E.BRAND_DARK,
    fontWeight: "700" as const,
  },
  webTimeSelectorsRow: {
    flexDirection: "row" as const,
    gap: 8,
  },
  webTimeSelector: {
    flex: 1,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.35)",
    padding: 6,
    gap: 4,
    backgroundColor: "#FCFDFC",
  },
  webTimeSelectorLabel: {
    fontSize: 11,
    color: E.TEXT_SEC,
    fontWeight: "600" as const,
  },
  webStepperRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "space-between" as const,
  },
  webStepperButton: {
    width: 30,
    height: 30,
    borderRadius: 7,
    borderWidth: 1,
    borderColor: "rgba(100, 116, 139, 0.26)",
    backgroundColor: "#FFFFFF",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  webStepperValue: {
    fontSize: 16,
    minWidth: 34,
    textAlign: "center" as const,
    fontWeight: "700" as const,
    color: E.TEXT,
  },
  webPresetRow: {
    flexDirection: "row" as const,
    flexWrap: "wrap" as const,
    gap: 6,
  },
  webPresetChip: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.24)",
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  webPresetChipText: {
    fontSize: 11,
    color: E.BRAND,
    fontWeight: "700" as const,
  },
  webActionsRow: {
    flexDirection: "row" as const,
    gap: 8,
    marginTop: 2,
  },
  webActionButton: {
    flex: 1,
    minHeight: 38,
    borderRadius: 8,
    borderWidth: 1,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    paddingHorizontal: 10,
  },
  webActionButtonSecondary: {
    backgroundColor: "#F8FAFC",
    borderColor: "rgba(100, 116, 139, 0.22)",
  },
  webActionButtonPrimary: {
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    borderColor: "rgba(0, 121, 107, 0.34)",
  },
  webActionButtonPressed: {
    opacity: 0.9,
  },
  webActionText: {
    fontSize: 12,
    fontWeight: "700" as const,
  },
  webActionTextSecondary: {
    color: E.TEXT_SEC,
  },
  webActionTextPrimary: {
    color: E.BRAND,
  },
  webInlineHint: {
    fontSize: 11,
    lineHeight: 15,
    color: E.TEXT_SEC,
  },
});

function parseToDate(iso: string): Date | null {
  const n = normalizeScheduledTimeIso(iso);
  if (!n) return null;
  const d = new Date(n.includes("T") ? n : `${n}T00:00:00`);
  return Number.isNaN(d.getTime()) ? null : d;
}

function toLocalIsoMinute(d: Date): string {
  const pad = (x: number) => String(x).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:00`;
}

function getMonthTitle(d: Date): string {
  return d.toLocaleDateString("fr-CH", {
    timeZone: SWISS_TZ,
    month: "long",
    year: "numeric",
  });
}

function startOfDay(d: Date): Date {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate(), 0, 0, 0, 0);
}

function isSameDay(a: Date, b: Date): boolean {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  );
}

function shiftMinutes(current: number, delta: number): number {
  const total = current + delta;
  const normalized = ((total % 60) + 60) % 60;
  return normalized;
}

function shiftHours(current: number, delta: number): number {
  const total = current + delta;
  const normalized = ((total % 24) + 24) % 24;
  return normalized;
}

function buildCalendarGrid(monthDate: Date) {
  const firstOfMonth = new Date(monthDate.getFullYear(), monthDate.getMonth(), 1);
  const firstWeekdayMondayBased = (firstOfMonth.getDay() + 6) % 7;
  const firstVisibleDay = new Date(firstOfMonth);
  firstVisibleDay.setDate(firstOfMonth.getDate() - firstWeekdayMondayBased);

  return Array.from({ length: 42 }, (_, idx) => {
    const day = new Date(firstVisibleDay);
    day.setDate(firstVisibleDay.getDate() + idx);
    return day;
  });
}

function formatSwissDisplay(iso: string): string {
  const n = normalizeScheduledTimeIso(iso);
  if (!n) return "";
  const d = parseToDate(n);
  if (!d) return "";
  return d.toLocaleString("fr-CH", {
    timeZone: SWISS_TZ,
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

type TimeDatePickerProps = {
  value: string;
  onChange: (value: string) => void;
};

export function TimeDatePicker({ value, onChange }: TimeDatePickerProps) {
  const t = useResponsiveTokens();
  const preview = useMemo(() => formatSwissDisplay(value), [value]);
  const [iosOpen, setIosOpen] = useState(false);
  const [androidStep, setAndroidStep] = useState<null | "date" | "time">(null);
  const [webEditorOpen, setWebEditorOpen] = useState(false);
  const baseDate = parseToDate(value) ?? new Date(Date.now() + 30 * 60 * 1000);
  const [webVisibleMonth, setWebVisibleMonth] = useState(baseDate);
  const today = startOfDay(new Date());
  const calendarDays = useMemo(() => buildCalendarGrid(webVisibleMonth), [webVisibleMonth]);

  const applyOffsetMinutes = (mins: number) => {
    const d = new Date();
    d.setMinutes(d.getMinutes() + mins);
    onChange(toLocalIsoMinute(d));
  };

  const applyTomorrowNine = () => {
    const d = new Date();
    d.setDate(d.getDate() + 1);
    d.setHours(9, 0, 0, 0);
    onChange(toLocalIsoMinute(d));
  };

  const onAndroidChange = (event: DateTimePickerEvent, selected?: Date) => {
    if (event.type === "dismissed") {
      setAndroidStep(null);
      return;
    }
    if (!selected) return;
    if (androidStep === "date") {
      const prev = parseToDate(value) ?? selected;
      const merged = new Date(selected);
      merged.setHours(prev.getHours(), prev.getMinutes(), 0, 0);
      onChange(toLocalIsoMinute(merged));
      setAndroidStep("time");
      return;
    }
    if (androidStep === "time") {
      const prev = parseToDate(value) ?? new Date();
      const merged = new Date(prev.getFullYear(), prev.getMonth(), prev.getDate());
      merged.setHours(selected.getHours(), selected.getMinutes(), 0, 0);
      onChange(toLocalIsoMinute(merged));
      setAndroidStep(null);
    }
  };

  const onIosChange = (event: DateTimePickerEvent, selected?: Date) => {
    if (event.type === "dismissed" || !selected) return;
    onChange(toLocalIsoMinute(selected));
  };

  const openPicker = () => {
    if (Platform.OS === "web") {
      const selected = parseToDate(value) ?? baseDate;
      setWebVisibleMonth(new Date(selected.getFullYear(), selected.getMonth(), 1));
      setWebEditorOpen((prev) => !prev);
      return;
    }
    if (Platform.OS === "ios") setIosOpen(true);
    else setAndroidStep("date");
  };

  const rowStyle: ViewStyle = {
    ...styles.row,
    minHeight: t.fieldShellMinHeight,
  };

  const selectedWebDate = parseToDate(value) ?? baseDate;
  const setWebDateTime = (nextDate: Date) => {
    onChange(toLocalIsoMinute(nextDate));
  };

  const setWebDay = (pickedDay: Date) => {
    const current = parseToDate(value) ?? baseDate;
    const merged = new Date(pickedDay);
    merged.setHours(current.getHours(), current.getMinutes(), 0, 0);
    setWebDateTime(merged);
    setWebVisibleMonth(new Date(pickedDay.getFullYear(), pickedDay.getMonth(), 1));
  };

  const setWebHour = (hour: number) => {
    const current = parseToDate(value) ?? baseDate;
    const merged = new Date(current);
    merged.setHours(hour, current.getMinutes(), 0, 0);
    setWebDateTime(merged);
  };

  const setWebMinute = (minute: number) => {
    const current = parseToDate(value) ?? baseDate;
    const merged = new Date(current);
    merged.setHours(current.getHours(), minute, 0, 0);
    setWebDateTime(merged);
  };

  const applyToday = () => {
    const current = parseToDate(value) ?? baseDate;
    const next = new Date();
    next.setHours(current.getHours(), current.getMinutes(), 0, 0);
    setWebDateTime(next);
    setWebVisibleMonth(new Date(next.getFullYear(), next.getMonth(), 1));
  };

  return (
    <View style={{ marginBottom: 4 }}>
      <AppText variant="label" style={styles.label}>
        Date & heure de départ *
      </AppText>
      <Pressable
        onPress={openPicker}
        style={({ pressed, hovered, focused }) => [
          rowStyle,
          preview ? styles.rowIdle : styles.rowEmpty,
          hovered && styles.rowHovered,
          focused && styles.rowFocused,
          pressed && styles.rowPressed,
        ]}
        accessibilityRole="button"
        accessibilityLabel="Choisir la date et l’heure de départ"
      >
        <View style={styles.leadingIconWrap}>
          <Ionicons name="calendar-outline" size={21} color={ICON_COLOR} />
        </View>
        <View style={styles.rowMuted}>
          <AppText style={[styles.rowValue, preview ? styles.rowValueDefined : styles.rowValueUndefined]}>
            {preview || "Non défini"}
          </AppText>
        </View>
        <View style={styles.trailingIconWrap}>
          <Ionicons name={webEditorOpen && Platform.OS === "web" ? "chevron-up" : "chevron-forward"} size={18} color={E.TEXT_MUTED} />
        </View>
      </Pressable>

      <View style={styles.chipRow}>
        <Pressable
          onPress={() => applyOffsetMinutes(30)}
          style={styles.chip}
          accessibilityRole="button"
          accessibilityLabel="Dans trente minutes"
        >
          <AppText style={styles.chipText}>Dans 30 min</AppText>
        </Pressable>
        <Pressable
          onPress={() => applyOffsetMinutes(60)}
          style={styles.chip}
          accessibilityRole="button"
          accessibilityLabel="Dans une heure"
        >
          <AppText style={styles.chipText}>Dans 1 h</AppText>
        </Pressable>
        <Pressable
          onPress={applyTomorrowNine}
          style={styles.chip}
          accessibilityRole="button"
          accessibilityLabel="Demain à neuf heures"
        >
          <AppText style={styles.chipText}>Demain 9 h</AppText>
        </Pressable>
      </View>

      {Platform.OS === "ios" && iosOpen ? (
        <Modal transparent animationType="slide" visible={iosOpen} onRequestClose={() => setIosOpen(false)}>
          <View style={{ flex: 1, justifyContent: "flex-end" }}>
            <Pressable
              style={{ flex: 1, backgroundColor: "rgba(0,0,0,0.35)" }}
              onPress={() => setIosOpen(false)}
              accessibilityLabel="Fermer le sélecteur"
            />
            <View
              style={{
                backgroundColor: "#fff",
                paddingBottom: 24,
                borderTopLeftRadius: 16,
                borderTopRightRadius: 16,
              }}
            >
              <View style={{ alignItems: "flex-end", padding: 12 }}>
                <Pressable onPress={() => setIosOpen(false)} hitSlop={12}>
                  <AppText variant="body" style={{ color: E.BRAND, fontWeight: "700" }}>
                    OK
                  </AppText>
                </Pressable>
              </View>
              <DateTimePicker value={baseDate} mode="datetime" display="spinner" onChange={onIosChange} />
            </View>
          </View>
        </Modal>
      ) : null}

      {Platform.OS === "android" && androidStep ? (
        <DateTimePicker
          value={baseDate}
          mode={androidStep}
          display="default"
          onChange={onAndroidChange}
        />
      ) : null}

      {Platform.OS === "web" && webEditorOpen ? (
        <View style={styles.webEditorCard}>
          <View style={styles.webInlineGrid}>
            <View style={styles.webPanel}>
              <View style={styles.webPanelHeaderRow}>
                <Pressable
                  onPress={() =>
                    setWebVisibleMonth(
                      (prev) => new Date(prev.getFullYear(), prev.getMonth() - 1, 1),
                    )
                  }
                  style={({ pressed }) => [styles.webPanelNavButton, pressed && styles.webActionButtonPressed]}
                  accessibilityRole="button"
                  accessibilityLabel="Mois précédent"
                >
                  <Ionicons name="chevron-back" size={16} color={E.TEXT_SEC} />
                </Pressable>
                <AppText style={styles.webPanelTitle}>{getMonthTitle(webVisibleMonth)}</AppText>
                <Pressable
                  onPress={() =>
                    setWebVisibleMonth(
                      (prev) => new Date(prev.getFullYear(), prev.getMonth() + 1, 1),
                    )
                  }
                  style={({ pressed }) => [styles.webPanelNavButton, pressed && styles.webActionButtonPressed]}
                  accessibilityRole="button"
                  accessibilityLabel="Mois suivant"
                >
                  <Ionicons name="chevron-forward" size={16} color={E.TEXT_SEC} />
                </Pressable>
              </View>

              <View style={styles.webWeekdayRow}>
                {WEEKDAY_LABELS.map((label) => (
                  <View key={label} style={styles.webWeekdayCell}>
                    <AppText style={styles.webWeekdayText}>{label}</AppText>
                  </View>
                ))}
              </View>

              <View style={styles.webDaysGrid}>
                {calendarDays.map((day) => {
                  const inMonth = day.getMonth() === webVisibleMonth.getMonth();
                  const selected = isSameDay(day, selectedWebDate);
                  const isToday = isSameDay(day, today);
                  return (
                    <Pressable
                      key={day.toISOString()}
                      onPress={() => setWebDay(day)}
                      style={({ pressed }) => [
                        styles.webDayCell,
                        !inMonth && styles.webDayCellOutsideMonth,
                        isToday && styles.webDayCellToday,
                        selected && styles.webDayCellSelected,
                        pressed && styles.webActionButtonPressed,
                      ]}
                      accessibilityRole="button"
                      accessibilityLabel={`Choisir le ${day.toLocaleDateString("fr-CH")}`}
                    >
                      <Text style={[styles.webDayText, selected && styles.webDayTextSelected]}>{day.getDate()}</Text>
                    </Pressable>
                  );
                })}
              </View>

              <View style={styles.webActionsRow}>
                <Pressable
                  onPress={applyToday}
                  style={({ pressed }) => [
                    styles.webActionButton,
                    styles.webActionButtonPrimary,
                    pressed && styles.webActionButtonPressed,
                  ]}
                  accessibilityRole="button"
                  accessibilityLabel="Aller à aujourd’hui"
                >
                  <AppText style={[styles.webActionText, styles.webActionTextPrimary]}>Aujourd&apos;hui</AppText>
                </Pressable>
              </View>
            </View>

            <View style={styles.webPanel}>
              <View style={styles.webTimeSelectorsRow}>
                <View style={styles.webTimeSelector}>
                  <AppText style={styles.webTimeSelectorLabel}>Heure</AppText>
                  <View style={styles.webStepperRow}>
                    <Pressable
                      onPress={() => setWebHour(shiftHours(selectedWebDate.getHours(), -1))}
                      style={({ pressed }) => [styles.webStepperButton, pressed && styles.webActionButtonPressed]}
                      accessibilityRole="button"
                      accessibilityLabel="Diminuer l’heure"
                    >
                      <Ionicons name="remove" size={16} color={E.TEXT_SEC} />
                    </Pressable>
                    <Text style={styles.webStepperValue}>{String(selectedWebDate.getHours()).padStart(2, "0")}</Text>
                    <Pressable
                      onPress={() => setWebHour(shiftHours(selectedWebDate.getHours(), 1))}
                      style={({ pressed }) => [styles.webStepperButton, pressed && styles.webActionButtonPressed]}
                      accessibilityRole="button"
                      accessibilityLabel="Augmenter l’heure"
                    >
                      <Ionicons name="add" size={16} color={E.TEXT_SEC} />
                    </Pressable>
                  </View>
                </View>

                <View style={styles.webTimeSelector}>
                  <AppText style={styles.webTimeSelectorLabel}>Minutes</AppText>
                  <View style={styles.webStepperRow}>
                    <Pressable
                      onPress={() => setWebMinute(shiftMinutes(selectedWebDate.getMinutes(), -5))}
                      style={({ pressed }) => [styles.webStepperButton, pressed && styles.webActionButtonPressed]}
                      accessibilityRole="button"
                      accessibilityLabel="Diminuer les minutes"
                    >
                      <Ionicons name="remove" size={16} color={E.TEXT_SEC} />
                    </Pressable>
                    <Text style={styles.webStepperValue}>{String(selectedWebDate.getMinutes()).padStart(2, "0")}</Text>
                    <Pressable
                      onPress={() => setWebMinute(shiftMinutes(selectedWebDate.getMinutes(), 5))}
                      style={({ pressed }) => [styles.webStepperButton, pressed && styles.webActionButtonPressed]}
                      accessibilityRole="button"
                      accessibilityLabel="Augmenter les minutes"
                    >
                      <Ionicons name="add" size={16} color={E.TEXT_SEC} />
                    </Pressable>
                  </View>
                </View>
              </View>

              <View style={styles.webPresetRow}>
                {TIME_PRESETS.map((hour) => (
                  <Pressable
                    key={`preset-${hour}`}
                    onPress={() => {
                      const next = new Date(selectedWebDate);
                      next.setHours(hour, 0, 0, 0);
                      setWebDateTime(next);
                    }}
                    style={({ pressed }) => [styles.webPresetChip, pressed && styles.webActionButtonPressed]}
                    accessibilityRole="button"
                    accessibilityLabel={`Régler l’heure sur ${hour} heures`}
                  >
                    <AppText style={styles.webPresetChipText}>{`${String(hour).padStart(2, "0")}:00`}</AppText>
                  </Pressable>
                ))}
              </View>

              <View style={styles.webActionsRow}>
                <Pressable
                  onPress={() => setWebDateTime(new Date())}
                  style={({ pressed }) => [
                    styles.webActionButton,
                    styles.webActionButtonPrimary,
                    pressed && styles.webActionButtonPressed,
                  ]}
                  accessibilityRole="button"
                  accessibilityLabel="Définir sur maintenant"
                >
                  <AppText style={[styles.webActionText, styles.webActionTextPrimary]}>Maintenant</AppText>
                </Pressable>
                <Pressable
                  onPress={() => onChange("")}
                  style={({ pressed }) => [
                    styles.webActionButton,
                    styles.webActionButtonSecondary,
                    pressed && styles.webActionButtonPressed,
                  ]}
                  accessibilityRole="button"
                  accessibilityLabel="Aucune date définie"
                >
                  <AppText style={[styles.webActionText, styles.webActionTextSecondary]}>À définir</AppText>
                </Pressable>
              </View>
            </View>
          </View>
          <AppText style={styles.webInlineHint}>
            Sélectionnez la date et l&apos;heure directement, sans saisie ISO manuelle.
          </AppText>
        </View>
      ) : null}
    </View>
  );
}
