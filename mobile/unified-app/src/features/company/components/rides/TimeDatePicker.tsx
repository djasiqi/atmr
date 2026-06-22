import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
  type ViewStyle,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useResponsiveTokens } from "../../../../design/responsive";
import { Modal } from "../../../../design/ui/LegacyModal";
import { AppText } from "../../../../design/ui/AppText";
import { E } from "../../theme/enterpriseOpsTheme";
import {
  buildGenevaScheduleFromLocalCalendarDay,
  clampZurichDayToToday,
  dateFromZurichWallParts,
  formatNaiveIsoInZurich,
  getTodayStartInZurich,
  isSameZurichDay,
  mergeZurichDayAndTime,
  parseScheduledTimeInstant,
  startOfZurichDay,
  zurichWallPartsFromDate,
} from "../../utils/companyDateUtils";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

const SWISS_TZ = "Europe/Zurich";
const ROW_RADIUS = 12;
const ICON_COLOR = E.TEXT_SEC;
const WEEKDAY_LABELS = ["Lu", "Ma", "Me", "Je", "Ve", "Sa", "Di"] as const;
/** Abréviations mois (fr), 3 caractères max — en-tête date compacte. */
const FR_MONTH_ABBR_3 = [
  "jan",
  "fév",
  "mar",
  "avr",
  "mai",
  "jun",
  "jul",
  "aoû",
  "sep",
  "oct",
  "nov",
  "déc",
] as const;
/** Bande horizontale continue : jours consécutifs ; largeur fixe par cellule. */
const DAY_STRIP_LEADING_WEEKS = 14;
const DAY_STRIP_TOTAL_DAYS = 224;
const DAY_CELL_WIDTH = 48;
const DAY_STRIP_PAST_VISIBLE_DAYS = 3;
/** Fenêtre de mois défilants (clic sur le mois dans l’en-tête). */
const MONTH_STRIP_LEADING = 18;
const MONTH_STRIP_TOTAL = 48;
const MONTH_CELL_WIDTH = 76;
const YEAR_STRIP_SPAN = 16;
const YEAR_CELL_WIDTH = 72;
/** Bandes défilantes heure / minute (onglet Heure mobile). */
const TIME_HOUR_CELL_WIDTH = YEAR_CELL_WIDTH;
const TIME_MINUTE_CELL_WIDTH = YEAR_CELL_WIDTH;
const TIME_MINUTE_STEP = 5;
const TIME_MINUTE_SLOTS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55] as const;
/** Même gabarit pour les carrousels jour / mois / année (évite les sauts au changement de mode). */
const PICKER_CAROUSEL_CARD_PADDING_V = 10;
const PICKER_CAROUSEL_CELL_MIN_HEIGHT = 56;
const PICKER_CAROUSEL_CARD_CONTENT_HEIGHT =
  PICKER_CAROUSEL_CARD_PADDING_V * 2 + PICKER_CAROUSEL_CELL_MIN_HEIGHT;
const PICKER_TAB_HEIGHT = 40;
const PICKER_ACTION_HEIGHT = 36;

const styles = StyleSheet.create({
  label: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "600" as const,
    color: E.TEXT,
    marginBottom: 4,
  },
  row: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    minHeight: 46,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.38)",
    borderRadius: ROW_RADIUS,
    paddingHorizontal: 11,
    backgroundColor: "#fff",
  },
  splitRow: {
    flexDirection: "row" as const,
    alignItems: "stretch" as const,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.38)",
    borderRadius: ROW_RADIUS,
    overflow: "hidden" as const,
    backgroundColor: "#FFFFFF",
  },
  splitRowTonal: {
    backgroundColor: "#FAFBFA",
  },
  splitCell: {
    flex: 1,
    minHeight: 46,
    flexDirection: "row" as const,
    alignItems: "center" as const,
    paddingHorizontal: 11,
    backgroundColor: "#FFFFFF",
  },
  splitCellTonal: {
    backgroundColor: "#FAFBFA",
  },
  splitCellPressed: {
    backgroundColor: "#F4FAF8",
  },
  splitCellDate: {
    flex: 1.55,
  },
  splitCellTime: {
    flex: 1,
  },
  splitDivider: {
    width: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(145, 165, 157, 0.32)",
  },
  splitValue: {
    flex: 1,
    marginRight: 4,
    minWidth: 0,
  },
  rowIdle: {
    backgroundColor: "#FFFFFF",
  },
  rowEmpty: {
    backgroundColor: "#FCFDFC",
    borderColor: "rgba(145, 165, 157, 0.3)",
  },
  rowTonal: {
    backgroundColor: "#FAFBFA",
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
    marginRight: 7,
  },
  rowMuted: {
    flex: 1,
    marginRight: 7,
  },
  rowValue: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "600" as const,
    lineHeight: 17,
  },
  rowValueDefined: {
    color: E.TEXT,
  },
  rowValueUndefined: {
    color: E.TEXT_MUTED,
    fontWeight: "500" as const,
  },
  trailingIconWrap: {
    width: 18,
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  requiredSlot: {
    width: 16,
    marginLeft: 4,
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  requiredMark: {
    color: "#DC2626",
    fontWeight: "700" as const,
    fontSize: 16,
    lineHeight: 18,
  },
  timeOnlyDateHint: {
    fontSize: FONT_SIZE.px12,
    color: E.TEXT_SEC,
    marginBottom: 6,
    lineHeight: 16,
  },
  timeOnlyRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    minHeight: 64,
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: ROW_RADIUS,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.22)",
    backgroundColor: "rgba(0, 121, 107, 0.06)",
    gap: 12,
  },
  timeOnlyRowEmpty: {
    borderColor: "rgba(145, 165, 157, 0.3)",
    backgroundColor: "#FCFDFC",
  },
  timeOnlyRowHovered: {
    borderColor: "rgba(0, 121, 107, 0.34)",
    backgroundColor: "rgba(0, 121, 107, 0.1)",
  },
  timeOnlyRowFocused: {
    borderColor: E.BRAND,
    shadowColor: E.BRAND,
    shadowOpacity: 0.16,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 2 },
    elevation: 1,
  },
  timeOnlyRowPressed: {
    backgroundColor: "rgba(0, 121, 107, 0.14)",
  },
  timeOnlyIconBadge: {
    width: 44,
    height: 44,
    borderRadius: 11,
    backgroundColor: "#FFFFFF",
    alignItems: "center" as const,
    justifyContent: "center" as const,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.15)",
  },
  timeOnlyBody: {
    flex: 1,
    minWidth: 0,
    gap: 2,
  },
  timeOnlyCaption: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "600" as const,
    color: E.TEXT_SEC,
    textTransform: "uppercase" as const,
    letterSpacing: 0.4,
  },
  timeOnlyValue: {
    fontSize: 28,
    fontWeight: "700" as const,
    color: E.TEXT,
    lineHeight: 32,
    fontVariant: ["tabular-nums"] as const,
  },
  timeOnlyValueEmpty: {
    fontSize: FONT_SIZE.px16,
    fontWeight: "600" as const,
    color: E.TEXT_MUTED,
    lineHeight: 22,
  },
  timeOnlyAction: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "600" as const,
    color: E.BRAND_DARK,
  },
  webActionsRow: {
    flexDirection: "row" as const,
    gap: 6,
    marginTop: 2,
  },
  webActionButton: {
    flex: 1,
    minHeight: 36,
    borderRadius: 9,
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
    backgroundColor: "rgba(0, 121, 107, 0.06)",
    borderColor: "rgba(0, 121, 107, 0.24)",
  },
  webActionButtonPressed: {
    opacity: 0.9,
  },
  webActionText: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "700" as const,
  },
  webActionTextSecondary: {
    color: E.TEXT_SEC,
  },
  webActionTextPrimary: {
    color: E.BRAND,
  },
  iosSheetHeader: {
    paddingHorizontal: 14,
    paddingTop: 10,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(145, 165, 157, 0.24)",
    gap: 8,
  },
  iosSheetTopRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "space-between" as const,
  },
  iosSheetTitle: {
    fontSize: FONT_SIZE.px14,
    fontWeight: "700" as const,
    color: E.TEXT,
  },
  iosSheetClose: {
    minHeight: 34,
    minWidth: 48,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(100, 116, 139, 0.2)",
    alignItems: "center" as const,
    justifyContent: "center" as const,
    paddingHorizontal: 10,
    backgroundColor: "#F8FAFC",
  },
  iosSheetCloseText: {
    fontSize: FONT_SIZE.px12,
    fontWeight: "700" as const,
    color: E.TEXT_SEC,
  },
  iosPreview: {
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    color: E.TEXT_SEC,
  },
  iosQuickRow: {
    flexDirection: "row" as const,
    flexWrap: "wrap" as const,
    gap: 6,
  },
  iosQuickChip: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.24)",
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  iosQuickChipText: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "700" as const,
    color: E.BRAND,
  },
  mobileModalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.32)",
    justifyContent: "flex-end" as const,
  },
  mobileModalCard: {
    flexGrow: 0,
    flexShrink: 1,
  },
  mobileModalHeader: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "space-between" as const,
    gap: 10,
    marginBottom: 8,
  },
  mobileModalHeaderText: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: 0,
  },
  mobileModalTitle: {
    fontSize: FONT_SIZE.px15,
    fontWeight: "700" as const,
    color: E.TEXT,
    lineHeight: 18,
  },
  mobileModalClose: {
    minHeight: 32,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(100, 116, 139, 0.2)",
    backgroundColor: "#F8FAFC",
    paddingHorizontal: 12,
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  mobileModalPreview: {
    fontSize: FONT_SIZE.px12,
    lineHeight: 14,
    color: E.TEXT_SEC,
    marginTop: 2,
  },
  mobileStepTabs: {
    flexDirection: "row" as const,
    gap: 6,
    marginBottom: 8,
  },
  mobileStepTab: {
    flex: 1,
    minHeight: PICKER_TAB_HEIGHT,
    borderRadius: 11,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.28)",
    backgroundColor: "#F9FBFA",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  mobileStepTabActive: {
    borderColor: "rgba(0, 121, 107, 0.4)",
    backgroundColor: "rgba(0, 121, 107, 0.08)",
  },
  mobileStepTabText: {
    fontSize: FONT_SIZE.px12,
    fontWeight: "700" as const,
    color: E.TEXT_SEC,
  },
  mobileStepTabTextActive: {
    color: E.BRAND_DARK,
  },
  mobileDateStepColumn: {
    gap: 6,
  },
  mobileDateShell: {
    borderRadius: 18,
    backgroundColor: "rgba(148, 163, 184, 0.2)",
    padding: 12,
    gap: 10,
  },
  mobileDateHeader: {
    flexDirection: "row" as const,
    justifyContent: "center" as const,
    alignItems: "center" as const,
    paddingHorizontal: 2,
    gap: 8,
    flexWrap: "nowrap" as const,
  },
  mobileDateSeg: {
    minWidth: 56,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    paddingVertical: 7,
    paddingHorizontal: 10,
    borderRadius: 10,
  },
  mobileDateSegActive: {
    backgroundColor: "rgba(0, 121, 107, 0.14)",
  },
  mobileDateSegText: {
    fontSize: FONT_SIZE.px24,
    fontWeight: "800" as const,
    color: E.TEXT,
    letterSpacing: -0.35,
    fontVariant: ["tabular-nums"],
  },
  mobileMonthStripOuter: {
    marginBottom: 4,
    marginTop: -4,
    minHeight: PICKER_CAROUSEL_CARD_CONTENT_HEIGHT,
    position: "relative" as const,
  },
  mobileCarouselFadeLeft: {
    position: "absolute" as const,
    left: 0,
    top: 0,
    bottom: 0,
    width: 14,
    borderTopLeftRadius: 14,
    borderBottomLeftRadius: 14,
    zIndex: 2,
    pointerEvents: "none" as const,
  },
  mobileCarouselFadeRight: {
    position: "absolute" as const,
    right: 0,
    top: 0,
    bottom: 0,
    width: 14,
    borderTopRightRadius: 14,
    borderBottomRightRadius: 14,
    zIndex: 2,
    pointerEvents: "none" as const,
  },
  mobilePickerCarouselCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(241,245,249,0.95)",
    paddingVertical: PICKER_CAROUSEL_CARD_PADDING_V,
    paddingHorizontal: 6,
    overflow: "hidden" as const,
  },
  mobileMonthStripRow: {
    flexDirection: "row" as const,
    alignItems: "stretch" as const,
  },
  mobileMonthStripCell: {
    width: MONTH_CELL_WIDTH,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    minHeight: PICKER_CAROUSEL_CELL_MIN_HEIGHT,
    borderRadius: 10,
    paddingHorizontal: 4,
    paddingVertical: 6,
    gap: 2,
  },
  mobileMonthStripCellActive: {
    backgroundColor: "rgba(0, 121, 107, 0.12)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.35)",
  },
  mobileMonthStripLabel: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "700" as const,
    color: E.TEXT_MUTED,
    letterSpacing: 0.2,
    textTransform: "capitalize" as const,
  },
  mobileMonthStripLabelActive: {
    color: E.TEXT,
  },
  mobileMonthStripYear: {
    fontSize: FONT_SIZE.px14,
    fontWeight: "700" as const,
    color: E.TEXT_MUTED,
    fontVariant: ["tabular-nums"],
  },
  mobileMonthStripYearActive: {
    color: E.TEXT,
  },
  mobileYearStripRow: {
    flexDirection: "row" as const,
    alignItems: "stretch" as const,
  },
  mobileYearStripCell: {
    width: YEAR_CELL_WIDTH,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    minHeight: PICKER_CAROUSEL_CELL_MIN_HEIGHT,
    borderRadius: 10,
    paddingVertical: 6,
  },
  mobileYearStripCellActive: {
    backgroundColor: "rgba(0, 121, 107, 0.12)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.35)",
  },
  mobileYearStripLabel: {
    fontSize: FONT_SIZE.px14,
    fontWeight: "700" as const,
    color: E.TEXT_MUTED,
    fontVariant: ["tabular-nums"],
  },
  mobileYearStripLabelActive: {
    color: E.TEXT,
  },
  mobileWeekNavigator: {
    width: "100%" as const,
  },
  mobileDayStripRow: {
    flexDirection: "row" as const,
    alignItems: "stretch" as const,
  },
  mobileDayStripCell: {
    width: DAY_CELL_WIDTH,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    minHeight: PICKER_CAROUSEL_CELL_MIN_HEIGHT,
    borderRadius: 10,
    paddingVertical: 6,
    gap: 2,
  },
  mobileDayStripCellSelected: {
    backgroundColor: E.BRAND,
  },
  mobileDayStripWeekday: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "700" as const,
    color: E.TEXT_MUTED,
    letterSpacing: 0.2,
  },
  mobileDayStripWeekdaySelected: {
    color: "rgba(255,255,255,0.92)",
  },
  mobileDayStripNumber: {
    fontSize: FONT_SIZE.px14,
    fontWeight: "700" as const,
    color: E.TEXT,
  },
  mobileDayStripNumberOutside: {
    color: E.TEXT_MUTED,
    fontWeight: "600" as const,
  },
  mobileDayStripNumberSelected: {
    color: "#FFFFFF",
  },
  mobileTimeHeaderColon: {
    fontSize: FONT_SIZE.px24,
    fontWeight: "800" as const,
    color: E.TEXT_SEC,
    letterSpacing: -0.35,
    paddingHorizontal: 2,
    lineHeight: 28,
    textAlignVertical: "center" as const,
  },
  mobileTimeHeaderColonWrap: {
    minHeight: 40,
    justifyContent: "center" as const,
    alignItems: "center" as const,
  },
  mobileTimeStripOuter: {
    marginBottom: 4,
    marginTop: -4,
    minHeight: PICKER_CAROUSEL_CARD_CONTENT_HEIGHT,
    position: "relative" as const,
  },
  mobileTimeHourStripRow: {
    flexDirection: "row" as const,
    alignItems: "stretch" as const,
  },
  mobileTimeMinuteStripRow: {
    flexDirection: "row" as const,
    alignItems: "stretch" as const,
  },
  mobileTimeStripCell: {
    alignItems: "center" as const,
    justifyContent: "center" as const,
    minHeight: PICKER_CAROUSEL_CELL_MIN_HEIGHT,
    borderRadius: 10,
    paddingVertical: 6,
  },
  mobileTimeHourStripCell: {
    width: TIME_HOUR_CELL_WIDTH,
  },
  mobileTimeMinuteStripCell: {
    width: TIME_MINUTE_CELL_WIDTH,
  },
  mobileTimeStripCellActive: {
    backgroundColor: "rgba(0, 121, 107, 0.12)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.35)",
  },
  mobileTimeStripValue: {
    fontSize: FONT_SIZE.px14,
    fontWeight: "700" as const,
    color: E.TEXT_MUTED,
    fontVariant: ["tabular-nums"],
  },
  mobileTimeStripValueActive: {
    color: E.TEXT,
  },
  mobileActionsDock: {
    flexDirection: "row" as const,
    gap: 6,
    marginTop: 6,
  },
  mobileActionDockButton: {
    flex: 1,
    minHeight: PICKER_ACTION_HEIGHT,
    borderRadius: 10,
    borderWidth: 1,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    paddingHorizontal: 8,
  },
  mobileActionDockPrimary: {
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    borderColor: "rgba(0, 121, 107, 0.32)",
  },
  mobileActionDockSecondary: {
    backgroundColor: "#F8FAFC",
    borderColor: "rgba(100, 116, 139, 0.2)",
  },
  mobileActionDockDanger: {
    backgroundColor: "#F8FAFC",
    borderColor: "rgba(148, 163, 184, 0.28)",
  },
  mobileActionDockText: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "700" as const,
  },
  mobileActionDockTextPrimary: {
    color: E.BRAND,
  },
  mobileActionDockTextSecondary: {
    color: E.TEXT_SEC,
  },
  mobileNextButton: {
    backgroundColor: "rgba(0, 121, 107, 0.06)",
    borderColor: "rgba(0, 121, 107, 0.24)",
  },
  mobileNextButtonText: {
    color: E.BRAND,
  },
  mobileClearButton: {
    backgroundColor: "#F8FAFC",
    borderColor: "rgba(148, 163, 184, 0.25)",
  },
  mobileModalCloseText: {
    fontSize: FONT_SIZE.px12,
    fontWeight: "700" as const,
    color: E.TEXT_SEC,
  },
});

function parseToDate(iso: string): Date | null {
  return parseScheduledTimeInstant(iso);
}

function toLocalIsoMinute(d: Date): string {
  return formatNaiveIsoInZurich(d);
}

function getMonthTitle(d: Date): string {
  return d.toLocaleDateString("fr-CH", {
    timeZone: SWISS_TZ,
    month: "long",
    year: "numeric",
  });
}

function getMonthAbbr3Fr(d: Date): string {
  return FR_MONTH_ABBR_3[d.getMonth()];
}

function capitalizeFirst(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return "";
  return trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
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

function startOfWeekMonday(d: Date): Date {
  const day = d.getDay();
  const mondayOffset = (day + 6) % 7;
  return new Date(d.getFullYear(), d.getMonth(), d.getDate() - mondayOffset, 0, 0, 0, 0);
}

function stripAnchorMonday(selected: Date): Date {
  const mon = startOfWeekMonday(selected);
  mon.setDate(mon.getDate() - DAY_STRIP_LEADING_WEEKS * 7);
  return mon;
}

/** Premier jour du mois dans la fenêtre défilante « mois » (centrée ~sur la date affichée). */
function monthStripWindowStartFor(selected: Date): Date {
  return new Date(selected.getFullYear(), selected.getMonth() - MONTH_STRIP_LEADING, 1);
}

/** Bande jours : ne commence pas avant aujourd’hui (pas de passé). */
function computeStripFirstDay(selected: Date, todayStart: Date): Date {
  const mon = stripAnchorMonday(selected);
  mon.setDate(mon.getDate() - DAY_STRIP_LEADING_WEEKS * 7);
  const candidate = startOfDay(mon);
  const t = startOfDay(todayStart);
  const minVisible = new Date(t);
  minVisible.setDate(minVisible.getDate() - DAY_STRIP_PAST_VISIBLE_DAYS);
  return candidate < minVisible ? minVisible : candidate;
}

function clampDayToToday(d: Date, todayStart: Date): Date {
  const sd = startOfDay(d);
  const t = startOfDay(todayStart);
  return sd < t ? new Date(t) : sd;
}

/** Le mois contient au moins un jour ≥ aujourd’hui. */
function monthHasSelectableDay(monthFirst: Date, todayStart: Date): boolean {
  const end = new Date(monthFirst.getFullYear(), monthFirst.getMonth() + 1, 0);
  return end >= startOfDay(todayStart);
}

function capitalizeFrShortMonth(label: string): string {
  if (!label.length) return label;
  const first = label.charAt(0).toUpperCase();
  if (label.length === 1) return first;
  return first + label.slice(1);
}

/** Minute affichée dans la bande : pas de 5 min ; valeur ISO peut être quelconque. */
function nearestFiveMinute(minute: number): number {
  const r = Math.round(minute / TIME_MINUTE_STEP) * TIME_MINUTE_STEP;
  if (r >= 60) return 55;
  return Math.max(0, r);
}

/** Délai minimum par rapport à « maintenant » pour une date/heure planifiable (évite passé / trop tôt). */
const SCHEDULE_MIN_LEAD_MS = 15 * 60 * 1000;

/**
 * Si la date est dans le passé ou le même jour mais trop tôt : jour courant (aujourd’hui) et
 * au moins maintenant + 15 minutes, aligné sur les créneaux de 5 minutes du picker.
 */
function enforceMinSchedule(candidate: Date, todayStartZurich: Date): Date {
  const minInstant = Date.now() + SCHEDULE_MIN_LEAD_MS;
  const candDay = startOfZurichDay(candidate);
  const todayDay = startOfZurichDay(todayStartZurich);

  // Jour futur à Genève : conserver la date/heure choisie (ne pas ramener à aujourd’hui).
  if (candDay.getTime() > todayDay.getTime()) {
    return candidate;
  }

  let result = new Date(candidate);
  let adjusted = false;

  if (candDay.getTime() < todayDay.getTime()) {
    result = new Date(minInstant);
    adjusted = true;
  } else if (isSameZurichDay(candidate, todayStartZurich) && result.getTime() < minInstant) {
    result = new Date(minInstant);
    adjusted = true;
  }

  if (adjusted || (isSameZurichDay(result, todayStartZurich) && result.getTime() < minInstant)) {
    let parts = zurichWallPartsFromDate(result);
    let m = nearestFiveMinute(parts.minute);
    result = dateFromZurichWallParts(parts.year, parts.month, parts.day, parts.hour, m, 0);
    while (result.getTime() < minInstant) {
      parts = zurichWallPartsFromDate(result);
      const nm = parts.minute + TIME_MINUTE_STEP;
      if (nm >= 60) {
        result = dateFromZurichWallParts(parts.year, parts.month, parts.day, parts.hour + 1, nm - 60, 0);
      } else {
        result = dateFromZurichWallParts(parts.year, parts.month, parts.day, parts.hour, nm, 0);
      }
    }
  }

  return result;
}

function formatSwissDisplay(iso: string): string {
  if (!iso.trim()) return "";
  const d = parseToDate(iso);
  if (!d) return "";
  return d.toLocaleString("fr-CH", {
    timeZone: SWISS_TZ,
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatSwissDateOnlyDisplay(iso: string): string {
  if (!iso.trim()) return "";
  const d = parseToDate(iso);
  if (!d) return "";
  return d.toLocaleDateString("fr-CH", {
    timeZone: SWISS_TZ,
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

function formatSwissDateCompactDisplay(iso: string): string {
  if (!iso.trim()) return "";
  const d = parseToDate(iso);
  if (!d) return "";
  const weekday = capitalizeFirst(
    d.toLocaleDateString("fr-CH", {
      timeZone: SWISS_TZ,
      weekday: "short",
    }).replace(/\.$/, ""),
  );
  const day = d.toLocaleDateString("fr-CH", {
    timeZone: SWISS_TZ,
    day: "2-digit",
  });
  const month = capitalizeFirst(
    d.toLocaleDateString("fr-CH", {
      timeZone: SWISS_TZ,
      month: "short",
    }).replace(/\.$/, ""),
  );
  const year = d.toLocaleDateString("fr-CH", {
    timeZone: SWISS_TZ,
    year: "numeric",
  });
  return `${weekday}. ${day} ${month} ${year}`;
}

function formatSwissTimeOnlyDisplay(iso: string): string {
  if (!iso.trim()) return "";
  const d = parseToDate(iso);
  if (!d) return "";
  return d.toLocaleTimeString("fr-CH", {
    timeZone: SWISS_TZ,
    hour: "2-digit",
    minute: "2-digit",
  });
}

type TimeDatePickerProps = {
  value: string;
  onChange: (value: string) => void;
  /** True: sélection date uniquement (pas d'étape heure). */
  dateOnly?: boolean;
  /** True: sélection heure uniquement — date figée via `emptyPreviewReferenceIso`. */
  timeOnly?: boolean;
  /** Rendu champ : une ligne combinée (défaut) ou deux cellules date/heure. */
  display?: "combined" | "split";
  /** Libellé au-dessus du champ. */
  label?: string;
  /** Texte affiché lorsque la valeur est vide. */
  emptyLabel?: string;
  /**
   * Si la valeur est vide, prévisualisation « date · … » à partir de cette ISO (ex. date/heure aller).
   * L’heure affichée vient du suffixe `emptyPreviewSuffix`, pas de cette ISO.
   */
  emptyPreviewReferenceIso?: string;
  /** Suffixe après la date lorsque la valeur est vide et qu’une référence est fournie (défaut : heure à définir). */
  emptyPreviewSuffix?: string;
  /** Titre de la feuille mobile (par défaut : le label sans astérisque final). */
  modalTitle?: string;
  accessibilityLabel?: string;
  /** Surface « tonal » (gris très clair) pour indiquer un champ révélé par un bouton optionnel. */
  tonal?: boolean;
  /** Affiche un astérisque rouge tant que la valeur est vide (champ obligatoire). */
  required?: boolean;
  /** Libellé accessibilité du bouton heure (mode split). */
  timeAccessibilityLabel?: string;
  /** Masque le champ déclencheur ; ouvrir l’éditeur via `openEditorSignal`. */
  standaloneEditor?: boolean;
  /** Incrémenter pour ouvrir l’éditeur (mode standalone). */
  openEditorSignal?: number;
  /** Validation explicite (bouton Valider). */
  onEditorConfirm?: (value: string) => void;
  /** Fermeture sans validation (backdrop, swipe). */
  onEditorDismiss?: () => void;
};

export function TimeDatePicker({
  value,
  onChange,
  dateOnly = false,
  timeOnly = false,
  display = "combined",
  label = "Date & heure de départ *",
  emptyLabel = "Non défini",
  emptyPreviewReferenceIso,
  emptyPreviewSuffix = " · heure à définir",
  modalTitle: modalTitleProp,
  accessibilityLabel = "Choisir la date et l’heure de départ",
  timeAccessibilityLabel = "Choisir l’heure",
  tonal = false,
  required = false,
  standaloneEditor = false,
  openEditorSignal = 0,
  onEditorConfirm,
  onEditorDismiss,
}: TimeDatePickerProps) {
  const modalTitleFallback = dateOnly ? "Date" : timeOnly ? "Heure" : "Date & heure";
  const modalTitle =
    (modalTitleProp ?? label.replace(/\s*\*\s*$/, "").trim()) || modalTitleFallback;
  const t = useResponsiveTokens();
  /** Retour A/R ou planification offre : date calendaire figée, seule l'heure est éditée. */
  const splitInheritsReferenceDate =
    display === "split" && !dateOnly && !timeOnly && Boolean(emptyPreviewReferenceIso?.trim());
  const locksDateToReference =
    splitInheritsReferenceDate
    || (timeOnly && Boolean(emptyPreviewReferenceIso?.trim()));
  const preview = useMemo(() => {
    if (timeOnly) {
      const time = formatSwissTimeOnlyDisplay(value);
      if (time) return time;
      return "";
    }
    const main = dateOnly ? formatSwissDateOnlyDisplay(value) : formatSwissDisplay(value);
    if (main) return main;
    const refIso = emptyPreviewReferenceIso?.trim();
    if (refIso) {
      const dateOnly = formatSwissDateOnlyDisplay(refIso);
      if (dateOnly) return `${dateOnly}${emptyPreviewSuffix}`;
    }
    return "";
  }, [value, emptyPreviewReferenceIso, emptyPreviewSuffix, dateOnly, timeOnly]);
  const splitDatePreview = useMemo(() => {
    if (splitInheritsReferenceDate) {
      const refDisplay = formatSwissDateCompactDisplay(emptyPreviewReferenceIso ?? "");
      if (refDisplay) return refDisplay;
    }
    return formatSwissDateCompactDisplay(value);
  }, [value, emptyPreviewReferenceIso, splitInheritsReferenceDate]);
  const splitTimePreview = useMemo(() => formatSwissTimeOnlyDisplay(value), [value]);
  const timeOnlyDateHint = useMemo(() => {
    if (!timeOnly) return "";
    const refIso = emptyPreviewReferenceIso?.trim();
    if (refIso) return formatSwissDateCompactDisplay(refIso);
    if (value.trim()) return formatSwissDateCompactDisplay(value);
    return "";
  }, [timeOnly, emptyPreviewReferenceIso, value]);
  const timeOnlyHasValue = preview.length > 0;
  const splitDateDefined = splitDatePreview.length > 0;
  const splitTimeDefined = splitTimePreview.length > 0;
  const [mobileEditorOpen, setMobileEditorOpen] = useState(false);
  const [mobileStep, setMobileStep] = useState<"date" | "time">("date");
  const baseDate = useMemo(() => {
    const parsed = parseToDate(value);
    if (parsed) return parsed;
    const refRaw = emptyPreviewReferenceIso?.trim();
    if (refRaw) {
      const refD = parseToDate(refRaw);
      if (refD) {
        return new Date(refD.getFullYear(), refD.getMonth(), refD.getDate(), 12, 0, 0, 0);
      }
    }
    return new Date(Date.now() + 30 * 60 * 1000);
  }, [value, emptyPreviewReferenceIso]);
  const [stripFirstDay, setStripFirstDay] = useState(() =>
    computeStripFirstDay(new Date(), startOfDay(new Date())),
  );
  const [stripViewportFocus, setStripViewportFocus] = useState(() => startOfDay(new Date()));
  const weekStripRef = useRef<ScrollView | null>(null);
  const stripScrollDoneRef = useRef(false);
  const [dateCarousel, setDateCarousel] = useState<"day" | "month" | "year" | null>("day");
  const [monthStripWindowStart, setMonthStripWindowStart] = useState(() =>
    monthStripWindowStartFor(new Date()),
  );
  const [carouselRailWidth, setCarouselRailWidth] = useState(0);
  const monthStripRef = useRef<ScrollView | null>(null);
  const yearStripRef = useRef<ScrollView | null>(null);
  const hourStripRef = useRef<ScrollView | null>(null);
  const minuteStripRef = useRef<ScrollView | null>(null);
  const [timeCarousel, setTimeCarousel] = useState<"hour" | "minute" | null>("hour");
  const [timeCarouselRailWidth, setTimeCarouselRailWidth] = useState(0);
  const stripViewportFocusRef = useRef(stripViewportFocus);
  stripViewportFocusRef.current = stripViewportFocus;
  const today = getTodayStartInZurich();
  const commitSchedule = useCallback(
    (next: Date) => {
      if (locksDateToReference) {
        const refParsed = parseToDate(emptyPreviewReferenceIso ?? "");
        if (refParsed) {
          onChange(toLocalIsoMinute(enforceMinSchedule(mergeZurichDayAndTime(refParsed, next), today)));
          return;
        }
      }
      onChange(toLocalIsoMinute(enforceMinSchedule(next, today)));
    },
    [onChange, today, locksDateToReference, emptyPreviewReferenceIso],
  );
  const stripDays = useMemo(
    () =>
      Array.from({ length: DAY_STRIP_TOTAL_DAYS }, (_, i) => {
        const d = new Date(stripFirstDay);
        d.setDate(stripFirstDay.getDate() + i);
        return startOfDay(d);
      }),
    [stripFirstDay],
  );

  const monthStripMonths = useMemo(() => {
    const raw = Array.from({ length: MONTH_STRIP_TOTAL }, (_, i) => {
      const d = new Date(monthStripWindowStart);
      d.setMonth(monthStripWindowStart.getMonth() + i);
      return new Date(d.getFullYear(), d.getMonth(), 1);
    });
    return raw.filter((m) => monthHasSelectableDay(m, today));
  }, [monthStripWindowStart, today]);

  const yearStripYears = useMemo(() => {
    const y0 = zurichWallPartsFromDate(today).year;
    return Array.from({ length: YEAR_STRIP_SPAN }, (_, i) => y0 + i);
  }, [today]);

  const openPicker = useCallback(() => {
    const parsed = parseToDate(value);
    const raw = parsed ?? baseDate;
    const fixed = enforceMinSchedule(raw, today);
    if (parsed && fixed.getTime() !== raw.getTime()) {
      onChange(toLocalIsoMinute(fixed));
    }
    const selected = clampZurichDayToToday(fixed, today);
    if (timeOnly) {
      setMobileStep("time");
      setTimeCarousel("hour");
    } else {
      setMobileStep("date");
      setStripFirstDay(computeStripFirstDay(selected, today));
      setStripViewportFocus(startOfDay(fixed));
      setMonthStripWindowStart(monthStripWindowStartFor(fixed));
      setDateCarousel("day");
    }
    stripScrollDoneRef.current = false;
    setMobileEditorOpen(true);
  }, [value, baseDate, today, onChange, timeOnly]);

  const openPickerRef = useRef(openPicker);
  openPickerRef.current = openPicker;

  const valueRef = useRef(value);
  valueRef.current = value;

  const closeEditor = useCallback(
    (reason: "confirm" | "dismiss") => {
      setMobileEditorOpen(false);
      if (reason === "confirm") {
        onEditorConfirm?.(valueRef.current);
      } else {
        onEditorDismiss?.();
      }
    },
    [onEditorConfirm, onEditorDismiss],
  );

  useEffect(() => {
    if (!openEditorSignal) return;
    // Ne pas dépendre de openPicker/value : sinon chaque changement d’heure rouvre l’éditeur sur l’onglet Date.
    openPickerRef.current();
  }, [openEditorSignal]);

  useEffect(() => {
    if (!mobileEditorOpen) {
      stripScrollDoneRef.current = false;
    }
  }, [mobileEditorOpen]);

  const goToTimeStep = useCallback(() => {
    if (dateOnly) return;
    setMobileStep("time");
    setTimeCarousel("hour");
  }, [dateOnly]);

  const rowStyle: ViewStyle = {
    ...styles.row,
    minHeight: t.fieldShellMinHeight,
  };

  const selectedWebDate = parseToDate(value) ?? baseDate;
  const selectedWebTimeParts = useMemo(
    () => zurichWallPartsFromDate(selectedWebDate),
    [selectedWebDate],
  );
  const setWebDateTime = (nextDate: Date) => {
    commitSchedule(nextDate);
  };

  const setWebDay = (pickedDay: Date) => {
    const current = parseToDate(value) ?? baseDate;
    const merged = buildGenevaScheduleFromLocalCalendarDay(pickedDay, current);
    setWebDateTime(merged);
    if (mobileEditorOpen) {
      if (dateOnly) closeEditor("confirm");
      else goToTimeStep();
    }
  };

  const setWebDayToday = () => {
    const t = zurichWallPartsFromDate(today);
    setWebDay(new Date(t.year, t.month - 1, t.day));
  };

  useEffect(() => {
    if (!mobileEditorOpen || carouselRailWidth <= 0 || stripScrollDoneRef.current) return;
    const selRaw = startOfZurichDay(selectedWebDate);
    const sel = selRaw.getTime() < today.getTime() ? today : selRaw;
    const idx = stripDays.findIndex((d) => isSameZurichDay(d, sel));
    if (idx < 0) return;
    const targetX = Math.max(
      idx * DAY_CELL_WIDTH + DAY_CELL_WIDTH / 2 - carouselRailWidth / 2,
    );
    requestAnimationFrame(() => {
      weekStripRef.current?.scrollTo({ x: Math.max(0, targetX), animated: false });
    });
    stripScrollDoneRef.current = true;
    setStripViewportFocus(startOfZurichDay(sel));
  }, [
    mobileEditorOpen,
    carouselRailWidth,
    stripFirstDay,
    stripDays,
    selectedWebDate,
    today,
  ]);

  const handleDayStripScroll = useCallback(
    (e: NativeSyntheticEvent<NativeScrollEvent>) => {
      const x = e.nativeEvent.contentOffset.x;
      const w = carouselRailWidth;
      if (!w || DAY_CELL_WIDTH <= 0) return;
      const centerX = x + w / 2;
      let idx = Math.floor(centerX / DAY_CELL_WIDTH);
      idx = Math.max(0, Math.min(stripDays.length - 1, idx));
      const d = stripDays[idx];
      setStripViewportFocus((prev) => (prev && isSameDay(prev, d) ? prev : d));
    },
    [carouselRailWidth, stripDays],
  );

  const handleDayStripMomentumEnd = useCallback(
    (e: NativeSyntheticEvent<NativeScrollEvent>) => {
      handleDayStripScroll(e);
    },
    [handleDayStripScroll],
  );

  const jumpToMonthFirst = useCallback(
    (monthFirst: Date) => {
      if (!monthHasSelectableDay(monthFirst, today)) return;
      let target = startOfZurichDay(monthFirst);
      if (target.getTime() < today.getTime()) target = new Date(today);
      const current = parseToDate(value) ?? baseDate;
      const merged = mergeZurichDayAndTime(target, current);
      commitSchedule(merged);
      setStripFirstDay(computeStripFirstDay(target, today));
      setStripViewportFocus(target);
      setMonthStripWindowStart(monthStripWindowStartFor(target));
      stripScrollDoneRef.current = false;
      setDateCarousel("day");
    },
    [today, value, baseDate, commitSchedule],
  );

  const jumpToYear = useCallback(
    (year: number) => {
      const yStart = new Date(year, 0, 1);
      let target = startOfZurichDay(yStart);
      if (target.getTime() < today.getTime()) target = new Date(today);
      const current = parseToDate(value) ?? baseDate;
      const merged = mergeZurichDayAndTime(target, current);
      commitSchedule(merged);
      setStripFirstDay(computeStripFirstDay(target, today));
      setStripViewportFocus(target);
      setMonthStripWindowStart(monthStripWindowStartFor(target));
      stripScrollDoneRef.current = false;
      setDateCarousel("month");
    },
    [today, value, baseDate, commitSchedule],
  );

  useEffect(() => {
    if (dateCarousel !== "month" || carouselRailWidth <= 0) return;
    const focus = stripViewportFocusRef.current;
    const ix = monthStripMonths.findIndex(
      (d) => d.getMonth() === focus.getMonth() && d.getFullYear() === focus.getFullYear(),
    );
    if (ix < 0) return;
    const targetX = ix * MONTH_CELL_WIDTH + MONTH_CELL_WIDTH / 2 - carouselRailWidth / 2;
    requestAnimationFrame(() => {
      monthStripRef.current?.scrollTo({ x: Math.max(0, targetX), animated: false });
    });
  }, [dateCarousel, carouselRailWidth, monthStripMonths]);

  useEffect(() => {
    if (dateCarousel !== "year" || carouselRailWidth <= 0) return;
    const y = stripViewportFocusRef.current.getFullYear();
    const ix = yearStripYears.indexOf(y);
    if (ix < 0) return;
    const targetX = ix * YEAR_CELL_WIDTH + YEAR_CELL_WIDTH / 2 - carouselRailWidth / 2;
    requestAnimationFrame(() => {
      yearStripRef.current?.scrollTo({ x: Math.max(0, targetX), animated: false });
    });
  }, [dateCarousel, carouselRailWidth, yearStripYears]);

  useEffect(() => {
    if (mobileStep !== "time" || timeCarouselRailWidth <= 0) return;
    const sel = parseToDate(value);
    const d = sel ?? new Date(Date.now() + 30 * 60 * 1000);
    const parts = zurichWallPartsFromDate(d);
    const h = parts.hour;
    const slotMin = nearestFiveMinute(parts.minute);
    const minuteIdx = slotMin / TIME_MINUTE_STEP;
    const targetHX = h * TIME_HOUR_CELL_WIDTH + TIME_HOUR_CELL_WIDTH / 2 - timeCarouselRailWidth / 2;
    const targetMX =
      minuteIdx * TIME_MINUTE_CELL_WIDTH +
      TIME_MINUTE_CELL_WIDTH / 2 -
      timeCarouselRailWidth / 2;
    requestAnimationFrame(() => {
      hourStripRef.current?.scrollTo({ x: Math.max(0, targetHX), animated: false });
      minuteStripRef.current?.scrollTo({ x: Math.max(0, targetMX), animated: false });
    });
  }, [mobileStep, timeCarouselRailWidth, value, timeCarousel]);

  const setWebTime = (hour: number, minute: number) => {
    const current = parseToDate(value) ?? baseDate;
    const day = zurichWallPartsFromDate(current);
    const merged = dateFromZurichWallParts(day.year, day.month, day.day, hour, minute, 0);
    setWebDateTime(merged);
  };

  return (
    <View style={{ marginBottom: 4 }}>
      {label.trim().length > 0 ? (
        <AppText variant="label" style={styles.label}>
          {label}
        </AppText>
      ) : null}
      {timeOnly && timeOnlyDateHint ? (
        <AppText style={styles.timeOnlyDateHint}>{timeOnlyDateHint}</AppText>
      ) : null}
      {!standaloneEditor && display === "split" && !dateOnly ? (
        <View style={[styles.splitRow, tonal && styles.splitRowTonal]}>
          {splitInheritsReferenceDate ? (
            <View
              style={[
                styles.splitCell,
                styles.splitCellDate,
                tonal && styles.splitCellTonal,
              ]}
              accessibilityElementsHidden={false}
              importantForAccessibility="yes"
              accessible
              accessibilityLabel={`Date de retour : ${splitDatePreview || emptyLabel}`}
            >
              <View style={styles.leadingIconWrap}>
                <Ionicons name="calendar-outline" size={19} color={ICON_COLOR} />
              </View>
              <View style={styles.splitValue}>
                <AppText
                  variant="label"
                  style={[
                    styles.rowValue,
                    splitDateDefined ? styles.rowValueDefined : styles.rowValueUndefined,
                  ]}
                  numberOfLines={1}
                >
                  {splitDatePreview || emptyLabel}
                </AppText>
              </View>
            </View>
          ) : (
            <Pressable
              onPress={openPicker}
              style={({ pressed }) => [
                styles.splitCell,
                styles.splitCellDate,
                tonal && styles.splitCellTonal,
                pressed && styles.splitCellPressed,
              ]}
              accessibilityRole="button"
              accessibilityLabel={accessibilityLabel}
            >
              <View style={styles.leadingIconWrap}>
                <Ionicons name="calendar-outline" size={19} color={ICON_COLOR} />
              </View>
              <View style={styles.splitValue}>
                <AppText
                  variant="label"
                  style={[
                    styles.rowValue,
                    splitDateDefined ? styles.rowValueDefined : styles.rowValueUndefined,
                  ]}
                  numberOfLines={1}
                >
                  {splitDatePreview || emptyLabel}
                </AppText>
              </View>
            </Pressable>
          )}
          <View style={styles.splitDivider} />
          <Pressable
            onPress={() => {
              openPicker();
              requestAnimationFrame(() => setMobileStep("time"));
            }}
            style={({ pressed }) => [
              styles.splitCell,
              styles.splitCellTime,
              tonal && styles.splitCellTonal,
              pressed && styles.splitCellPressed,
            ]}
            accessibilityRole="button"
            accessibilityLabel={timeAccessibilityLabel}
          >
            <View style={styles.leadingIconWrap}>
              <Ionicons name="time-outline" size={19} color={ICON_COLOR} />
            </View>
            <View style={styles.splitValue}>
              <AppText
                variant="label"
                style={[
                  styles.rowValue,
                  splitTimeDefined ? styles.rowValueDefined : styles.rowValueUndefined,
                ]}
                numberOfLines={1}
              >
                {splitTimeDefined
                  ? splitTimePreview
                  : splitInheritsReferenceDate
                    ? emptyLabel
                    : "--:--"}
              </AppText>
            </View>
            <View style={styles.trailingIconWrap}>
              <Ionicons name="chevron-forward" size={16} color={E.TEXT_MUTED} />
            </View>
            <View style={styles.requiredSlot}>
              {required && !preview ? (
                <AppText
                  variant="label"
                  accessibilityLabel="Champ obligatoire"
                  style={styles.requiredMark}
                >
                  *
                </AppText>
              ) : null}
            </View>
          </Pressable>
        </View>
      ) : !standaloneEditor && timeOnly ? (
        <Pressable
          onPress={openPicker}
          style={({ pressed, hovered, focused }) => [
            styles.timeOnlyRow,
            timeOnlyHasValue ? null : styles.timeOnlyRowEmpty,
            hovered && styles.timeOnlyRowHovered,
            focused && styles.timeOnlyRowFocused,
            pressed && styles.timeOnlyRowPressed,
          ]}
          accessibilityRole="button"
          accessibilityLabel={accessibilityLabel}
        >
          <View style={styles.timeOnlyIconBadge}>
            <Ionicons name="time-outline" size={22} color={E.BRAND} />
          </View>
          <View style={styles.timeOnlyBody}>
            <AppText style={styles.timeOnlyCaption}>Prise en charge</AppText>
            <AppText
              style={timeOnlyHasValue ? styles.timeOnlyValue : styles.timeOnlyValueEmpty}
              numberOfLines={1}
            >
              {preview || emptyLabel}
            </AppText>
          </View>
          <AppText style={styles.timeOnlyAction}>
            {timeOnlyHasValue ? "Modifier" : "Choisir"}
          </AppText>
          <View style={styles.requiredSlot}>
            {required && !timeOnlyHasValue ? (
              <AppText
                variant="label"
                accessibilityLabel="Champ obligatoire"
                style={styles.requiredMark}
              >
                *
              </AppText>
            ) : null}
          </View>
        </Pressable>
      ) : !standaloneEditor ? (
        <Pressable
          onPress={openPicker}
          style={({ pressed, hovered, focused }) => [
            rowStyle,
            preview ? styles.rowIdle : styles.rowEmpty,
            tonal && styles.rowTonal,
            hovered && styles.rowHovered,
            focused && styles.rowFocused,
            pressed && styles.rowPressed,
          ]}
          accessibilityRole="button"
          accessibilityLabel={accessibilityLabel}
        >
          <View style={styles.leadingIconWrap}>
            <Ionicons name="calendar-outline" size={20} color={ICON_COLOR} />
          </View>
          <View style={styles.rowMuted}>
            <AppText style={[styles.rowValue, preview ? styles.rowValueDefined : styles.rowValueUndefined]}>
              {preview || emptyLabel}
            </AppText>
          </View>
          <View style={styles.trailingIconWrap}>
            <Ionicons name="chevron-forward" size={17} color={E.TEXT_MUTED} />
          </View>
          <View style={styles.requiredSlot}>
            {required && !preview ? (
              <AppText
                variant="label"
                accessibilityLabel="Champ obligatoire"
                style={styles.requiredMark}
              >
                *
              </AppText>
            ) : null}
          </View>
        </Pressable>
      ) : null}

      <Modal
        visible={mobileEditorOpen}
        title=""
        onClose={() => closeEditor("dismiss")}
        presentation="bottomSheet"
        sheetBodyMaxHeightRatio={0.88}
        renderHeader={() => (
          <View style={styles.mobileModalHeader}>
            <View style={styles.mobileModalHeaderText}>
              <AppText style={styles.mobileModalTitle} numberOfLines={1}>
                {modalTitle}
              </AppText>
              {preview ? (
                <AppText style={styles.mobileModalPreview} numberOfLines={1}>
                  {preview}
                </AppText>
              ) : null}
            </View>
            <Pressable
              style={styles.mobileModalClose}
              onPress={() => closeEditor("confirm")}
              accessibilityRole="button"
              accessibilityLabel="Valider et fermer le sélecteur"
            >
              <AppText style={styles.mobileModalCloseText}>Valider</AppText>
            </Pressable>
          </View>
        )}
        footer={null}
      >
        <View style={styles.mobileModalCard}>
              {!dateOnly && !timeOnly ? (
              <View style={styles.mobileStepTabs}>
                <Pressable
                  onPress={() => setMobileStep("date")}
                  style={({ pressed }) => [
                    styles.mobileStepTab,
                    mobileStep === "date" ? styles.mobileStepTabActive : null,
                    pressed ? styles.webActionButtonPressed : null,
                  ]}
                  accessibilityRole="button"
                  accessibilityLabel="Onglet date"
                >
                  <AppText
                    style={[
                      styles.mobileStepTabText,
                      mobileStep === "date" ? styles.mobileStepTabTextActive : null,
                    ]}
                  >
                    Date
                  </AppText>
                </Pressable>
                <Pressable
                  onPress={goToTimeStep}
                  style={({ pressed }) => [
                    styles.mobileStepTab,
                    mobileStep === "time" ? styles.mobileStepTabActive : null,
                    pressed ? styles.webActionButtonPressed : null,
                  ]}
                  accessibilityRole="button"
                  accessibilityLabel="Onglet heure"
                >
                  <AppText
                    style={[
                      styles.mobileStepTabText,
                      mobileStep === "time" ? styles.mobileStepTabTextActive : null,
                    ]}
                  >
                    Heure
                  </AppText>
                </Pressable>
              </View>
              ) : null}
              <ScrollView showsVerticalScrollIndicator={false} nestedScrollEnabled>
                <View>
                  {!timeOnly && (mobileStep === "date" || dateOnly) ? (
                    <>
                    <View style={styles.mobileDateStepColumn}>
                        <View style={styles.mobileDateShell}>
                          <View style={styles.mobileDateHeader}>
                            <Pressable
                              style={[
                                styles.mobileDateSeg,
                                dateCarousel === "day" ? styles.mobileDateSegActive : null,
                              ]}
                              onPress={() =>
                                setDateCarousel((c) => (c === "day" ? null : "day"))
                              }
                              accessibilityRole="button"
                              accessibilityLabel="Choisir le jour : liste défilante"
                              hitSlop={{ top: 8, bottom: 8, left: 6, right: 6 }}
                            >
                              <AppText style={styles.mobileDateSegText}>
                                {String(stripViewportFocus.getDate()).padStart(2, "0")}
                              </AppText>
                            </Pressable>
                            <Pressable
                              style={[
                                styles.mobileDateSeg,
                                dateCarousel === "month" ? styles.mobileDateSegActive : null,
                              ]}
                              onPress={() =>
                                setDateCarousel((c) => (c === "month" ? null : "month"))
                              }
                              accessibilityRole="button"
                              accessibilityLabel="Choisir le mois : liste défilante"
                              hitSlop={{ top: 8, bottom: 8, left: 6, right: 6 }}
                            >
                              <AppText style={styles.mobileDateSegText}>
                                {getMonthAbbr3Fr(stripViewportFocus)}
                              </AppText>
                            </Pressable>
                            <Pressable
                              style={[
                                styles.mobileDateSeg,
                                dateCarousel === "year" ? styles.mobileDateSegActive : null,
                              ]}
                              onPress={() =>
                                setDateCarousel((c) => (c === "year" ? null : "year"))
                              }
                              accessibilityRole="button"
                              accessibilityLabel="Choisir l’année : liste défilante"
                              hitSlop={{ top: 8, bottom: 8, left: 6, right: 6 }}
                            >
                              <AppText style={styles.mobileDateSegText}>
                                {String(stripViewportFocus.getFullYear())}
                              </AppText>
                            </Pressable>
                          </View>
                          <View
                            style={styles.mobileMonthStripOuter}
                            onLayout={(e) => setCarouselRailWidth(e.nativeEvent.layout.width)}
                          >
                            {carouselRailWidth > 0 && dateCarousel === "month" ? (
                              <ScrollView
                                ref={monthStripRef}
                                horizontal
                                snapToInterval={MONTH_CELL_WIDTH}
                                snapToAlignment="start"
                                decelerationRate="normal"
                                nestedScrollEnabled
                                keyboardShouldPersistTaps="handled"
                                showsHorizontalScrollIndicator={false}
                                scrollEventThrottle={16}
                                style={{ width: carouselRailWidth }}
                              >
                                <View style={styles.mobilePickerCarouselCard}>
                                  <View style={styles.mobileMonthStripRow}>
                                    {monthStripMonths.map((monthStart) => {
                                      const active =
                                        monthStart.getMonth() === stripViewportFocus.getMonth() &&
                                        monthStart.getFullYear() === stripViewportFocus.getFullYear();
                                      const shortMonth = capitalizeFrShortMonth(
                                        monthStart.toLocaleDateString("fr-CH", { month: "short" }),
                                      );
                                      return (
                                        <Pressable
                                          key={`month-strip-${monthStart.getFullYear()}-${monthStart.getMonth()}`}
                                          onPress={() => jumpToMonthFirst(monthStart)}
                                          style={({ pressed }) => [
                                            styles.mobileMonthStripCell,
                                            active ? styles.mobileMonthStripCellActive : null,
                                            pressed ? styles.webActionButtonPressed : null,
                                          ]}
                                          accessibilityRole="button"
                                          accessibilityLabel={`Aller à ${getMonthTitle(monthStart)}`}
                                        >
                                          <AppText
                                            style={[
                                              styles.mobileMonthStripLabel,
                                              active ? styles.mobileMonthStripLabelActive : null,
                                            ]}
                                          >
                                            {shortMonth}
                                          </AppText>
                                          <AppText
                                            style={[
                                              styles.mobileMonthStripYear,
                                              active ? styles.mobileMonthStripYearActive : null,
                                            ]}
                                          >
                                            {monthStart.getFullYear()}
                                          </AppText>
                                        </Pressable>
                                      );
                                    })}
                                  </View>
                                </View>
                              </ScrollView>
                            ) : null}
                            {carouselRailWidth > 0 && dateCarousel === "year" ? (
                              <ScrollView
                                ref={yearStripRef}
                                horizontal
                                snapToInterval={YEAR_CELL_WIDTH}
                                snapToAlignment="start"
                                decelerationRate="normal"
                                nestedScrollEnabled
                                keyboardShouldPersistTaps="handled"
                                showsHorizontalScrollIndicator={false}
                                scrollEventThrottle={16}
                                style={{ width: carouselRailWidth }}
                              >
                                <View style={styles.mobilePickerCarouselCard}>
                                  <View style={styles.mobileYearStripRow}>
                                    {yearStripYears.map((y) => {
                                      const active =
                                        stripViewportFocus.getFullYear() === y;
                                      return (
                                        <Pressable
                                          key={`year-strip-${y}`}
                                          onPress={() => jumpToYear(y)}
                                          style={({ pressed }) => [
                                            styles.mobileYearStripCell,
                                            active ? styles.mobileYearStripCellActive : null,
                                            pressed ? styles.webActionButtonPressed : null,
                                          ]}
                                          accessibilityRole="button"
                                          accessibilityLabel={`Année ${y}`}
                                        >
                                          <AppText
                                            style={[
                                              styles.mobileYearStripLabel,
                                              active ? styles.mobileYearStripLabelActive : null,
                                            ]}
                                          >
                                            {y}
                                          </AppText>
                                        </Pressable>
                                      );
                                    })}
                                  </View>
                                </View>
                              </ScrollView>
                            ) : null}
                            {carouselRailWidth > 0 && dateCarousel === "day" ? (
                              <ScrollView
                                ref={weekStripRef}
                                horizontal
                                snapToInterval={DAY_CELL_WIDTH}
                                snapToAlignment="start"
                                decelerationRate="normal"
                                nestedScrollEnabled
                                keyboardShouldPersistTaps="handled"
                                showsHorizontalScrollIndicator={false}
                                scrollEventThrottle={16}
                                style={{ width: carouselRailWidth }}
                                onScroll={handleDayStripScroll}
                                onMomentumScrollEnd={handleDayStripMomentumEnd}
                              >
                                <View style={styles.mobilePickerCarouselCard}>
                                  <View style={styles.mobileDayStripRow}>
                                    {stripDays.map((day) => {
                                      const selected = isSameZurichDay(day, selectedWebDate);
                                      const wd = WEEKDAY_LABELS[(day.getDay() + 6) % 7];
                                      const outsideFocus =
                                        day.getMonth() !== stripViewportFocus.getMonth() ||
                                        day.getFullYear() !== stripViewportFocus.getFullYear();
                                      const isPastDay = startOfZurichDay(day).getTime() < today.getTime();
                                      return (
                                        <Pressable
                                          key={`day-strip-${day.toISOString()}`}
                                          disabled={isPastDay}
                                          onPress={() => setWebDay(day)}
                                          style={({ pressed }) => [
                                            styles.mobileDayStripCell,
                                            selected ? styles.mobileDayStripCellSelected : null,
                                            isPastDay && { opacity: 0.35 },
                                            pressed ? styles.webActionButtonPressed : null,
                                          ]}
                                          accessibilityRole="button"
                                          accessibilityLabel={`Choisir le ${day.toLocaleDateString("fr-CH")}`}
                                        >
                                          <AppText
                                            style={[
                                              styles.mobileDayStripWeekday,
                                              selected ? styles.mobileDayStripWeekdaySelected : null,
                                            ]}
                                          >
                                            {wd}
                                          </AppText>
                                          <Text
                                            style={[
                                              styles.mobileDayStripNumber,
                                              outsideFocus && !selected
                                                ? styles.mobileDayStripNumberOutside
                                                : null,
                                              selected ? styles.mobileDayStripNumberSelected : null,
                                            ]}
                                          >
                                            {String(day.getDate()).padStart(2, "0")}
                                          </Text>
                                        </Pressable>
                                      );
                                    })}
                                  </View>
                                </View>
                              </ScrollView>
                            ) : carouselRailWidth > 0 && dateCarousel == null ? (
                              <View style={{ height: PICKER_CAROUSEL_CARD_CONTENT_HEIGHT }} />
                            ) : carouselRailWidth === 0 ? (
                              <View style={{ minHeight: PICKER_CAROUSEL_CARD_CONTENT_HEIGHT }} />
                            ) : null}
                            <LinearGradient
                              colors={["rgba(248, 250, 252, 0.82)", "rgba(248, 250, 252, 0)"]}
                              start={{ x: 0, y: 0.5 }}
                              end={{ x: 1, y: 0.5 }}
                              style={styles.mobileCarouselFadeLeft}
                            />
                            <LinearGradient
                              colors={["rgba(248, 250, 252, 0)", "rgba(248, 250, 252, 0.82)"]}
                              start={{ x: 0, y: 0.5 }}
                              end={{ x: 1, y: 0.5 }}
                              style={styles.mobileCarouselFadeRight}
                            />
                          </View>
                        <View style={styles.mobileActionsDock}>
                        <Pressable
                          onPress={setWebDayToday}
                          style={({ pressed }) => [
                            styles.mobileActionDockButton,
                            styles.mobileActionDockPrimary,
                            pressed && styles.webActionButtonPressed,
                          ]}
                          accessibilityRole="button"
                          accessibilityLabel="Aller à aujourd’hui"
                        >
                          <AppText style={[styles.mobileActionDockText, styles.mobileActionDockTextPrimary]}>
                            Aujourd&apos;hui
                          </AppText>
                        </Pressable>
                        <Pressable
                          onPress={dateOnly ? () => closeEditor("confirm") : goToTimeStep}
                          style={({ pressed }) => [
                            styles.mobileActionDockButton,
                            styles.mobileActionDockPrimary,
                            pressed && styles.webActionButtonPressed,
                          ]}
                          accessibilityRole="button"
                          accessibilityLabel={dateOnly ? "Valider la date" : "Passer à la sélection de l’heure"}
                        >
                          <AppText style={[styles.mobileActionDockText, styles.mobileActionDockTextPrimary]}>
                            {dateOnly ? "Valider" : "Continuer"}
                          </AppText>
                        </Pressable>
                        </View>
                    </View>
                    </View>
                    </>
                  ) : (
                    <View style={styles.mobileDateStepColumn}>
                    <View style={styles.mobileDateShell}>
                      <View style={styles.mobileDateHeader}>
                        <Pressable
                          style={[
                            styles.mobileDateSeg,
                            timeCarousel === "hour" ? styles.mobileDateSegActive : null,
                          ]}
                          onPress={() =>
                            setTimeCarousel((c) => (c === "hour" ? null : "hour"))
                          }
                          accessibilityRole="button"
                          accessibilityLabel="Choisir les heures : liste défilante"
                          hitSlop={{ top: 8, bottom: 8, left: 6, right: 6 }}
                        >
                          <AppText style={styles.mobileDateSegText}>
                            {String(selectedWebTimeParts.hour).padStart(2, "0")}
                          </AppText>
                        </Pressable>
                        <View style={styles.mobileTimeHeaderColonWrap}>
                          <AppText style={styles.mobileTimeHeaderColon}>:</AppText>
                        </View>
                        <Pressable
                          style={[
                            styles.mobileDateSeg,
                            timeCarousel === "minute" ? styles.mobileDateSegActive : null,
                          ]}
                          onPress={() =>
                            setTimeCarousel((c) => (c === "minute" ? null : "minute"))
                          }
                          accessibilityRole="button"
                          accessibilityLabel="Choisir les minutes : liste défilante"
                          hitSlop={{ top: 8, bottom: 8, left: 6, right: 6 }}
                        >
                          <AppText style={styles.mobileDateSegText}>
                            {String(selectedWebTimeParts.minute).padStart(2, "0")}
                          </AppText>
                        </Pressable>
                      </View>
                      <View
                        style={styles.mobileTimeStripOuter}
                        onLayout={(e) => setTimeCarouselRailWidth(e.nativeEvent.layout.width)}
                      >
                        {timeCarouselRailWidth > 0 && timeCarousel === "hour" ? (
                          <ScrollView
                            ref={hourStripRef}
                            horizontal
                            snapToInterval={TIME_HOUR_CELL_WIDTH}
                            snapToAlignment="start"
                            decelerationRate="normal"
                            nestedScrollEnabled
                            keyboardShouldPersistTaps="handled"
                            showsHorizontalScrollIndicator={false}
                            scrollEventThrottle={16}
                            style={{ width: timeCarouselRailWidth }}
                          >
                            <View style={styles.mobilePickerCarouselCard}>
                              <View style={styles.mobileTimeHourStripRow}>
                                {Array.from({ length: 24 }, (_, hour) => {
                                  const active = selectedWebTimeParts.hour === hour;
                                  return (
                                    <Pressable
                                      key={`hour-strip-${hour}`}
                                      onPress={() => {
                                        setWebTime(hour, selectedWebTimeParts.minute);
                                        setTimeCarousel("minute");
                                      }}
                                      style={({ pressed }) => [
                                        styles.mobileTimeStripCell,
                                        styles.mobileTimeHourStripCell,
                                        active ? styles.mobileTimeStripCellActive : null,
                                        pressed ? styles.webActionButtonPressed : null,
                                      ]}
                                      accessibilityRole="button"
                                      accessibilityLabel={`Choisir ${String(hour).padStart(2, "0")} heures`}
                                    >
                                      <AppText
                                        style={[
                                          styles.mobileTimeStripValue,
                                          active ? styles.mobileTimeStripValueActive : null,
                                        ]}
                                      >
                                        {String(hour).padStart(2, "0")}
                                      </AppText>
                                    </Pressable>
                                  );
                                })}
                              </View>
                            </View>
                          </ScrollView>
                        ) : null}
                        {timeCarouselRailWidth > 0 && timeCarousel === "minute" ? (
                          <ScrollView
                            ref={minuteStripRef}
                            horizontal
                            snapToInterval={TIME_MINUTE_CELL_WIDTH}
                            snapToAlignment="start"
                            decelerationRate="normal"
                            nestedScrollEnabled
                            keyboardShouldPersistTaps="handled"
                            showsHorizontalScrollIndicator={false}
                            scrollEventThrottle={16}
                            style={{ width: timeCarouselRailWidth }}
                          >
                            <View style={styles.mobilePickerCarouselCard}>
                              <View style={styles.mobileTimeMinuteStripRow}>
                                {TIME_MINUTE_SLOTS.map((minute) => {
                                  const active =
                                    nearestFiveMinute(selectedWebTimeParts.minute) === minute;
                                  return (
                                    <Pressable
                                      key={`minute-strip-${minute}`}
                                      onPress={() => setWebTime(selectedWebTimeParts.hour, minute)}
                                      style={({ pressed }) => [
                                        styles.mobileTimeStripCell,
                                        styles.mobileTimeMinuteStripCell,
                                        active ? styles.mobileTimeStripCellActive : null,
                                        pressed ? styles.webActionButtonPressed : null,
                                      ]}
                                      accessibilityRole="button"
                                      accessibilityLabel={`Choisir la minute ${String(minute).padStart(2, "0")}`}
                                    >
                                      <AppText
                                        style={[
                                          styles.mobileTimeStripValue,
                                          active ? styles.mobileTimeStripValueActive : null,
                                        ]}
                                      >
                                        {String(minute).padStart(2, "0")}
                                      </AppText>
                                    </Pressable>
                                  );
                                })}
                              </View>
                            </View>
                          </ScrollView>
                        ) : null}
                        {timeCarouselRailWidth > 0 && timeCarousel == null ? (
                          <View style={{ height: PICKER_CAROUSEL_CARD_CONTENT_HEIGHT }} />
                        ) : null}
                        {timeCarouselRailWidth === 0 ? (
                          <View style={{ minHeight: PICKER_CAROUSEL_CARD_CONTENT_HEIGHT }} />
                        ) : null}
                        <LinearGradient
                          colors={["rgba(248, 250, 252, 0.82)", "rgba(248, 250, 252, 0)"]}
                          start={{ x: 0, y: 0.5 }}
                          end={{ x: 1, y: 0.5 }}
                          style={styles.mobileCarouselFadeLeft}
                        />
                        <LinearGradient
                          colors={["rgba(248, 250, 252, 0)", "rgba(248, 250, 252, 0.82)"]}
                          start={{ x: 0, y: 0.5 }}
                          end={{ x: 1, y: 0.5 }}
                          style={styles.mobileCarouselFadeRight}
                        />
                      </View>
                      <View style={styles.mobileActionsDock}>
                        {!timeOnly ? (
                        <Pressable
                          onPress={() => setMobileStep("date")}
                          style={({ pressed }) => [
                            styles.mobileActionDockButton,
                            styles.mobileActionDockSecondary,
                            pressed && styles.webActionButtonPressed,
                          ]}
                          accessibilityRole="button"
                          accessibilityLabel="Retour à la sélection de date"
                        >
                          <AppText
                            style={[styles.mobileActionDockText, styles.mobileActionDockTextSecondary]}
                          >
                            Retour date
                          </AppText>
                        </Pressable>
                        ) : null}
                        <Pressable
                          onPress={() => commitSchedule(new Date())}
                          style={({ pressed }) => [
                            styles.mobileActionDockButton,
                            styles.mobileActionDockPrimary,
                            pressed && styles.webActionButtonPressed,
                          ]}
                          accessibilityRole="button"
                          accessibilityLabel="Définir sur maintenant"
                        >
                          <AppText
                            style={[styles.mobileActionDockText, styles.mobileActionDockTextPrimary]}
                          >
                            Maintenant
                          </AppText>
                        </Pressable>
                        {!timeOnly ? (
                        <Pressable
                          onPress={() => onChange("")}
                          style={({ pressed }) => [
                            styles.mobileActionDockButton,
                            styles.mobileClearButton,
                            pressed && styles.webActionButtonPressed,
                          ]}
                          accessibilityRole="button"
                          accessibilityLabel="Aucune date définie"
                        >
                          <AppText
                            style={[styles.mobileActionDockText, styles.mobileActionDockTextSecondary]}
                          >
                            À définir
                          </AppText>
                        </Pressable>
                        ) : null}
                      </View>
                    </View>
                    </View>
                  )}
                </View>
            </ScrollView>
          </View>
      </Modal>
    </View>
  );
}
