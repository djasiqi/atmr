import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Animated, Easing, Keyboard, Platform, Pressable, StyleSheet, View } from "react-native";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { AxiosError } from "axios";
import { AppButton, Modal, ModalFooterActions, useAccessibilityScale } from "../../../../design/responsive";
import { AppInput } from "../../../../design/ui/AppInput";
import { AppSelect } from "../../../../design/ui/AppSelect";
import { AppText } from "../../../../design/ui/AppText";
import { isFeatureEnabled } from "../../../../core/featureFlags/registry";
import { apiClient } from "../../../../core/api/client";
import { E } from "../../theme/enterpriseOpsTheme";
import {
  normalizeScheduledTimeIso,
  useCompanyBillingPricingContext,
  useCompanyClientDetail,
  useCompanyPricingSimulation,
  useRideCreate,
  useRideFormState,
} from "../../useRideForms";
import type { RideAddressOption, RideClientOption } from "../../useRideForms";
import { useActiveCompanyContextId } from "../../hooks";
import { searchCompanyAddresses } from "../../api/companyApi";
import { AddressFieldTrigger, AddressPickerSheet } from "./AddressPickerSheet";
import { ClientPickerSheet, CreateClientTrigger } from "./ClientPickerSheet";
import {
  applyCreateRideActiveField,
  createRideMissingHint,
  type CreateRideActiveField,
} from "./createRideActiveField";
import { RecurrenceSelector } from "./RecurrenceSelector";
import { TimeDatePicker } from "./TimeDatePicker";
import { ClientCreateModal } from "./ClientCreateModal";
import { RideCreateSection } from "./RideCreateSection";
import { RideRoutePreview } from "./RideRoutePreview";
import {
  analyzePricingSimulation,
  backendWeekdayFromScheduledIso,
  buildRideCreatePayload,
  computeRecurrencePreview,
  parseMedicalHintsFromAddress,
  parseSimulationAmount,
  resolvePreferentialBookingAmount,
} from "./rideCreateHelpers";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import { createShadow } from "../../../../styles/shadowStyles";

type RideCreateModalProps = {
  visible: boolean;
  onClose: () => void;
  onCreated?: () => void;
};

const NOTES_MAX = 500;
const SIM_DEBOUNCE_MS = 180;
const SIM_CACHE_TTL_MS = 60 * 1000;
const SIM_CACHE_MAX_SIZE = 50;
const COORD_PRECISION = 5;
/** Jours backend 0 = lun … 6 = dim. */
const WEEKDAY_SHORT = ["Lu", "Ma", "Me", "Je", "Ve", "Sa", "Di"] as const;
const ROW_RADIUS = 12;
const COMPACT_CONTROL_RADIUS = 11;
const COMPACT_CHIP_HEIGHT = 32;
const COMPACT_CHIP_SMALL_HEIGHT = 40;
const COMPACT_ACTION_HEIGHT = 46;
const COMPACT_MULTILINE_MEDIUM_HEIGHT = 72;
const COMPACT_MULTILINE_MEDIUM_INPUT_HEIGHT = 56;
const FIELD_ICON_SIZE = 18;
const BACK_BOX = {
  width: 32,
  height: 32,
  borderRadius: 10,
  backgroundColor: "transparent",
  alignItems: "center" as const,
  justifyContent: "center" as const,
};

const s = StyleSheet.create({
  form: { gap: 6, paddingBottom: 2 },
  sectionDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(15, 23, 42, 0.06)",
    marginVertical: 0,
  },
  sectionLabel: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "600" as const,
    color: E.TEXT,
    marginBottom: 4,
  },
  sectionHelper: {
    fontSize: FONT_SIZE.px12,
    color: E.TEXT_MUTED,
    lineHeight: 17,
  },
  fieldBlock: { gap: 6 },
  formGroup: { gap: 8 },
  tonalGroup: {
    backgroundColor: "#FAFBFA",
    borderRadius: ROW_RADIUS,
    paddingHorizontal: 10,
    paddingVertical: 10,
  },
  formGroupDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(145, 165, 157, 0.35)",
    marginVertical: 8,
  },
  fieldLabel: {
    fontSize: FONT_SIZE.px12,
    fontWeight: "600" as const,
    color: E.TEXT_SEC,
    letterSpacing: 0.1,
  },
  fieldRequired: { color: E.BRAND },

  /* ---------- Header ---------- */
  headerRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 8,
    paddingTop: 0,
    paddingBottom: 2,
    marginTop: -4,
  },
  headerCenter: { flex: 1, alignItems: "center" as const },
  headerTitle: {
    fontSize: FONT_SIZE.px15,
    fontWeight: "700" as const,
    color: E.TEXT,
    letterSpacing: 0.1,
    textAlign: "center" as const,
  },
  headerIconBtn: {
    width: 30,
    height: 30,
    borderRadius: 15,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    backgroundColor: "rgba(0, 121, 107, 0.06)",
  },

  /* ---------- Pickup / Dropoff split ---------- */
  pickupDropoffSplit: {
    flexDirection: "row" as const,
    alignItems: "stretch" as const,
    gap: 6,
  },
  addressColumnLeft: { flex: 1, minWidth: 0, gap: 8 },
  addressActionsColumn: {
    width: 36,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    gap: 8,
  },
  swapBtnRound: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.22)",
    backgroundColor: "rgba(0, 121, 107, 0.10)",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  actionRoundBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 1,
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  actionRoundBtnInactive: {
    borderColor: "rgba(148, 163, 184, 0.4)",
    backgroundColor: "#FFFFFF",
  },
  actionRoundBtnActive: {
    borderColor: E.BRAND,
    backgroundColor: E.BRAND,
  },
  actionRoundBtnSpacer: {
    width: 36,
    height: 36,
  },
  inlineActionRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 6,
  },
  inlineActionGrow: { flex: 1, minWidth: 0 },
  /* ---------- Date / Heure ---------- */
  dateTimeRow: {
    flexDirection: "row" as const,
    gap: 8,
  },
  dateTimeSlot: { flex: 1, minWidth: 0 },

  /* ---------- Section Prix ---------- */
  priceRow: {
    flexDirection: "row" as const,
    gap: 6,
    alignItems: "stretch" as const,
  },
  priceCardEstimate: {
    flex: 1.4,
    borderRadius: 9,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.20)",
    backgroundColor: "rgba(0, 121, 107, 0.04)",
    paddingHorizontal: 8,
    paddingVertical: 5,
    gap: 1,
    justifyContent: "center" as const,
  },
  priceCardManual: {
    flex: 1,
    borderRadius: 9,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.32)",
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 8,
    paddingVertical: 5,
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "space-between" as const,
    gap: 4,
  },
  priceCardManualActive: {
    borderColor: E.BRAND,
    backgroundColor: "rgba(0, 121, 107, 0.04)",
  },
  priceCardLabelRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 5,
    flexWrap: "wrap" as const,
  },
  priceCardLabel: {
    fontSize: FONT_SIZE.px10,
    color: E.TEXT_SEC,
    fontWeight: "700" as const,
    letterSpacing: 0.3,
    textTransform: "uppercase" as const,
    lineHeight: 12,
  },
  priceCardLabelActive: { color: E.BRAND_DARK },
  priceCardBadgeRecommended: {
    paddingHorizontal: 4,
    paddingVertical: 0,
    borderRadius: 999,
    backgroundColor: "rgba(0, 121, 107, 0.12)",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(0, 121, 107, 0.22)",
  },
  priceCardBadgeRecommendedText: {
    color: E.BRAND_DARK,
    fontSize: FONT_SIZE.px10,
    fontWeight: "700" as const,
    lineHeight: 12,
  },
  priceCardAmountRow: {
    flexDirection: "row" as const,
    alignItems: "baseline" as const,
    gap: 3,
  },
  priceCardAmount: {
    color: E.BRAND_DARK,
    fontSize: FONT_SIZE.px15,
    fontWeight: "800" as const,
    lineHeight: 18,
  },
  priceCardAmountUnit: {
    color: E.BRAND_DARK,
    fontSize: FONT_SIZE.px10,
    fontWeight: "700" as const,
  },
  priceCardSubtext: {
    color: E.TEXT_MUTED,
    fontSize: FONT_SIZE.px10,
    lineHeight: 12,
  },
  priceCardManualCol: { flex: 1, minWidth: 0 },
  priceCardManualLabel: {
    fontSize: FONT_SIZE.px12,
    fontWeight: "700" as const,
    color: E.TEXT,
    lineHeight: 14,
  },
  priceCardManualLabelActive: { color: E.BRAND_DARK },
  priceCardManualHint: {
    color: E.TEXT_MUTED,
    fontSize: FONT_SIZE.px10,
    lineHeight: 12,
    marginTop: 0,
  },

  /* ---------- Section 4 : accordéon ---------- */
  extraInfoHint: {
    color: E.TEXT_MUTED,
    fontSize: FONT_SIZE.px12,
    lineHeight: 15,
    marginTop: 2,
  },
  extraInfoBody: { gap: 12 },

  /* ---------- Sub-block flat (clinique, AR, récurrence, médical, notes) ---------- */
  subCard: {
    paddingVertical: 4,
    gap: 8,
  },
  subCardBordered: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.28)",
    backgroundColor: "#FAFBFA",
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  subBlockDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(15, 23, 42, 0.08)",
    marginVertical: 2,
  },
  subCardTitleRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 8,
  },
  subCardTitle: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "700" as const,
    color: E.TEXT,
  },
  subCardValue: {
    fontSize: FONT_SIZE.px13,
    color: E.TEXT,
    fontWeight: "600" as const,
  },
  subCardHint: {
    fontSize: FONT_SIZE.px12,
    color: E.TEXT_MUTED,
    lineHeight: 16,
  },

  /* ---------- Generic chip (still used by clinic toggle, wheelchair) ---------- */
  chip: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: COMPACT_CONTROL_RADIUS,
    borderWidth: StyleSheet.hairlineWidth,
    minHeight: COMPACT_CHIP_HEIGHT,
  },
  chipOn: {
    backgroundColor: "rgba(0, 121, 107, 0.14)",
    borderColor: E.BRAND,
  },
  chipOff: {
    backgroundColor: "#FFFFFF",
    borderColor: "rgba(0, 121, 107, 0.28)",
  },
  chipLabelOn: { color: E.BRAND, fontWeight: "700" as const, fontSize: FONT_SIZE.px13, lineHeight: 16 },
  chipLabelOff: { color: E.TEXT_SEC, fontWeight: "600" as const, fontSize: FONT_SIZE.px13, lineHeight: 16 },
  chipBlueOn: {
    backgroundColor: "rgba(14, 165, 233, 0.12)",
    borderColor: E.TRANSFER,
  },
  chipBlueOff: {
    backgroundColor: "#FFFFFF",
    borderColor: "rgba(14, 165, 233, 0.32)",
  },
  chipBlueLabelOn: {
    color: E.TRANSFER,
    fontWeight: "700" as const,
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
  },
  chipOrangeOn: {
    backgroundColor: "rgba(245, 158, 11, 0.14)",
    borderColor: E.URGENT,
  },
  chipOrangeOff: {
    backgroundColor: "#FFFFFF",
    borderColor: "rgba(245, 158, 11, 0.36)",
  },
  chipOrangeLabelOn: {
    color: E.URGENT,
    fontWeight: "700" as const,
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
  },

  /* ---------- Recurrence ---------- */
  recurrenceWeekdayRow: {
    flexDirection: "row" as const,
    flexWrap: "nowrap" as const,
    gap: 6,
    justifyContent: "space-between" as const,
  },
  recurrenceWeekdayChip: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 0,
    minWidth: 0,
    minHeight: COMPACT_CHIP_SMALL_HEIGHT,
    paddingVertical: 8,
    paddingHorizontal: 6,
    borderRadius: COMPACT_CONTROL_RADIUS,
    borderWidth: 1,
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  recurrenceWeekdayChipOn: {
    backgroundColor: E.BRAND,
    borderColor: E.BRAND,
  },
  recurrenceWeekdayChipOff: {
    backgroundColor: "#FFFFFF",
    borderColor: "rgba(0, 121, 107, 0.35)",
  },
  recurrenceWeekdayChipTextOn: { color: "#FFFFFF", fontWeight: "700" as const, fontSize: FONT_SIZE.px12, lineHeight: 15 },
  recurrenceWeekdayChipTextOff: { color: E.BRAND, fontWeight: "600" as const, fontSize: FONT_SIZE.px12, lineHeight: 15 },
  recurrenceModeCard: {
    flexDirection: "row" as const,
    alignItems: "flex-start" as const,
    gap: 10,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    borderColor: "rgba(0, 121, 107, 0.20)",
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 12,
  },
  recurrenceModeIconWrap: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: "rgba(0, 121, 107, 0.14)",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  recurrenceModeTitle: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "700" as const,
    color: E.TEXT,
    lineHeight: 16,
  },
  recurrenceModeDesc: {
    fontSize: FONT_SIZE.px11,
    color: E.TEXT_SEC,
    lineHeight: 14,
    marginTop: 2,
  },
  recurrenceDateField: { gap: 4 },
  recurrenceDateFieldLabel: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "700" as const,
    color: E.TEXT,
    letterSpacing: 0.2,
  },
  recurrenceSummaryCard: {
    gap: 8,
    backgroundColor: "#FFFFFF",
    borderColor: "rgba(0, 121, 107, 0.32)",
    borderWidth: 1,
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 12,
  },
  recurrenceSummaryHeader: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 6,
  },
  recurrenceSummaryEyebrow: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "700" as const,
    color: E.BRAND_DARK,
    letterSpacing: 0.6,
    textTransform: "uppercase" as const,
  },
  recurrenceSummaryBody: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 10,
  },
  recurrenceSummaryTitle: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "700" as const,
    color: E.TEXT,
    lineHeight: 16,
  },
  recurrenceSummaryWindow: {
    fontSize: FONT_SIZE.px11,
    color: E.TEXT_SEC,
    lineHeight: 14,
    marginTop: 2,
  },
  recurrenceCountBadge: {
    minWidth: 78,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 10,
    backgroundColor: "rgba(0, 121, 107, 0.10)",
    alignItems: "center" as const,
  },
  recurrenceCountBadgePending: {
    backgroundColor: "rgba(148, 163, 184, 0.16)",
  },
  recurrenceCountBadgeNum: {
    fontSize: FONT_SIZE.px16,
    fontWeight: "800" as const,
    color: E.BRAND_DARK,
    lineHeight: 18,
  },
  recurrenceCountBadgeNumPending: {
    color: E.TEXT_MUTED,
  },
  recurrenceCountBadgeLabel: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "600" as const,
    color: E.BRAND_DARK,
    textAlign: "center" as const,
    lineHeight: 12,
    marginTop: 2,
  },
  recurrenceCountBadgeLabelPending: {
    color: E.TEXT_MUTED,
  },
  recurrencePreviewEyebrow: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "700" as const,
    color: E.TEXT_SEC,
    letterSpacing: 0.6,
    textTransform: "uppercase" as const,
  },
  recurrencePreviewRow: {
    flexDirection: "row" as const,
    flexWrap: "wrap" as const,
    gap: 6,
  },
  recurrencePreviewChip: {
    width: 56,
    paddingVertical: 6,
    paddingHorizontal: 4,
    borderRadius: 10,
    backgroundColor: "#FFFFFF",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(145, 165, 157, 0.38)",
    alignItems: "center" as const,
  },
  recurrencePreviewChipWd: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "600" as const,
    color: E.TEXT_SEC,
    lineHeight: 12,
  },
  recurrencePreviewChipDay: {
    fontSize: FONT_SIZE.px16,
    fontWeight: "800" as const,
    color: E.TEXT,
    lineHeight: 18,
  },
  recurrencePreviewChipMonth: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "700" as const,
    color: E.TEXT_SEC,
    letterSpacing: 0.4,
    lineHeight: 12,
  },
  recurrencePreviewChipMore: {
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    borderColor: "rgba(0, 121, 107, 0.28)",
    justifyContent: "center" as const,
  },
  recurrencePreviewChipMoreNum: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "800" as const,
    color: E.BRAND_DARK,
    lineHeight: 16,
  },
  recurrencePreviewChipMoreLabel: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "600" as const,
    color: E.BRAND_DARK,
    lineHeight: 12,
  },
  recurrenceHint: {
    flexDirection: "row" as const,
    alignItems: "flex-start" as const,
    gap: 10,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: ROW_RADIUS,
    backgroundColor: "rgba(0, 121, 107, 0.06)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.18)",
  },
  recurrenceHintText: {
    flex: 1,
    fontSize: FONT_SIZE.px12,
    color: E.TEXT_SEC,
    lineHeight: 17,
    fontWeight: "500" as const,
  },
  recurrenceSubLabel: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "700" as const,
    color: E.TEXT,
    marginTop: 2,
    marginBottom: 0,
  },

  /* ---------- Wheelchair row ---------- */
  wheelchairRow: {
    flexDirection: "row" as const,
    gap: 6,
    flexWrap: "wrap" as const,
  },

  /* ---------- Notes ---------- */
  /* ---------- Amount meta ---------- */
  amountMetaRow: {
    flexDirection: "row" as const,
    flexWrap: "wrap" as const,
    gap: 6,
    alignItems: "center" as const,
    marginTop: 2,
  },
  amountBadge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    borderWidth: 1,
  },
  amountBadgeText: {
    fontSize: FONT_SIZE.px12,
    fontWeight: "700" as const,
  },

  /* ---------- Footer ---------- */
  footerCol: { gap: 10 },
  summaryPanel: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.18)",
    backgroundColor: "rgba(0, 121, 107, 0.04)",
    paddingHorizontal: 10,
    paddingVertical: 7,
    gap: 4,
  },
  summaryPanelHeader: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 6,
    marginBottom: 1,
  },
  summaryPanelTitle: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "700" as const,
    color: E.TEXT_SEC,
    letterSpacing: 0.35,
    textTransform: "uppercase" as const,
  },
  summaryPanelDivider: {
    flex: 1,
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(0, 121, 107, 0.18)",
  },
  summaryRow: {
    flexDirection: "row" as const,
    alignItems: "baseline" as const,
    gap: 10,
  },
  summaryRowStacked: {
    flexDirection: "column" as const,
    alignItems: "stretch" as const,
  },
  summaryCell: {
    flex: 1,
    minWidth: 0,
    flexShrink: 1,
    flexDirection: "row" as const,
    alignItems: "baseline" as const,
    flexWrap: "wrap" as const,
    gap: 6,
  },
  summaryCellLabel: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "600" as const,
    color: E.TEXT_MUTED,
    letterSpacing: 0.2,
    flexShrink: 0,
  },
  summaryCellValue: {
    flex: 1,
    minWidth: 0,
    flexShrink: 1,
    fontSize: FONT_SIZE.px12,
    fontWeight: "700" as const,
    color: E.TEXT,
    lineHeight: 15,
  },
  summaryCellValueMuted: {
    color: E.TEXT_MUTED,
    fontWeight: "600" as const,
  },
  summaryPriceRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "space-between" as const,
    marginTop: 3,
    paddingTop: 4,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(0, 121, 107, 0.18)",
  },
  summaryPriceLabel: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "700" as const,
    color: E.TEXT_SEC,
    letterSpacing: 0.2,
  },
  summaryPriceValue: {
    fontSize: FONT_SIZE.px14,
    fontWeight: "800" as const,
    color: E.BRAND_DARK,
    lineHeight: 18,
  },
  footerHint: {
    fontSize: FONT_SIZE.px12,
    color: E.TEXT_MUTED,
    textAlign: "left" as const,
  },
  footerBtnSecondary: {
    minHeight: 48,
    borderRadius: 13,
  },
  footerBtnPrimary: {
    minHeight: 52,
    borderRadius: 14,
    ...createShadow({
      shadowColor: E.BRAND_DARK,
      shadowOffset: { width: 0, height: 6 },
      shadowOpacity: 0.22,
      shadowRadius: 14,
      elevation: 6,
    }),
  },

  /* ---------- Misc ---------- */
  linkNewClient: {
    marginTop: 4,
    alignSelf: "flex-start" as const,
    paddingVertical: 4,
  },
  manualEditWrap: { gap: 6 },
  error: { marginTop: 4 },
});

const OUTLINE_SECONDARY = {
  minHeight: COMPACT_ACTION_HEIGHT,
  borderRadius: COMPACT_CONTROL_RADIUS,
  borderColor: "rgba(0, 121, 107, 0.32)",
} as const;

function parseOptionalAmount(raw: string): number | null {
  const t = raw.trim().replace(",", ".");
  if (!t) return null;
  const n = Number.parseFloat(t);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

function isValidPreferentialAmount(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function hasValidCoords(address: RideAddressOption | null | undefined): boolean {
  if (!address) return false;
  const lat = address.latitude;
  const lng = address.longitude;
  if (typeof lat !== "number" || typeof lng !== "number") return false;
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return false;
  // Une coordonnée (0, 0) côté Suisse/Genève signale toujours une donnée manquante.
  if (lat === 0 && lng === 0) return false;
  // Bornes plausibles (filtrer NaN/erreurs venant de parsers).
  if (lat < -90 || lat > 90) return false;
  if (lng < -180 || lng > 180) return false;
  return true;
}

function toRoundedCoord(value: number): number {
  return Number(value.toFixed(COORD_PRECISION));
}

function pruneSimulationCache(
  cache: Map<string, { amount: number; warningMessage: string | null; cachedAt: number }>
) {
  const now = Date.now();
  for (const [key, entry] of cache.entries()) {
    if (now - entry.cachedAt > SIM_CACHE_TTL_MS) {
      cache.delete(key);
    }
  }
  if (cache.size <= SIM_CACHE_MAX_SIZE) return;
  const oldest = [...cache.entries()].sort((a, b) => a[1].cachedAt - b[1].cachedAt);
  const toDelete = oldest.slice(0, cache.size - SIM_CACHE_MAX_SIZE);
  toDelete.forEach(([key]) => cache.delete(key));
}

function toResolvedAddressOption(raw: unknown, fallbackLabel: string): RideAddressOption | null {
  if (!raw || typeof raw !== "object") return null;
  const row = raw as Record<string, unknown>;
  const properties =
    row.properties && typeof row.properties === "object"
      ? (row.properties as Record<string, unknown>)
      : undefined;
  const geometry =
    row.geometry && typeof row.geometry === "object"
      ? (row.geometry as Record<string, unknown>)
      : undefined;
  const geometryCoords = Array.isArray(geometry?.coordinates) ? geometry.coordinates : null;

  const labelCandidate =
    (typeof row.label === "string" && row.label.trim()) ||
    (typeof row.description === "string" && row.description.trim()) ||
    (typeof row.address === "string" && row.address.trim()) ||
    (typeof row.display_name === "string" && row.display_name.trim()) ||
    (typeof properties?.label === "string" && properties.label.trim()) ||
    (typeof properties?.name === "string" && properties.name.trim()) ||
    fallbackLabel;

  const latCandidate =
    row.lat ??
    row.latitude ??
    properties?.lat ??
    (geometryCoords && geometryCoords.length > 1 ? geometryCoords[1] : null);
  const lonCandidate =
    row.lon ??
    row.lng ??
    row.longitude ??
    properties?.lon ??
    properties?.lng ??
    (geometryCoords && geometryCoords.length > 0 ? geometryCoords[0] : null);

  const latitude =
    typeof latCandidate === "number"
      ? latCandidate
      : typeof latCandidate === "string"
        ? Number.parseFloat(latCandidate)
        : NaN;
  const longitude =
    typeof lonCandidate === "number"
      ? lonCandidate
      : typeof lonCandidate === "string"
        ? Number.parseFloat(lonCandidate)
        : NaN;

  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    return null;
  }

  const idCandidate =
    row.id ??
    row.place_id ??
    row.placeId ??
    row.photon_id ??
    row.osm_id ??
    properties?.id;
  const parsedId =
    typeof idCandidate === "number"
      ? idCandidate
      : typeof idCandidate === "string"
        ? Number.parseInt(idCandidate, 10)
        : NaN;

  return {
    id: Number.isFinite(parsedId) ? parsedId : Math.abs(Math.floor(latitude * 100000)),
    label: String(labelCandidate),
    placeId:
      typeof row.place_id === "string"
        ? row.place_id
        : typeof row.placeId === "string"
          ? row.placeId
          : typeof properties?.place_id === "string"
            ? properties.place_id
            : null,
    latitude,
    longitude,
  };
}

async function enrichAddressWithPlaceDetails(address: RideAddressOption): Promise<RideAddressOption> {
  if (hasValidCoords(address) || !address.placeId) {
    return address;
  }
  try {
    const response = await apiClient.get(
      `geocode/place-details?place_id=${encodeURIComponent(address.placeId)}`
    );
    const details = response?.data as
      | {
          lat?: number | string;
          lon?: number | string;
          label?: string;
          address?: string;
          name?: string;
          address_components?: { long_name?: string; types?: string[] }[];
        }
      | undefined;
    const latCandidate = Number(details?.lat);
    const lonCandidate = Number(details?.lon);
    if (!Number.isFinite(latCandidate) || !Number.isFinite(lonCandidate)) {
      return address;
    }
    const comps = Array.isArray(details?.address_components) ? details.address_components : [];
    const pickComp = (type: string) =>
      comps.find((c) => Array.isArray(c.types) && c.types.includes(type))?.long_name?.trim() || "";
    const streetNumber = pickComp("street_number");
    const route = pickComp("route");
    const postcode = pickComp("postal_code");
    const city = pickComp("locality") || pickComp("administrative_area_level_2");
    const streetAddress = [route, streetNumber].filter(Boolean).join(" ").trim();
    const structuredAddress = [streetAddress, [postcode, city].filter(Boolean).join(" ").trim()]
      .filter(Boolean)
      .join(", ");
    const placeName =
      (typeof details?.name === "string" && details.name.trim()) ||
      address.mainText ||
      "";
    const nextLabel =
      (structuredAddress
        ? placeName && placeName.toLowerCase() !== streetAddress.toLowerCase()
          ? `${placeName}, ${structuredAddress}`
          : structuredAddress
        : "") ||
      (typeof details?.label === "string" && details.label.trim()) ||
      (typeof details?.address === "string" && details.address.trim()) ||
      address.label;
    return {
      ...address,
      label: nextLabel,
      mainText: placeName || address.mainText || null,
      secondaryText: structuredAddress || address.secondaryText || null,
      latitude: latCandidate,
      longitude: lonCandidate,
    };
  } catch {
    return address;
  }
}

function formatSwissDateTime(iso: string): string {
  const n = normalizeScheduledTimeIso(iso);
  if (!n) return "Non défini";
  const d = new Date(n);
  if (Number.isNaN(d.getTime())) return "Non défini";
  return d.toLocaleString("fr-CH", {
    timeZone: "Europe/Zurich",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatSwissTimeOnly(iso: string): string {
  const n = normalizeScheduledTimeIso(iso);
  if (!n) return "";
  const d = new Date(n);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString("fr-CH", {
    timeZone: "Europe/Zurich",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function toSubmitErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const data = error.response?.data as
      | { message?: unknown; error?: unknown; error_message?: unknown }
      | undefined;
    const candidate = [data?.message, data?.error, data?.error_message, error.message].find(
      (value): value is string => typeof value === "string" && value.trim().length > 0
    );
    if (candidate) return candidate.trim();
  }
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message.trim();
  }
  return "Création de la réservation impossible.";
}

export function RideCreateModal({ visible, onClose, onCreated }: RideCreateModalProps) {
  const { isVeryLargeText, shouldStackRows } = useAccessibilityScale();
  const activeContextId = useActiveCompanyContextId();
  const createRide = useRideCreate();
  const form = useRideFormState();
  const [error, setError] = useState<string | null>(null);
  const [selectedClientLabel, setSelectedClientLabel] = useState("");
  const [selectedClientSubtitle, setSelectedClientSubtitle] = useState("");
  const [amountSource, setAmountSource] = useState<"preferential" | "simulated" | "manual" | null>(null);
  const [amountLocked, setAmountLocked] = useState(false);
  const [pricingWarning, setPricingWarning] = useState("");
  const [selectedClientPreferentialRate, setSelectedClientPreferentialRate] = useState<number | null>(null);
  const [billToPatient, setBillToPatient] = useState(false);
  const [createClientVisible, setCreateClientVisible] = useState(false);
  const [extraInfoOpen, setExtraInfoOpen] = useState(false);
  const [priceOpen, setPriceOpen] = useState(false);
  const [manualPriceOpen, setManualPriceOpen] = useState(false);
  const [routePointsForPricing, setRoutePointsForPricing] = useState<{ lat: number; lng: number }[]>([]);
  const [routeDistanceMeters, setRouteDistanceMeters] = useState<number | null>(null);
  const [routeDurationSeconds, setRouteDurationSeconds] = useState<number | null>(null);
  const [routePricingReady, setRoutePricingReady] = useState(false);
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const [activeField, setActiveField] = useState<CreateRideActiveField>(null);
  const priceAutoOpenedRef = useRef(false);
  const swapRotation = useRef(new Animated.Value(0)).current;
  const swapRotationTargetRef = useRef(0);

  const pickerSheetOpen =
    activeField === "client" || activeField === "pickup" || activeField === "dropoff";
  const parentKeyboardActive = keyboardVisible && !pickerSheetOpen;

  const setFieldActive = useCallback(
    (field: Exclude<CreateRideActiveField, null>, open: boolean) => {
      setActiveField((prev) => applyCreateRideActiveField(prev, field, open));
    },
    []
  );

  const handleSwapAddresses = useCallback(() => {
    form.swapAddresses();
    swapRotationTargetRef.current += 180;
    Animated.timing(swapRotation, {
      toValue: swapRotationTargetRef.current,
      duration: 280,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, [form, swapRotation]);

  const swapRotateStyle = useMemo(
    () => ({
      transform: [
        {
          rotate: swapRotation.interpolate({
            inputRange: [0, 360],
            outputRange: ["0deg", "360deg"],
            extrapolate: "extend" as const,
          }),
        },
      ],
    }),
    [swapRotation],
  );

  useEffect(() => {
    const showEvent = Platform.OS === "ios" ? "keyboardWillShow" : "keyboardDidShow";
    const hideEvent = Platform.OS === "ios" ? "keyboardWillHide" : "keyboardDidHide";
    const showSub = Keyboard.addListener(showEvent, () => setKeyboardVisible(true));
    const hideSub = Keyboard.addListener(hideEvent, () => {
      setKeyboardVisible(false);
    });
    return () => {
      showSub.remove();
      hideSub.remove();
    };
  }, []);

  useEffect(() => {
    if (!visible) {
      setActiveField(null);
    }
  }, [visible]);
  const clientDetailHydrationKeyRef = useRef<string>("");
  const completedSimulationKeyRef = useRef<string>("");
  const activeSimulationKeyRef = useRef<string>("");
  const simulationRequestSeqRef = useRef(0);
  const amountLockedRef = useRef(false);
  const amountSourceRef = useRef<"preferential" | "simulated" | "manual" | null>(null);
  const simulationCacheRef = useRef(
    new Map<string, { amount: number; warningMessage: string | null; cachedAt: number }>()
  );
  const lastGeocodePickupKeyRef = useRef<string>("");
  const lastGeocodeDropoffKeyRef = useRef<string>("");
  const structuredPayloadEnabled = isFeatureEnabled("company_mobile_structured_ride_payload_enabled");
  const clientDetailQuery = useCompanyClientDetail(form.clientId);
  const pricingContextQuery = useCompanyBillingPricingContext();
  const pricingSimulation = useCompanyPricingSimulation();
  const pricingSimulateMutateRef = useRef(pricingSimulation.mutate);
  pricingSimulateMutateRef.current = pricingSimulation.mutate;
  const {
    pickup,
    clientId,
    pickupAccessNotes,
    dropoffAccessNotes,
    notesMedical,
    establishment,
    hospitalService,
    doctorName,
    wheelchairClient,
    wheelchairProvide,
    amountInput,
    scheduledAt,
    isRoundTrip,
    isMaterialDelivery,
    pickupAddress,
    dropoffAddress,
    setPickupAccessNotes,
    setDropoffAccessNotes,
    setNotesMedical,
    setEstablishment,
    setHospitalService,
    setDoctorName,
    setWheelchairClient,
    setWheelchairProvide,
    setAmountInput,
    recurrenceDays,
    setRecurrenceDays,
    recurrenceOccurrences,
    setRecurrenceOccurrences,
    recurrenceEndDate,
    setRecurrenceEndDate,
    recurrenceIntervalWeeks,
    setRecurrenceIntervalWeeks,
  } = form;

  const scheduledOk = useMemo(() => {
    const n = normalizeScheduledTimeIso(form.scheduledAt);
    return Boolean(n && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(n));
  }, [form.scheduledAt]);

  const recurringOn = form.recurrence !== "none";

  useEffect(() => {
    amountLockedRef.current = amountLocked;
  }, [amountLocked]);

  useEffect(() => {
    amountSourceRef.current = amountSource;
  }, [amountSource]);

  useEffect(() => {
    if (form.recurrence !== "custom" && form.recurrence !== "weekly") return;
    setRecurrenceDays((prev) => {
      if (prev.length > 0) return prev;
      const wd = backendWeekdayFromScheduledIso(form.scheduledAt);
      return wd != null ? [wd] : prev;
    });
  }, [form.recurrence, form.scheduledAt, setRecurrenceDays]);

  useEffect(() => {
    setRecurrenceOccurrences((prev) => Math.min(52, Math.max(1, Math.floor(Number(prev)) || 1)));
  }, [setRecurrenceOccurrences]);

  const recurrenceEndValid = /^\d{4}-\d{2}-\d{2}$/.test(recurrenceEndDate.trim());
  const recurrenceApiLimitMode = recurrenceEndDate.trim().length > 0 ? "until" : "count";
  const recurrenceSelectedDaysCount = recurrenceDays.length;

  const recurrenceSummary = useMemo(() => {
    if (!recurringOn) return "";
    const timePart = scheduledOk ? formatSwissTimeOnly(form.scheduledAt) : "";
    const head = timePart ? `Horaire: ${timePart}. ` : "";
    if (form.recurrence === "daily") {
      return `${head}Trajet répété tous les jours.`;
    }
    if (form.recurrence === "weekly") {
      return `${head}Trajet répété chaque semaine.`;
    }
    const days = form.recurrenceDays.map((d) => WEEKDAY_SHORT[d] ?? "?").join(", ");
    const intervalLabel =
      recurrenceIntervalWeeks > 1 ? `Toutes les ${recurrenceIntervalWeeks} semaines. ` : "";
    return `${head}${intervalLabel}${days ? `Jours actifs: ${days}.` : "Sélectionnez au moins un jour."}`;
  }, [
    recurringOn,
    form.recurrence,
    form.recurrenceDays,
    form.scheduledAt,
    scheduledOk,
    recurrenceIntervalWeeks,
  ]);

  const recurrencePreview = useMemo(() => {
    if (!recurringOn || !scheduledOk) return { total: 0, dates: [] as Date[] };
    if (form.recurrence === "custom" && recurrenceSelectedDaysCount === 0) {
      return { total: 0, dates: [] as Date[] };
    }
    return computeRecurrencePreview({
      scheduledAt: form.scheduledAt,
      recurrence: form.recurrence,
      days: recurrenceDays,
      endDate: recurrenceEndValid ? recurrenceEndDate.trim() : undefined,
      intervalWeeks: form.recurrence === "custom" ? recurrenceIntervalWeeks : 1,
    });
  }, [
    recurringOn,
    scheduledOk,
    form.recurrence,
    form.scheduledAt,
    recurrenceDays,
    recurrenceSelectedDaysCount,
    recurrenceEndDate,
    recurrenceEndValid,
    recurrenceIntervalWeeks,
  ]);

  const recurrenceLongLabel = useMemo(() => {
    if (!recurringOn) return "";
    if (form.recurrence === "daily") return "Tous les jours";
    if (recurrenceDays.length === 0) {
      return form.recurrence === "weekly" ? "Une fois par semaine" : "Aucun jour sélectionné";
    }
    const isCustomMulti = form.recurrence === "custom" && recurrenceIntervalWeeks > 1;
    const longNames = [
      "lundis",
      "mardis",
      "mercredis",
      "jeudis",
      "vendredis",
      "samedis",
      "dimanches",
    ];
    const buildDaysPhrase = (lead: string) => {
      if (recurrenceDays.length === 7) return lead ? `${lead}, tous les jours` : "Tous les jours";
      const picked = recurrenceDays.map((d) => longNames[d] ?? "?");
      const tailWord = picked[picked.length - 1];
      const head = picked.slice(0, -1).join(", ");
      const daysText =
        picked.length === 1
          ? lead
            ? `le ${tailWord.slice(0, -1)}`
            : `Tous les ${tailWord}`
          : lead
            ? `les ${head} et ${tailWord}`
            : `Tous les ${head} et ${tailWord}`;
      return lead ? `${lead}, ${daysText}` : daysText;
    };
    if (isCustomMulti) {
      return buildDaysPhrase(`Toutes les ${recurrenceIntervalWeeks} semaines`);
    }
    return buildDaysPhrase("");
  }, [recurringOn, form.recurrence, recurrenceDays, recurrenceIntervalWeeks]);

  const recurrenceWindowLabel = useMemo(() => {
    if (recurrencePreview.dates.length === 0) return "";
    const first = recurrencePreview.dates[0];
    const last = recurrencePreview.dates[recurrencePreview.dates.length - 1];
    const fmt = (d: Date) =>
      d.toLocaleDateString("fr-CH", { day: "2-digit", month: "long", year: "numeric" });
    return `Du ${fmt(first)} au ${fmt(last)}`;
  }, [recurrencePreview.dates]);

  const recurrenceValid = useMemo(() => {
    if (!recurringOn) return true;
    if (
      (form.recurrence === "custom" || form.recurrence === "weekly") &&
      recurrenceSelectedDaysCount === 0
    )
      return false;
    if (recurrenceEndDate.trim().length > 0) return recurrenceEndValid;
    const n = Math.floor(Number(recurrenceOccurrences));
    if (!Number.isFinite(n) || n < 1 || n > 52) return false;
    return true;
  }, [recurringOn, form.recurrence, recurrenceSelectedDaysCount, recurrenceOccurrences, recurrenceEndDate, recurrenceEndValid]);

  const toggleRecurrenceWeekday = (day: number) => {
    const set = new Set(recurrenceDays);
    if (set.has(day)) set.delete(day);
    else set.add(day);
    setRecurrenceDays(Array.from(set).sort((a, b) => a - b));
  };

  const amountValue = parseOptionalAmount(form.amountInput);
  const amountValid = amountValue != null && amountValue > 0;

  const canSubmit =
    Boolean(form.clientId) &&
    form.pickup.trim().length > 0 &&
    form.dropoff.trim().length > 0 &&
    scheduledOk &&
    (form.isMaterialDelivery || amountValid) &&
    form.internalNotes.length <= NOTES_MAX &&
    (!form.isMaterialDelivery || form.deliveryDescription.trim().length > 0) &&
    recurrenceValid;

  const section1Complete =
    Boolean(form.clientId) &&
    form.pickup.trim().length > 0 &&
    form.dropoff.trim().length > 0 &&
    scheduledOk;
  const section3Complete = form.isMaterialDelivery
    ? form.deliveryDescription.trim().length > 0
    : amountValid;

  useEffect(() => {
    if (section1Complete && !priceAutoOpenedRef.current && !form.isMaterialDelivery) {
      priceAutoOpenedRef.current = true;
      setPriceOpen(true);
    }
  }, [section1Complete, form.isMaterialDelivery]);

  const handlePickupAddressSelected = useCallback(async (address: RideAddressOption) => {
    form.selectPickupAddress(address);
    setActiveField(null);
    const enriched = await enrichAddressWithPlaceDetails(address);
    if (enriched !== address && hasValidCoords(enriched)) {
      form.selectPickupAddress(enriched);
      completedSimulationKeyRef.current = "";
    }
  }, [form]);

  const handleDropoffAddressSelected = useCallback(async (address: RideAddressOption) => {
    form.selectDropoffAddress(address);
    setActiveField(null);
    const enriched = await enrichAddressWithPlaceDetails(address);
    if (enriched !== address && hasValidCoords(enriched)) {
      form.selectDropoffAddress(enriched);
      completedSimulationKeyRef.current = "";
    }
  }, [form]);

  const handleClientSelected = (client: RideClientOption) => {
    Keyboard.dismiss();
    form.setClientId(client.id);
    setSelectedClientLabel(client.label);
    setSelectedClientSubtitle(client.pickupAddressCandidate?.label?.trim() ?? "");
    setActiveField(null);
    setSelectedClientPreferentialRate(
      isValidPreferentialAmount(client.preferentialRate) ? client.preferentialRate : null
    );
    setBillToPatient(false);
    if (client.pickupAddressCandidate) {
      void handlePickupAddressSelected(client.pickupAddressCandidate);
    }
    if (form.pickupAccessNotes.trim().length === 0 && client.pickupAccessNotes) {
      form.setPickupAccessNotes(client.pickupAccessNotes);
    }
    if (form.dropoffAccessNotes.trim().length === 0 && client.dropoffAccessNotes) {
      form.setDropoffAccessNotes(client.dropoffAccessNotes);
    }
    if (form.notesMedical.trim().length === 0 && client.notesMedical) {
      form.setNotesMedical(client.notesMedical);
    }
    if (form.establishment.trim().length === 0 && client.establishment) {
      form.setEstablishment(client.establishment);
      setExtraInfoOpen(true);
    }
    if (form.hospitalService.trim().length === 0 && client.hospitalService) {
      form.setHospitalService(client.hospitalService);
    }
    if (form.doctorName.trim().length === 0 && client.doctorName) {
      form.setDoctorName(client.doctorName);
    }
    if (!form.wheelchairClient && client.wheelchairClient) {
      form.setWheelchairClient(true);
      form.setWheelchairProvide(false);
    } else if (!form.wheelchairProvide && client.wheelchairProvide) {
      form.setWheelchairProvide(true);
      form.setWheelchairClient(false);
    }
    if (parseOptionalAmount(form.amountInput) == null && client.preferentialRate && client.preferentialRate > 0) {
      form.setAmountInput(
        resolvePreferentialBookingAmount(client.preferentialRate, form.isRoundTrip).toFixed(2),
      );
      setAmountSource("preferential");
      setAmountLocked(false);
    }
  };

  useEffect(() => {
    const detail = clientDetailQuery.data;
    if (!detail) return;
    const hydrationKey = [
      String(clientId ?? ""),
      String(Boolean(detail.hasActiveStay)),
      String(Boolean(detail.clinicAddress)),
      String(Boolean(detail.pickupAddressCandidate)),
      String(Boolean(billToPatient)),
    ].join("|");
    if (clientDetailHydrationKeyRef.current === hydrationKey) {
      return;
    }
    clientDetailHydrationKeyRef.current = hydrationKey;
    if (detail.hasActiveStay && !billToPatient && detail.clinicAddress) {
      void handlePickupAddressSelected(detail.clinicAddress);
      if (establishment.trim().length === 0 && detail.clinicName) setEstablishment(detail.clinicName);
      if (hospitalService.trim().length === 0 && detail.clinicService) {
        setHospitalService(detail.clinicService);
      }
      const accessHint = [detail.clinicFloor ? `Étage ${detail.clinicFloor}` : "", detail.clinicRoom ? `Chambre ${detail.clinicRoom}` : ""]
        .filter(Boolean)
        .join(" · ");
      if (pickupAccessNotes.trim().length === 0 && accessHint) setPickupAccessNotes(accessHint);
      setExtraInfoOpen(true);
    }
    if ((!detail.hasActiveStay || billToPatient) && detail.pickupAddressCandidate) {
      void handlePickupAddressSelected(detail.pickupAddressCandidate);
    }
    if (pickupAccessNotes.trim().length === 0 && detail.pickupAccessNotes) {
      setPickupAccessNotes(detail.pickupAccessNotes);
    }
    if (dropoffAccessNotes.trim().length === 0 && detail.dropoffAccessNotes) {
      setDropoffAccessNotes(detail.dropoffAccessNotes);
    }
    if (notesMedical.trim().length === 0 && detail.notesMedical) {
      setNotesMedical(detail.notesMedical);
    }
    if (establishment.trim().length === 0 && detail.establishment) {
      setEstablishment(detail.establishment);
      setExtraInfoOpen(true);
    }
    if (hospitalService.trim().length === 0 && detail.hospitalService) {
      setHospitalService(detail.hospitalService);
    }
    if (doctorName.trim().length === 0 && detail.doctorName) {
      setDoctorName(detail.doctorName);
    }
    if (!wheelchairClient && !wheelchairProvide) {
      if (detail.wheelchairClient) setWheelchairClient(true);
      if (detail.wheelchairProvide) setWheelchairProvide(true);
    }
    if (parseOptionalAmount(amountInput) == null && detail.preferentialRate && detail.preferentialRate > 0) {
      setAmountInput(
        resolvePreferentialBookingAmount(detail.preferentialRate, isRoundTrip).toFixed(2),
      );
      setAmountSource("preferential");
      setAmountLocked(false);
    }
  }, [
    amountInput,
    clientId,
    clientDetailQuery.data,
    doctorName,
    dropoffAccessNotes,
    establishment,
    hospitalService,
    isRoundTrip,
    notesMedical,
    pickup,
    pickupAccessNotes,
    wheelchairClient,
    wheelchairProvide,
    setAmountInput,
    setDoctorName,
    setDropoffAccessNotes,
    setEstablishment,
    setHospitalService,
    setNotesMedical,
    setPickupAccessNotes,
    setWheelchairClient,
    setWheelchairProvide,
    billToPatient,
    handlePickupAddressSelected,
  ]);

  useEffect(() => {
    if (!form.clientId) {
      setAmountSource(null);
      setAmountLocked(false);
      setPricingWarning("");
      setSelectedClientPreferentialRate(null);
      setBillToPatient(false);
      clientDetailHydrationKeyRef.current = "";
      completedSimulationKeyRef.current = "";
      activeSimulationKeyRef.current = "";
      simulationRequestSeqRef.current = 0;
      simulationCacheRef.current.clear();
      lastGeocodePickupKeyRef.current = "";
      lastGeocodeDropoffKeyRef.current = "";
    }
  }, [form.clientId]);

  useEffect(() => {
    let cancelled = false;
    const loadRouteData = async () => {
      setRoutePricingReady(false);
      if (!hasValidCoords(pickupAddress) || !hasValidCoords(dropoffAddress)) {
        setRoutePointsForPricing([]);
        setRouteDistanceMeters(null);
        setRouteDurationSeconds(null);
        if (!cancelled) setRoutePricingReady(true);
        return;
      }
      const pickupLat = Number(pickupAddress?.latitude);
      const pickupLon = Number(pickupAddress?.longitude);
      const dropoffLat = Number(dropoffAddress?.latitude);
      const dropoffLon = Number(dropoffAddress?.longitude);

      // Garde-fou supplémentaire : refuse les coordonnées dégénérées (0,0) qui
      // produisent un fallback haversine absurde côté backend (~180 km figés).
      const coordsLook =
        Number.isFinite(pickupLat) &&
        Number.isFinite(pickupLon) &&
        Number.isFinite(dropoffLat) &&
        Number.isFinite(dropoffLon) &&
        !(pickupLat === 0 && pickupLon === 0) &&
        !(dropoffLat === 0 && dropoffLon === 0);
      if (!coordsLook) {
        setRoutePointsForPricing([]);
        setRouteDistanceMeters(null);
        setRouteDurationSeconds(null);
        if (!cancelled) setRoutePricingReady(true);
        return;
      }

      const params = {
        pickup_lat: pickupLat,
        pickup_lon: pickupLon,
        dropoff_lat: dropoffLat,
        dropoff_lon: dropoffLon,
      };

      try {
        const { data } = await apiClient.get("/osrm/route", { params, timeout: 6000 });
        if (cancelled) return;

        const route = Array.isArray(data?.route)
          ? data.route
              .filter((pair: unknown) => Array.isArray(pair) && pair.length >= 2)
              .map((pair: unknown): { lat: number; lng: number } => {
                const [lat, lng] = pair as [number, number];
                return { lat: Number(lat), lng: Number(lng) };
              })
              .filter((pt: { lat: number; lng: number }) => Number.isFinite(pt.lat) && Number.isFinite(pt.lng))
          : [];
        setRoutePointsForPricing(route);

        const distanceRaw = Number(data?.distance);
        const durationRaw = Number(data?.duration);
        setRouteDistanceMeters(Number.isFinite(distanceRaw) && distanceRaw > 0 ? distanceRaw : null);
        setRouteDurationSeconds(Number.isFinite(durationRaw) && durationRaw > 0 ? durationRaw : null);
      } catch {
        if (cancelled) return;
        setRoutePointsForPricing([]);
        setRouteDistanceMeters(null);
        setRouteDurationSeconds(null);
      } finally {
        if (!cancelled) setRoutePricingReady(true);
      }
    };
    void loadRouteData();
    return () => {
      cancelled = true;
    };
  }, [pickupAddress, dropoffAddress]);

  useEffect(() => {
    if (!activeContextId) return;
    let cancelled = false;

    const normalizeRows = (payload: unknown): unknown[] => {
      if (Array.isArray(payload)) return payload;
      if (!payload || typeof payload !== "object") return [];
      const raw = payload as Record<string, unknown>;
      const buckets = [
        raw.items,
        raw.results,
        raw.data,
        raw.clients,
        raw.addresses,
        raw.features,
        raw.predictions,
        raw.suggestions,
      ];
      const list = buckets.find((entry) => Array.isArray(entry));
      return Array.isArray(list) ? list : [];
    };

    const resolveAddressCoords = async (
      kind: "pickup" | "dropoff",
      rawLabel: string,
      setResolved: (addr: RideAddressOption) => void
    ) => {
      const q = rawLabel.trim();
      if (q.length < 4) return;
      const key = `${kind}|${q.toLowerCase()}`;
      const keyRef = kind === "pickup" ? lastGeocodePickupKeyRef : lastGeocodeDropoffKeyRef;
      if (keyRef.current === key) return;
      keyRef.current = key;
      try {
        const payload = await searchCompanyAddresses({ contextId: activeContextId, q });
        if (cancelled) return;
        const rows = normalizeRows(payload)
          .map((row) => toResolvedAddressOption(row, q))
          .filter((row): row is RideAddressOption => row != null);
        const scored = rows
          .map((row) => {
            const label = (row.label || "").toLowerCase();
            const mainText = (row.mainText || "").toLowerCase();
            const startsWith = label.startsWith(q.toLowerCase()) || mainText.startsWith(q.toLowerCase());
            const hasCoords = hasValidCoords(row);
            const isGoogle = row.source === "google_places" || row.source === "google";
            const score = (hasCoords ? 1000 : 0) + (isGoogle ? 300 : 0) + (startsWith ? 150 : 0);
            return { row, score };
          })
          .sort((a, b) => b.score - a.score);
        const resolved = scored[0]?.row ?? null;
        if (!resolved || cancelled) return;
        setResolved(resolved);
        completedSimulationKeyRef.current = "";
      } catch {
        // Best effort: si l'auto-résolution échoue, l'utilisateur peut saisir manuellement.
      }
    };

    if (!hasValidCoords(pickupAddress) && pickup.trim().length > 0) {
      void resolveAddressCoords("pickup", pickup, (addr) => {
        void handlePickupAddressSelected(addr);
      });
    }
    if (!hasValidCoords(dropoffAddress) && form.dropoff.trim().length > 0) {
      void resolveAddressCoords("dropoff", form.dropoff, (addr) => {
        void handleDropoffAddressSelected(addr);
      });
    }

    return () => {
      cancelled = true;
    };
  }, [
    activeContextId,
    dropoffAddress,
    form.dropoff,
    form.selectDropoffAddress,
    pickup,
    pickupAddress,
    handlePickupAddressSelected,
    handleDropoffAddressSelected,
  ]);

  const activePreferentialAmount = useMemo(() => {
    if (!form.clientId || isMaterialDelivery) return null;
    const detail = clientDetailQuery.data;
    const detailRate = detail?.preferentialRate ?? null;

    // Si le client est hospitalisé et qu'on ne force pas la facturation patient,
    // priorité au tarif de la clinique/séjour.
    if (!billToPatient && detail?.hasActiveStay && isValidPreferentialAmount(detailRate)) {
      return detailRate;
    }

    // En facturation patient (ou sans séjour), utiliser le tarif préférentiel client si disponible.
    if (isValidPreferentialAmount(selectedClientPreferentialRate)) {
      return selectedClientPreferentialRate;
    }

    // Fallback sur le détail quand il n'y a pas de séjour actif explicite.
    if ((!detail || !detail.hasActiveStay) && isValidPreferentialAmount(detailRate)) {
      return detailRate;
    }

    return null;
  }, [
    billToPatient,
    clientDetailQuery.data,
    form.clientId,
    isMaterialDelivery,
    selectedClientPreferentialRate,
  ]);

  useEffect(() => {
    if (isMaterialDelivery) return;
    if (activePreferentialAmount != null) {
      if (!amountLocked) {
        setAmountInput(
          resolvePreferentialBookingAmount(activePreferentialAmount, isRoundTrip).toFixed(2),
        );
        setAmountSource("preferential");
        setPricingWarning("");
      }
      return;
    }
    if (!amountLocked && amountSource === "preferential") {
      setAmountInput("");
      setAmountSource(null);
    }
  }, [
    activePreferentialAmount,
    amountLocked,
    amountSource,
    isMaterialDelivery,
    isRoundTrip,
    setAmountInput,
  ]);

  useEffect(() => {
    if (isMaterialDelivery || amountLocked || amountSource === "preferential") return;
    if (!pickupAddress || !dropoffAddress || !scheduledOk) return;
    if (!routePricingReady) return;
    if (pricingContextQuery.isLoading) return;
    const pickupLat = Number(pickupAddress.latitude);
    const pickupLng = Number(pickupAddress.longitude);
    const dropoffLat = Number(dropoffAddress.latitude);
    const dropoffLng = Number(dropoffAddress.longitude);
    const hasValidCoords =
      Number.isFinite(pickupLat) &&
      Number.isFinite(pickupLng) &&
      Number.isFinite(dropoffLat) &&
      Number.isFinite(dropoffLng);
    if (!hasValidCoords) {
      setPricingWarning("Sélectionnez les adresses dans la liste pour calculer automatiquement le montant.");
      return;
    }
    const pricingProfileVersionId = pricingContextQuery.data?.pricingProfileVersionId;
    if (!pricingProfileVersionId) {
      setPricingWarning("Profil tarifaire introuvable: montant manuel requis.");
      return;
    }
    const simulationKey = [
      String(pricingProfileVersionId),
      normalizeScheduledTimeIso(scheduledAt),
      String(Boolean(isRoundTrip)),
      toRoundedCoord(pickupLat).toFixed(COORD_PRECISION),
      toRoundedCoord(pickupLng).toFixed(COORD_PRECISION),
      toRoundedCoord(dropoffLat).toFixed(COORD_PRECISION),
      toRoundedCoord(dropoffLng).toFixed(COORD_PRECISION),
      String(routePointsForPricing.length),
      String(routeDistanceMeters ?? ""),
    ].join("|");
    if (completedSimulationKeyRef.current === simulationKey) {
      return;
    }
    const cached = simulationCacheRef.current.get(simulationKey);
    if (cached && Date.now() - cached.cachedAt <= SIM_CACHE_TTL_MS) {
      setAmountInput(cached.amount.toFixed(2));
      setAmountSource("simulated");
      setPricingWarning(cached.warningMessage || "");
      completedSimulationKeyRef.current = simulationKey;
      return;
    }
    activeSimulationKeyRef.current = simulationKey;
    simulationRequestSeqRef.current += 1;
    const requestSeq = simulationRequestSeqRef.current;
    setPricingWarning("");
    const timer = setTimeout(() => {
      if (completedSimulationKeyRef.current === simulationKey) return;
      const payload = {
        pricing_profile_version_id: pricingProfileVersionId,
        booking: {
          pickup_at: normalizeScheduledTimeIso(scheduledAt),
          is_round_trip: isRoundTrip,
          pickup_lat: toRoundedCoord(pickupLat),
          pickup_lng: toRoundedCoord(pickupLng),
          dropoff_lat: toRoundedCoord(dropoffLat),
          dropoff_lng: toRoundedCoord(dropoffLng),
          route_points:
            Array.isArray(routePointsForPricing) && routePointsForPricing.length > 1
              ? routePointsForPricing
              : undefined,
        },
      };
      pricingSimulateMutateRef.current(payload, {
        onSuccess: (response) => {
          if (
            requestSeq !== simulationRequestSeqRef.current ||
            activeSimulationKeyRef.current !== simulationKey ||
            amountLockedRef.current ||
            amountSourceRef.current === "manual"
          ) {
            return;
          }
          const analysis = analyzePricingSimulation(response);
          if (analysis.warningMessage) {
            setPricingWarning(analysis.warningMessage);
          } else {
            setPricingWarning("");
          }
          if (analysis.blocked) {
            completedSimulationKeyRef.current = "";
            return;
          }
          const amount = analysis.amount ?? parseSimulationAmount(response);
          if (amount == null) {
            setPricingWarning("Calcul auto indisponible: saisissez un montant.");
            completedSimulationKeyRef.current = "";
            return;
          }
          setAmountInput(amount.toFixed(2));
          setAmountSource("simulated");
          completedSimulationKeyRef.current = simulationKey;
          simulationCacheRef.current.set(simulationKey, {
            amount,
            warningMessage: analysis.warningMessage,
            cachedAt: Date.now(),
          });
          pruneSimulationCache(simulationCacheRef.current);
        },
        onError: () => {
          if (
            requestSeq !== simulationRequestSeqRef.current ||
            activeSimulationKeyRef.current !== simulationKey ||
            amountLockedRef.current ||
            amountSourceRef.current === "manual"
          ) {
            return;
          }
          completedSimulationKeyRef.current = "";
          setPricingWarning("Calcul auto indisponible: saisissez un montant.");
        },
      });
    }, SIM_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [
    amountLocked,
    amountSource,
    dropoffAddress,
    isMaterialDelivery,
    isRoundTrip,
    pickupAddress,
    scheduledAt,
    pricingContextQuery.data?.pricingProfileVersionId,
    pricingContextQuery.isLoading,
    routeDistanceMeters,
    routePointsForPricing,
    routePricingReady,
    scheduledOk,
    setAmountInput,
  ]);

  const submit = async () => {
    if (!canSubmit) {
      if (!recurrenceValid && recurringOn) {
        setError(
          "Vérifiez la récurrence : au moins un jour en mode « Perso », nombre de répétitions entre 1 et 52, et date de fin au format JJ-MM-AAAA si renseignée.",
        );
      } else {
        setError("Renseignez le client, les lieux, la date/heure et respectez la limite des notes.");
      }
      return;
    }
    try {
      const resolveAddressForSubmit = async (
        label: string,
        current: RideAddressOption | null
      ): Promise<RideAddressOption | null> => {
        if (hasValidCoords(current)) {
          return current;
        }
        if (!activeContextId || label.trim().length < 4) {
          return current;
        }
        try {
          const payload = await searchCompanyAddresses({ contextId: activeContextId, q: label.trim() });
          const rows = (Array.isArray(payload) ? payload : []) as unknown[];
          const normalizedRows =
            rows.length > 0
              ? rows
              : payload && typeof payload === "object"
                ? ([
                    (payload as Record<string, unknown>).items,
                    (payload as Record<string, unknown>).results,
                    (payload as Record<string, unknown>).data,
                    (payload as Record<string, unknown>).addresses,
                    (payload as Record<string, unknown>).predictions,
                    (payload as Record<string, unknown>).suggestions,
                  ].find((entry) => Array.isArray(entry)) as unknown[] | undefined) ?? []
                : [];
          const candidates = normalizedRows
            .map((row) => toResolvedAddressOption(row, label))
            .filter((row): row is RideAddressOption => row != null)
            .map((row) => {
              const startsWith = row.label.toLowerCase().startsWith(label.trim().toLowerCase());
              const isGoogle = row.source === "google_places" || row.source === "google";
              const score = (hasValidCoords(row) ? 1000 : 0) + (startsWith ? 150 : 0) + (isGoogle ? 300 : 0);
              return { row, score };
            })
            .sort((a, b) => b.score - a.score);
          const best = candidates[0]?.row ?? current;
          if (!best) return null;
          return enrichAddressWithPlaceDetails(best);
        } catch {
          return current;
        }
      };

      const resolvedPickupAddress = await resolveAddressForSubmit(form.pickup, form.pickupAddress);
      const resolvedDropoffAddress = await resolveAddressForSubmit(form.dropoff, form.dropoffAddress);
      if (!hasValidCoords(resolvedPickupAddress) || !hasValidCoords(resolvedDropoffAddress)) {
        setError(
          "Coordonnées GPS introuvables pour le départ ou la destination. Sélectionnez une suggestion d’adresse avant de créer la réservation."
        );
        return;
      }
      if (resolvedPickupAddress && resolvedPickupAddress !== form.pickupAddress) {
        form.selectPickupAddress(resolvedPickupAddress);
      }
      if (resolvedDropoffAddress && resolvedDropoffAddress !== form.dropoffAddress) {
        form.selectDropoffAddress(resolvedDropoffAddress);
      }

      const debugTraceId = `ride_create_${Date.now()}`;
      const scheduled_time = normalizeScheduledTimeIso(form.scheduledAt);
      const normalizedReturnScheduledAt = normalizeScheduledTimeIso(form.returnScheduledAt);
      const payload = buildRideCreatePayload({
        structuredPayloadEnabled,
        clientId: form.clientId,
        pickup: form.pickup,
        dropoff: form.dropoff,
        pickupAddress: resolvedPickupAddress,
        dropoffAddress: resolvedDropoffAddress,
        scheduledTime: scheduled_time,
        isRoundTrip: form.isRoundTrip,
        recurrence: form.recurrence,
        notesMedical: form.notesMedical,
        establishment: form.establishment,
        hospitalService: form.hospitalService,
        doctorName: form.doctorName,
        pickupAccessNotes: form.pickupAccessNotes,
        dropoffAccessNotes: form.dropoffAccessNotes,
        wheelchairClient: form.wheelchairClient,
        wheelchairProvide: form.wheelchairProvide,
        internalNotes: form.internalNotes,
        notesMax: NOTES_MAX,
        amountInput: form.amountInput,
        amountSource,
        pricingProfileId: pricingContextQuery.data?.pricingProfileId ?? null,
        pricingProfileVersionId: pricingContextQuery.data?.pricingProfileVersionId ?? null,
        isMaterialDelivery: form.isMaterialDelivery,
        deliveryDescription: form.deliveryDescription,
        returnScheduledAt: normalizedReturnScheduledAt,
        billToPatient,
        hasActiveStay: Boolean(clientDetailQuery.data?.hasActiveStay),
        clinicBillingPartyId: clientDetailQuery.data?.clinicBillingPartyId ?? null,
        recurrenceLimitMode: recurrenceApiLimitMode,
        recurrenceOccurrences,
        recurrenceEndDate,
        recurrenceDays,
        recurrenceIntervalWeeks,
      });

      if (__DEV__) {
        console.log("[RideCreateModal] submit payload", {
          trace_id: debugTraceId,
          is_recurring: payload.is_recurring ?? false,
          recurrence_type: payload.recurrence_type ?? null,
          recurrence_days: payload.recurrence_days ?? null,
          recurrence_end_date: payload.recurrence_end_date ?? null,
          occurrences: payload.occurrences ?? null,
          recurrence_series_length: payload.recurrence_series_length ?? null,
          is_round_trip: payload.is_round_trip ?? false,
          return_date: payload.return_date ?? null,
          return_time: payload.return_time ?? null,
          scheduled_time: payload.scheduled_time ?? null,
        });
      }

      await createRide.mutateAsync(payload);
      form.reset();
      setSelectedClientLabel("");
      setSelectedClientSubtitle("");
      setAmountSource(null);
      setAmountLocked(false);
      setPricingWarning("");
      setBillToPatient(false);
      setExtraInfoOpen(false);
      setPriceOpen(false);
      setManualPriceOpen(false);
      priceAutoOpenedRef.current = false;
      setRouteDistanceMeters(null);
      setRouteDurationSeconds(null);
      setError(null);
      onCreated?.();
      onClose();
    } catch (e) {
      setError(toSubmitErrorMessage(e));
    }
  };

  const footerSummaryText = useMemo(
    () =>
      createRideMissingHint({
        hasClient: Boolean(form.clientId),
        hasPickup: form.pickup.trim().length > 0,
        hasDropoff: form.dropoff.trim().length > 0,
        hasSchedule: scheduledOk,
        hasAmount: form.isMaterialDelivery || amountValid,
      }),
    [amountValid, form.clientId, form.dropoff, form.isMaterialDelivery, form.pickup, scheduledOk]
  );

  const summaryData = useMemo(() => {
    const clientLabel = selectedClientLabel || (form.clientId ? `Client #${form.clientId}` : "");
    const pickupShort = form.pickup.trim().split(",")[0] || "";
    const dropoffShort = form.dropoff.trim().split(",")[0] || "";
    const dateLabel = scheduledOk ? formatSwissDateTime(form.scheduledAt) : "";
    const amount = parseOptionalAmount(form.amountInput);
    const amountText = form.isMaterialDelivery
      ? "Livraison"
      : amount != null
        ? `${amount.toFixed(2).replace(".", ",")} CHF`
        : "";
    return {
      client: clientLabel,
      pickup: pickupShort,
      dropoff: dropoffShort,
      date: dateLabel,
      amount: amountText,
    };
  }, [
    form.amountInput,
    form.clientId,
    form.dropoff,
    form.isMaterialDelivery,
    form.pickup,
    form.scheduledAt,
    scheduledOk,
    selectedClientLabel,
  ]);

  const amountBadgeMeta = useMemo(() => {
    if (!amountSource) return null;
    if (amountSource === "preferential") {
      return {
        label: isRoundTrip ? "Tarif préférentiel · total A/R" : "Tarif préférentiel",
        borderColor: "rgba(14, 116, 144, 0.34)",
        backgroundColor: "rgba(14, 116, 144, 0.10)",
        textColor: "#0E7490",
      };
    }
    if (amountSource === "simulated") {
      return {
        label: "Calculé automatiquement",
        borderColor: "rgba(0, 121, 107, 0.34)",
        backgroundColor: "rgba(0, 121, 107, 0.10)",
        textColor: E.BRAND,
      };
    }
    return {
      label: "Modifié manuellement",
      borderColor: "rgba(249, 115, 22, 0.36)",
      backgroundColor: "rgba(249, 115, 22, 0.10)",
      textColor: "#C2410C",
    };
  }, [amountSource, isRoundTrip]);

  const priceEstimateSubtext = useMemo(() => {
    if (amountSource === "preferential") {
      if (isRoundTrip && activePreferentialAmount != null) {
        const perLeg = activePreferentialAmount.toFixed(2).replace(".", ",");
        return `Tarif préférentiel · ${perLeg} CHF × 2 trajets`;
      }
      return "Tarif préférentiel · par trajet";
    }
    if (amountSource === "simulated") return "Tarif conseillé";
    if (pricingSimulation.isPending && !amountLocked) return "Calcul en cours…";
    if (pricingWarning.trim().length > 0) return pricingWarning;
    if (section1Complete && !routePricingReady) return "Calcul de l'itinéraire…";
    if (!scheduledOk) return "Renseignez la date et l'heure de départ";
    if (!pickupAddress || !dropoffAddress) return "Renseignez les adresses";
    return "En attente du calcul";
  }, [
    activePreferentialAmount,
    amountLocked,
    amountSource,
    isRoundTrip,
    pickupAddress,
    dropoffAddress,
    pricingSimulation.isPending,
    pricingWarning,
    routePricingReady,
    scheduledOk,
    section1Complete,
  ]);

  const header = () => (
    <View style={s.headerRow}>
      <Pressable
        onPress={onClose}
        style={BACK_BOX}
        accessibilityRole="button"
        accessibilityLabel="Retour"
        hitSlop={6}
      >
        <Ionicons name="chevron-back" size={20} color={E.BRAND} />
      </Pressable>
      <View style={s.headerCenter}>
        <AppText variant="sectionTitle" style={s.headerTitle}>
          Créer une réservation
        </AppText>
      </View>
      <Pressable
        onPress={() => setExtraInfoOpen((v) => !v)}
        style={s.headerIconBtn}
        accessibilityRole="button"
        accessibilityLabel="Aide et informations complémentaires"
        accessibilityHint="Ouvre la section informations complémentaires"
        hitSlop={6}
      >
        <Ionicons name="help-circle-outline" size={18} color={E.BRAND} />
      </Pressable>
    </View>
  );

  const summaryCriticalLines = isVeryLargeText ? undefined : 1;
  const summaryRowStyle = shouldStackRows ? [s.summaryRow, s.summaryRowStacked] : s.summaryRow;

  const footer = (
    <View style={s.footerCol}>
      {parentKeyboardActive ? null : (
      <View style={s.summaryPanel}>
        <View style={s.summaryPanelHeader}>
          <Ionicons name="receipt-outline" size={14} color={E.BRAND_DARK} />
          <AppText variant="label" style={s.summaryPanelTitle} scaleRole="chrome">Résumé</AppText>
          <View style={s.summaryPanelDivider} />
        </View>
        <View style={summaryRowStyle}>
          <View style={s.summaryCell}>
            <AppText variant="label" style={s.summaryCellLabel} scaleRole="chrome">Client</AppText>
            <AppText
              variant="body"
              style={[s.summaryCellValue, !summaryData.client ? s.summaryCellValueMuted : null]}
              numberOfLines={summaryCriticalLines}
            >
              {summaryData.client || "—"}
            </AppText>
          </View>
          <View style={s.summaryCell}>
            <AppText variant="label" style={s.summaryCellLabel} scaleRole="chrome">Date</AppText>
            <AppText
              variant="body"
              style={[s.summaryCellValue, !summaryData.date ? s.summaryCellValueMuted : null]}
              numberOfLines={summaryCriticalLines}
            >
              {summaryData.date || "—"}
            </AppText>
          </View>
        </View>
        <View style={summaryRowStyle}>
          <View style={s.summaryCell}>
            <AppText variant="label" style={s.summaryCellLabel} scaleRole="chrome">Départ</AppText>
            <AppText
              variant="body"
              style={[s.summaryCellValue, !summaryData.pickup ? s.summaryCellValueMuted : null]}
              numberOfLines={summaryCriticalLines}
            >
              {summaryData.pickup || "—"}
            </AppText>
          </View>
          <View style={s.summaryCell}>
            <AppText variant="label" style={s.summaryCellLabel} scaleRole="chrome">Destination</AppText>
            <AppText
              variant="body"
              style={[s.summaryCellValue, !summaryData.dropoff ? s.summaryCellValueMuted : null]}
              numberOfLines={summaryCriticalLines}
            >
              {summaryData.dropoff || "—"}
            </AppText>
          </View>
        </View>
        <View style={s.summaryPriceRow}>
          <AppText variant="label" style={s.summaryPriceLabel} scaleRole="chrome">Prix</AppText>
          <AppText variant="sectionTitle" style={s.summaryPriceValue}>{summaryData.amount || "—"}</AppText>
        </View>
      </View>
      )}
      <ModalFooterActions
        stacked={shouldStackRows}
        hint={
          !canSubmit && !createRide.isPending && !parentKeyboardActive ? (
            <AppText variant="caption" style={s.footerHint}>
              {footerSummaryText}
            </AppText>
          ) : null
        }
        secondary={
          <AppButton
            title="Annuler"
            variant="secondary"
            onPress={onClose}
            style={{ ...s.footerBtnSecondary, ...OUTLINE_SECONDARY }}
          />
        }
        primary={
          <AppButton
            title={createRide.isPending ? "Création…" : "Confirmer la réservation"}
            variant="primary"
            disabled={!canSubmit || createRide.isPending}
            loading={createRide.isPending}
            onPress={() => void submit()}
            style={s.footerBtnPrimary}
            leftIcon={
              <Ionicons
                name="checkmark-circle-outline"
                size={20}
                color={!canSubmit || createRide.isPending ? "rgba(255,255,255,0.85)" : "#fff"}
              />
            }
          />
        }
      />
    </View>
  );

  return (
    <>
      <Modal
        visible={visible}
        title=""
        onClose={onClose}
        presentation="bottomSheet"
        renderHeader={header}
        footer={footer}
        sheetBodyMaxHeightRatio={0.9}
      >
        <View style={s.form}>
          {/* ============================================== */}
          {/* Section 1 — Informations essentielles          */}
          {/* ============================================== */}
          <RideCreateSection
            number={1}
            title="Informations essentielles"
            gap={0}
            complete={section1Complete}
          >
            <View style={s.formGroup}>
              <View style={s.inlineActionRow}>
                <View style={s.inlineActionGrow}>
                  <CreateClientTrigger
                    selectedId={form.clientId}
                    selectedLabel={selectedClientLabel}
                    selectedSubtitle={selectedClientSubtitle || undefined}
                    onPress={() => setFieldActive("client", true)}
                    onClear={() => {
                      form.setClientId(null);
                      setSelectedClientLabel("");
                      setSelectedClientSubtitle("");
                      setSelectedClientPreferentialRate(null);
                      setFieldActive("client", true);
                    }}
                    leftSlot={<Ionicons name="person-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
                  />
                </View>
                <Pressable
                  onPress={() => form.setIsMaterialDelivery(!form.isMaterialDelivery)}
                  style={[
                    s.actionRoundBtn,
                    form.isMaterialDelivery ? s.actionRoundBtnActive : s.actionRoundBtnInactive,
                  ]}
                  accessibilityRole="button"
                  accessibilityState={{ selected: form.isMaterialDelivery }}
                  accessibilityLabel="Livraison de matériel"
                  hitSlop={6}
                >
                  <Ionicons
                    name="cube-outline"
                    size={18}
                    color={form.isMaterialDelivery ? "#FFFFFF" : E.TEXT_SEC}
                  />
                </Pressable>
              </View>
            </View>

            <>
              {form.isMaterialDelivery ? (
                <View style={s.formGroup}>
                <View style={s.inlineActionRow}>
                  <View style={s.inlineActionGrow}>
                    <AppInput
                      value={form.deliveryDescription}
                      onChangeText={form.setDeliveryDescription}
                      placeholder="Description de la livraison (ex : dossiers médicaux, matériel orthopédique…)"
                      accessibilityLabel="Description de la livraison"
                      leftSlot={
                        <Ionicons name="cube-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />
                      }
                      rightSlot={
                        <View
                          style={{ width: 16, alignItems: "center", justifyContent: "center" }}
                          accessibilityElementsHidden={form.deliveryDescription.trim().length !== 0}
                        >
                          {form.deliveryDescription.trim().length === 0 ? (
                            <AppText
                              variant="label"
                              accessibilityLabel="Champ obligatoire"
                              style={{
                                color: "#DC2626",
                                fontWeight: "700",
                                fontSize: 16,
                                lineHeight: 18,
                              }}
                            >
                              *
                            </AppText>
                          ) : null}
                        </View>
                      }
                      shellStyle={{
                        borderRadius: ROW_RADIUS,
                        minHeight: 48,
                        paddingHorizontal: 10,
                        backgroundColor: "#FAFBFA",
                      }}
                    />
                  </View>
                  <View
                    style={s.actionRoundBtnSpacer}
                    accessibilityElementsHidden
                    importantForAccessibility="no"
                  />
                </View>
                </View>
              ) : null}
              {clientDetailQuery.data?.hasActiveStay ? (
                <View style={[s.subCard, s.subCardBordered]}>
                  <View style={s.subCardTitleRow}>
                    <Ionicons name="medkit-outline" size={16} color={E.BRAND} />
                    <AppText variant="label" style={s.subCardTitle} numberOfLines={2}>
                      Client hospitalisé
                      {clientDetailQuery.data.clinicName ? ` · ${clientDetailQuery.data.clinicName}` : ""}
                    </AppText>
                  </View>
                  <AppText variant="caption" style={s.subCardHint}>
                    Départ établissement prioritaire.
                  </AppText>
                  <Pressable
                    onPress={() => setBillToPatient((v) => !v)}
                    style={[s.chip, billToPatient ? s.chipOn : s.chipOff]}
                    accessibilityRole="button"
                    accessibilityState={{ selected: billToPatient }}
                  >
                    <AppText
                      variant="label"
                      style={billToPatient ? s.chipLabelOn : s.chipLabelOff}
                    >
                      {billToPatient ? "Facturation patient (override)" : "Facturation clinique"}
                    </AppText>
                  </Pressable>
                </View>
              ) : null}

            <View style={s.formGroupDivider} />

            <View style={s.pickupDropoffSplit}>
              <View style={s.addressColumnLeft}>
                <AddressFieldTrigger
                  value={form.pickup}
                  placeholder="Adresse de départ…"
                  required
                  onPress={() => setFieldActive("pickup", true)}
                  onClear={() => form.setPickup("")}
                  leftSlot={<Ionicons name="navigate-outline" size={16} color={E.TEXT_SEC} />}
                />
                <AddressFieldTrigger
                  value={form.dropoff}
                  placeholder="Adresse de destination…"
                  required
                  onPress={() => setFieldActive("dropoff", true)}
                  onClear={() => form.setDropoff("")}
                  leftSlot={<Ionicons name="location-outline" size={16} color={E.TEXT_SEC} />}
                />
              </View>
              <View style={s.addressActionsColumn}>
                <Animated.View style={swapRotateStyle}>
                  <Pressable
                    onPress={handleSwapAddresses}
                    style={s.swapBtnRound}
                    accessibilityRole="button"
                    accessibilityLabel="Inverser pickup/destination"
                    hitSlop={6}
                  >
                    <Ionicons name="swap-vertical-outline" size={18} color={E.BRAND} />
                  </Pressable>
                </Animated.View>
                <Pressable
                  onPress={() => {
                    const nextRound = !form.isRoundTrip;
                    form.setIsRoundTrip(nextRound);
                    if (nextRound) setExtraInfoOpen(true);
                    if (
                      !amountLocked &&
                      amountSource === "preferential" &&
                      activePreferentialAmount != null
                    ) {
                      setAmountInput(
                        resolvePreferentialBookingAmount(
                          activePreferentialAmount,
                          nextRound,
                        ).toFixed(2),
                      );
                    }
                  }}
                  style={[
                    s.actionRoundBtn,
                    form.isRoundTrip ? s.actionRoundBtnActive : s.actionRoundBtnInactive,
                  ]}
                  accessibilityRole="button"
                  accessibilityState={{ selected: form.isRoundTrip }}
                  accessibilityLabel="Aller-retour"
                  hitSlop={6}
                >
                  <Ionicons
                    name="repeat-outline"
                    size={18}
                    color={form.isRoundTrip ? "#FFFFFF" : E.TEXT_SEC}
                  />
                </Pressable>
              </View>
            </View>

            <View style={{ marginTop: 10 }}>
              <RideRoutePreview
                pickupLat={pickupAddress ? Number(pickupAddress.latitude) : null}
                pickupLng={pickupAddress ? Number(pickupAddress.longitude) : null}
                dropoffLat={dropoffAddress ? Number(dropoffAddress.latitude) : null}
                dropoffLng={dropoffAddress ? Number(dropoffAddress.longitude) : null}
                routePoints={routePointsForPricing}
                distanceMeters={routeDistanceMeters}
                durationSeconds={routeDurationSeconds}
                routeKind="Le plus rapide"
              />
            </View>

            <View style={s.formGroupDivider} />

            <View style={s.formGroup}>
            <View style={s.inlineActionRow}>
              <View style={s.inlineActionGrow}>
                <TimeDatePicker
                  value={form.scheduledAt}
                  onChange={form.setScheduledAt}
                  label=""
                  display="split"
                  required
                />
              </View>
              <Pressable
                onPress={() => {
                  if (recurringOn) {
                    form.setRecurrence("none");
                  } else {
                    form.setRecurrence("daily");
                    setExtraInfoOpen(true);
                  }
                }}
                style={[
                  s.actionRoundBtn,
                  recurringOn ? s.actionRoundBtnActive : s.actionRoundBtnInactive,
                ]}
                accessibilityRole="button"
                accessibilityState={{ selected: recurringOn }}
                accessibilityLabel="Course récurrente"
                hitSlop={6}
              >
                <Ionicons
                  name="calendar-outline"
                  size={18}
                  color={recurringOn ? "#FFFFFF" : E.TEXT_SEC}
                />
              </Pressable>
            </View>

            {form.isRoundTrip ? (
              <View style={s.inlineActionRow}>
                <View style={s.inlineActionGrow}>
                  <TimeDatePicker
                    value={form.returnScheduledAt}
                    onChange={form.setReturnScheduledAt}
                    label=""
                    display="split"
                    emptyLabel="À définir"
                    emptyPreviewReferenceIso={form.scheduledAt}
                    modalTitle="Heure de retour"
                    accessibilityLabel="Choisir la date et l’heure de retour"
                    timeAccessibilityLabel="Choisir l’heure de retour"
                    tonal
                  />
                </View>
                <View
                  style={s.actionRoundBtnSpacer}
                  accessibilityElementsHidden
                  importantForAccessibility="no"
                />
              </View>
            ) : null}

            {recurringOn ? (
              <View style={[s.subCard, s.tonalGroup]}>
                <RecurrenceSelector
                  showLabel={false}
                  value={
                    form.recurrence === "daily" ||
                    form.recurrence === "weekly" ||
                    form.recurrence === "custom"
                      ? form.recurrence
                      : "daily"
                  }
                  onChange={(v) => {
                    form.setRecurrence(v);
                    if (v !== "custom" && v !== "weekly") setRecurrenceDays([]);
                    if (v !== "custom") setRecurrenceIntervalWeeks(2);
                  }}
                />

                <View style={s.recurrenceModeCard}>
                  <View style={s.recurrenceModeIconWrap}>
                    <Ionicons
                      name={
                        form.recurrence === "daily"
                          ? "calendar-outline"
                          : form.recurrence === "weekly"
                            ? "calendar-clear-outline"
                            : "options-outline"
                      }
                      size={18}
                      color={E.BRAND_DARK}
                    />
                  </View>
                  <View style={{ flex: 1, minWidth: 0 }}>
                    <AppText variant="label" style={s.recurrenceModeTitle}>
                      {form.recurrence === "daily"
                        ? "Tous les jours"
                        : form.recurrence === "weekly"
                          ? "Certaines journées"
                          : "Règles personnalisées"}
                    </AppText>
                    <AppText variant="caption" style={s.recurrenceModeDesc}>
                      {form.recurrence === "daily"
                        ? "Une réservation sera créée chaque jour jusqu'à la date de fin."
                        : form.recurrence === "weekly"
                          ? "Choisissez les jours de la semaine où la réservation doit se répéter."
                          : "Créez des règles de récurrence avancées selon vos besoins."}
                    </AppText>
                  </View>
                </View>

                {form.recurrence === "custom" ? (
                  <View style={{ gap: 6 }}>
                    <AppText variant="label" style={s.recurrenceDateFieldLabel}>
                      Répéter
                    </AppText>
                    <AppSelect
                      value={String(recurrenceIntervalWeeks)}
                      onChange={(v) => {
                        const n = Math.max(1, Math.min(12, Math.floor(Number(v) || 1)));
                        setRecurrenceIntervalWeeks(n);
                      }}
                      options={[
                        { value: "1", label: "Toutes les semaines" },
                        { value: "2", label: "Toutes les 2 semaines" },
                        { value: "3", label: "Toutes les 3 semaines" },
                        { value: "4", label: "Toutes les 4 semaines" },
                        { value: "6", label: "Toutes les 6 semaines" },
                        { value: "8", label: "Toutes les 8 semaines" },
                        { value: "12", label: "Toutes les 12 semaines" },
                      ]}
                    />
                  </View>
                ) : null}

                {form.recurrence === "custom" || form.recurrence === "weekly" ? (
                  <View style={{ gap: 8 }}>
                    <AppText variant="label" style={s.recurrenceSubLabel}>
                      {form.recurrence === "custom" ? "Jours concernés" : "Sélectionnez les jours"}
                      {" ("}
                      {recurrenceSelectedDaysCount} sélectionné
                      {recurrenceSelectedDaysCount > 1 ? "s" : ""})
                    </AppText>
                    <View style={s.recurrenceWeekdayRow}>
                      {WEEKDAY_SHORT.map((label, day) => {
                        const on = recurrenceDays.includes(day);
                        return (
                          <Pressable
                            key={label}
                            onPress={() => toggleRecurrenceWeekday(day)}
                            style={[
                              s.recurrenceWeekdayChip,
                              on ? s.recurrenceWeekdayChipOn : s.recurrenceWeekdayChipOff,
                            ]}
                            accessibilityRole="button"
                            accessibilityState={{ selected: on }}
                            accessibilityLabel={`${label}${on ? ", sélectionné" : ""}`}
                          >
                            <AppText
                              variant="label"
                              style={
                                on
                                  ? s.recurrenceWeekdayChipTextOn
                                  : s.recurrenceWeekdayChipTextOff
                              }
                            >
                              {label}
                            </AppText>
                          </Pressable>
                        );
                      })}
                    </View>
                    {recurrenceSelectedDaysCount === 0 ? (
                      <AppText
                        variant="caption"
                        style={{ color: "#B91C1C", fontSize: FONT_SIZE.px12, fontWeight: "600" }}
                      >
                        ⚠️ Veuillez sélectionner au moins un jour
                      </AppText>
                    ) : null}
                  </View>
                ) : null}

                <View style={s.recurrenceDateField}>
                  <AppText variant="label" style={s.recurrenceDateFieldLabel}>
                    Date de fin
                  </AppText>
                  <TimeDatePicker
                    value={recurrenceEndDate ? `${recurrenceEndDate}T12:00:00` : ""}
                    onChange={(v) => {
                      const n = normalizeScheduledTimeIso(v);
                      const [d] = n.split("T");
                      if (d) setRecurrenceEndDate(d);
                    }}
                    dateOnly
                    label=""
                    emptyLabel="JJ-MM-AAAA"
                    modalTitle="Date de fin"
                    accessibilityLabel="Choisir la date de fin de récurrence"
                    tonal
                  />
                </View>
                {recurrenceEndDate.trim().length > 0 && !recurrenceEndValid ? (
                  <AppText
                    variant="caption"
                    style={{ color: "#B91C1C", fontSize: FONT_SIZE.px12, fontWeight: "600" }}
                  >
                    ⚠️ Format attendu: JJ-MM-AAAA
                  </AppText>
                ) : null}

                <View style={s.recurrenceSummaryCard}>
                  <View style={s.recurrenceSummaryHeader}>
                    <Ionicons
                      name={recurrencePreview.total > 0 ? "checkmark-circle" : "time-outline"}
                      size={16}
                      color={recurrencePreview.total > 0 ? E.BRAND : E.TEXT_MUTED}
                    />
                    <AppText variant="caption" style={s.recurrenceSummaryEyebrow}>
                      Résumé de la récurrence
                    </AppText>
                  </View>
                  <View style={s.recurrenceSummaryBody}>
                    <View style={{ flex: 1, minWidth: 0 }}>
                      <AppText variant="label" style={s.recurrenceSummaryTitle}>
                        {recurrencePreview.total > 0
                          ? recurrenceLongLabel
                          : !scheduledOk
                            ? "En attente"
                            : form.recurrence === "custom" && recurrenceSelectedDaysCount === 0
                              ? "Sélectionnez au moins un jour"
                              : recurrenceLongLabel || "Récurrence"}
                      </AppText>
                      <AppText variant="caption" style={s.recurrenceSummaryWindow}>
                        {recurrencePreview.total > 0
                          ? recurrenceWindowLabel
                          : !scheduledOk
                            ? "Choisissez la date et l'heure de départ pour voir le résumé."
                            : form.recurrence === "custom" && recurrenceSelectedDaysCount === 0
                              ? "Aucun jour sélectionné dans la semaine."
                              : "Saisissez une date de fin pour calculer la série."}
                      </AppText>
                    </View>
                    <View
                      style={[
                        s.recurrenceCountBadge,
                        recurrencePreview.total === 0 && s.recurrenceCountBadgePending,
                      ]}
                    >
                      <AppText
                        variant="label"
                        style={[
                          s.recurrenceCountBadgeNum,
                          recurrencePreview.total === 0 && s.recurrenceCountBadgeNumPending,
                        ]}
                      >
                        {recurrencePreview.total > 0 ? recurrencePreview.total : "—"}
                      </AppText>
                      <AppText
                        variant="caption"
                        style={[
                          s.recurrenceCountBadgeLabel,
                          recurrencePreview.total === 0 && s.recurrenceCountBadgeLabelPending,
                        ]}
                      >
                        {recurrencePreview.total > 1 ? "trajets" : "trajet"}{"\n"}seront créés
                      </AppText>
                    </View>
                  </View>
                </View>

                {recurrencePreview.dates.length > 0 ? (
                  <View style={{ gap: 6 }}>
                    <AppText variant="caption" style={s.recurrencePreviewEyebrow}>
                      Aperçu des prochaines dates
                    </AppText>
                    <View style={s.recurrencePreviewRow}>
                      {recurrencePreview.dates.slice(0, 5).map((d, i) => {
                        const weekday = d
                          .toLocaleDateString("fr-CH", { weekday: "short" })
                          .replace(".", "")
                          .replace(/^\w/, (c) => c.toUpperCase());
                        const day = d.toLocaleDateString("fr-CH", { day: "2-digit" });
                        const month = d
                          .toLocaleDateString("fr-CH", { month: "short" })
                          .replace(".", "")
                          .toUpperCase();
                        return (
                          <View key={`${d.toISOString()}-${i}`} style={s.recurrencePreviewChip}>
                            <AppText variant="caption" style={s.recurrencePreviewChipWd}>
                              {weekday}
                            </AppText>
                            <AppText variant="label" style={s.recurrencePreviewChipDay}>
                              {day}
                            </AppText>
                            <AppText variant="caption" style={s.recurrencePreviewChipMonth}>
                              {month}
                            </AppText>
                          </View>
                        );
                      })}
                      {recurrencePreview.total > 5 ? (
                        <View style={[s.recurrencePreviewChip, s.recurrencePreviewChipMore]}>
                          <AppText variant="label" style={s.recurrencePreviewChipMoreNum}>
                            + {recurrencePreview.total - 5}
                          </AppText>
                          <AppText variant="caption" style={s.recurrencePreviewChipMoreLabel}>
                            autres
                          </AppText>
                        </View>
                      ) : null}
                    </View>
                  </View>
                ) : null}

                <View style={s.recurrenceHint}>
                  <Ionicons name="information-circle-outline" size={18} color={E.BRAND} />
                  <AppText variant="caption" style={s.recurrenceHintText}>
                    {recurrencePreview.total > 0
                      ? `Une réservation sera créée ${
                          recurrenceLongLabel
                            ? recurrenceLongLabel.charAt(0).toLowerCase() +
                              recurrenceLongLabel.slice(1)
                            : ""
                        }. ${recurrencePreview.total} réservation${recurrencePreview.total > 1 ? "s" : ""} seront créées au total.`
                      : recurrenceSummary}
                  </AppText>
                </View>
              </View>
            ) : null}
            </View>
            </>
          </RideCreateSection>

          <View style={s.sectionDivider} />

          {/* ============================================== */}
          {/* Section 2 — Prix de la course                  */}
          {/* ============================================== */}
          {!form.isMaterialDelivery ? (
            <RideCreateSection
              number={2}
              title="Prix de la course"
              gap={10}
              complete={section3Complete}
              open={priceOpen}
              hideBody={!priceOpen}
              onTogglePress={() => setPriceOpen((v) => !v)}
            >
              <View style={s.priceRow}>
                <Pressable
                  onPress={() => {
                    setManualPriceOpen(false);
                    if (amountLocked) {
                      setAmountLocked(false);
                      setAmountSource(null);
                    }
                    completedSimulationKeyRef.current = "";
                    activeSimulationKeyRef.current = "";
                    setPricingWarning("");
                  }}
                  style={s.priceCardEstimate}
                  accessibilityRole="button"
                  accessibilityLabel="Utiliser l'estimation automatique"
                  hitSlop={4}
                >
                  <View style={s.priceCardLabelRow}>
                    <AppText variant="label" style={[s.priceCardLabel, s.priceCardLabelActive]}>
                      Estimation auto
                    </AppText>
                    {amountSource !== "manual" ? (
                      <View style={s.priceCardBadgeRecommended}>
                        <AppText variant="label" style={s.priceCardBadgeRecommendedText}>Recommandé</AppText>
                      </View>
                    ) : null}
                  </View>
                  <View style={s.priceCardAmountRow}>
                    <AppText variant="sectionTitle" style={s.priceCardAmount}>
                      {amountValue != null ? amountValue.toFixed(2).replace(".", ",") : "—"}
                    </AppText>
                    <AppText variant="label" style={s.priceCardAmountUnit}>CHF</AppText>
                  </View>
                  <AppText variant="caption" style={s.priceCardSubtext} numberOfLines={2}>
                    {priceEstimateSubtext}
                  </AppText>
                </Pressable>
                <Pressable
                  onPress={() => setManualPriceOpen((v) => !v)}
                  style={[s.priceCardManual, manualPriceOpen && s.priceCardManualActive]}
                  accessibilityRole="button"
                  accessibilityLabel="Saisir un montant manuellement"
                  hitSlop={4}
                >
                  <View style={s.priceCardManualCol}>
                    <AppText
                      variant="body"
                      style={[s.priceCardManualLabel, manualPriceOpen && s.priceCardManualLabelActive]}
                      numberOfLines={1}
                    >
                      Saisie manuelle
                    </AppText>
                    <AppText variant="caption" style={s.priceCardManualHint} numberOfLines={1}>
                      Définir un montant
                    </AppText>
                  </View>
                  <Ionicons
                    name={manualPriceOpen ? "chevron-up" : "chevron-forward"}
                    size={14}
                    color={manualPriceOpen ? E.BRAND_DARK : E.TEXT_SEC}
                  />
                </Pressable>
              </View>
              {manualPriceOpen ? (
                <View style={s.manualEditWrap}>
                  <AppInput
                    label="Montant *"
                    value={form.amountInput}
                    onChangeText={(value) => {
                      form.setAmountInput(value);
                      setAmountSource("manual");
                      setAmountLocked(true);
                    }}
                    placeholder="Ex : 45.00"
                    keyboardType="decimal-pad"
                    leftSlot={<Ionicons name="cash-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
                    shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
                  />
                  {amountLocked ? (
                    <Pressable
                      onPress={() => {
                        setAmountLocked(false);
                        setAmountSource(null);
                        completedSimulationKeyRef.current = "";
                      }}
                      style={s.linkNewClient}
                      accessibilityRole="button"
                      accessibilityLabel="Réactiver le calcul automatique du montant"
                    >
                      <AppText variant="label" style={{ color: E.BRAND, fontWeight: "600" }}>
                        Recalculer automatiquement
                      </AppText>
                    </Pressable>
                  ) : null}
                </View>
              ) : null}
              {amountBadgeMeta ? (
                <View style={s.amountMetaRow}>
                  <View
                    style={[
                      s.amountBadge,
                      {
                        borderColor: amountBadgeMeta.borderColor,
                        backgroundColor: amountBadgeMeta.backgroundColor,
                      },
                    ]}
                  >
                    <AppText
                      variant="caption"
                      style={[s.amountBadgeText, { color: amountBadgeMeta.textColor }]}
                    >
                      {amountBadgeMeta.label}
                    </AppText>
                  </View>
                </View>
              ) : null}
              {pricingSimulation.isPending && !amountLocked ? (
                <AppText variant="caption" style={s.sectionHelper}>
                  {amountSource === "simulated" && form.amountInput.trim().length > 0
                    ? "Mise à jour du montant exact en cours…"
                    : "Calcul du montant en cours…"}
                </AppText>
              ) : null}
              {pricingWarning && amountSource != null ? (
                <AppText variant="caption" style={s.sectionHelper}>{pricingWarning}</AppText>
              ) : null}
            </RideCreateSection>
          ) : null}

          <View style={s.sectionDivider} />

          {/* ============================================== */}
          {/* Section 3 — Informations complémentaires       */}
          {/* ============================================== */}
          <RideCreateSection
            number={3}
            title="Informations complémentaires"
            subtitle="Médical, notes internes…"
            open={extraInfoOpen}
            hideBody={!extraInfoOpen}
            onTogglePress={() => setExtraInfoOpen((v) => !v)}
            gap={14}
          >
            <View style={s.subCard}>
              <View style={{ gap: 8 }}>
                  <AppInput
                    value={form.establishment}
                    onChangeText={form.setEstablishment}
                    placeholder="Établissement (optionnel)"
                    leftSlot={<Ionicons name="business-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
                    shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
                  />
                  <AppInput
                    value={form.hospitalService}
                    onChangeText={form.setHospitalService}
                    placeholder="Service hospitalier (optionnel)"
                    leftSlot={<Ionicons name="medkit-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
                    shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
                  />
                  <AppInput
                    value={form.doctorName}
                    onChangeText={form.setDoctorName}
                    placeholder="Médecin référent (optionnel)"
                    leftSlot={<Ionicons name="person-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
                    shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
                  />
                  <AppInput
                    value={form.notesMedical}
                    onChangeText={form.setNotesMedical}
                    placeholder="Instructions particulières, bâtiment, étage…"
                    multiline
                    textAlignVertical="top"
                    shellStyle={{
                      borderRadius: ROW_RADIUS,
                      minHeight: COMPACT_MULTILINE_MEDIUM_HEIGHT,
                      alignItems: "flex-start",
                      backgroundColor: "#FFFFFF",
                    }}
                    style={{ minHeight: COMPACT_MULTILINE_MEDIUM_INPUT_HEIGHT }}
                  />
                  <AppInput
                    value={form.pickupAccessNotes}
                    onChangeText={form.setPickupAccessNotes}
                    placeholder="Comment accéder au point de départ ?"
                    accessibilityLabel="Accès au point de départ"
                    leftSlot={<Ionicons name="navigate-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
                    shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
                  />
                  <AppInput
                    value={form.dropoffAccessNotes}
                    onChangeText={form.setDropoffAccessNotes}
                    placeholder="Comment accéder à la destination ?"
                    accessibilityLabel="Accès à la destination"
                    leftSlot={<Ionicons name="location-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
                    shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
                  />
                <View style={s.wheelchairRow}>
                  <Pressable
                    onPress={() => form.setWheelchairClient(!form.wheelchairClient)}
                    style={[
                      s.chip,
                      form.wheelchairClient ? s.chipBlueOn : s.chipBlueOff,
                    ]}
                    accessibilityRole="button"
                    accessibilityState={{ selected: form.wheelchairClient }}
                    accessibilityLabel="Client en chaise roulante"
                  >
                    <MaterialCommunityIcons
                      name="human-wheelchair"
                      size={17}
                      color={form.wheelchairClient ? E.TRANSFER : E.TEXT_SEC}
                    />
                    <AppText
                      variant="label"
                      style={
                        form.wheelchairClient ? s.chipBlueLabelOn : s.chipLabelOff
                      }
                    >
                      En chaise
                    </AppText>
                  </Pressable>
                  <Pressable
                    onPress={() => form.setWheelchairProvide(!form.wheelchairProvide)}
                    style={[
                      s.chip,
                      form.wheelchairProvide ? s.chipOrangeOn : s.chipOrangeOff,
                    ]}
                    accessibilityRole="button"
                    accessibilityState={{ selected: form.wheelchairProvide }}
                    accessibilityLabel="Fournir une chaise roulante"
                  >
                    <MaterialCommunityIcons
                      name="wheelchair"
                      size={17}
                      color={form.wheelchairProvide ? E.URGENT : E.TEXT_SEC}
                    />
                    <AppText
                      variant="label"
                      style={
                        form.wheelchairProvide
                          ? s.chipOrangeLabelOn
                          : s.chipLabelOff
                      }
                    >
                      Fournir chaise
                    </AppText>
                  </Pressable>
                </View>
              </View>
            </View>

          </RideCreateSection>

          {error ? (
            <AppText variant="error" style={s.error}>
              {error}
            </AppText>
          ) : null}
        </View>
      </Modal>
      <ClientPickerSheet
        visible={activeField === "client"}
        selectedId={form.clientId}
        onClose={() => setFieldActive("client", false)}
        onSelect={(client) => {
          handleClientSelected(client);
        }}
        onCreateClient={() => {
          Keyboard.dismiss();
          setFieldActive("client", false);
          setCreateClientVisible(true);
        }}
      />
      <AddressPickerSheet
        visible={activeField === "pickup"}
        title="Adresse de départ"
        value={form.pickup}
        onClose={() => setFieldActive("pickup", false)}
        onChange={form.setPickup}
        onSelect={(address) => {
          Keyboard.dismiss();
          void handlePickupAddressSelected(address);
        }}
      />
      <AddressPickerSheet
        visible={activeField === "dropoff"}
        title="Adresse de destination"
        value={form.dropoff}
        onClose={() => setFieldActive("dropoff", false)}
        onChange={form.setDropoff}
        onSelect={(address) => {
          Keyboard.dismiss();
          void handleDropoffAddressSelected(address);
          const hints = parseMedicalHintsFromAddress(address.label);
          if (hints.establishment && form.establishment.trim().length === 0) {
            form.setEstablishment(hints.establishment);
            setExtraInfoOpen(true);
          }
          if (hints.doctorName && form.doctorName.trim().length === 0) {
            form.setDoctorName(hints.doctorName);
            setExtraInfoOpen(true);
          }
          if (hints.hospitalService && form.hospitalService.trim().length === 0) {
            form.setHospitalService(hints.hospitalService);
            setExtraInfoOpen(true);
          }
          if (hints.notesMedical && form.notesMedical.trim().length === 0) {
            form.setNotesMedical(hints.notesMedical);
          }
        }}
      />
      <ClientCreateModal
        visible={createClientVisible}
        onClose={() => setCreateClientVisible(false)}
        onCreated={() => setError(null)}
      />
    </>
  );
}
