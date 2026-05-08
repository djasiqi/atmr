import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AxiosError } from "axios";
import { AppButton, Modal } from "../../../../design/responsive";
import { AppInput } from "../../../../design/ui/AppInput";
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
import { AddressSelector } from "./AddressSelector";
import { ClientSelector } from "./ClientSelector";
import { RecurrenceSelector } from "./RecurrenceSelector";
import { TimeDatePicker } from "./TimeDatePicker";
import { ClientCreateModal } from "./ClientCreateModal";
import {
  analyzePricingSimulation,
  backendWeekdayFromScheduledIso,
  buildRideCreatePayload,
  parseMedicalHintsFromAddress,
  parseSimulationAmount,
} from "./rideCreateHelpers";

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
const COMPACT_CHIP_HEIGHT = 42;
const COMPACT_CHIP_SMALL_HEIGHT = 40;
const COMPACT_ACTION_HEIGHT = 46;
const COMPACT_MULTILINE_MEDIUM_HEIGHT = 72;
const COMPACT_MULTILINE_LARGE_HEIGHT = 88;
const COMPACT_MULTILINE_MEDIUM_INPUT_HEIGHT = 56;
const COMPACT_MULTILINE_LARGE_INPUT_HEIGHT = 72;
const FIELD_ICON_SIZE = 18;
const BACK_BOX = {
  width: 40,
  height: 40,
  borderRadius: 12,
  backgroundColor: "transparent",
  alignItems: "center" as const,
  justifyContent: "center" as const,
};

const s = StyleSheet.create({
  form: { gap: 12 },
  sectionBlock: { gap: 6 },
  sectionLabel: {
    fontSize: 13,
    fontWeight: "600" as const,
    color: E.TEXT,
    marginBottom: 2,
  },
  sectionHelper: {
    fontSize: 12,
    color: E.TEXT_MUTED,
    lineHeight: 17,
  },
  sectionDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(148, 163, 184, 0.24)",
    marginVertical: 0,
  },
  headerRow: {
    flexDirection: "row" as const,
    alignItems: "flex-start" as const,
    gap: 12,
    marginBottom: 2,
    paddingBottom: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "rgba(148, 163, 184, 0.28)",
  },
  headerCenter: { flex: 1, gap: 4 },
  headerTitle: {
    fontSize: 18,
    fontWeight: "700" as const,
    color: E.TEXT,
    letterSpacing: 0.15,
  },
  toggleRow: {
    flexDirection: "row" as const,
    flexWrap: "wrap" as const,
    gap: 5,
    alignItems: "center" as const,
  },
  chip: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
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
  chipLabelOn: { color: E.BRAND, fontWeight: "700" as const, fontSize: 13, lineHeight: 16 },
  chipLabelOff: { color: E.TEXT_SEC, fontWeight: "600" as const, fontSize: 13, lineHeight: 16 },
  card: {
    borderRadius: ROW_RADIUS,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.35)",
    backgroundColor: "#FAFBFA",
    paddingHorizontal: 12,
    paddingVertical: 8,
    gap: 6,
  },
  cardRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "space-between" as const,
    gap: 12,
  },
  cardLabel: { fontSize: 12, color: E.TEXT_MUTED },
  cardValue: {
    fontSize: 13,
    color: E.TEXT,
    fontWeight: "600" as const,
  },
  recurrenceHeader: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 8,
    marginBottom: 0,
  },
  recurrenceSectionTitle: {
    fontSize: 14,
    fontWeight: "700" as const,
    color: E.TEXT,
    letterSpacing: 0.15,
  },
  recurrenceMetaSheet: {
    borderRadius: ROW_RADIUS,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.35)",
    backgroundColor: "#FFFFFF",
    overflow: "hidden" as const,
  },
  recurrenceMetaRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "space-between" as const,
    gap: 12,
    paddingVertical: 9,
    paddingHorizontal: 12,
  },
  recurrenceMetaLeft: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 6,
    flexShrink: 1,
  },
  recurrenceMetaIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  recurrenceMetaLabel: {
    fontSize: 13,
    fontWeight: "600" as const,
    color: E.TEXT,
    lineHeight: 16,
  },
  recurrenceMetaValue: {
    fontSize: 13,
    fontWeight: "700" as const,
    color: E.BRAND,
    lineHeight: 16,
  },
  recurrenceMetaValueMuted: {
    fontSize: 13,
    fontWeight: "600" as const,
    color: E.TEXT_MUTED,
    fontStyle: "italic" as const,
    lineHeight: 16,
  },
  recurrenceMetaDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(148, 163, 184, 0.35)",
    marginLeft: 52,
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
    fontSize: 12,
    color: E.TEXT_SEC,
    lineHeight: 17,
    fontWeight: "500" as const,
  },
  recurrenceSubLabel: {
    fontSize: 13,
    fontWeight: "700" as const,
    color: E.TEXT,
    marginTop: 2,
    marginBottom: 0,
  },
  recurrenceLimitRow: {
    flexDirection: "row" as const,
    flexWrap: "wrap" as const,
    gap: 6,
    alignItems: "center" as const,
  },
  recurrenceLimitChip: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: COMPACT_CONTROL_RADIUS,
    borderWidth: 1,
    minHeight: COMPACT_CHIP_SMALL_HEIGHT,
  },
  recurrenceLimitChipOn: {
    backgroundColor: "rgba(0, 121, 107, 0.14)",
    borderColor: E.BRAND,
  },
  recurrenceLimitChipOff: {
    backgroundColor: "#FFFFFF",
    borderColor: "rgba(148, 163, 184, 0.45)",
  },
  recurrenceLimitChipLabelOn: { color: E.BRAND, fontWeight: "700" as const, fontSize: 12, lineHeight: 15 },
  recurrenceLimitChipLabelOff: { color: E.TEXT_SEC, fontWeight: "600" as const, fontSize: 12, lineHeight: 15 },
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
  recurrenceWeekdayChipTextOn: { color: "#FFFFFF", fontWeight: "700" as const, fontSize: 12, lineHeight: 15 },
  recurrenceWeekdayChipTextOff: { color: E.BRAND, fontWeight: "600" as const, fontSize: 12, lineHeight: 15 },
  swapBtn: {
    alignSelf: "flex-end" as const,
    width: 40,
    height: 40,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.22)",
    backgroundColor: "rgba(0, 121, 107, 0.06)",
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  addressBlock: { gap: 6 },
  pickupDropoffRow: {
    flexDirection: "row" as const,
    alignItems: "flex-start" as const,
    flexWrap: "wrap" as const,
    columnGap: 8,
    rowGap: 6,
  },
  addressFieldsColumn: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 280,
    minWidth: 220,
    gap: 6,
  },
  swapColumn: {
    minWidth: 44,
    alignSelf: "stretch" as const,
    alignItems: "flex-end" as const,
    justifyContent: "center" as const,
  },
  compactSectionBlock: { gap: 3 },
  compactAddressContainer: { gap: 3 },
  compactAddressShell: {
    minHeight: 42,
    paddingHorizontal: 10,
  },
  compactAddressInput: {
    paddingVertical: 7,
  },
  medicalFields: { gap: 8, paddingTop: 2 },
  wheelchairRow: {
    flexDirection: "row" as const,
    gap: 6,
    flexWrap: "wrap" as const,
  },
  returnBlock: { gap: 6 },
  returnLabel: {
    fontSize: 13,
    fontWeight: "600" as const,
    color: E.TEXT,
    marginBottom: 2,
  },
  medicalRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    justifyContent: "space-between" as const,
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: ROW_RADIUS,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.35)",
    backgroundColor: "#FAFBFA",
  },
  medicalCard: {
    borderRadius: ROW_RADIUS,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.35)",
    backgroundColor: "#FAFBFA",
    padding: 8,
    gap: 8,
  },
  footerRow: { gap: 6 },
  footerSummary: {
    fontSize: 13,
    color: E.TEXT,
    fontWeight: "600" as const,
  },
  footerHint: {
    fontSize: 12,
    color: E.TEXT_MUTED,
    textAlign: "left" as const,
  },
  footerButtons: {
    flexDirection: "row" as const,
    gap: 8,
    alignItems: "stretch" as const,
  },
  footerBtn: { flex: 1, minHeight: COMPACT_ACTION_HEIGHT, borderRadius: COMPACT_CONTROL_RADIUS },
  linkNewClient: {
    marginTop: 4,
    alignSelf: "flex-start" as const,
    paddingVertical: 4,
  },
  notesCounter: {
    alignSelf: "flex-end" as const,
    marginBottom: 4,
    fontSize: 12,
    color: E.TEXT_MUTED,
  },
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
    fontSize: 12,
    fontWeight: "700" as const,
  },
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
  return Number.isFinite(Number(address.latitude)) && Number.isFinite(Number(address.longitude));
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
          address_components?: Array<{ long_name?: string; types?: string[] }>;
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
  const activeContextId = useActiveCompanyContextId();
  const createRide = useRideCreate();
  const form = useRideFormState();
  const [error, setError] = useState<string | null>(null);
  const [selectedClientLabel, setSelectedClientLabel] = useState("");
  const [amountSource, setAmountSource] = useState<"preferential" | "simulated" | "manual" | null>(null);
  const [amountLocked, setAmountLocked] = useState(false);
  const [pricingWarning, setPricingWarning] = useState("");
  const [selectedClientPreferentialRate, setSelectedClientPreferentialRate] = useState<number | null>(null);
  const [billToPatient, setBillToPatient] = useState(false);
  const [createClientVisible, setCreateClientVisible] = useState(false);
  const [medicalOpen, setMedicalOpen] = useState(false);
  const [routePointsForPricing, setRoutePointsForPricing] = useState<Array<{ lat: number; lng: number }>>([]);
  const clientDetailHydrationKeyRef = useRef<string>("");
  const lastSimulationKeyRef = useRef<string>("");
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
    selectPickupAddress,
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
    if (form.recurrence !== "custom") return;
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
    return `${head}${days ? `Jours actifs: ${days}.` : "Sélectionnez au moins un jour."}`;
  }, [recurringOn, form.recurrence, form.recurrenceDays, form.scheduledAt, scheduledOk]);

  const recurrenceValid = useMemo(() => {
    if (!recurringOn) return true;
    if (form.recurrence === "custom" && recurrenceSelectedDaysCount === 0) return false;
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

  const handlePickupAddressSelected = useCallback(async (address: RideAddressOption) => {
    form.selectPickupAddress(address);
    const enriched = await enrichAddressWithPlaceDetails(address);
    if (enriched !== address && hasValidCoords(enriched)) {
      form.selectPickupAddress(enriched);
      lastSimulationKeyRef.current = "";
    }
  }, [form]);

  const handleDropoffAddressSelected = useCallback(async (address: RideAddressOption) => {
    form.selectDropoffAddress(address);
    const enriched = await enrichAddressWithPlaceDetails(address);
    if (enriched !== address && hasValidCoords(enriched)) {
      form.selectDropoffAddress(enriched);
      lastSimulationKeyRef.current = "";
    }
  }, [form]);

  const handleClientSelected = (client: RideClientOption) => {
    form.setClientId(client.id);
    setSelectedClientLabel(client.label);
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
      setMedicalOpen(true);
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
      form.setAmountInput(client.preferentialRate.toFixed(2));
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
      setMedicalOpen(true);
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
      setMedicalOpen(true);
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
      setAmountInput(detail.preferentialRate.toFixed(2));
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
  ]);

  useEffect(() => {
    if (!form.clientId) {
      setAmountSource(null);
      setAmountLocked(false);
      setPricingWarning("");
      setSelectedClientPreferentialRate(null);
      setBillToPatient(false);
      clientDetailHydrationKeyRef.current = "";
      lastSimulationKeyRef.current = "";
      activeSimulationKeyRef.current = "";
      simulationRequestSeqRef.current = 0;
      simulationCacheRef.current.clear();
      lastGeocodePickupKeyRef.current = "";
      lastGeocodeDropoffKeyRef.current = "";
    }
  }, [form.clientId]);

  useEffect(() => {
    let cancelled = false;
    const loadRoutePoints = async () => {
      if (!hasValidCoords(pickupAddress) || !hasValidCoords(dropoffAddress)) {
        setRoutePointsForPricing([]);
        return;
      }
      try {
        const response = await apiClient.get("/osrm/route", {
          params: {
            pickup_lat: Number(pickupAddress?.latitude),
            pickup_lon: Number(pickupAddress?.longitude),
            dropoff_lat: Number(dropoffAddress?.latitude),
            dropoff_lon: Number(dropoffAddress?.longitude),
          },
          timeout: 4000,
        });
        if (cancelled) return;
        const route = Array.isArray(response?.data?.route)
          ? response.data.route
              .filter((pair: unknown) => Array.isArray(pair) && pair.length >= 2)
              .map((pair: unknown) => {
                const [lat, lng] = pair as [number, number];
                return { lat: Number(lat), lng: Number(lng) };
              })
              .filter((pt) => Number.isFinite(pt.lat) && Number.isFinite(pt.lng))
          : [];
        setRoutePointsForPricing(route);
      } catch {
        if (!cancelled) {
          setRoutePointsForPricing([]);
        }
      }
    };
    void loadRoutePoints();
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
        lastSimulationKeyRef.current = "";
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
        setAmountInput(activePreferentialAmount.toFixed(2));
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
    setAmountInput,
  ]);

  useEffect(() => {
    if (isMaterialDelivery || amountLocked || amountSource === "preferential") return;
    if (!pickupAddress || !dropoffAddress || !scheduledOk) return;
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
    ].join("|");
    if (lastSimulationKeyRef.current === simulationKey) {
      return;
    }
    const cached = simulationCacheRef.current.get(simulationKey);
    if (cached && Date.now() - cached.cachedAt <= SIM_CACHE_TTL_MS) {
      setAmountInput(cached.amount.toFixed(2));
      setAmountSource("simulated");
      setPricingWarning(cached.warningMessage || "");
      lastSimulationKeyRef.current = simulationKey;
      return;
    }
    lastSimulationKeyRef.current = simulationKey;
    activeSimulationKeyRef.current = simulationKey;
    simulationRequestSeqRef.current += 1;
    const requestSeq = simulationRequestSeqRef.current;
    setPricingWarning("");
    const timer = setTimeout(() => {
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
      pricingSimulation.mutate(payload, {
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
            return;
          }
          const amount = analysis.amount ?? parseSimulationAmount(response);
          if (amount == null) {
            setPricingWarning("Calcul auto indisponible: saisissez un montant.");
            return;
          }
          setAmountInput(amount.toFixed(2));
          setAmountSource("simulated");
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
    pricingSimulation,
    routePointsForPricing,
    scheduledOk,
    setAmountInput,
    clientId,
  ]);

  const submit = async () => {
    if (!canSubmit) {
      if (!recurrenceValid && recurringOn) {
        setError(
          "Vérifiez la récurrence : au moins un jour en mode « Perso », nombre de répétitions entre 1 et 52, et date de fin au format AAAA-MM-JJ si renseignée.",
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
      setAmountSource(null);
      setAmountLocked(false);
      setPricingWarning("");
      setBillToPatient(false);
      setMedicalOpen(false);
      setError(null);
      onCreated?.();
      onClose();
    } catch (e) {
      setError(toSubmitErrorMessage(e));
    }
  };

  const footerSummaryText = useMemo(() => {
    if (!form.clientId) return "Client non sélectionné";
    if (!scheduledOk) return "Date/heure manquante";
    const client = selectedClientLabel || `Client #${form.clientId}`;
    const pickup = form.pickup.trim().split(",")[0] || "…";
    const dropoff = form.dropoff.trim().split(",")[0] || "…";
    const datetime = formatSwissDateTime(form.scheduledAt);
    const amount = parseOptionalAmount(form.amountInput);
    const badges = [
      form.isRoundTrip ? "A/R" : "",
      form.recurrence !== "none" ? "Récurrente" : "",
      form.isMaterialDelivery ? "Livraison" : "",
      clientDetailQuery.data?.hasActiveStay ? "Départ établissement" : "",
      clientDetailQuery.data?.hasActiveStay && !billToPatient ? "Facturation clinique" : "",
    ]
      .filter(Boolean)
      .join(" · ");
    const amountText = amount != null ? `${amount.toFixed(2)} CHF` : "Montant manquant";
    return [client, `${pickup} → ${dropoff}`, datetime, badges, amountText].filter(Boolean).join(" · ");
  }, [
    form.amountInput,
    form.clientId,
    form.dropoff,
    form.isMaterialDelivery,
    form.isRoundTrip,
    form.pickup,
    form.recurrence,
    form.scheduledAt,
    scheduledOk,
    selectedClientLabel,
    clientDetailQuery.data?.hasActiveStay,
    billToPatient,
  ]);

  const amountBadgeMeta = useMemo(() => {
    if (!amountSource) return null;
    if (amountSource === "preferential") {
      return {
        label: "Tarif préférentiel",
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
  }, [amountSource]);

  const header = () => (
    <View style={s.headerRow}>
      <Pressable
        onPress={onClose}
        style={BACK_BOX}
        accessibilityRole="button"
        accessibilityLabel="Retour"
      >
        <Ionicons name="chevron-back" size={22} color={E.BRAND} />
      </Pressable>
      <View style={s.headerCenter}>
        <AppText style={s.headerTitle}>Créer une réservation</AppText>
      </View>
    </View>
  );

  const footer = (
    <View style={s.footerRow}>
      <AppText style={s.footerSummary}>{footerSummaryText}</AppText>
      {!canSubmit && !createRide.isPending ? (
        <AppText style={s.footerHint}>Complétez les champs marqués d’un *</AppText>
      ) : null}
      <View style={s.footerButtons}>
        <AppButton
          title="Annuler"
          variant="secondary"
          onPress={onClose}
          style={{ ...s.footerBtn, ...OUTLINE_SECONDARY }}
        />
        <AppButton
          title={createRide.isPending ? "Création…" : "Créer la réservation"}
          variant="primary"
          disabled={!canSubmit || createRide.isPending}
          loading={createRide.isPending}
          onPress={() => void submit()}
          style={s.footerBtn}
          leftIcon={
            <Ionicons
              name="checkmark-circle-outline"
              size={20}
              color={!canSubmit || createRide.isPending ? "rgba(255,255,255,0.85)" : "#fff"}
            />
          }
        />
      </View>
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
        sheetBodyMaxHeightRatio={0.68}
      >
        <View style={s.form}>
          <View style={s.sectionBlock}>
            <AppText style={s.sectionLabel}>Client *</AppText>
            <ClientSelector
              showFieldLabel={false}
              value={form.clientId}
              onChange={form.setClientId}
              onSelectClient={handleClientSelected}
              onCreateClient={() => setCreateClientVisible(true)}
              leftSlot={<Ionicons name="search-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
            />
            {clientDetailQuery.data?.hasActiveStay ? (
              <View style={s.card}>
                <AppText style={s.cardValue}>
                  Client hospitalisé{clientDetailQuery.data.clinicName ? ` · ${clientDetailQuery.data.clinicName}` : ""}
                </AppText>
                <AppText style={s.sectionHelper}>
                  Départ établissement prioritaire.
                </AppText>
                <Pressable
                  onPress={() => setBillToPatient((v) => !v)}
                  style={[s.chip, billToPatient ? s.chipOn : s.chipOff]}
                  accessibilityRole="button"
                  accessibilityState={{ selected: billToPatient }}
                >
                  <AppText style={billToPatient ? s.chipLabelOn : s.chipLabelOff}>
                    {billToPatient ? "Facturation patient (override)" : "Facturation clinique"}
                  </AppText>
                </Pressable>
              </View>
            ) : null}
          </View>

          <View style={s.pickupDropoffRow}>
            <View style={s.addressFieldsColumn}>
              <View style={[s.sectionBlock, s.compactSectionBlock]}>
                <AppText style={s.sectionLabel}>Lieu de prise en charge *</AppText>
                <AddressSelector
                  label=""
                  value={form.pickup}
                  onChange={form.setPickup}
                  onSelectAddress={(address) => {
                    void handlePickupAddressSelected(address);
                  }}
                  placeholder="Rechercher une adresse ou un lieu…"
                  leftSlot={<Ionicons name="navigate-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
                  containerStyle={s.compactAddressContainer}
                  shellStyle={s.compactAddressShell}
                  inputStyle={s.compactAddressInput}
                />
              </View>

              <View style={[s.sectionBlock, s.compactSectionBlock]}>
                <AppText style={s.sectionLabel}>Lieu de destination *</AppText>
                <AddressSelector
                  label=""
                  value={form.dropoff}
                  onChange={form.setDropoff}
                  onSelectAddress={(address) => {
                    void handleDropoffAddressSelected(address);
                    const hints = parseMedicalHintsFromAddress(address.label);
                    if (hints.establishment && form.establishment.trim().length === 0) {
                      form.setEstablishment(hints.establishment);
                      setMedicalOpen(true);
                    }
                    if (hints.doctorName && form.doctorName.trim().length === 0) {
                      form.setDoctorName(hints.doctorName);
                      setMedicalOpen(true);
                    }
                    if (hints.hospitalService && form.hospitalService.trim().length === 0) {
                      form.setHospitalService(hints.hospitalService);
                      setMedicalOpen(true);
                    }
                    if (hints.notesMedical && form.notesMedical.trim().length === 0) {
                      form.setNotesMedical(hints.notesMedical);
                    }
                  }}
                  placeholder="Rechercher une adresse ou un lieu…"
                  leftSlot={<Ionicons name="location-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
                  containerStyle={s.compactAddressContainer}
                  shellStyle={s.compactAddressShell}
                  inputStyle={s.compactAddressInput}
                />
              </View>
            </View>

            <View style={s.swapColumn}>
              <Pressable
                onPress={form.swapAddresses}
                style={s.swapBtn}
                accessibilityRole="button"
                accessibilityLabel="Inverser pickup/destination"
              >
                <Ionicons name="swap-vertical-outline" size={20} color={E.BRAND} />
              </Pressable>
            </View>
          </View>

          <TimeDatePicker value={form.scheduledAt} onChange={form.setScheduledAt} />
          <View style={s.sectionDivider} />

          <View style={s.toggleRow}>
            <Pressable
              onPress={() => form.setIsRoundTrip(!form.isRoundTrip)}
              style={[s.chip, form.isRoundTrip ? s.chipOn : s.chipOff]}
              accessibilityRole="button"
              accessibilityState={{ selected: form.isRoundTrip }}
            >
              <Ionicons
                name="swap-horizontal-outline"
                size={17}
                color={form.isRoundTrip ? E.BRAND : E.TEXT_SEC}
              />
              <AppText style={form.isRoundTrip ? s.chipLabelOn : s.chipLabelOff}>Trajet AR</AppText>
            </Pressable>
            <Pressable
              onPress={() => {
                if (recurringOn) form.setRecurrence("none");
                else form.setRecurrence("daily");
              }}
              style={[s.chip, recurringOn ? s.chipOn : s.chipOff]}
              accessibilityRole="button"
              accessibilityState={{ selected: recurringOn }}
            >
              <Ionicons
                name="repeat-outline"
                size={17}
                color={recurringOn ? E.BRAND : E.TEXT_SEC}
              />
              <AppText style={recurringOn ? s.chipLabelOn : s.chipLabelOff}>Récurrente</AppText>
            </Pressable>
            <Pressable
              onPress={() => form.setIsMaterialDelivery(!form.isMaterialDelivery)}
              style={[s.chip, form.isMaterialDelivery ? s.chipOn : s.chipOff]}
              accessibilityRole="button"
              accessibilityState={{ selected: form.isMaterialDelivery }}
            >
              <Ionicons
                name="cube-outline"
                size={17}
                color={form.isMaterialDelivery ? E.BRAND : E.TEXT_SEC}
              />
              <AppText style={form.isMaterialDelivery ? s.chipLabelOn : s.chipLabelOff}>Livraison</AppText>
            </Pressable>
          </View>

          {form.isRoundTrip ? (
            <View style={s.card}>
              <TimeDatePicker
                value={form.returnScheduledAt}
                onChange={form.setReturnScheduledAt}
                label="Date retour (optionnel)"
                emptyLabel="À définir"
                emptyPreviewReferenceIso={form.scheduledAt}
                modalTitle="Date retour"
                accessibilityLabel="Choisir la date et l’heure de retour"
              />
            </View>
          ) : null}

          {recurringOn ? (
            <View style={s.card}>
              <View style={s.recurrenceHeader}>
                <Ionicons name="repeat-outline" size={20} color={E.BRAND} />
                <AppText style={s.recurrenceSectionTitle}>Récurrence</AppText>
              </View>
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
                  if (v !== "custom") setRecurrenceDays([]);
                }}
              />
              {form.recurrence === "custom" ? (
                <View style={{ gap: 8 }}>
                  <AppText style={s.recurrenceSubLabel}>
                    Sélectionner les jours ({recurrenceSelectedDaysCount} sélectionné{recurrenceSelectedDaysCount > 1 ? "s" : ""})
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
                          <AppText style={on ? s.recurrenceWeekdayChipTextOn : s.recurrenceWeekdayChipTextOff}>
                            {label}
                          </AppText>
                        </Pressable>
                      );
                    })}
                  </View>
                  {recurrenceSelectedDaysCount === 0 ? (
                    <AppText style={{ color: "#B91C1C", fontSize: 12, fontWeight: "600" }}>
                      ⚠️ Veuillez sélectionner au moins un jour
                    </AppText>
                  ) : null}
                </View>
              ) : null}
              <TimeDatePicker
                value={recurrenceEndDate ? `${recurrenceEndDate}T12:00:00` : ""}
                onChange={(v) => {
                  const n = normalizeScheduledTimeIso(v);
                  const [d] = n.split("T");
                  if (d) setRecurrenceEndDate(d);
                }}
                dateOnly
                label="Jusqu’au (optionnel)"
                emptyLabel="AAAA-MM-JJ"
                modalTitle="Date de fin"
                accessibilityLabel="Choisir la date de fin de récurrence"
              />
              <AppText style={s.sectionHelper}>
                ℹ️ Sans date de fin, la série utilise une fenêtre par défaut côté serveur.
              </AppText>
              {recurrenceEndDate.trim().length > 0 && !recurrenceEndValid ? (
                <AppText style={{ color: "#B91C1C", fontSize: 12, fontWeight: "600" }}>
                  ⚠️ Format attendu: AAAA-MM-JJ
                </AppText>
              ) : null}
              <View style={s.recurrenceHint}>
                <Ionicons name="information-circle-outline" size={20} color={E.BRAND} />
                <AppText style={s.recurrenceHintText}>{recurrenceSummary}</AppText>
              </View>
            </View>
          ) : null}

          {/* Description livraison (quand mode Livraison actif) */}
          {form.isMaterialDelivery ? (
            <AppInput
              label="Description de la livraison *"
              value={form.deliveryDescription}
              onChangeText={form.setDeliveryDescription}
              placeholder="Ex : dossiers médicaux, matériel orthopédique…"
              leftSlot={<Ionicons name="cube-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
              shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FAFBFA" }}
            />
          ) : (
            <>
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
                shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FAFBFA" }}
              />
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
                    <AppText style={[s.amountBadgeText, { color: amountBadgeMeta.textColor }]}>
                      {amountBadgeMeta.label}
                    </AppText>
                  </View>
                </View>
              ) : null}
              {pricingSimulation.isPending && !amountLocked ? (
                <AppText style={s.sectionHelper}>
                  {amountSource === "simulated" && form.amountInput.trim().length > 0
                    ? "Mise à jour du montant exact en cours…"
                    : "Calcul du montant en cours…"}
                </AppText>
              ) : null}
              {pricingWarning ? <AppText style={s.sectionHelper}>{pricingWarning}</AppText> : null}
              {amountLocked && !form.isMaterialDelivery ? (
                <>
                  <AppText style={s.sectionHelper}>
                    Montant verrouillé manuellement.
                  </AppText>
                  <Pressable
                    onPress={() => {
                      setAmountLocked(false);
                      setAmountSource(null);
                      lastSimulationKeyRef.current = "";
                    }}
                    style={s.linkNewClient}
                    accessibilityRole="button"
                    accessibilityLabel="Réactiver le calcul automatique du montant"
                  >
                    <AppText style={{ color: E.BRAND, fontWeight: "600" }}>
                      Recalculer automatiquement
                    </AppText>
                  </Pressable>
                </>
              ) : null}
            </>
          )}
          <View style={s.sectionDivider} />

          <View style={s.sectionBlock}>
            <Pressable
              onPress={() => setMedicalOpen(!medicalOpen)}
              style={s.medicalRow}
              accessibilityRole="button"
              accessibilityState={{ expanded: medicalOpen }}
            >
              <AppText variant="label">Informations médicales</AppText>
              <Ionicons
                name={medicalOpen ? "chevron-up" : "chevron-down"}
                size={20}
                color={E.TEXT_MUTED}
              />
            </Pressable>
            {medicalOpen ? (
              <View style={s.medicalCard}>
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
                  label="Accès pickup"
                  value={form.pickupAccessNotes}
                  onChangeText={form.setPickupAccessNotes}
                  placeholder="Ex: entrée arrière, sonner à…, appeler avant…"
                  leftSlot={<Ionicons name="navigate-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
                  shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
                />
                <AppInput
                  label="Accès destination"
                  value={form.dropoffAccessNotes}
                  onChangeText={form.setDropoffAccessNotes}
                  placeholder="Ex: entrée B, étage 2, service…, appeler secrétariat…"
                  leftSlot={<Ionicons name="location-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
                  shellStyle={{ borderRadius: ROW_RADIUS, backgroundColor: "#FFFFFF" }}
                />
                <AppText style={s.sectionLabel}>Chaise roulante</AppText>
                <View style={s.wheelchairRow}>
                  <Pressable
                    onPress={() => form.setWheelchairClient(!form.wheelchairClient)}
                    style={[s.chip, form.wheelchairClient ? s.chipOn : s.chipOff]}
                    accessibilityRole="button"
                    accessibilityState={{ selected: form.wheelchairClient }}
                  >
                    <AppText style={form.wheelchairClient ? s.chipLabelOn : s.chipLabelOff}>
                      {"♿"} En chaise
                    </AppText>
                  </Pressable>
                  <Pressable
                    onPress={() => form.setWheelchairProvide(!form.wheelchairProvide)}
                    style={[s.chip, form.wheelchairProvide ? s.chipOn : s.chipOff]}
                    accessibilityRole="button"
                    accessibilityState={{ selected: form.wheelchairProvide }}
                  >
                    <AppText style={form.wheelchairProvide ? s.chipLabelOn : s.chipLabelOff}>
                      {"🏥"} Fournir chaise
                    </AppText>
                  </Pressable>
                </View>
              </View>
            ) : null}
          </View>

          <View style={s.sectionBlock}>
            <AppText variant="label">Notes internes</AppText>
            <AppText style={s.notesCounter}>
              {form.internalNotes.length}/{NOTES_MAX}
            </AppText>
            <AppInput
              value={form.internalNotes}
              onChangeText={(t) =>
                form.setInternalNotes(t.length > NOTES_MAX ? t.slice(0, NOTES_MAX) : t)
              }
              placeholder="Ajouter des notes…"
              multiline
              maxLength={NOTES_MAX}
              leftSlot={<Ionicons name="create-outline" size={FIELD_ICON_SIZE} color={E.TEXT_SEC} />}
              textAlignVertical="top"
              shellStyle={{
                borderRadius: ROW_RADIUS,
                minHeight: COMPACT_MULTILINE_LARGE_HEIGHT,
                alignItems: "flex-start",
                backgroundColor: "#FAFBFA",
              }}
              style={{ minHeight: COMPACT_MULTILINE_LARGE_INPUT_HEIGHT }}
            />
          </View>

          {error ? (
            <AppText variant="error" style={s.error}>
              {error}
            </AppText>
          ) : null}
        </View>
      </Modal>
      <ClientCreateModal
        visible={createClientVisible}
        onClose={() => setCreateClientVisible(false)}
        onCreated={() => setError(null)}
      />
    </>
  );
}
