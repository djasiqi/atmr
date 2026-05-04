import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  AppText,
  brandPrimary,
  useAppViewport,
  useResponsiveTokens,
} from "../../../../src/design/responsive";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Ionicons } from "@expo/vector-icons";
import {
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { useClientBottomContentPadding } from "../../../../src/features/client/navigation/ClientFloatingAppBar";
import { useLocalSearchParams, useRouter } from "expo-router";
import DateTimePicker, {
  DateTimePickerEvent,
} from "@react-native-community/datetimepicker";
import { PermissionGuard } from "../../../../src/core/guards";
import { useSession } from "../../../../src/core/sessionProvider";
import {
  CLIENT_SURFACE_CONTRACT_VERSIONS,
  logContractMismatchEvent,
} from "../../../../src/core/contracts/clientSurfaceVersions";
import {
  autocompleteAddress,
  createClientBooking,
  getBookingDetail,
  getGeocodePlaceDetails,
  postIndicativeFareEstimate,
  previewClientBooking,
  reverseGeocodeFromCoordinates,
} from "../../../../src/features/client/api";
import { useActiveClientContextId, useClientProfileQuery } from "../../../../src/features/client/hooks";
import { invalidateClientQueries } from "../../../../src/features/client/queryKeys";
import {
  collectAlternativeAddressQueries,
  effectiveSuggestionLon,
  findGeocodedInAddressBook,
  getDomicileLatLon,
  isGeocodedSuggestion,
  normAddrKey,
  recordGeocodedInAddressBook,
  type ResolveFieldForSubmitResult,
  tryResolveSingleUnambiguousSuggestion,
  uniqueGeocodedByPlaceOrCoord,
} from "../../../../src/features/client/booking/bookingAddressResolution";
import {
  buildClientNoteFromLegs,
  formatClientDomicile,
  formatDateYmd,
  formatTimeHm,
  isCoordinatePairLabel,
  MAX_CLIENT_NOTE_LEN,
  MAX_CLIENT_NOTE_LEG,
  parseCoordInput,
} from "../../../../src/features/client/booking/bookingDraftFormatting";
import { trackClientKpiEvent } from "../../../../src/features/client/statusEvents";
import {
  AddressAutocompleteSuggestion,
  BookingDraftPayload,
  BookingPreviewResponse,
  CanonicalAddressPrecisionLevel,
  ClientApiError,
  IndicativeFareEstimateResponse,
  RecurrenceType,
} from "../../../../src/features/client/types";
import {
  consumePublicPreRequestDraft,
  fetchPublicPreRequestDraft,
} from "../../../../src/core/api/client";
import {
  clearPublicPreRequestDraft,
  loadPublicPreRequestDraft,
} from "../../../../src/core/public/preRequestDraft";

/** Aligné `BookingCreateSchema.hospital_service` (colonne SQL 100). */
const MAX_HOSPITAL_SERVICE_LEN = 100;
const PREVIEW_FRESHNESS_TTL_MS = 5 * 60 * 1000;
const DEFAULT_CANONICAL_MATRIX: Record<
  CanonicalAddressPrecisionLevel,
  "allow" | "warn" | "block"
> = {
  rooftop: "allow",
  entrance: "allow",
  street: "warn",
  locality: "block",
  approximate: "block",
};
const RECURRENCE_WEEK_DAYS = [
  { id: 0, short: "L", label: "Lundi" },
  { id: 1, short: "Ma", label: "Mardi" },
  { id: 2, short: "Me", label: "Mercredi" },
  { id: 3, short: "J", label: "Jeudi" },
  { id: 4, short: "V", label: "Vendredi" },
  { id: 5, short: "S", label: "Samedi" },
  { id: 6, short: "D", label: "Dimanche" },
];

const MSG_ADDRESS_CHOOSE_SUGGESTION =
  "Choisissez l'adresse dans les suggestions pour confirmer le point exact.";

const MSG_ADDRESS_CONFIRM_EXACT =
  "Nous avons besoin de confirmer le point exact. Choisissez une adresse proposée dans la liste.";

const MSG_ADDRESS_LIST_HELPER =
  "Sélectionnez une adresse dans la liste pour confirmer le point exact.";

const MSG_DOMICILE_CONFIRM = "Adresse du domicile à confirmer : choisissez la même adresse dans la liste.";

const MSG_SUGGESTION_NEEDS_GEO = "Cette adresse doit être précisée. Choisissez une proposition avec un point exact.";

/** Échec place-details / géoloc incomplète après sélection explicite. */
const MSG_ADDRESS_NEEDS_PRECISION = "Adresse à préciser";

const MSG_ADDRESS_LOCALIZE_FAILED =
  "Nous n’avons pas pu localiser précisément cette adresse. Essayez d’ajouter la ville, le numéro ou choisissez une autre proposition.";

const SUGGESTION_SUBLABEL_NEEDS_GEO = "Adresse à préciser";

function __logSuggestionDebug(
  field: "pickup" | "dropoff",
  item: AddressAutocompleteSuggestion,
  selection: { validated: boolean; lat?: number; lon?: number } | null
) {
  if (typeof __DEV__ === "undefined" || !__DEV__) {
    return;
  }
  const la = item.lat;
  const lo = effectiveSuggestionLon(item);
  // eslint-disable-next-line no-console
  console.log({
    field,
    label: item.label,
    address: item.address,
    lat: la,
    lon: lo,
    lng: item.lng,
    isGeocoded: isGeocodedSuggestion(item),
    selection,
  });
}

function localToIso(value: string): string | null {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return null;
  return d.toISOString();
}

function parseYmdHmToDate(date: string, time: string): Date | null {
  const parsed = new Date(`${date}T${time}:00`);
  return Number.isFinite(parsed.getTime()) ? parsed : null;
}

function extractPrimaryPlaceLabel(value: string): string {
  const raw = String(value || "").trim();
  if (!raw) return "";
  const firstSegment = raw
    .split(",")
    .map((part) => part.trim())
    .find(Boolean);
  return firstSegment || raw;
}

function isHospitalLikeDestination(value: string): boolean {
  const lower = String(value || "").toLowerCase();
  return ["hopital", "hôpital", "hug", "clinique", "hospital", "chuv"].some((k) =>
    lower.includes(k)
  );
}

function destinationHasDoctorHint(value: string): boolean {
  const lower = String(value || "").toLowerCase();
  return ["docteur", "dr", "dr.", "dr med", "dr méd", "medecin", "médecin"].some((k) =>
    lower.includes(k)
  );
}

function oneHourLaterRoundedToFiveMinutes() {
  const now = new Date();
  const oneHourLater = new Date(now.getTime() + 60 * 60 * 1000);
  oneHourLater.setMinutes(Math.ceil(oneHourLater.getMinutes() / 5) * 5, 0, 0);
  const yyyy = oneHourLater.getFullYear();
  const mm = String(oneHourLater.getMonth() + 1).padStart(2, "0");
  const dd = String(oneHourLater.getDate()).padStart(2, "0");
  const hh = String(oneHourLater.getHours()).padStart(2, "0");
  const min = String(oneHourLater.getMinutes()).padStart(2, "0");
  return {
    date: `${yyyy}-${mm}-${dd}`,
    time: `${hh}:${min}`,
    local: `${yyyy}-${mm}-${dd}T${hh}:${min}`,
  };
}

/** Parité texte web (ClientDashboard) : indisponibilité de l’indicatif. */
const INDICATIVE_FARE_UNAVAILABLE_UX =
  "L'estimation indicative est momentanément indisponible.";

function jsDateToRecurrenceDayId(d: Date): number {
  return (d.getDay() + 6) % 7;
}

/**
 * Occurrences entre deux dates (inclus), plafonnées — aligné sur le portail web.
 */
function estimatedOccurrencesForRecurrence({
  startYmd,
  endYmd,
  recurrenceType,
  recurrenceDays,
}: {
  startYmd: string;
  endYmd: string;
  recurrenceType: RecurrenceType;
  recurrenceDays: number[];
}): number {
  const start = new Date(`${startYmd}T12:00:00`);
  const end = new Date(`${endYmd}T12:00:00`);
  if (!Number.isFinite(start.getTime()) || !Number.isFinite(end.getTime()) || end < start) {
    return 1;
  }
  if (recurrenceType === "daily") {
    const msPerDay = 24 * 60 * 60 * 1000;
    const n = Math.floor((end.getTime() - start.getTime()) / msPerDay) + 1;
    return Math.min(52, Math.max(1, n));
  }
  if (recurrenceType === "weekly") {
    let n = 0;
    const cur = new Date(start);
    while (cur <= end) {
      n += 1;
      cur.setDate(cur.getDate() + 7);
    }
    return Math.min(52, Math.max(1, n));
  }
  if (recurrenceType === "custom" && recurrenceDays.length > 0) {
    const set = new Set(recurrenceDays);
    let n = 0;
    const cur = new Date(start);
    while (cur <= end) {
      if (set.has(jsDateToRecurrenceDayId(cur))) n += 1;
      cur.setDate(cur.getDate() + 1);
    }
    return Math.min(52, Math.max(1, n));
  }
  return 1;
}

function roundChfToFiveRappen(value: number): number {
  const x = Number(value);
  if (!Number.isFinite(x)) return x;
  return Math.round((x + Number.EPSILON) * 20) / 20;
}

const ACCENT = brandPrimary;
const ACCENT_SOFT = "#e6f6fb";
const BORDER = "#e2e8f0";

const styles = StyleSheet.create({
  scroll: {
    flex: 1,
    backgroundColor: "#f1f5f9",
  },
  scrollContent: {
    paddingHorizontal: 18,
    maxWidth: 560,
    width: "100%",
    alignSelf: "center",
  },
  heroPanel: {
    backgroundColor: "#fff",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: BORDER,
    paddingVertical: 16,
    paddingHorizontal: 16,
    gap: 5,
    marginBottom: 8,
    borderTopWidth: 3,
    borderTopColor: ACCENT,
    ...Platform.select({
      ios: {
        shadowColor: "#0f172a",
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.04,
        shadowRadius: 8,
      },
      android: { elevation: 1 },
      default: {},
    }),
  },
  title: {
    fontSize: 20,
    fontWeight: "700",
    color: "#0f172a",
    letterSpacing: -0.35,
    lineHeight: 26,
  },
  lead: {
    fontSize: 13,
    lineHeight: 18,
    color: "#64748b",
    fontWeight: "500",
  },
  formCard: {
    backgroundColor: "#fff",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: BORDER,
    padding: 16,
    gap: 14,
    marginTop: 14,
    ...Platform.select({
      ios: {
        shadowColor: "#0f172a",
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.04,
        shadowRadius: 6,
      },
      android: { elevation: 1 },
      default: {},
    }),
  },
  fieldStack: {
    gap: 10,
  },
  routeDivider: {
    height: 1,
    backgroundColor: "#f1f5f9",
    marginVertical: 2,
  },
  cardEyebrow: {
    fontSize: 11,
    fontWeight: "600",
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: 0.45,
    marginBottom: 2,
  },
  sectionLabel: {
    fontSize: 15,
    fontWeight: "600",
    color: "#0f172a",
    letterSpacing: -0.12,
  },
  sectionHint: {
    fontSize: 13,
    color: "#64748b",
    lineHeight: 19,
    marginTop: 2,
  },
  addressBlock: {
    gap: 8,
  },
  destinationTopRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
  },
  destinationLabelCluster: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    minWidth: 0,
  },
  swapBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 999,
    paddingVertical: 8,
    paddingHorizontal: 11,
    backgroundColor: "#fff",
    flexShrink: 0,
  },
  swapBtnText: {
    fontSize: 13,
    fontWeight: "600",
    color: "#475569",
  },
  fieldLabelRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  fieldLabelIconWrap: {
    width: 30,
    height: 30,
    borderRadius: 8,
    backgroundColor: ACCENT_SOFT,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "rgba(10, 126, 164, 0.18)",
  },
  pickupActionsColumn: {
    gap: 8,
  },
  locationBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: "#fafbfc",
  },
  locationBtnText: {
    color: ACCENT,
    fontWeight: "600",
    fontSize: 13,
  },
  input: {
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 16,
    color: "#0f172a",
    backgroundColor: "#fafbfc",
    minHeight: 48,
  },
  inputRecognized: {
    borderColor: "rgba(5, 150, 105, 0.45)",
    backgroundColor: "rgba(5, 150, 105, 0.06)",
  },
  inputNeedsConfirm: {
    borderColor: "rgba(217, 119, 6, 0.5)",
    backgroundColor: "rgba(254, 252, 232, 0.7)",
  },
  addressFieldStatus: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    marginTop: 4,
  },
  addressFieldStatusText: {
    fontSize: 12,
    lineHeight: 17,
  },
  addressFieldStatusTextOk: {
    color: "#166534",
  },
  addressFieldStatusTextWarn: {
    color: "#b45309",
  },
  addressFieldStatusTextInfo: {
    color: "#64748b",
  },
  rowBetween: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 12,
  },
  suggestionList: {
    gap: 8,
  },
  suggestionPress: {
    borderRadius: 10,
    overflow: "hidden",
  },
  suggestionInner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 12,
    paddingHorizontal: 14,
    backgroundColor: "#fafbfc",
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 10,
  },
  suggestionText: {
    color: "#0f172a",
    fontSize: 14,
    fontWeight: "500",
    lineHeight: 20,
  },
  suggestionTextCol: {
    flex: 1,
    minWidth: 0,
    gap: 2,
  },
  suggestionSub: {
    fontSize: 11,
    fontWeight: "500",
    color: "#94a3b8",
  },
  instructionsToggleTextCol: {
    flex: 1,
    minWidth: 0,
    gap: 3,
    justifyContent: "center",
  },
  instructionsCharCount: {
    color: "#64748b",
    textAlign: "right",
    fontSize: 13,
    fontWeight: "500",
  },
  instructionsNoteFooterHint: {
    fontSize: 12,
    color: "#94a3b8",
    lineHeight: 17,
    marginTop: 2,
  },
  planningBody: {
    gap: 0,
  },
  planningSection: {
    gap: 10,
  },
  planningLabelRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 2,
  },
  planningDivider: {
    height: 1,
    backgroundColor: "#f1f5f9",
    marginVertical: 14,
  },
  planningMicroHint: {
    fontSize: 12,
    color: "#94a3b8",
    lineHeight: 17,
    marginTop: -2,
  },
  segmentRow: {
    flexDirection: "row",
    gap: 10,
  },
  segment: {
    flex: 1,
    borderWidth: 1,
    borderColor: BORDER,
    backgroundColor: "#fff",
    borderRadius: 10,
    paddingVertical: 13,
    paddingHorizontal: 10,
    minHeight: 50,
    justifyContent: "center",
  },
  segmentActive: {
    borderColor: ACCENT,
    backgroundColor: ACCENT_SOFT,
  },
  segmentText: {
    textAlign: "center",
    fontWeight: "600",
    fontSize: 14,
    color: "#475569",
  },
  segmentTextActive: {
    color: "#0c4a6e",
  },
  dateRow: {
    flexDirection: "row",
    gap: 10,
  },
  dateTile: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 12,
    backgroundColor: "#fafbfc",
    minHeight: 64,
  },
  dateTileIconWrap: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: ACCENT_SOFT,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "rgba(10, 126, 164, 0.18)",
  },
  dateTileTextCol: {
    flex: 1,
    minWidth: 0,
  },
  dateTileLabel: {
    fontSize: 10,
    fontWeight: "700",
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginBottom: 3,
  },
  dateTileValue: {
    fontSize: 15,
    fontWeight: "600",
    color: "#0f172a",
  },
  dateTilePlaceholder: {
    color: "#94a3b8",
    fontWeight: "500",
  },
  recurrenceToggle: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    borderWidth: 1,
    borderColor: BORDER,
    backgroundColor: "#fff",
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 9,
  },
  recurrenceToggleOn: {
    borderColor: ACCENT,
    backgroundColor: ACCENT_SOFT,
  },
  recurrenceToggleText: {
    fontWeight: "600",
    fontSize: 13,
    color: "#334155",
  },
  recurrenceToggleTextOn: {
    color: "#0c4a6e",
  },
  recurrenceTypeRow: {
    flexDirection: "row",
    gap: 8,
  },
  recurrenceTypeBtn: {
    flex: 1,
    borderWidth: 1,
    borderColor: BORDER,
    backgroundColor: "#fff",
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 6,
  },
  recurrenceTypeBtnOn: {
    borderColor: ACCENT,
    backgroundColor: ACCENT_SOFT,
  },
  recurrenceTypeText: {
    textAlign: "center",
    fontSize: 11,
    fontWeight: "600",
    color: "#475569",
  },
  dayChip: {
    width: 42,
    borderWidth: 1,
    borderColor: BORDER,
    backgroundColor: "#fff",
    borderRadius: 10,
    paddingVertical: 10,
  },
  dayChipOn: {
    borderColor: ACCENT,
    backgroundColor: ACCENT_SOFT,
  },
  dayChipText: {
    textAlign: "center",
    fontWeight: "700",
    fontSize: 13,
    color: "#475569",
  },
  primaryButton: {
    alignSelf: "stretch",
    width: "100%",
    backgroundColor: ACCENT,
    borderRadius: 10,
    paddingVertical: 15,
    paddingHorizontal: 16,
    alignItems: "center",
    marginTop: 0,
    ...Platform.select({
      ios: {
        shadowColor: "#0f172a",
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.06,
        shadowRadius: 4,
      },
      android: { elevation: 2 },
      default: {},
    }),
  },
  primaryButtonDisabled: {
    opacity: 0.5,
  },
  primaryButtonText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 15,
    letterSpacing: 0.2,
  },
  cancelButton: {
    paddingVertical: 14,
    alignItems: "center",
    marginBottom: 4,
  },
  cancelButtonText: {
    color: "#64748b",
    fontWeight: "500",
    fontSize: 14,
  },
  errorBanner: {
    backgroundColor: "#fef2f2",
    borderWidth: 1,
    borderColor: "#fecaca",
    borderRadius: 10,
    padding: 14,
    marginTop: 8,
  },
  errorText: {
    color: "#991b1b",
    fontSize: 14,
    lineHeight: 20,
  },
  warnBanner: {
    backgroundColor: "#fffbeb",
    borderWidth: 1,
    borderColor: "#fde68a",
    borderRadius: 10,
    padding: 12,
    marginTop: 8,
  },
  warnText: {
    color: "#92400e",
    fontSize: 14,
    lineHeight: 20,
  },
  indicativeCard: {
    borderWidth: 1,
    borderColor: BORDER,
    backgroundColor: "#fafbfc",
    borderRadius: 12,
    borderLeftWidth: 3,
    borderLeftColor: ACCENT,
    padding: 16,
    gap: 8,
    marginTop: 16,
    ...Platform.select({
      ios: {
        shadowColor: "#0f172a",
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.03,
        shadowRadius: 4,
      },
      android: { elevation: 0 },
      default: {},
    }),
  },
  indicativeTitle: {
    fontWeight: "600",
    fontSize: 11,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  indicativeAmount: {
    fontSize: 20,
    fontWeight: "700",
    color: "#0f172a",
    letterSpacing: -0.3,
  },
  indicativeMeta: {
    fontSize: 13,
    color: "#475569",
  },
  indicativeFoot: {
    fontSize: 12,
    color: "#64748b",
    lineHeight: 17,
  },
  actionsBlock: {
    marginTop: 28,
    paddingTop: 20,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "#e2e8f0",
    gap: 6,
  },
  stepPill: {
    fontSize: 12,
    fontWeight: "600",
    color: "#64748b",
    letterSpacing: 0.2,
  },
  summaryLegal: {
    fontSize: 14,
    lineHeight: 20,
    color: "#0f172a",
    fontWeight: "600",
    marginTop: 4,
  },
  summaryRow: {
    flexDirection: "row",
    gap: 10,
    alignItems: "flex-start",
    paddingVertical: 4,
  },
  summaryLabel: {
    width: 100,
    fontSize: 12,
    fontWeight: "600",
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: 0.4,
    paddingTop: 2,
  },
  summaryValue: {
    flex: 1,
    fontSize: 15,
    color: "#0f172a",
    lineHeight: 22,
    fontWeight: "500",
  },
  linkButton: {
    paddingVertical: 14,
    alignItems: "center",
  },
  linkButtonText: {
    color: ACCENT,
    fontWeight: "600",
    fontSize: 15,
  },
  pickupActionRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  recurrenceCompactLabel: {
    fontSize: 13,
    fontWeight: "600",
    color: "#475569",
  },
  recurrenceCompactHint: {
    fontSize: 11,
    color: "#94a3b8",
    lineHeight: 15,
  },
  fieldLabelIconWrapSm: {
    width: 26,
    height: 26,
    borderRadius: 7,
    backgroundColor: ACCENT_SOFT,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "rgba(10, 126, 164, 0.15)",
  },
  carrierSubLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: "#64748b",
    marginTop: 4,
  },
});

function bookingPreviewErrorMessage(err: unknown): string {
  const c = (err as Partial<ClientApiError> | null)?.code;
  if (c === "preview_unavailable") {
    return (
      "Nous n’arrivons pas à calculer le détail du tarif pour le paiement pour ce type de compte. " +
        "L’indicatif ci-dessus reste informatif. " +
        "L’exploitant doit paramétrer l’entreprise de référence pour le portail, ou contactez l’assistance."
    );
  }
  return ((err as { message?: string } | null)?.message?.trim() || "") || "Impossible de prévisualiser le prix.";
}

export type ClientFormStep = "details" | "summary";

export type ClientBookingCreateScreenTestProps = {
  /** Réservé aux tests (Expo ne passe jamais cette prop). */
  _testFormStep?: ClientFormStep;
};

type ClientBookingCreateScreenProps = ClientBookingCreateScreenTestProps;

export default function ClientBookingCreateScreen(
  props: ClientBookingCreateScreenProps = {}
) {
  const { _testFormStep: testFormStep } = props;
  const router = useRouter();
  const { topInset } = useAppViewport();
  const t = useResponsiveTokens();
  const bottomBarPad = useClientBottomContentPadding();
  const params = useLocalSearchParams<{ publicDraftId?: string }>();
  const queryClient = useQueryClient();
  const { activeContext, bootstrap, bootstrapSession } = useSession();
  const contextId = useActiveClientContextId();
  const profileQuery = useClientProfileQuery();
  const scrollRef = useRef<ScrollView | null>(null);
  const [formStep, setFormStep] = useState<ClientFormStep>(
    () => testFormStep ?? "details"
  );
  const [optionalDetailsExpanded, setOptionalDetailsExpanded] = useState(false);
  const [pickupRefineMessage, setPickupRefineMessage] = useState<string | null>(null);
  const [dropoffRefineMessage, setDropoffRefineMessage] = useState<string | null>(null);
  const [suggestionResolving, setSuggestionResolving] = useState<"pickup" | "dropoff" | null>(null);

  const [pickupLocation, setPickupLocation] = useState("");
  const [dropoffLocation, setDropoffLocation] = useState("");
  /** Même valeur que les champs, pour vérification après `await` sans effet de bord async. */
  const pickupLocationRef = useRef("");
  const dropoffLocationRef = useRef("");
  /** Résout les requêtes obsolètes d’autocomplete (changement de texte). */
  const pickupAutocompleteRequestRef = useRef(0);
  const dropoffAutocompleteRequestRef = useRef(0);
  const [selectedDate, setSelectedDate] = useState("");
  const [selectedTime, setSelectedTime] = useState("");
  const [asap, setAsap] = useState(false);
  const [isRoundTrip, setIsRoundTrip] = useState(false);
  const [returnTime, setReturnTime] = useState("");
  const [returnDate, setReturnDate] = useState("");
  const [isRecurring, setIsRecurring] = useState(false);
  const [recurrenceType, setRecurrenceType] = useState<RecurrenceType>("weekly");
  const [recurrenceDays, setRecurrenceDays] = useState<number[]>([]);
  const [recurrenceLength, setRecurrenceLength] = useState("4");
  const [recurrenceEndDate, setRecurrenceEndDate] = useState("");
  const [clientNoteDeparture, setClientNoteDeparture] = useState("");
  const [clientNoteArrival, setClientNoteArrival] = useState("");
  const [medicalFacility, setMedicalFacility] = useState("");
  const [hospitalService, setHospitalService] = useState("");
  const [doctorName, setDoctorName] = useState("");
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [showReturnDatePicker, setShowReturnDatePicker] = useState(false);
  const [showReturnTimePicker, setShowReturnTimePicker] = useState(false);
  const [precisionWarning, setPrecisionWarning] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [previewData, setPreviewData] = useState<BookingPreviewResponse | null>(null);
  const [previewMeta, setPreviewMeta] = useState<{
    hash: string;
    previewedAtMs: number;
    previewContractVersion: string | null;
  } | null>(null);
  const invalidatePreview = useCallback(() => {
    setPreviewData(null);
    setPreviewMeta(null);
    setPrecisionWarning(null);
  }, []);
  const [pickupSuggestions, setPickupSuggestions] = useState<AddressAutocompleteSuggestion[]>([]);
  const [dropoffSuggestions, setDropoffSuggestions] = useState<AddressAutocompleteSuggestion[]>([]);
  const [indicativeUi, setIndicativeUi] = useState<
    | { kind: "idle" }
    | { kind: "loading" }
    | { kind: "ok"; data: IndicativeFareEstimateResponse }
    | { kind: "unavailable" }
  >({ kind: "idle" });
  const [pickupSelection, setPickupSelection] = useState<{
    validated: boolean;
    lat?: number;
    lon?: number;
  } | null>(null);
  const [destinationSelection, setDestinationSelection] = useState<{
    validated: boolean;
    lat?: number;
    lon?: number;
  } | null>(null);
  const [pickupTouched, setPickupTouched] = useState(false);
  /** Incrémenté à chaque inversion : sur le web, force le remount du champ départ (TextInput contrôlé parfois désynchronisé). */
  const [pickupInputRevision, setPickupInputRevision] = useState(0);
  const [locationBias, setLocationBias] = useState<{ lat: number; lon: number } | null>(null);
  const [isResolvingLocation, setIsResolvingLocation] = useState(false);
  const [verifyingFieldOnContinue, setVerifyingFieldOnContinue] = useState<"pickup" | "dropoff" | null>(null);
  /** Suggestions API déjà vues (géoloc OK) : permet de valider le texte même si le 2e appel /geocode/autocomplete ne renvoie pas le même libellé. */
  const pickupGeocodeBookRef = useRef(new Map<string, AddressAutocompleteSuggestion>());
  const dropoffGeocodeBookRef = useRef(new Map<string, AddressAutocompleteSuggestion>());
  const customerName = useMemo(
    () =>
      profileQuery.data?.full_name?.trim() ||
      `${profileQuery.data?.first_name ?? ""} ${profileQuery.data?.last_name ?? ""}`.trim() ||
      "Client",
    [profileQuery.data?.full_name, profileQuery.data?.first_name, profileQuery.data?.last_name]
  );

  const draftPayload = useMemo<BookingDraftPayload>(() => {
    const scheduledLocal = selectedDate && selectedTime ? `${selectedDate}T${selectedTime}` : "";
    const scheduledIso = asap ? null : localToIso(scheduledLocal);
    const returnTimeIso =
      returnDate && returnTime ? localToIso(`${returnDate}T${returnTime}`) : null;
    const recurrenceSeriesLength = Math.min(52, Math.max(1, Number(recurrenceLength) || 1));
    const mergedClientNote = buildClientNoteFromLegs(clientNoteDeparture, clientNoteArrival);
    return {
      customer_name: customerName,
      pickup_location: pickupLocation.trim(),
      dropoff_location: dropoffLocation.trim(),
      scheduled_time: scheduledIso,
      asap,
      is_round_trip: isRoundTrip,
      return_time: returnTimeIso,
      return_date: returnDate.trim() || null,
      client_note: mergedClientNote,
      is_recurring: isRecurring,
      recurrence_type: isRecurring ? recurrenceType : null,
      recurrence_days:
        isRecurring && recurrenceType === "custom"
          ? recurrenceDays
          : null,
      recurrence_end_date: isRecurring ? recurrenceEndDate.trim() || null : null,
      recurrence_series_length: isRecurring ? recurrenceSeriesLength : null,
    };
  }, [
    asap,
    clientNoteArrival,
    clientNoteDeparture,
    customerName,
    dropoffLocation,
    isRecurring,
    isRoundTrip,
    pickupLocation,
    recurrenceDays,
    recurrenceEndDate,
    recurrenceLength,
    recurrenceType,
    returnDate,
    returnTime,
    selectedDate,
    selectedTime,
  ]);

  const previewMutation = useMutation({
    mutationFn: previewClientBooking,
    onSuccess: (data) => {
      const contracts = data.contracts;
      if (
        contracts?.status_dictionary_version &&
        contracts.status_dictionary_version !==
        CLIENT_SURFACE_CONTRACT_VERSIONS.statusDictionaryVersion
      ) {
        logContractMismatchEvent(
          "status",
          CLIENT_SURFACE_CONTRACT_VERSIONS.statusDictionaryVersion,
          contracts.status_dictionary_version
        );
      }
      if (
        contracts?.pricing_contract_version &&
        contracts.pricing_contract_version !==
        CLIENT_SURFACE_CONTRACT_VERSIONS.pricingContractVersion
      ) {
        logContractMismatchEvent(
          "pricing",
          CLIENT_SURFACE_CONTRACT_VERSIONS.pricingContractVersion,
          contracts.pricing_contract_version
        );
      }
      if (
        contracts?.canonical_address_contract_version &&
        contracts.canonical_address_contract_version !==
        CLIENT_SURFACE_CONTRACT_VERSIONS.canonicalAddressContractVersion
      ) {
        logContractMismatchEvent(
          "canonical_address",
          CLIENT_SURFACE_CONTRACT_VERSIONS.canonicalAddressContractVersion,
          contracts.canonical_address_contract_version
        );
      }
      if (
        contracts?.preview_contract_version &&
        contracts.preview_contract_version !==
        CLIENT_SURFACE_CONTRACT_VERSIONS.previewContractVersion
      ) {
        logContractMismatchEvent(
          "preview",
          CLIENT_SURFACE_CONTRACT_VERSIONS.previewContractVersion,
          contracts.preview_contract_version
        );
      }
      setErrorMessage(null);
      setPreviewData(data);
    },
    onError: (error: unknown) => {
      invalidatePreview();
      setErrorMessage(bookingPreviewErrorMessage(error));
    },
  });

  const createMutation = useMutation({
    mutationFn: createClientBooking,
    onSuccess: async (response) => {
      if (!contextId) return;
      invalidateClientQueries(queryClient, contextId);
      const bookingId = response.booking_id ?? response.booking?.id;
      if (!bookingId) {
        setErrorMessage("Réservation créée mais identifiant introuvable.");
        return;
      }
      const bookingDetail = await getBookingDetail(Number(bookingId)).catch(() => null);
      const paymentRequired = Boolean(bookingDetail?.payment_required);
      const paymentStatus = bookingDetail?.payment_status ?? "unknown";
      if (paymentRequired && paymentStatus !== "paid") {
        router.replace({
          pathname: "/(app)/(client)/payment",
          params: { bookingId: String(bookingId) },
        });
        return;
      }
      const pricingReason = response.booking?.pricing_adjustment_reason;
      router.replace({
        pathname: "/(app)/(client)/booking/[bookingId]",
        params: {
          bookingId: String(bookingId),
          ...(pricingReason ? { pricingReason } : {}),
          created: "1",
        },
      });
    },
    onError: (error: Error) => {
      setErrorMessage(error.message ?? "Impossible de créer la réservation.");
    },
  });

  async function loadSuggestions(
    field: "pickup" | "dropoff",
    value: string
  ): Promise<void> {
    if (value.trim().length < 2) {
      if (field === "pickup") setPickupSuggestions([]);
      else setDropoffSuggestions([]);
      return;
    }
    if (field === "pickup") {
      pickupAutocompleteRequestRef.current += 1;
    } else {
      dropoffAutocompleteRequestRef.current += 1;
    }
    const thisReq =
      field === "pickup"
        ? pickupAutocompleteRequestRef.current
        : dropoffAutocompleteRequestRef.current;
    try {
      const results = await autocompleteAddress(
        value.trim(),
        field === "pickup" && locationBias
          ? { lat: locationBias.lat, lon: locationBias.lon, limit: 8 }
          : undefined
      );
      if (typeof __DEV__ !== "undefined" && __DEV__) {
        // eslint-disable-next-line no-console
        console.log({
          tag: "autocompleteAddress",
          field,
          value: value.trim(),
          resultStatus: "ok",
          results: results.map((r) => ({
            field,
            value: value.trim(),
            label: r.label,
            place_id: r.place_id,
            lat: r.lat,
            lon: r.lon,
            lng: r.lng,
            isGeocoded: isGeocodedSuggestion(r),
          })),
        });
      }
      if (
        field === "pickup" &&
        thisReq !== pickupAutocompleteRequestRef.current
      ) {
        return;
      }
      if (
        field === "dropoff" &&
        thisReq !== dropoffAutocompleteRequestRef.current
      ) {
        return;
      }
      const book = field === "pickup" ? pickupGeocodeBookRef : dropoffGeocodeBookRef;
      for (const it of results) {
        recordGeocodedInAddressBook(book.current, it);
      }
      if (field === "pickup") setPickupSuggestions(results);
      else setDropoffSuggestions(results);

      const vNow =
        (field === "pickup" ? pickupLocationRef : dropoffLocationRef).current
          .trim();
      if (vNow !== value.trim()) {
        return;
      }
      const toApply = tryResolveSingleUnambiguousSuggestion(
        vNow,
        results
      );
      if (toApply) {
        recordGeocodedInAddressBook(book.current, toApply);
        applyResolvedFromSuggestion(field, toApply);
      } else {
        if (field === "pickup" && thisReq !== pickupAutocompleteRequestRef.current) {
          return;
        }
        if (field === "dropoff" && thisReq !== dropoffAutocompleteRequestRef.current) {
          return;
        }
        const vPreAsync = (field === "pickup" ? pickupLocationRef : dropoffLocationRef)
          .current
          .trim();
        if (vPreAsync !== value.trim()) {
          return;
        }
        const ok = await tryAutoresolveFromSingleNongeocodedResult(field, results);
        if (!ok) {
          return;
        }
        if (field === "pickup" && thisReq !== pickupAutocompleteRequestRef.current) {
          return;
        }
        if (field === "dropoff" && thisReq !== dropoffAutocompleteRequestRef.current) {
          return;
        }
      }
    } catch {
      if (
        field === "pickup" &&
        thisReq !== pickupAutocompleteRequestRef.current
      ) {
        return;
      }
      if (
        field === "dropoff" &&
        thisReq !== dropoffAutocompleteRequestRef.current
      ) {
        return;
      }
      if (field === "pickup") setPickupSuggestions([]);
      else setDropoffSuggestions([]);
    }
  }

  async function mergeItemWithPlaceDetails(
    item: AddressAutocompleteSuggestion
  ): Promise<AddressAutocompleteSuggestion | null> {
    if (isGeocodedSuggestion(item)) {
      return item;
    }
    if (!item.place_id) {
      return null;
    }
    const d = await getGeocodePlaceDetails(item.place_id);
    if (!d) {
      return null;
    }
    const la = d.lat;
    const lo = effectiveSuggestionLon(d);
    if (la == null || lo == null) {
      return null;
    }
    return {
      ...item,
      address: d.address || item.address,
      label: item.label || d.address || item.label,
      lat: la,
      lon: d.lon != null ? d.lon : lo,
    };
  }

  async function tryAutoresolveFromSingleNongeocodedResult(
    field: "pickup" | "dropoff",
    results: AddressAutocompleteSuggestion[]
  ): Promise<boolean> {
    if (results.length !== 1) {
      return false;
    }
    const u = results[0]!;
    if (isGeocodedSuggestion(u)) {
      // Une seule proposition géolocalisée ne suffit pas : il faut un match univoque
      // (tryResolveSingleUnambiguous) ou un choix explicite dans la liste.
      return false;
    }
    if (!u.place_id) {
      return false;
    }
    const merged = await mergeItemWithPlaceDetails(u);
    if (merged && isGeocodedSuggestion(merged)) {
      const book = field === "pickup" ? pickupGeocodeBookRef : dropoffGeocodeBookRef;
      recordGeocodedInAddressBook(book.current, merged);
      applyResolvedFromSuggestion(field, merged);
      if (field === "pickup") {
        setPickupRefineMessage(null);
      } else {
        setDropoffRefineMessage(null);
      }
      return true;
    }
    return false;
  }

  async function onSuggestionSelected(
    field: "pickup" | "dropoff",
    item: AddressAutocompleteSuggestion
  ): Promise<void> {
    const selection =
      field === "pickup" ? pickupSelection : destinationSelection;
    const la0 = item.lat;
    const lo0 = effectiveSuggestionLon(item);
    __logSuggestionDebug(field, item, selection);
    if (typeof __DEV__ !== "undefined" && __DEV__) {
      // eslint-disable-next-line no-console
      console.log({
        tag: "onSuggestionSelected",
        phase: "before",
        field,
        value: (item.address ?? item.label).trim(),
        label: item.label,
        place_id: item.place_id,
        lat: la0,
        lon: item.lon,
        lng: item.lng,
        isGeocoded: isGeocodedSuggestion(item),
        selection,
        resultStatus: "pending",
      });
    }
    if (isGeocodedSuggestion(item)) {
      if (field === "pickup") {
        setPickupTouched(true);
      }
      recordGeocodedInAddressBook(
        field === "pickup" ? pickupGeocodeBookRef.current : dropoffGeocodeBookRef.current,
        item
      );
      if (field === "pickup") {
        setPickupRefineMessage(null);
      } else {
        setDropoffRefineMessage(null);
      }
      applyResolvedFromSuggestion(field, item);
      invalidatePreview();
      setErrorMessage(null);
      if (typeof __DEV__ !== "undefined" && __DEV__) {
        // eslint-disable-next-line no-console
        console.log({
          tag: "onSuggestionSelected",
          phase: "after",
          field,
          resultStatus: "geocoded_direct",
        });
      }
      return;
    }
    if (item.place_id) {
      if (field === "pickup") {
        setSuggestionResolving("pickup");
      } else {
        setSuggestionResolving("dropoff");
      }
      try {
        if (typeof __DEV__ !== "undefined" && __DEV__) {
          // eslint-disable-next-line no-console
          console.log({
            tag: "onSuggestionSelected",
            phase: "placeDetails_fetch",
            field,
            place_id: item.place_id,
            label: item.label,
          });
        }
        const merged = await mergeItemWithPlaceDetails(item);
        if (merged && isGeocodedSuggestion(merged)) {
          if (field === "pickup") {
            setPickupTouched(true);
          }
          recordGeocodedInAddressBook(
            field === "pickup" ? pickupGeocodeBookRef.current : dropoffGeocodeBookRef.current,
            merged
          );
          if (field === "pickup") {
            setPickupRefineMessage(null);
          } else {
            setDropoffRefineMessage(null);
          }
          applyResolvedFromSuggestion(field, merged);
          invalidatePreview();
          setErrorMessage(null);
          if (typeof __DEV__ !== "undefined" && __DEV__) {
            // eslint-disable-next-line no-console
            console.log({
              tag: "onSuggestionSelected",
              phase: "after",
              field,
              place_id: merged.place_id,
              lat: merged.lat,
              lon: merged.lon,
              resultStatus: "placeDetails_ok",
            });
          }
          return;
        }
        const pl = (item.address ?? item.label).trim();
        if (field === "pickup") {
          setPickupTouched(true);
          pickupLocationRef.current = pl;
          setPickupLocation(pl);
          setPickupSelection({ validated: false, lat: undefined, lon: undefined });
          setPickupRefineMessage(MSG_ADDRESS_NEEDS_PRECISION);
          setPickupSuggestions([]);
        } else {
          dropoffLocationRef.current = pl;
          setDropoffLocation(pl);
          setDestinationSelection({ validated: false, lat: undefined, lon: undefined });
          setDropoffRefineMessage(MSG_ADDRESS_NEEDS_PRECISION);
          setDropoffSuggestions([]);
        }
        invalidatePreview();
        setErrorMessage(null);
        if (typeof __DEV__ !== "undefined" && __DEV__) {
          // eslint-disable-next-line no-console
          console.log({
            tag: "onSuggestionSelected",
            phase: "after",
            field,
            place_id: item.place_id,
            resultStatus: "placeDetails_failed",
          });
        }
        return;
      } finally {
        setSuggestionResolving(null);
      }
    }
    const pl = (item.address ?? item.label).trim();
    if (field === "pickup") {
      setPickupTouched(true);
      pickupLocationRef.current = pl;
      setPickupLocation(pl);
      setPickupSelection({ validated: false, lat: undefined, lon: undefined });
      setPickupRefineMessage(MSG_SUGGESTION_NEEDS_GEO);
      setPickupSuggestions([]);
    } else {
      dropoffLocationRef.current = pl;
      setDropoffLocation(pl);
      setDestinationSelection({ validated: false, lat: undefined, lon: undefined });
      setDropoffRefineMessage(MSG_SUGGESTION_NEEDS_GEO);
      setDropoffSuggestions([]);
    }
    invalidatePreview();
    setErrorMessage(null);
  }

  async function applyCurrentLocationToPickup(): Promise<void> {
    if (!globalThis.navigator?.geolocation?.getCurrentPosition) {
      setErrorMessage("La localisation n'est pas disponible sur cet appareil.");
      return;
    }
    setIsResolvingLocation(true);
    setErrorMessage(null);
    globalThis.navigator.geolocation.getCurrentPosition(
      async (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        setLocationBias({ lat, lon });
        try {
          const reversed = await reverseGeocodeFromCoordinates(lat, lon);
          const fromReverse = (reversed?.address ?? reversed?.label ?? "").trim();
          const revLon = reversed != null ? effectiveSuggestionLon(reversed) : undefined;
          if (fromReverse && !isCoordinatePairLabel(fromReverse) && reversed?.lat != null && revLon != null) {
            pickupLocationRef.current = fromReverse;
            setPickupLocation(fromReverse);
            setPickupSelection({
              validated: true,
              lat: reversed!.lat!,
              lon: revLon,
            });
          } else {
            const nearest = await autocompleteAddress(`${lat},${lon}`, {
              lat,
              lon,
              limit: 3,
            });
            const pick = nearest.find((item) => {
              const line = (item.address ?? item.label ?? "").trim();
              return line.length > 0 && !isCoordinatePairLabel(line);
            });
            if (pick) {
              const line = (pick.address ?? pick.label ?? "").trim();
              const pickLon = effectiveSuggestionLon(pick);
              pickupLocationRef.current = line;
              setPickupLocation(line);
              setPickupSelection({
                validated: isGeocodedSuggestion(pick),
                lat: pick.lat ?? lat,
                lon: pickLon ?? lon,
              });
            } else {
              setErrorMessage(
                "Impossible de convertir votre position en adresse postale. Saisissez l'adresse de départ manuellement ou réessayez."
              );
            }
          }
        } catch {
          setErrorMessage(
            "Impossible de convertir votre position en adresse postale. Saisissez l'adresse de départ manuellement."
          );
        } finally {
          setPickupTouched(true);
          invalidatePreview();
          setPickupSuggestions([]);
          setIsResolvingLocation(false);
        }
      },
      () => {
        setIsResolvingLocation(false);
        setErrorMessage("Impossible de récupérer votre position actuelle.");
      },
      {
        enableHighAccuracy: true,
        timeout: 12000,
        maximumAge: 15000,
      }
    );
  }

  function applyResolvedFromSuggestion(
    field: "pickup" | "dropoff",
    firstValid: AddressAutocompleteSuggestion
  ): void {
    const la = firstValid.lat;
    const lo = effectiveSuggestionLon(firstValid);
    if (la == null || lo == null) {
      return;
    }
    const resolvedLabel = firstValid.address ?? firstValid.label;
    const nextSelection: { validated: true; lat: number; lon: number } = {
      validated: true,
      lat: la,
      lon: lo,
    };
    if (field === "pickup") {
      pickupLocationRef.current = resolvedLabel;
      setPickupLocation(resolvedLabel);
      setPickupSelection(nextSelection);
      setPickupSuggestions([]);
    } else {
      dropoffLocationRef.current = resolvedLabel;
      setDropoffLocation(resolvedLabel);
      setDestinationSelection(nextSelection);
      setDropoffSuggestions([]);
    }
  }

  async function resolveFieldForSubmit(
    field: "pickup" | "dropoff",
    valueOverride?: string
  ): Promise<ResolveFieldForSubmitResult> {
    const fromState = (field === "pickup" ? pickupLocation : dropoffLocation).trim();
    const value = (valueOverride != null && valueOverride.trim() !== "" ? valueOverride : fromState).trim();
    const selection = field === "pickup" ? pickupSelection : destinationSelection;

    const logResolve = (
      r: ResolveFieldForSubmitResult,
      extra?: { label?: string; place_id?: string; lat?: number | null; lon?: number | null; lng?: number | null; isGeocoded?: boolean }
    ) => {
      if (typeof __DEV__ === "undefined" || !__DEV__) {
        return;
      }
      // eslint-disable-next-line no-console
      console.log({
        tag: "resolveFieldForSubmit",
        phase: "end",
        field,
        value,
        label: extra?.label ?? value,
        place_id: extra?.place_id,
        lat: extra?.lat,
        lon: extra?.lon,
        lng: extra?.lng,
        isGeocoded: extra?.isGeocoded,
        selection,
        resultStatus: r.status,
        localizationOnly: r.status === "unresolved" ? r.localizationOnly : undefined,
        needsPickCount: r.status === "needs_pick_from_list" ? r.items.length : undefined,
      });
    };

    if (typeof __DEV__ !== "undefined" && __DEV__) {
      // eslint-disable-next-line no-console
      console.log({
        tag: "resolveFieldForSubmit",
        phase: "start",
        field,
        value,
        label: value,
        place_id: undefined,
        lat: undefined,
        lon: undefined,
        lng: undefined,
        isGeocoded: false,
        selection,
        resultStatus: "pending",
      });
    }

    if (
      selection &&
      selection.validated &&
      typeof selection.lat === "number" &&
      typeof selection.lon === "number" &&
      Number.isFinite(selection.lat) &&
      Number.isFinite(selection.lon)
    ) {
      const out: ResolveFieldForSubmitResult = { status: "ok" };
      logResolve(out, { isGeocoded: true });
      return out;
    }

    const fromCoords = parseCoordInput(value);
    if (fromCoords) {
      if (field === "pickup") {
        pickupLocationRef.current = value;
        setPickupLocation(value);
        setPickupSelection({
          validated: true,
          lat: fromCoords.lat,
          lon: fromCoords.lon,
        });
      } else {
        dropoffLocationRef.current = value;
        setDropoffLocation(value);
        setDestinationSelection({
          validated: true,
          lat: fromCoords.lat,
          lon: fromCoords.lon,
        });
      }
      const out: ResolveFieldForSubmitResult = { status: "ok" };
      logResolve(out, {
        isGeocoded: true,
        lat: fromCoords.lat,
        lon: fromCoords.lon,
      });
      return out;
    }

    if (value.length < 2) {
      const out: ResolveFieldForSubmitResult = { status: "unresolved" };
      logResolve(out);
      return out;
    }

    const dText = formatClientDomicile(profileQuery.data);
    const dCoords = getDomicileLatLon(profileQuery.data);
    if (field === "pickup" && dText && dCoords) {
      const vKey = normAddrKey(value);
      const dKey = normAddrKey(dText);
      if (vKey.length >= 2 && dKey.length >= 2 && vKey === dKey) {
        setPickupSelection({ validated: true, lat: dCoords.lat, lon: dCoords.lon });
        const out: ResolveFieldForSubmitResult = { status: "ok" };
        logResolve(out, {
          isGeocoded: true,
          lat: dCoords.lat,
          lon: dCoords.lon,
        });
        return out;
      }
    }

    const book = field === "pickup" ? pickupGeocodeBookRef : dropoffGeocodeBookRef;
    const fromBook = findGeocodedInAddressBook(book.current, value);
    if (fromBook && isGeocodedSuggestion(fromBook)) {
      recordGeocodedInAddressBook(book.current, fromBook);
      applyResolvedFromSuggestion(field, fromBook);
      const out: ResolveFieldForSubmitResult = { status: "ok" };
      logResolve(out, {
        isGeocoded: true,
        lat: fromBook.lat,
        lon: effectiveSuggestionLon(fromBook),
      });
      return out;
    }

    const options =
      field === "pickup" && locationBias
        ? { lat: locationBias.lat, lon: locationBias.lon, limit: 12 }
        : { limit: 12 };

    let sawAnyAutocompleteResults = false;
    let lastNonEmptyResults: AddressAutocompleteSuggestion[] = [];
    for (const q of collectAlternativeAddressQueries(value)) {
      try {
        const results = await autocompleteAddress(q, options);
        if (results.length > 0) {
          lastNonEmptyResults = results;
          sawAnyAutocompleteResults = true;
        }
        for (const it of results) {
          recordGeocodedInAddressBook(book.current, it);
        }
        const geo = uniqueGeocodedByPlaceOrCoord(results);
        const placeOnly = results.filter(
          (r) => Boolean(r.place_id) && !isGeocodedSuggestion(r)
        );
        if (typeof __DEV__ !== "undefined" && __DEV__) {
          // eslint-disable-next-line no-console
          console.log({
            tag: "resolveFieldForSubmit",
            afterAutocomplete: true,
            field,
            query: q,
            value,
            selection,
            resultsLength: results.length,
            geoLength: geo.length,
            placeOnlyLength: placeOnly.length,
            hasPlaceId: results.some((r) => Boolean(r.place_id)),
            results: results.map((r) => ({
              label: r.label,
              place_id: r.place_id,
              lat: r.lat,
              lon: r.lon,
              lng: r.lng,
              isGeocoded: isGeocodedSuggestion(r),
            })),
          });
        }
        if (geo.length === 1) {
          applyResolvedFromSuggestion(field, geo[0]!);
          if (field === "pickup") {
            setPickupRefineMessage(null);
          } else {
            setDropoffRefineMessage(null);
          }
          const out: ResolveFieldForSubmitResult = { status: "ok" };
          logResolve(out, {
            isGeocoded: true,
            label: (geo[0]!.address ?? geo[0]!.label) ?? value,
            place_id: geo[0]!.place_id,
            lat: geo[0]!.lat,
            lon: effectiveSuggestionLon(geo[0]!),
          });
          return out;
        }
        if (geo.length > 1) {
          if (field === "pickup") {
            setPickupRefineMessage(null);
          } else {
            setDropoffRefineMessage(null);
          }
          const out: ResolveFieldForSubmitResult = {
            status: "needs_pick_from_list",
            field,
            items: geo.slice(0, 8),
          };
          logResolve(out);
          return out;
        }
        if (geo.length === 0 && results.length > 1 && placeOnly.length >= 2) {
          if (field === "pickup") {
            setPickupRefineMessage(null);
          } else {
            setDropoffRefineMessage(null);
          }
          const out: ResolveFieldForSubmitResult = {
            status: "needs_pick_from_list",
            field,
            items: placeOnly.slice(0, 8),
          };
          logResolve(out);
          return out;
        }
        if (results.length === 1) {
          const u = results[0]!;
          if (isGeocodedSuggestion(u)) {
            applyResolvedFromSuggestion(field, u);
            if (field === "pickup") {
              setPickupRefineMessage(null);
            } else {
              setDropoffRefineMessage(null);
            }
            const out: ResolveFieldForSubmitResult = { status: "ok" };
            logResolve(out, {
              isGeocoded: true,
              label: (u.address ?? u.label) ?? value,
              place_id: u.place_id,
              lat: u.lat,
              lon: effectiveSuggestionLon(u),
            });
            return out;
          }
          if (u.place_id) {
            const merged = await mergeItemWithPlaceDetails(u);
            if (merged && isGeocodedSuggestion(merged)) {
              recordGeocodedInAddressBook(book.current, merged);
              applyResolvedFromSuggestion(field, merged);
              if (field === "pickup") {
                setPickupRefineMessage(null);
              } else {
                setDropoffRefineMessage(null);
              }
              const out: ResolveFieldForSubmitResult = { status: "ok" };
              logResolve(out, {
                isGeocoded: true,
                label: (merged.address ?? merged.label) ?? value,
                place_id: merged.place_id,
                lat: merged.lat,
                lon: effectiveSuggestionLon(merged),
              });
              return out;
            }
          }
        }
      } catch {
        /* tente la variante suivante */
      }
    }
    if (sawAnyAutocompleteResults && lastNonEmptyResults.length > 0) {
      if (field === "pickup") {
        setPickupSuggestions(lastNonEmptyResults.slice(0, 8));
      } else {
        setDropoffSuggestions(lastNonEmptyResults.slice(0, 8));
      }
    }
    const out: ResolveFieldForSubmitResult = {
      status: "unresolved",
      localizationOnly: sawAnyAutocompleteResults,
    };
    logResolve(out);
    return out;
  }

  async function ensureAddressResolvedForSubmit(
    field: "pickup" | "dropoff",
    valueOverride?: string
  ): Promise<boolean> {
    const r = await resolveFieldForSubmit(field, valueOverride);
    if (r.status === "needs_pick_from_list") {
      if (r.field === "pickup") setPickupSuggestions(r.items);
      else setDropoffSuggestions(r.items);
    }
    return r.status === "ok";
  }

  function validateDraft(validation?: {
    hasValidatedPickup?: boolean;
    hasValidatedDestination?: boolean;
  }): string | null {
    const domCoords = getDomicileLatLon(profileQuery.data);
    const profileDomicile = formatClientDomicile(profileQuery.data);
    const textNormMatchesDomicile = Boolean(
      profileDomicile &&
        normAddrKey(pickupLocation) === normAddrKey(String(profileDomicile))
    );
    const hasValidatedDomicileCoords = Boolean(textNormMatchesDomicile && domCoords);
    const selectionHasReliableGeocode = (s: {
      validated?: boolean;
      lat?: number;
      lon?: number;
    } | null) =>
      Boolean(
        s &&
          s.validated &&
          typeof s.lat === "number" &&
          typeof s.lon === "number" &&
          Number.isFinite(s.lat) &&
          Number.isFinite(s.lon)
      );
    const hasValidatedPickup =
      validation?.hasValidatedPickup ??
      Boolean(
        selectionHasReliableGeocode(pickupSelection) ||
          hasValidatedDomicileCoords ||
          parseCoordInput(pickupLocation)
      );
    const hasValidatedDestination =
      validation?.hasValidatedDestination ??
      Boolean(
        selectionHasReliableGeocode(destinationSelection) || parseCoordInput(dropoffLocation)
      );
    if (!draftPayload.pickup_location || !draftPayload.dropoff_location) {
      return "Les adresses de départ et destination sont requises.";
    }
    if (!hasValidatedPickup || !hasValidatedDestination) {
      return MSG_ADDRESS_CHOOSE_SUGGESTION;
    }
    if (!asap && (!selectedDate || !selectedTime)) {
      return "Veuillez sélectionner une date et une heure.";
    }
    let outboundMs = Date.now();
    if (!asap) {
      const outboundDateTime = new Date(`${selectedDate}T${selectedTime}:00`);
      if (!Number.isFinite(outboundDateTime.getTime())) {
        return "Date/heure de départ invalide.";
      }
      if (outboundDateTime.getTime() < Date.now() - 60 * 1000) {
        return "Veuillez choisir une date et une heure futures.";
      }
      outboundMs = outboundDateTime.getTime();
    }
    if (isRoundTrip) {
      if (!returnDate.trim()) {
        return "Pour un aller-retour, indiquez au moins la date de retour.";
      }
      const outboundDateStr = !asap && selectedDate ? selectedDate : todayDateMin;
      if (returnDate.trim() < outboundDateStr.trim()) {
        return "La date de retour ne peut pas être antérieure au départ.";
      }
      if (returnTime.trim()) {
        const returnDateTime = new Date(`${returnDate}T${returnTime}:00`);
        if (!Number.isFinite(returnDateTime.getTime())) {
          return "Date/heure de retour invalides.";
        }
        if (returnDateTime.getTime() <= outboundMs) {
          return "L'heure de retour doit être après le départ.";
        }
      }
    }
    if (isRecurring) {
      if (!draftPayload.recurrence_type) {
        return "Type de récurrence requis.";
      }
      if (
        draftPayload.recurrence_type === "custom" &&
        (!draftPayload.recurrence_days || draftPayload.recurrence_days.length === 0)
      ) {
        return "Sélectionnez au moins un jour pour la récurrence personnalisée.";
      }
      if (recurrenceEndDate.trim()) {
        const startDate = !asap && selectedDate ? selectedDate : todayDateMin;
        if (recurrenceEndDate.trim() < startDate.trim()) {
          return "La date de fin de série ne peut pas précéder le premier départ.";
        }
      } else {
        const n = Number(recurrenceLength);
        if (!Number.isFinite(n) || n < 1 || n > 52) {
          return "Le nombre de répétitions doit être compris entre 1 et 52.";
        }
      }
    }
    return null;
  }

  async function applyDomicileAsPickup(): Promise<void> {
    const d = formatClientDomicile(profileQuery.data);
    if (!d) return;
    const coords = getDomicileLatLon(profileQuery.data);
    setPickupTouched(true);
    pickupLocationRef.current = d;
    setPickupLocation(d);
    setPickupRefineMessage(null);
    if (coords) {
      setPickupSelection({ validated: true, lat: coords.lat, lon: coords.lon });
    } else {
      setPickupSelection({ validated: false, lat: undefined, lon: undefined });
    }
    invalidatePreview();
    if (Platform.OS === "web") {
      setPickupInputRevision((n) => n + 1);
    }
    await loadSuggestions("pickup", d);
  }

  function handleBackToDetails(): void {
    setFormStep("details");
    invalidatePreview();
    setIndicativeUi({ kind: "idle" });
  }

  async function handleGoToSummary(): Promise<void> {
    setErrorMessage(null);
    if (typeof __DEV__ !== "undefined" && __DEV__) {
      // eslint-disable-next-line no-console
      console.log({
        goToSummary: "pickup",
        label: pickupLocation,
        address: pickupLocation,
        selection: pickupSelection,
      });
      // eslint-disable-next-line no-console
      console.log({
        goToSummary: "dropoff",
        label: dropoffLocation,
        address: dropoffLocation,
        selection: destinationSelection,
      });
    }
    setVerifyingFieldOnContinue("pickup");
    const r1 = await resolveFieldForSubmit("pickup");
    if (r1.status === "needs_pick_from_list") {
      setVerifyingFieldOnContinue(null);
      setPickupSuggestions(r1.items);
      setErrorMessage(MSG_ADDRESS_CONFIRM_EXACT);
      return;
    }
    if (r1.status === "unresolved") {
      setVerifyingFieldOnContinue(null);
      void loadSuggestions("pickup", pickupLocation);
      setErrorMessage(
        r1.localizationOnly
          ? MSG_ADDRESS_LOCALIZE_FAILED
          : MSG_ADDRESS_CHOOSE_SUGGESTION
      );
      return;
    }
    setVerifyingFieldOnContinue("dropoff");
    const r2 = await resolveFieldForSubmit("dropoff");
    if (r2.status === "needs_pick_from_list") {
      setVerifyingFieldOnContinue(null);
      setDropoffSuggestions(r2.items);
      setErrorMessage(MSG_ADDRESS_CONFIRM_EXACT);
      return;
    }
    if (r2.status === "unresolved") {
      setVerifyingFieldOnContinue(null);
      void loadSuggestions("dropoff", dropoffLocation);
      setErrorMessage(
        r2.localizationOnly
          ? MSG_ADDRESS_LOCALIZE_FAILED
          : MSG_ADDRESS_CHOOSE_SUGGESTION
      );
      return;
    }
    setVerifyingFieldOnContinue(null);
    const validationError = validateDraft({
      hasValidatedPickup: true,
      hasValidatedDestination: true,
    });
    if (validationError) {
      setErrorMessage(validationError);
      return;
    }
    setErrorMessage(null);
    setFormStep("summary");
    setTimeout(() => {
      scrollRef.current?.scrollTo({ y: 0, animated: true });
    }, 0);
  }

  async function handleProceedToPayment(): Promise<void> {
    setErrorMessage(null);
    setVerifyingFieldOnContinue("pickup");
    const r1 = await resolveFieldForSubmit("pickup");
    if (r1.status === "needs_pick_from_list") {
      setVerifyingFieldOnContinue(null);
      setFormStep("details");
      setPickupSuggestions(r1.items);
      setErrorMessage(MSG_ADDRESS_CONFIRM_EXACT);
      return;
    }
    if (r1.status === "unresolved") {
      setVerifyingFieldOnContinue(null);
      setFormStep("details");
      void loadSuggestions("pickup", pickupLocation);
      setErrorMessage(
        r1.localizationOnly
          ? MSG_ADDRESS_LOCALIZE_FAILED
          : MSG_ADDRESS_CHOOSE_SUGGESTION
      );
      return;
    }
    setVerifyingFieldOnContinue("dropoff");
    const r2 = await resolveFieldForSubmit("dropoff");
    if (r2.status === "needs_pick_from_list") {
      setVerifyingFieldOnContinue(null);
      setFormStep("details");
      setDropoffSuggestions(r2.items);
      setErrorMessage(MSG_ADDRESS_CONFIRM_EXACT);
      return;
    }
    if (r2.status === "unresolved") {
      setVerifyingFieldOnContinue(null);
      setFormStep("details");
      void loadSuggestions("dropoff", dropoffLocation);
      setErrorMessage(
        r2.localizationOnly
          ? MSG_ADDRESS_LOCALIZE_FAILED
          : MSG_ADDRESS_CHOOSE_SUGGESTION
      );
      return;
    }
    setVerifyingFieldOnContinue(null);
    const validationError = validateDraft({
      hasValidatedPickup: true,
      hasValidatedDestination: true,
    });
    if (validationError) {
      setErrorMessage(validationError);
      setFormStep("details");
      return;
    }
    try {
      setErrorMessage(null);
      trackClientKpiEvent("booking_submit_clicked", {
        asap,
        isRoundTrip,
        isRecurring,
      });
      const activePreviewContractVersion =
        bootstrap?.preview_contract_version ??
        CLIENT_SURFACE_CONTRACT_VERSIONS.previewContractVersion;
      const hasFreshPreview =
        previewData &&
        previewMeta &&
        Date.now() - previewMeta.previewedAtMs <= PREVIEW_FRESHNESS_TTL_MS &&
        previewMeta.hash === currentFormHash &&
        previewMeta.previewContractVersion === activePreviewContractVersion;
      const preview = hasFreshPreview
        ? previewData
        : await previewMutation.mutateAsync(draftPayload);
      const previewContractVersion = preview.contracts?.preview_contract_version ?? null;
      if (
        previewContractVersion &&
        previewContractVersion !== activePreviewContractVersion
      ) {
        await bootstrapSession();
        setErrorMessage(
          "Le contrat de prévisualisation a changé. Rechargez la page puis continuez vers le paiement."
        );
        trackClientKpiEvent("preview_contract_version_mismatch", {
          expected: activePreviewContractVersion,
          received: previewContractVersion,
        });
        return;
      }
      setPreviewMeta({
        hash: currentFormHash,
        previewedAtMs: Date.now(),
        previewContractVersion,
      });
      const previewAmount = Number(preview?.pricing?.amount);
      if (!Number.isFinite(previewAmount) || previewAmount <= 0) {
        setErrorMessage("Prévisualisation tarifaire indisponible. Réessayez.");
        return;
      }
      const pickupCanonical = preview.canonical_addresses?.pickup;
      const dropoffCanonical = preview.canonical_addresses?.dropoff;
      const pickupPolicy = getCanonicalPolicy(preview, pickupCanonical?.precision_level);
      const dropoffPolicy = getCanonicalPolicy(preview, dropoffCanonical?.precision_level);
      if (
        !pickupCanonical?.canonical_hash ||
        !dropoffCanonical?.canonical_hash ||
        pickupPolicy === "block" ||
        dropoffPolicy === "block"
      ) {
        setErrorMessage(
          "Les adresses doivent être suffisamment précises pour continuer. Veuillez affiner la sélection."
        );
        trackClientKpiEvent("canonical_precision_blocked", {
          pickupPrecision: pickupCanonical?.precision_level ?? null,
          dropoffPrecision: dropoffCanonical?.precision_level ?? null,
        });
        return;
      }
      if (pickupPolicy === "warn" || dropoffPolicy === "warn") {
        setPrecisionWarning(
          "Adresse validée au niveau rue: vous pouvez continuer, mais une précision supplémentaire est recommandée."
        );
      } else {
        setPrecisionWarning(null);
      }
      const customerName =
        draftPayload.customer_name?.trim() || "Client";

      await createMutation.mutateAsync({
        ...draftPayload,
        customer_name: customerName,
        amount: previewAmount,
        preview_amount: previewAmount,
        ...(medicalFacility.trim() ? { medical_facility: medicalFacility.trim() } : {}),
        ...(doctorName.trim() ? { doctor_name: doctorName.trim() } : {}),
        ...(hospitalService.trim() ? { hospital_service: hospitalService.trim() } : {}),
        pickup_location: pickupCanonical?.label ?? draftPayload.pickup_location,
        dropoff_location: dropoffCanonical?.label ?? draftPayload.dropoff_location,
      });
      trackClientKpiEvent("booking_create_success", {
        pricingStatus: preview.pricing?.pricing_status ?? null,
        transmissionRequiresClientAction:
          preview.workflow?.transmission_requires_client_action ?? null,
      });
    } catch (error) {
      const e = error as Partial<ClientApiError> | { message?: string } | null;
      if (e && "code" in e && e.code === "preview_unavailable") {
        setErrorMessage(bookingPreviewErrorMessage(error));
        return;
      }
      const m = (e as { message?: string } | null)?.message?.trim();
      setErrorMessage(
        m || "Impossible de finaliser l'étape. Réessayez ou modifiez les informations."
      );
    }
  }

  const canProceedToPayment = !createMutation.isPending && !previewMutation.isPending;
  const currentFormHash = useMemo(
    () =>
      JSON.stringify({
        draft: draftPayload,
        medical_facility: medicalFacility.trim() || null,
        doctor_name: doctorName.trim() || null,
        hospital_service: hospitalService.trim() || null,
      }),
    [draftPayload, medicalFacility, doctorName, hospitalService]
  );
  const clientNotePreview = useMemo(
    () => buildClientNoteFromLegs(clientNoteDeparture, clientNoteArrival),
    [clientNoteArrival, clientNoteDeparture]
  );
  const clientNotePreviewLen = clientNotePreview.length;
  const todayDateMin = useMemo(() => new Date().toISOString().split("T")[0], []);

  const recurrenceStartYmd = !asap && selectedDate ? selectedDate : todayDateMin;
  const recurrenceSeriesMultiplier = useMemo(() => {
    if (!isRecurring) return 1;
    const end = String(recurrenceEndDate || "").trim();
    if (end) {
      return Math.max(
        1,
        estimatedOccurrencesForRecurrence({
          startYmd: recurrenceStartYmd,
          endYmd: end,
          recurrenceType,
          recurrenceDays,
        })
      );
    }
    const n = Math.min(52, Math.max(1, Math.floor(Number(recurrenceLength)) || 1));
    if (recurrenceType === "custom" && recurrenceDays.length > 0) {
      return Math.max(1, n * recurrenceDays.length);
    }
    return Math.max(1, n);
  }, [
    isRecurring,
    recurrenceEndDate,
    recurrenceStartYmd,
    recurrenceLength,
    recurrenceType,
    recurrenceDays,
  ]);

  const indicativeDisplayChf = useMemo(() => {
    if (indicativeUi.kind !== "ok") return null;
    let v = indicativeUi.data.indicative_amount_chf;
    if (isRoundTrip) v *= 2;
    if (recurrenceSeriesMultiplier > 1) v *= recurrenceSeriesMultiplier;
    return roundChfToFiveRappen(v);
  }, [indicativeUi, isRoundTrip, recurrenceSeriesMultiplier]);

  const indicativeLegalLine = useMemo(() => {
    if (indicativeUi.kind !== "ok" || indicativeDisplayChf == null) return "";
    const chunks: string[] = [];
    if (isRoundTrip) chunks.push("aller + retour (×2)");
    if (isRecurring) {
      chunks.push(
        recurrenceSeriesMultiplier > 1
          ? `série décrite (×${recurrenceSeriesMultiplier})`
          : "série indiquée"
      );
    }
    const tail =
      " Indicatif, non contractuel. Le prix final est confirmé à la prévisualisation (avant demande de transport).";
    if (chunks.length === 0) {
      return `Indicatif avant validation transporteur.${tail}`;
    }
    return `Indicatif : ${chunks.join(" · ")}, ordre de grandeur${tail}`;
  }, [indicativeUi, indicativeDisplayChf, isRoundTrip, isRecurring, recurrenceSeriesMultiplier]);

  useEffect(() => {
    const p = pickupLocation.trim();
    const d = dropoffLocation.trim();
    if (!p || !d) {
      setIndicativeUi({ kind: "idle" });
      return;
    }

    // Paire modifiée : ne pas conserver l'ancien montant pendant le debounce
    // (parité web).
    setIndicativeUi({ kind: "idle" });

    const t = setTimeout(() => {
      void (async () => {
        setIndicativeUi({ kind: "loading" });
        const r = await postIndicativeFareEstimate({ pickup_location: p, dropoff_location: d });
        if (r.success) {
          setIndicativeUi({ kind: "ok", data: r.data });
        } else {
          setIndicativeUi({ kind: "unavailable" });
        }
      })();
    }, 2000);
    return () => clearTimeout(t);
  }, [pickupLocation, dropoffLocation]);

  function toggleRecurrenceDay(dayId: number) {
    setRecurrenceDays((prev) =>
      prev.includes(dayId)
        ? prev.filter((d) => d !== dayId)
        : [...prev, dayId].sort((a, b) => a - b)
    );
    invalidatePreview();
  }

  function handleSwapAddresses() {
    const nextPickup = dropoffLocation;
    const nextDropoff = pickupLocation;
    const nextPickupSel = destinationSelection;
    const nextDropSel = pickupSelection;
    const tBook = pickupGeocodeBookRef.current;
    pickupGeocodeBookRef.current = dropoffGeocodeBookRef.current;
    dropoffGeocodeBookRef.current = tBook;
    setPickupTouched(true);
    setPickupSuggestions([]);
    setDropoffSuggestions([]);
    setPickupLocation(nextPickup);
    setDropoffLocation(nextDropoff);
    setPickupSelection(nextPickupSel);
    setDestinationSelection(nextDropSel);
    invalidatePreview();
    setErrorMessage(null);
    if (Platform.OS === "web") {
      setPickupInputRevision((n) => n + 1);
    }
  }

  function getCanonicalPolicy(
    preview: BookingPreviewResponse,
    level: CanonicalAddressPrecisionLevel | null | undefined
  ): "allow" | "warn" | "block" {
    if (!level) return "allow";
    const matrix = preview.validation?.canonical_precision_acceptance_matrix;
    return matrix?.[level] ?? DEFAULT_CANONICAL_MATRIX[level] ?? "allow";
  }

  function handleDatePickerChange(
    event: DateTimePickerEvent,
    value: Date | undefined
  ) {
    setShowDatePicker(false);
    if (event.type === "dismissed" || !value) return;
    setSelectedDate(formatDateYmd(value));
    invalidatePreview();
  }

  function handleTimePickerChange(
    event: DateTimePickerEvent,
    value: Date | undefined
  ) {
    setShowTimePicker(false);
    if (event.type === "dismissed" || !value) return;
    setSelectedTime(formatTimeHm(value));
    invalidatePreview();
  }

  function handleReturnDatePickerChange(
    event: DateTimePickerEvent,
    value: Date | undefined
  ) {
    setShowReturnDatePicker(false);
    if (event.type === "dismissed" || !value) return;
    setReturnDate(formatDateYmd(value));
    invalidatePreview();
  }

  function handleReturnTimePickerChange(
    event: DateTimePickerEvent,
    value: Date | undefined
  ) {
    setShowReturnTimePicker(false);
    if (event.type === "dismissed" || !value) return;
    setReturnTime(formatTimeHm(value));
    invalidatePreview();
  }

  useEffect(() => {
    if (pickupTouched || pickupLocation.trim().length > 0) return;
    const domicile = formatClientDomicile(profileQuery.data);
    if (!domicile) return;
    const domCoords = getDomicileLatLon(profileQuery.data);
    pickupLocationRef.current = domicile;
    setPickupLocation(domicile);
    if (domCoords) {
      setPickupSelection({ validated: true, lat: domCoords.lat, lon: domCoords.lon });
      setPickupRefineMessage(null);
    } else {
      setPickupSelection({ validated: false, lat: undefined, lon: undefined });
      setPickupRefineMessage(null);
      void loadSuggestions("pickup", domicile);
    }
  }, [pickupTouched, pickupLocation, profileQuery.data]);

  useEffect(() => {
    if (!previewData) {
      setPreviewMeta(null);
      setPrecisionWarning(null);
    }
  }, [previewData]);

  useEffect(() => {
    const defaults = oneHourLaterRoundedToFiveMinutes();
    setSelectedDate((prev) => prev || defaults.date);
    setSelectedTime((prev) => prev || defaults.time);
  }, []);

  useEffect(() => {
    const incomingDraftId = typeof params.publicDraftId === "string" ? params.publicDraftId.trim() : "";
    if (!incomingDraftId) return;
    void (async () => {
      let serverDraft = await fetchPublicPreRequestDraft(incomingDraftId);
      if (!serverDraft) {
        const localDraft = await loadPublicPreRequestDraft();
        if (localDraft && localDraft.draft_id === incomingDraftId) {
          serverDraft = {
            draft_id: localDraft.draft_id,
            departure: localDraft.departure,
            destination: localDraft.destination,
            date: localDraft.date,
            transport_type: localDraft.transport_type,
            special_requirements: localDraft.special_requirements ?? null,
            contact_email: localDraft.contact_email ?? null,
            contact_phone: localDraft.contact_phone ?? null,
            service_area_status: localDraft.service_area_status ?? null,
          };
        }
      }
      if (!serverDraft) return;
      setPickupTouched(true);
      const dep = (serverDraft.departure ?? "").trim();
      const dest = (serverDraft.destination ?? "").trim();
      pickupLocationRef.current = dep;
      dropoffLocationRef.current = dest;
      setPickupLocation(dep);
      setDropoffLocation(dest);
      if (serverDraft.date) {
        setSelectedDate(serverDraft.date);
      }
      if (serverDraft.pickup_time) {
        setSelectedTime(serverDraft.pickup_time);
      }
      setIsRoundTrip(serverDraft.trip_type === "round_trip");
      if (serverDraft.special_requirements) {
        setClientNoteDeparture(serverDraft.special_requirements);
      }
      await consumePublicPreRequestDraft(incomingDraftId).catch(() => undefined);
      await clearPublicPreRequestDraft().catch(() => undefined);
      if (dep) {
        await ensureAddressResolvedForSubmit("pickup", dep);
      }
      if (dest) {
        await ensureAddressResolvedForSubmit("dropoff", dest);
      }
      setErrorMessage(null);
    })();
  }, [params.publicDraftId]);

  useEffect(() => {
    if (!isRoundTrip) {
      setReturnDate("");
      setReturnTime("");
      return;
    }
    setReturnDate((prev) => prev || selectedDate || todayDateMin);
  }, [isRoundTrip, selectedDate, todayDateMin]);

  useEffect(() => {
    if (!dropoffLocation.trim()) {
      setMedicalFacility("");
      setDoctorName("");
      setHospitalService("");
      return;
    }
    const primaryPlaceLabel = extractPrimaryPlaceLabel(dropoffLocation);
    if (isHospitalLikeDestination(dropoffLocation)) {
      setMedicalFacility(primaryPlaceLabel);
    } else {
      setMedicalFacility("");
    }
    if (destinationHasDoctorHint(dropoffLocation)) {
      setDoctorName(primaryPlaceLabel);
    } else {
      setDoctorName("");
    }
  }, [dropoffLocation]);

  const domicileTextForUi = formatClientDomicile(profileQuery.data);
  const domicileServerCoords = getDomicileLatLon(profileQuery.data);

  const pickupAddressFieldLine = useMemo(() => {
    const t = pickupLocation.trim();
    if (
      isResolvingLocation ||
      verifyingFieldOnContinue === "pickup" ||
      suggestionResolving === "pickup"
    ) {
      return { kind: "verifying" as const };
    }
    if (!t) {
      return { kind: "empty" as const };
    }
    if (pickupRefineMessage) {
      return { kind: "refine" as const, text: pickupRefineMessage };
    }
    if (parseCoordInput(t)) {
      return { kind: "ok" as const };
    }
    if (
      pickupSelection &&
      pickupSelection.validated &&
      typeof pickupSelection.lat === "number" &&
      typeof pickupSelection.lon === "number"
    ) {
      return { kind: "ok" as const };
    }
    if (domicileTextForUi) {
      const domMatch =
        t.length >= 2 &&
        domicileTextForUi.length >= 2 &&
        normAddrKey(t) === normAddrKey(domicileTextForUi);
      if (domMatch && domicileServerCoords) {
        return { kind: "ok" as const };
      }
      if (domMatch && !domicileServerCoords) {
        return { kind: "domicile_confirm" as const };
      }
    }
    return { kind: "helper" as const };
  }, [
    domicileServerCoords,
    domicileTextForUi,
    isResolvingLocation,
    pickupLocation,
    pickupRefineMessage,
    pickupSelection,
    suggestionResolving,
    verifyingFieldOnContinue,
  ]);

  const dropoffAddressFieldLine = useMemo(() => {
    const t = dropoffLocation.trim();
    if (verifyingFieldOnContinue === "dropoff" || suggestionResolving === "dropoff") {
      return { kind: "verifying" as const };
    }
    if (!t) {
      return { kind: "empty" as const };
    }
    if (dropoffRefineMessage) {
      return { kind: "refine" as const, text: dropoffRefineMessage };
    }
    if (parseCoordInput(t)) {
      return { kind: "ok" as const };
    }
    if (
      destinationSelection &&
      destinationSelection.validated &&
      typeof destinationSelection.lat === "number" &&
      typeof destinationSelection.lon === "number"
    ) {
      return { kind: "ok" as const };
    }
    return { kind: "helper" as const };
  }, [
    destinationSelection,
    dropoffLocation,
    dropoffRefineMessage,
    suggestionResolving,
    verifyingFieldOnContinue,
  ]);

  const canGoToSummary = useMemo(
    () => !isResolvingLocation && suggestionResolving == null,
    [isResolvingLocation, suggestionResolving]
  );

  if (!activeContext || activeContext.context_type !== "client") {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center", padding: t.pageGap }}>
        <AppText variant="body">Contexte client requis.</AppText>
      </View>
    );
  }

  const hasTransporteurDetailsInSummary =
    Boolean(
      medicalFacility.trim() ||
        hospitalService.trim() ||
        doctorName.trim() ||
        buildClientNoteFromLegs(clientNoteDeparture, clientNoteArrival).trim()
    );

  const summaryHoraireLabel = asap
    ? "Dès que possible"
    : selectedDate && selectedTime
      ? `${new Date(`${selectedDate}T${selectedTime}:00`).toLocaleString("fr-CH", {
          weekday: "short",
          day: "2-digit",
          month: "short",
          hour: "2-digit",
          minute: "2-digit",
        })}`
      : "—";

  return (
    <PermissionGuard permission="booking:create">
      <ScrollView
        ref={scrollRef}
        style={styles.scroll}
        contentContainerStyle={[
          styles.scrollContent,
          {
            paddingTop: Math.max(16, topInset + 8),
            paddingBottom: Math.max(bottomBarPad, 24),
          },
        ]}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroPanel}>
          <AppText variant="sectionTitle" style={styles.title}>
            Transport médical
          </AppText>
          <Text testID="booking-form-step" style={styles.stepPill}>
            {formStep === "details" ? "1/2 · Demande" : "2/2 · Récapitulatif"}
          </Text>
          {formStep === "details" ? (
            <AppText variant="bodyMuted" style={styles.lead}>
              Indiquez le trajet et l’horaire. Vous pourrez ajouter des détails utiles si nécessaire.
            </AppText>
          ) : (
            <AppText variant="bodyMuted" style={styles.summaryLegal}>
              Votre demande sera transmise après le paiement.
            </AppText>
          )}
        </View>

        {formStep === "details" ? (
          <View>
        <View style={styles.formCard}>
          <AppText variant="caption" style={styles.cardEyebrow}>
            Itinéraire
          </AppText>
          <View style={styles.addressBlock}>
            <View style={styles.fieldLabelRow}>
              <View style={styles.fieldLabelIconWrap}>
                <Ionicons name="location-outline" size={18} color={ACCENT} />
              </View>
              <AppText variant="sectionTitle" style={styles.sectionLabel}>
                Prise en charge
              </AppText>
            </View>
            <TextInput
            key={Platform.OS === "web" ? `client-booking-pickup-${pickupInputRevision}` : undefined}
            autoComplete="off"
            value={pickupLocation}
            onChangeText={(value) => {
              pickupLocationRef.current = value;
              setPickupTouched(true);
              setPickupLocation(value);
              setPickupSelection(null);
              setPickupRefineMessage(null);
              invalidatePreview();
              setErrorMessage(null);
              void loadSuggestions("pickup", value);
            }}
            placeholder="Adresse ou établissement (ex. HUG)"
            placeholderTextColor="#94a3b8"
            style={[
              styles.input,
              pickupAddressFieldLine.kind === "ok" ? styles.inputRecognized : null,
              pickupAddressFieldLine.kind === "helper" ||
                pickupAddressFieldLine.kind === "domicile_confirm" ||
                pickupAddressFieldLine.kind === "refine"
                ? styles.inputNeedsConfirm
                : null,
            ]}
            accessibilityLabel="Adresse de prise en charge"
          />
          {pickupAddressFieldLine.kind === "verifying" ? (
            <View style={styles.addressFieldStatus}>
              <Ionicons name="time-outline" size={14} color="#64748b" />
              <Text style={[styles.addressFieldStatusText, styles.addressFieldStatusTextInfo]}>
                Vérification de l’adresse…
              </Text>
            </View>
          ) : pickupAddressFieldLine.kind === "ok" ? (
            <View style={styles.addressFieldStatus}>
              <Ionicons name="checkmark-circle" size={15} color="#16a34a" />
              <Text style={[styles.addressFieldStatusText, styles.addressFieldStatusTextOk]}>
                Adresse reconnue
              </Text>
            </View>
          ) : pickupAddressFieldLine.kind === "domicile_confirm" ? (
            <Text
              style={[styles.addressFieldStatusText, { marginTop: 4, color: "#b45309" }]}
            >
              {MSG_DOMICILE_CONFIRM}
            </Text>
          ) : pickupAddressFieldLine.kind === "refine" ? (
            <Text
              style={[styles.addressFieldStatusText, { marginTop: 4, color: "#b45309" }]}
            >
              {pickupAddressFieldLine.text}
            </Text>
          ) : pickupAddressFieldLine.kind === "helper" ? (
            <Text
              style={[styles.addressFieldStatusText, { marginTop: 4, color: "#b45309" }]}
            >
              {MSG_ADDRESS_LIST_HELPER}
            </Text>
          ) : null}
          <View style={styles.pickupActionsColumn}>
            <View style={styles.pickupActionRow}>
              <Pressable
                onPress={() => void applyCurrentLocationToPickup()}
                disabled={isResolvingLocation}
                accessibilityRole="button"
                accessibilityLabel="Utiliser ma position actuelle comme adresse de départ"
                android_ripple={{ color: "rgba(10, 126, 164, 0.12)" }}
                testID="booking-pickup-my-position"
                style={({ pressed }) => [
                  styles.locationBtn,
                  { flex: 1, minWidth: 120, opacity: isResolvingLocation ? 0.55 : 1 },
                  pressed && !isResolvingLocation && { backgroundColor: "#f8fafc" },
                ]}
              >
                <Ionicons
                  name={isResolvingLocation ? "hourglass-outline" : "navigate-outline"}
                  size={17}
                  color={ACCENT}
                />
                <Text style={styles.locationBtnText}>
                  {isResolvingLocation ? "En cours" : "Ma position"}
                </Text>
              </Pressable>
              {formatClientDomicile(profileQuery.data) ? (
                <Pressable
                  onPress={() => {
                    void applyDomicileAsPickup();
                  }}
                  accessibilityRole="button"
                  accessibilityLabel="Utiliser mon adresse de domicile enregistrée"
                  android_ripple={{ color: "rgba(10, 126, 164, 0.12)" }}
                  testID="booking-pickup-home"
                  style={({ pressed }) => [
                    styles.locationBtn,
                    { flex: 1, minWidth: 120 },
                    pressed && { backgroundColor: "#f8fafc" },
                  ]}
                >
                  <Ionicons name="home-outline" size={17} color={ACCENT} />
                  <Text style={styles.locationBtnText}>Domicile</Text>
                </Pressable>
              ) : null}
            </View>
          </View>
          {pickupSuggestions.length > 0 ? (
            <View style={styles.suggestionList}>
              {pickupSuggestions.slice(0, 4).map((item, sugIdx) => (
                <Pressable
                  key={`pickup-${item.place_id ?? item.label}`}
                  testID={`booking-suggestion-pickup-${sugIdx}`}
                  onPress={() => {
                    void onSuggestionSelected("pickup", item);
                  }}
                  {...(Platform.OS === "web"
                    ? ({
                        onMouseDown: (e: { preventDefault: () => void }) => e.preventDefault(),
                      } as Record<string, unknown>)
                    : {})}
                  style={({ pressed }) => [styles.suggestionPress, pressed && { opacity: 0.85 }]}
                >
                  <View style={styles.suggestionInner}>
                    <Ionicons name="location-outline" size={22} color={ACCENT} />
                    <View style={styles.suggestionTextCol}>
                      <Text style={styles.suggestionText} numberOfLines={3}>
                        {item.label}
                      </Text>
                      {!isGeocodedSuggestion(item) ? (
                        <Text style={styles.suggestionSub} numberOfLines={1}>
                          {SUGGESTION_SUBLABEL_NEEDS_GEO}
                        </Text>
                      ) : null}
                    </View>
                    <Ionicons name="chevron-forward" size={18} color="#94a3b8" />
                  </View>
                </Pressable>
              ))}
            </View>
          ) : null}
          </View>

          <View style={styles.routeDivider} />

          <View style={styles.addressBlock}>
          <View style={styles.destinationTopRow}>
            <View style={styles.destinationLabelCluster}>
              <View style={styles.fieldLabelIconWrap}>
                <Ionicons name="flag-outline" size={18} color={ACCENT} />
              </View>
              <AppText variant="sectionTitle" style={styles.sectionLabel} numberOfLines={1}>
                Arrivée
              </AppText>
            </View>
            <Pressable
              onPress={handleSwapAddresses}
              accessibilityRole="button"
              accessibilityLabel="Inverser les adresses de départ et d'arrivée"
              android_ripple={{ color: "rgba(15, 23, 42, 0.08)" }}
              style={({ pressed, hovered }) => [
                styles.swapBtn,
                hovered && Platform.OS === "web"
                  ? { borderColor: ACCENT, backgroundColor: "rgba(10, 126, 164, 0.06)" }
                  : null,
                pressed && { backgroundColor: "#f1f5f9", borderColor: ACCENT },
              ]}
            >
              <Ionicons name="swap-vertical" size={16} color="#64748b" />
              <Text style={styles.swapBtnText}>Inverser</Text>
            </Pressable>
          </View>
          <TextInput
            autoComplete="off"
            value={dropoffLocation}
            onChangeText={(value) => {
              dropoffLocationRef.current = value;
              setDropoffLocation(value);
              setDestinationSelection(null);
              setDropoffRefineMessage(null);
              invalidatePreview();
              setErrorMessage(null);
              void loadSuggestions("dropoff", value);
            }}
            placeholder="Établissement ou adresse d'arrivée"
            placeholderTextColor="#94a3b8"
            style={[
              styles.input,
              dropoffAddressFieldLine.kind === "ok" ? styles.inputRecognized : null,
              dropoffAddressFieldLine.kind === "helper" || dropoffAddressFieldLine.kind === "refine"
                ? styles.inputNeedsConfirm
                : null,
            ]}
            accessibilityLabel="Adresse de destination"
          />
          {dropoffAddressFieldLine.kind === "verifying" ? (
            <View style={styles.addressFieldStatus}>
              <Ionicons name="time-outline" size={14} color="#64748b" />
              <Text style={[styles.addressFieldStatusText, styles.addressFieldStatusTextInfo]}>
                Vérification de l’adresse…
              </Text>
            </View>
          ) : dropoffAddressFieldLine.kind === "ok" ? (
            <View style={styles.addressFieldStatus}>
              <Ionicons name="checkmark-circle" size={15} color="#16a34a" />
              <Text style={[styles.addressFieldStatusText, styles.addressFieldStatusTextOk]}>
                Adresse reconnue
              </Text>
            </View>
          ) : dropoffAddressFieldLine.kind === "refine" ? (
            <Text
              style={[styles.addressFieldStatusText, { marginTop: 4, color: "#b45309" }]}
            >
              {dropoffAddressFieldLine.text}
            </Text>
          ) : dropoffAddressFieldLine.kind === "helper" ? (
            <Text
              style={[styles.addressFieldStatusText, { marginTop: 4, color: "#b45309" }]}
            >
              {MSG_ADDRESS_LIST_HELPER}
            </Text>
          ) : null}
          {dropoffSuggestions.length > 0 ? (
            <View style={styles.suggestionList}>
              {dropoffSuggestions.slice(0, 4).map((item) => (
                <Pressable
                  key={`dropoff-${item.place_id ?? item.label}`}
                  onPress={() => {
                    void onSuggestionSelected("dropoff", item);
                  }}
                  {...(Platform.OS === "web"
                    ? ({
                        onMouseDown: (e: { preventDefault: () => void }) => e.preventDefault(),
                      } as Record<string, unknown>)
                    : {})}
                  style={({ pressed }) => [styles.suggestionPress, pressed && { opacity: 0.85 }]}
                >
                  <View style={styles.suggestionInner}>
                    <Ionicons name="navigate-outline" size={22} color={ACCENT} />
                    <View style={styles.suggestionTextCol}>
                      <Text style={styles.suggestionText} numberOfLines={3}>
                        {item.label}
                      </Text>
                      {!isGeocodedSuggestion(item) ? (
                        <Text style={styles.suggestionSub} numberOfLines={1}>
                          {SUGGESTION_SUBLABEL_NEEDS_GEO}
                        </Text>
                      ) : null}
                    </View>
                    <Ionicons name="chevron-forward" size={18} color="#94a3b8" />
                  </View>
                </Pressable>
              ))}
            </View>
          ) : null}
          </View>
        </View>

        <View style={styles.formCard}>
          <AppText variant="caption" style={styles.cardEyebrow}>
            Planification
          </AppText>
          <View style={styles.planningBody}>
            <View style={styles.planningSection}>
              <View style={styles.planningLabelRow}>
                <View style={styles.fieldLabelIconWrap}>
                  <Ionicons name="time-outline" size={18} color={ACCENT} />
                </View>
                <AppText variant="sectionTitle" style={styles.sectionLabel}>
                  Horaire du transport
                </AppText>
              </View>
              <View style={styles.segmentRow}>
                <Pressable
                  onPress={() => {
                    setAsap(true);
                    invalidatePreview();
                  }}
                  accessibilityRole="button"
                  accessibilityState={{ selected: asap }}
                  accessibilityLabel="Dès que possible"
                  android_ripple={{ color: "rgba(10, 126, 164, 0.12)" }}
                  style={({ pressed, hovered }) => [
                    styles.segment,
                    asap && styles.segmentActive,
                    hovered &&
                      Platform.OS === "web" &&
                      !asap && {
                        borderColor: "rgba(10, 126, 164, 0.35)",
                        backgroundColor: "#f8fafc",
                      },
                    pressed && { opacity: 0.92 },
                  ]}
                >
                  <Text style={[styles.segmentText, asap && styles.segmentTextActive]}>
                    Dès que possible
                  </Text>
                </Pressable>
                <Pressable
                  onPress={() => {
                    setAsap(false);
                    invalidatePreview();
                  }}
                  accessibilityRole="button"
                  accessibilityState={{ selected: !asap }}
                  accessibilityLabel="Planifier une date et une heure"
                  android_ripple={{ color: "rgba(10, 126, 164, 0.12)" }}
                  style={({ pressed, hovered }) => [
                    styles.segment,
                    !asap && styles.segmentActive,
                    hovered &&
                      Platform.OS === "web" &&
                      asap && {
                        borderColor: "rgba(10, 126, 164, 0.35)",
                        backgroundColor: "#f8fafc",
                      },
                    pressed && { opacity: 0.92 },
                  ]}
                >
                  <Text style={[styles.segmentText, !asap && styles.segmentTextActive]}>
                    Planifier un horaire
                  </Text>
                </Pressable>
              </View>
              {asap ? (
                <Text style={styles.planningMicroHint}>
                  Le créneau précis sera convenu avec le transporteur selon disponibilités.
                </Text>
              ) : null}
              {!asap ? (
                <View style={styles.dateRow}>
                  <Pressable
                    onPress={() => setShowDatePicker(true)}
                    accessibilityRole="button"
                    accessibilityLabel="Choisir la date du transport"
                    android_ripple={{ color: "rgba(10, 126, 164, 0.08)" }}
                  >
                    {({ pressed, hovered }) => (
                      <View
                        style={[
                          styles.dateTile,
                          hovered &&
                            Platform.OS === "web" && {
                              borderColor: "rgba(10, 126, 164, 0.35)",
                              backgroundColor: "#fafbfc",
                            },
                          pressed && { backgroundColor: "#f8fafc" },
                        ]}
                      >
                        <View style={styles.dateTileIconWrap}>
                          <Ionicons name="calendar-outline" size={18} color={ACCENT} />
                        </View>
                        <View style={styles.dateTileTextCol}>
                          <Text style={styles.dateTileLabel}>Date</Text>
                          <Text
                            style={[
                              styles.dateTileValue,
                              !selectedDate && styles.dateTilePlaceholder,
                            ]}
                            numberOfLines={1}
                          >
                            {selectedDate
                              ? new Date(selectedDate + "T12:00:00").toLocaleDateString(
                                  "fr-CH",
                                  {
                                    weekday: "short",
                                    day: "2-digit",
                                    month: "short",
                                    year: "numeric",
                                  }
                                )
                              : "Choisir"}
                          </Text>
                        </View>
                      </View>
                    )}
                  </Pressable>
                  <Pressable
                    onPress={() => setShowTimePicker(true)}
                    accessibilityRole="button"
                    accessibilityLabel="Choisir l'heure du transport"
                    android_ripple={{ color: "rgba(10, 126, 164, 0.08)" }}
                  >
                    {({ pressed, hovered }) => (
                      <View
                        style={[
                          styles.dateTile,
                          hovered &&
                            Platform.OS === "web" && {
                              borderColor: "rgba(10, 126, 164, 0.35)",
                              backgroundColor: "#fafbfc",
                            },
                          pressed && { backgroundColor: "#f8fafc" },
                        ]}
                      >
                        <View style={styles.dateTileIconWrap}>
                          <Ionicons name="alarm-outline" size={18} color={ACCENT} />
                        </View>
                        <View style={styles.dateTileTextCol}>
                          <Text style={styles.dateTileLabel}>Heure</Text>
                          <Text
                            style={[
                              styles.dateTileValue,
                              !selectedTime && styles.dateTilePlaceholder,
                            ]}
                            numberOfLines={1}
                          >
                            {selectedTime || "Choisir"}
                          </Text>
                        </View>
                      </View>
                    )}
                  </Pressable>
                </View>
              ) : null}
            </View>

            <View style={styles.planningDivider} />

            <View style={styles.planningSection}>
              <View style={styles.planningLabelRow}>
                <View style={styles.fieldLabelIconWrap}>
                  <Ionicons name="swap-horizontal-outline" size={18} color={ACCENT} />
                </View>
                <AppText variant="sectionTitle" style={styles.sectionLabel}>
                  Aller / retour
                </AppText>
              </View>
              <View style={styles.segmentRow}>
                <Pressable
                  onPress={() => {
                    setIsRoundTrip(false);
                    invalidatePreview();
                  }}
                  accessibilityRole="button"
                  accessibilityState={{ selected: !isRoundTrip }}
                  accessibilityLabel="Aller simple"
                  android_ripple={{ color: "rgba(10, 126, 164, 0.12)" }}
                  style={({ pressed, hovered }) => [
                    styles.segment,
                    !isRoundTrip && styles.segmentActive,
                    hovered &&
                      Platform.OS === "web" &&
                      isRoundTrip && {
                        borderColor: "rgba(10, 126, 164, 0.35)",
                        backgroundColor: "#f8fafc",
                      },
                    pressed && { opacity: 0.92 },
                  ]}
                >
                  <Text
                    style={[styles.segmentText, !isRoundTrip && styles.segmentTextActive]}
                  >
                    Aller simple
                  </Text>
                </Pressable>
                <Pressable
                  onPress={() => {
                    setIsRoundTrip(true);
                    invalidatePreview();
                  }}
                  accessibilityRole="button"
                  accessibilityState={{ selected: isRoundTrip }}
                  accessibilityLabel="Trajet avec retour planifié"
                  android_ripple={{ color: "rgba(10, 126, 164, 0.12)" }}
                  style={({ pressed, hovered }) => [
                    styles.segment,
                    isRoundTrip && styles.segmentActive,
                    hovered &&
                      Platform.OS === "web" &&
                      !isRoundTrip && {
                        borderColor: "rgba(10, 126, 164, 0.35)",
                        backgroundColor: "#f8fafc",
                      },
                    pressed && { opacity: 0.92 },
                  ]}
                >
                  <Text
                    style={[styles.segmentText, isRoundTrip && styles.segmentTextActive]}
                  >
                    Avec retour planifié
                  </Text>
                </Pressable>
              </View>
              {isRoundTrip ? (
                <View style={styles.dateRow}>
                  <Pressable
                    onPress={() => setShowReturnDatePicker(true)}
                    accessibilityRole="button"
                    accessibilityLabel="Choisir la date de retour"
                    android_ripple={{ color: "rgba(10, 126, 164, 0.08)" }}
                  >
                    {({ pressed, hovered }) => (
                      <View
                        style={[
                          styles.dateTile,
                          hovered &&
                            Platform.OS === "web" && {
                              borderColor: "rgba(10, 126, 164, 0.35)",
                              backgroundColor: "#fafbfc",
                            },
                          pressed && { backgroundColor: "#f8fafc" },
                        ]}
                      >
                        <View style={styles.dateTileIconWrap}>
                          <Ionicons name="calendar-outline" size={18} color={ACCENT} />
                        </View>
                        <View style={styles.dateTileTextCol}>
                          <Text style={styles.dateTileLabel}>Retour — date</Text>
                          <Text
                            style={[
                              styles.dateTileValue,
                              !returnDate && styles.dateTilePlaceholder,
                            ]}
                            numberOfLines={1}
                          >
                            {returnDate
                              ? new Date(returnDate + "T12:00:00").toLocaleDateString(
                                  "fr-CH",
                                  {
                                    weekday: "short",
                                    day: "2-digit",
                                    month: "short",
                                    year: "numeric",
                                  }
                                )
                              : "Choisir"}
                          </Text>
                        </View>
                      </View>
                    )}
                  </Pressable>
                  <Pressable
                    onPress={() => setShowReturnTimePicker(true)}
                    accessibilityRole="button"
                    accessibilityLabel="Choisir l'heure de retour, optionnel"
                    android_ripple={{ color: "rgba(10, 126, 164, 0.08)" }}
                  >
                    {({ pressed, hovered }) => (
                      <View
                        style={[
                          styles.dateTile,
                          hovered &&
                            Platform.OS === "web" && {
                              borderColor: "rgba(10, 126, 164, 0.35)",
                              backgroundColor: "#fafbfc",
                            },
                          pressed && { backgroundColor: "#f8fafc" },
                        ]}
                      >
                        <View style={styles.dateTileIconWrap}>
                          <Ionicons name="alarm-outline" size={18} color={ACCENT} />
                        </View>
                        <View style={styles.dateTileTextCol}>
                          <Text style={styles.dateTileLabel}>Retour — heure</Text>
                          <Text
                            style={[
                              styles.dateTileValue,
                              !returnTime && styles.dateTilePlaceholder,
                            ]}
                            numberOfLines={1}
                          >
                            {returnTime || "Optionnel"}
                          </Text>
                        </View>
                      </View>
                    )}
                  </Pressable>
                </View>
              ) : null}
            </View>

            <View style={styles.planningDivider} />

            <View style={styles.planningSection}>
              <View style={styles.rowBetween}>
                <View
                  style={{
                    flex: 1,
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 8,
                    minWidth: 0,
                  }}
                >
                  <View style={styles.fieldLabelIconWrapSm}>
                    <Ionicons name="repeat-outline" size={16} color={ACCENT} />
                  </View>
                  <View style={{ flex: 1, minWidth: 0, gap: 1 }}>
                    <Text style={styles.recurrenceCompactLabel}>Récurrence (option)</Text>
                    <Text style={styles.recurrenceCompactHint} numberOfLines={1}>
                      Même trajet sur plusieurs dates
                    </Text>
                  </View>
                </View>
                <Pressable
                  onPress={() => {
                    setIsRecurring((value) => !value);
                    invalidatePreview();
                  }}
                  accessibilityRole="button"
                  accessibilityState={{ selected: isRecurring }}
                  accessibilityLabel={
                    isRecurring
                      ? "Désactiver la demande récurrente"
                      : "Activer la demande récurrente"
                  }
                  android_ripple={{ color: "rgba(10, 126, 164, 0.12)" }}
                  style={({ pressed, hovered }) => [
                    styles.recurrenceToggle,
                    isRecurring && styles.recurrenceToggleOn,
                    { paddingVertical: 7, paddingHorizontal: 10 },
                    hovered &&
                      Platform.OS === "web" &&
                      !isRecurring && {
                        borderColor: "rgba(10, 126, 164, 0.4)",
                        backgroundColor: "#f8fafc",
                      },
                    pressed && { opacity: 0.92 },
                  ]}
                >
                  <Ionicons
                    name={isRecurring ? "checkmark-circle" : "ellipse-outline"}
                    size={16}
                    color={isRecurring ? ACCENT : "#94a3b8"}
                  />
                  <Text
                    style={[
                      styles.recurrenceToggleText,
                      { fontSize: 12 },
                      isRecurring && styles.recurrenceToggleTextOn,
                    ]}
                  >
                    Activer
                  </Text>
                </Pressable>
              </View>
              {isRecurring ? (
                <>
                  <View style={styles.recurrenceTypeRow}>
                    {(
                      [
                        ["daily", "Tous les jours"],
                        ["weekly", "Toutes les semaines"],
                        ["custom", "Jours perso."],
                      ] as const
                    ).map(([value, label]) => (
                      <Pressable
                        key={value}
                        onPress={() => {
                          setRecurrenceType(value);
                          invalidatePreview();
                        }}
                        accessibilityRole="button"
                        accessibilityState={{ selected: recurrenceType === value }}
                        accessibilityLabel={label}
                        android_ripple={{ color: "rgba(10, 126, 164, 0.1)" }}
                        style={({ pressed, hovered }) => [
                          styles.recurrenceTypeBtn,
                          recurrenceType === value && styles.recurrenceTypeBtnOn,
                          hovered &&
                            Platform.OS === "web" &&
                            recurrenceType !== value && {
                              borderColor: "rgba(10, 126, 164, 0.3)",
                              backgroundColor: "#fafbfc",
                            },
                          pressed && { opacity: 0.92 },
                        ]}
                      >
                        <Text style={styles.recurrenceTypeText}>{label}</Text>
                      </Pressable>
                    ))}
                  </View>
                  {recurrenceType === "custom" ? (
                    <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
                      {RECURRENCE_WEEK_DAYS.map((day) => {
                        const selected = recurrenceDays.includes(day.id);
                        return (
                          <Pressable
                            key={day.id}
                            onPress={() => toggleRecurrenceDay(day.id)}
                            style={({ pressed }) => [
                              styles.dayChip,
                              selected && styles.dayChipOn,
                              pressed && { opacity: 0.9 },
                            ]}
                          >
                            <Text
                              style={[
                                styles.dayChipText,
                                selected && styles.segmentTextActive,
                              ]}
                            >
                              {day.short}
                            </Text>
                          </Pressable>
                        );
                      })}
                    </View>
                  ) : null}
                  <TextInput
                    value={recurrenceLength}
                    onChangeText={(value) => {
                      setRecurrenceLength(value);
                      invalidatePreview();
                    }}
                    keyboardType="number-pad"
                    placeholder="Nombre de répétitions (1–52)"
                    placeholderTextColor="#94a3b8"
                    style={styles.input}
                  />
                  <TextInput
                    value={recurrenceEndDate}
                    onChangeText={(value) => {
                      setRecurrenceEndDate(value);
                      invalidatePreview();
                    }}
                    placeholder="Fin de série (AAAA-MM-JJ, optionnel)"
                    placeholderTextColor="#94a3b8"
                    style={styles.input}
                  />
                </>
              ) : null}
            </View>
          </View>
        </View>

        <View testID="booking-optional-details-card" style={styles.formCard}>
          <Pressable
            onPress={() => setOptionalDetailsExpanded((v) => !v)}
            testID="booking-optional-details-toggle"
            accessibilityRole="button"
            accessibilityState={{ expanded: optionalDetailsExpanded }}
            accessibilityLabel="Détails facultatifs pour le transport, afficher ou masquer"
            android_ripple={{ color: "rgba(10, 126, 164, 0.08)" }}
            style={({ pressed, hovered }) => [
              styles.rowBetween,
              { alignItems: "center", paddingVertical: 4, gap: 8 },
              pressed && { opacity: 0.9 },
              hovered && Platform.OS === "web" && { opacity: 0.96 },
            ]}
          >
            <View style={styles.instructionsToggleTextCol}>
              <AppText variant="sectionTitle" style={styles.sectionLabel}>
                Détails (facultatif)
              </AppText>
              <AppText variant="bodyMuted" style={[styles.sectionHint, { marginTop: 0 }]}>
                Ajoutez ces informations seulement si elles peuvent aider le transporteur.
              </AppText>
            </View>
            <Ionicons
              name={optionalDetailsExpanded ? "chevron-up" : "chevron-down"}
              size={20}
              color={ACCENT}
            />
          </Pressable>
          {optionalDetailsExpanded ? (
            <View style={styles.fieldStack}>
              <TextInput
                value={medicalFacility}
                onChangeText={(value) => {
                  setMedicalFacility(value.slice(0, 200));
                  invalidatePreview();
                }}
                placeholder="Établissement (hôpital, clinique, cabinet…)"
                placeholderTextColor="#94a3b8"
                style={styles.input}
                maxLength={200}
                accessibilityLabel="Établissement ou lieu de soins"
              />
              <TextInput
                value={hospitalService}
                onChangeText={(value) => {
                  setHospitalService(value.slice(0, MAX_HOSPITAL_SERVICE_LEN));
                  invalidatePreview();
                }}
                placeholder="Service ou unité (ex. urgences, cardiologie, imagerie)"
                placeholderTextColor="#94a3b8"
                style={styles.input}
                maxLength={MAX_HOSPITAL_SERVICE_LEN}
                accessibilityLabel="Service ou unité hospitalière"
              />
              <TextInput
                value={doctorName}
                onChangeText={(value) => {
                  setDoctorName(value.slice(0, 200));
                  invalidatePreview();
                }}
                placeholder="Médecin référent ou praticien (nom)"
                placeholderTextColor="#94a3b8"
                style={styles.input}
                maxLength={200}
                accessibilityLabel="Nom du médecin ou du praticien"
              />
              <Text style={styles.carrierSubLabel}>Notes au transporteur (départ / arrivée)</Text>
              <TextInput
                value={clientNoteDeparture}
                onChangeText={(value) => {
                  setClientNoteDeparture(value.slice(0, MAX_CLIENT_NOTE_LEG));
                  invalidatePreview();
                }}
                placeholder="Au départ (ex. RDV 9h, parking visiteurs)"
                placeholderTextColor="#94a3b8"
                style={styles.input}
                maxLength={MAX_CLIENT_NOTE_LEG}
                accessibilityLabel="Note pour le transporteur au départ"
              />
              <TextInput
                value={clientNoteArrival}
                onChangeText={(value) => {
                  setClientNoteArrival(value.slice(0, MAX_CLIENT_NOTE_LEG));
                  invalidatePreview();
                }}
                placeholder="À l'arrivée (ex. bât. B, 3e étage)"
                placeholderTextColor="#94a3b8"
                style={styles.input}
                maxLength={MAX_CLIENT_NOTE_LEG}
                accessibilityLabel="Note pour le transporteur à l’arrivée"
              />
              <Text style={styles.instructionsCharCount}>
                {clientNotePreviewLen} / {MAX_CLIENT_NOTE_LEN}
              </Text>
              {isRecurring ? (
                <Text style={styles.instructionsNoteFooterHint}>
                  Une ligne récapitulative (récurrence) est ajoutée côté serveur au-dessus de ce
                  texte dans la note transporteur.
                </Text>
              ) : null}
            </View>
          ) : null}
        </View>

        {showDatePicker ? (
          <DateTimePicker
            value={
              parseYmdHmToDate(
                selectedDate || todayDateMin,
                selectedTime || "09:00"
              ) ?? new Date()
            }
            mode="date"
            minimumDate={new Date(todayDateMin)}
            onChange={handleDatePickerChange}
          />
        ) : null}
        {showTimePicker ? (
          <DateTimePicker
            value={
              parseYmdHmToDate(
                selectedDate || todayDateMin,
                selectedTime || "09:00"
              ) ?? new Date()
            }
            mode="time"
            onChange={handleTimePickerChange}
          />
        ) : null}
        {showReturnDatePicker ? (
          <DateTimePicker
            value={
              parseYmdHmToDate(returnDate || selectedDate || todayDateMin, returnTime || "09:00") ??
              new Date()
            }
            mode="date"
            minimumDate={
              parseYmdHmToDate(selectedDate || todayDateMin, selectedTime || "00:00") ??
              new Date(todayDateMin)
            }
            onChange={handleReturnDatePickerChange}
          />
        ) : null}
        {showReturnTimePicker ? (
          <DateTimePicker
            value={
              parseYmdHmToDate(returnDate || selectedDate || todayDateMin, returnTime || "09:00") ??
              new Date()
            }
            mode="time"
            onChange={handleReturnTimePickerChange}
          />
        ) : null}
        </View>
        ) : null}

        {formStep === "summary" ? (
        <View>
        <View style={styles.formCard} testID="booking-summary-card">
          <Text style={styles.cardEyebrow}>Synthèse</Text>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Départ</Text>
            <Text style={styles.summaryValue} numberOfLines={4}>
              {pickupLocation.trim() || "—"}
            </Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Arrivée</Text>
            <Text style={styles.summaryValue} numberOfLines={4}>
              {dropoffLocation.trim() || "—"}
            </Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Horaire</Text>
            <Text style={styles.summaryValue} numberOfLines={3}>
              {summaryHoraireLabel}
            </Text>
          </View>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Trajet</Text>
            <Text style={styles.summaryValue}>
              {isRoundTrip
                ? returnDate.trim()
                  ? `Avec retour (retour le ${new Date(`${returnDate}T12:00:00`).toLocaleDateString("fr-CH", {
                      weekday: "short",
                      day: "2-digit",
                      month: "short",
                    })}${returnTime.trim() ? ` · ${returnTime}` : ""})`
                  : "Avec retour planifié"
                : "Aller simple"}
            </Text>
          </View>
          {isRecurring ? (
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Récurrence</Text>
              <Text style={styles.summaryValue} numberOfLines={4}>
                {recurrenceType === "daily"
                  ? "Tous les jours"
                  : recurrenceType === "weekly"
                    ? "Toutes les semaines"
                    : "Jours personnalisés"}
                {recurrenceEndDate.trim()
                  ? ` · fin ${recurrenceEndDate.trim()}`
                  : ` · ${recurrenceLength} occurrence(s) max.`}
              </Text>
            </View>
          ) : null}
          {hasTransporteurDetailsInSummary ? (
            <View style={{ marginTop: 4, gap: 6 }}>
              <Text style={[styles.summaryLabel, { width: "100%" }]}>Détails transport</Text>
              {medicalFacility.trim() ? (
                <Text style={styles.summaryValue}>
                  <Text style={{ color: "#64748b" }}>Établ. </Text>
                  {medicalFacility.trim()}
                </Text>
              ) : null}
              {hospitalService.trim() ? (
                <Text style={styles.summaryValue}>
                  <Text style={{ color: "#64748b" }}>Service </Text>
                  {hospitalService.trim()}
                </Text>
              ) : null}
              {doctorName.trim() ? (
                <Text style={styles.summaryValue}>
                  <Text style={{ color: "#64748b" }}>Praticien </Text>
                  {doctorName.trim()}
                </Text>
              ) : null}
              {buildClientNoteFromLegs(clientNoteDeparture, clientNoteArrival).trim() ? (
                <Text style={styles.summaryValue}>
                  {buildClientNoteFromLegs(clientNoteDeparture, clientNoteArrival)}
                </Text>
              ) : null}
            </View>
          ) : null}
        </View>

        {pickupLocation.trim() && dropoffLocation.trim() ? (
          <View style={styles.indicativeCard}>
            <Text style={styles.indicativeTitle}>Indicatif (ordre de grandeur)</Text>
            {indicativeUi.kind === "loading" ? (
              <Text style={styles.indicativeMeta}>Calcul en cours…</Text>
            ) : null}
            {indicativeUi.kind === "unavailable" ? (
              <Text style={styles.warnText}>{INDICATIVE_FARE_UNAVAILABLE_UX}</Text>
            ) : null}
            {indicativeUi.kind === "ok" && indicativeDisplayChf != null ? (
              <>
                <Text style={styles.indicativeAmount}>
                  {indicativeDisplayChf.toFixed(2)} CHF
                </Text>
                {indicativeUi.data.distance_m != null ? (
                  <Text style={styles.indicativeMeta}>
                    ≈ {Math.round((indicativeUi.data.duration_s || 0) / 60)} min ·{" "}
                    {((indicativeUi.data.distance_m || 0) / 1000).toFixed(1)} km
                  </Text>
                ) : null}
                <Text style={styles.indicativeFoot}>
                  Config v{indicativeUi.data.config_version} · {indicativeLegalLine}
                </Text>
              </>
            ) : null}
            <Text style={styles.indicativeFoot}>
              Indicatif serveur — le montant exact est confirmé à l’étape suivante (avant règlement en
              ligne).
            </Text>
          </View>
        ) : null}
        </View>
        ) : null}

        {errorMessage ? (
          <View style={styles.errorBanner}>
            <AppText variant="error" style={styles.errorText}>
              {errorMessage}
            </AppText>
          </View>
        ) : null}
        {precisionWarning ? (
          <View style={styles.warnBanner}>
            <AppText variant="body" style={styles.warnText}>
              {precisionWarning}
            </AppText>
          </View>
        ) : null}

        <View style={styles.actionsBlock}>
          {formStep === "details" ? (
            <Pressable
              onPress={() => void handleGoToSummary()}
              disabled={!canGoToSummary}
              testID="booking-cta-go-summary"
              accessibilityRole="button"
              accessibilityState={{ disabled: !canGoToSummary }}
              accessibilityLabel="Continuer vers le récapitulatif"
              android_ripple={
                canGoToSummary ? { color: "rgba(255, 255, 255, 0.2)" } : undefined
              }
              style={({ pressed, hovered }) => [
                styles.primaryButton,
                !canGoToSummary && styles.primaryButtonDisabled,
                hovered && Platform.OS === "web" && canGoToSummary && { opacity: 0.94 },
                pressed && canGoToSummary && { opacity: 0.9 },
              ]}
            >
              <Text style={styles.primaryButtonText}>Continuer vers le récapitulatif</Text>
            </Pressable>
          ) : null}

          {formStep === "summary" ? (
            <Pressable
              onPress={() => void handleProceedToPayment()}
              disabled={!canProceedToPayment}
              testID="booking-cta-proceed-payment"
              accessibilityRole="button"
              accessibilityState={{ disabled: !canProceedToPayment }}
              accessibilityLabel={
                createMutation.isPending || previewMutation.isPending
                  ? "Préparation du paiement en cours"
                  : "Continuer vers le paiement en ligne"
              }
              android_ripple={
                canProceedToPayment ? { color: "rgba(255, 255, 255, 0.2)" } : undefined
              }
              style={({ pressed, hovered }) => [
                styles.primaryButton,
                !canProceedToPayment && styles.primaryButtonDisabled,
                hovered && Platform.OS === "web" && canProceedToPayment && { opacity: 0.94 },
                pressed && canProceedToPayment && { opacity: 0.9 },
              ]}
            >
              <Text style={styles.primaryButtonText}>
                {createMutation.isPending || previewMutation.isPending
                  ? "Préparation du paiement…"
                  : "Continuer vers le paiement"}
              </Text>
            </Pressable>
          ) : null}

          {formStep === "summary" ? (
            <Pressable
              onPress={handleBackToDetails}
              testID="booking-cta-back-details"
              accessibilityRole="button"
              accessibilityLabel="Modifier ma demande de transport"
              android_ripple={{ color: "rgba(10, 126, 164, 0.1)" }}
              style={({ pressed, hovered }) => [
                styles.linkButton,
                hovered && Platform.OS === "web" && { opacity: 0.9 },
                pressed && { opacity: 0.85 },
              ]}
            >
              <Text style={styles.linkButtonText}>Modifier ma demande</Text>
            </Pressable>
          ) : null}

          <Pressable
            onPress={() => router.back()}
            accessibilityRole="button"
            accessibilityLabel="Retour sans enregistrer"
            android_ripple={{ color: "rgba(15, 23, 42, 0.06)" }}
            style={({ pressed, hovered }) => [
              styles.cancelButton,
              hovered && Platform.OS === "web" && { opacity: 0.85 },
              pressed && { opacity: 0.75 },
            ]}
          >
            <Text style={styles.cancelButtonText}>Annuler</Text>
          </Pressable>
        </View>
      </ScrollView>
    </PermissionGuard>
  );
}
