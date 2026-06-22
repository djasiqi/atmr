import { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  RefreshControl,
  StyleSheet,
  View,
} from "react-native";
import NetInfo from "@react-native-community/netinfo";
import { useLocalSearchParams, useRouter } from "expo-router";
import { AxiosError } from "axios";
import { Screen, AppText, AppButton } from "../../../../src/design/responsive";
import {
  useAcceptInstitutionOfferMutation,
  useInstitutionOfferDetailQuery,
  useRejectInstitutionOfferMutation,
} from "../../../../src/features/company/hooks";
import { parseInstitutionOfferApiError } from "../../../../src/features/company/api/institutionOffersApi";
import { resolveInstitutionOfferTerminalState } from "../../../../src/features/company/utils/institutionOfferResponse";
import { getOfferPushPreview } from "../../../../src/features/company/push/companyPush";
import { E } from "../../../../src/features/company/theme/enterpriseOpsTheme";
import {
  buildInstitutionMobilityChips,
  buildInstitutionRoutePoints,
  buildInstitutionScheduleLabel,
  buildInstitutionTripBadge,
  buildOfferStatusLabel,
  formatBirthDateCH,
  formatInstantDateTimeCH,
  formatMissionTypeLabel,
  formatPriceEstimateLabel,
  resolveInstitutionPatientName,
} from "../../../../src/features/company/utils/institutionOfferDisplay";
import { resolveInstitutionOfferActions } from "../../../../src/features/company/utils/institutionOfferActions";
import { computeAcceptNowPickupIso } from "../../../../src/features/company/utils/institutionOfferProposeTime";
import { PlanOfferTimeModal } from "../../../../src/features/company/components/offers/ProposeOfferTimeModal";
import { InstitutionOfferStateNotice } from "../../../../src/features/company/components/offers/InstitutionOfferStateNotice";

type OfferUiState =
  | "loading"
  | "active"
  | "offline_preview"
  | "already_accepted"
  | "already_rejected"
  | "unavailable"
  | "expired"
  | "cancelled"
  | "converted"
  | "error";

function resolveUiState(input: {
  isLoading: boolean;
  isError: boolean;
  online: boolean;
  offer: ReturnType<typeof useInstitutionOfferDetailQuery>["data"];
  conflictCode?: string;
}): OfferUiState {
  if (input.isLoading) return "loading";
  if (input.conflictCode === "OFFER_ALREADY_ACCEPTED") return "already_accepted";
  if (input.conflictCode === "OFFER_REJECTED") return "already_rejected";
  if (input.conflictCode === "OFFER_UNAVAILABLE") return "unavailable";
  if (input.conflictCode === "OFFER_EXPIRED") return "expired";
  if (input.conflictCode === "REQUEST_CANCELLED") return "cancelled";
  if (input.conflictCode === "REQUEST_CONVERTED") return "converted";
  if (input.isError && !input.online) return "offline_preview";
  if (input.isError) return "error";
  if (!input.offer) return "error";
  const status = String(input.offer.status ?? "").toUpperCase();
  if (status === "ACCEPTED") return "already_accepted";
  if (status === "REJECTED") return "already_rejected";

  const terminal = resolveInstitutionOfferTerminalState(input.offer);
  if (terminal === "expired") return "expired";
  if (terminal === "unavailable") return "unavailable";
  if (!input.online) return "offline_preview";
  return "active";
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={s.summaryRow}>
      <AppText variant="label" style={s.summaryLabel}>
        {label}
      </AppText>
      <AppText variant="body" style={s.summaryValue}>
        {value}
      </AppText>
    </View>
  );
}

export default function InstitutionOfferDetailScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ offerId?: string; request?: string }>();
  const offerId = Number(params.offerId);
  const [online, setOnline] = useState(true);
  const [conflictCode, setConflictCode] = useState<string | undefined>();
  const [actionError, setActionError] = useState<string | null>(null);
  const [bookingIdFromConflict, setBookingIdFromConflict] = useState<number | null>(null);
  const [planModalOpen, setPlanModalOpen] = useState(false);

  const preview = useMemo(
    () => (Number.isFinite(offerId) ? getOfferPushPreview(offerId) : undefined),
    [offerId]
  );

  const detailQuery = useInstitutionOfferDetailQuery(Number.isFinite(offerId) ? offerId : null);
  const acceptMutation = useAcceptInstitutionOfferMutation();
  const rejectMutation = useRejectInstitutionOfferMutation();

  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      setOnline(Boolean(state.isConnected) && state.isInternetReachable !== false);
    });
    void NetInfo.fetch().then((state) => {
      setOnline(Boolean(state.isConnected) && state.isInternetReachable !== false);
    });
    return unsubscribe;
  }, []);

  const offer = detailQuery.data;
  const req = offer?.transport_request;
  const uiState = resolveUiState({
    isLoading: detailQuery.isLoading,
    isError: detailQuery.isError,
    online,
    offer,
    conflictCode,
  });

  const institutionName = req?.institution_name ?? preview?.institution_name ?? "Institution";
  const patientName = resolveInstitutionPatientName(req, preview);
  const scheduleLabel = buildInstitutionScheduleLabel(req, preview);
  const routePoints = buildInstitutionRoutePoints(req);
  const tripBadge = buildInstitutionTripBadge(req, routePoints);
  const mobilityChips = buildInstitutionMobilityChips(req);
  const priceEstimate = formatPriceEstimateLabel(offer);
  const statusLabel = buildOfferStatusLabel(offer);
  const offerTerminal = resolveInstitutionOfferTerminalState(offer);
  const expiresLabel =
    formatInstantDateTimeCH(offer?.expires_at ?? preview?.expires_at) ?? null;
  const birthDate =
    formatBirthDateCH(req?.patient?.dob ?? req?.patient?.birth_date) ?? null;
  const missionType = formatMissionTypeLabel(req?.mission_type);
  const notes = req?.notes?.trim();

  const offerActions = useMemo(
    () => resolveInstitutionOfferActions(offer),
    [offer]
  );

  const onValidate = async () => {
    if (!Number.isFinite(offerId) || !online) return;
    setActionError(null);
    try {
      const result = await acceptMutation.mutateAsync({ offerId });
      if (result.booking_id != null) {
        router.replace({
          pathname: "/(app)/(company)/ride-details",
          params: { rideId: String(result.booking_id) },
        });
      }
    } catch (error) {
      const parsed = parseInstitutionOfferApiError(error);
      if (parsed.code) setConflictCode(parsed.code);
      if (parsed.booking_id != null) setBookingIdFromConflict(parsed.booking_id);
      setActionError(parsed.message);
    }
  };

  const onAcceptNow = async () => {
    if (!Number.isFinite(offerId) || !online) return;
    setActionError(null);
    try {
      const result = await acceptMutation.mutateAsync({
        offerId,
        proposedPickupTime: computeAcceptNowPickupIso(),
      });
      if (result.booking_id != null) {
        router.replace({
          pathname: "/(app)/(company)/ride-details",
          params: { rideId: String(result.booking_id) },
        });
      }
    } catch (error) {
      const parsed = parseInstitutionOfferApiError(error);
      if (parsed.code) setConflictCode(parsed.code);
      if (parsed.booking_id != null) setBookingIdFromConflict(parsed.booking_id);
      setActionError(parsed.message);
    }
  };

  const onReject = async () => {
    if (!Number.isFinite(offerId) || !online) return;
    setActionError(null);
    try {
      await rejectMutation.mutateAsync({ offerId });
      setConflictCode("OFFER_REJECTED");
    } catch (error) {
      const parsed = parseInstitutionOfferApiError(error);
      if (parsed.code) setConflictCode(parsed.code);
      setActionError(parsed.message);
    }
  };

  // Planifier = acceptation avec définition du pickup (pas de validation institution).
  const onPlanConfirm = async (_offerId: number, proposedPickupIso: string) => {
    if (!Number.isFinite(offerId) || !online) return;
    setActionError(null);
    try {
      const result = await acceptMutation.mutateAsync({
        offerId,
        proposedPickupTime: proposedPickupIso,
      });
      setPlanModalOpen(false);
      if (result.booking_id != null) {
        router.replace({
          pathname: "/(app)/(company)/ride-details",
          params: { rideId: String(result.booking_id) },
        });
      }
    } catch (error) {
      const parsed = parseInstitutionOfferApiError(error);
      if (parsed.code) setConflictCode(parsed.code);
      if (parsed.booking_id != null) setBookingIdFromConflict(parsed.booking_id);
      setActionError(parsed.message);
    }
  };

  const goToBooking = (bookingId: number) => {
    router.push({
      pathname: "/(app)/(company)/ride-details",
      params: { rideId: String(bookingId) },
    });
  };

  const goToOffers = () => {
    router.push("/(app)/(company)/offers");
  };

  return (
    <Screen
      scroll
      backgroundColor={E.BG}
      refreshControl={
        <RefreshControl
          refreshing={detailQuery.isFetching}
          onRefresh={() => void detailQuery.refetch()}
          tintColor={E.BRAND}
        />
      }
    >
      <AppText variant="sectionTitle" style={s.pageTitle}>
        Demande institution
      </AppText>

      {uiState === "loading" ? (
        <ActivityIndicator color={E.BRAND} style={{ marginTop: 24 }} />
      ) : null}

      {!online || uiState === "offline_preview" ? (
        <View style={s.banner}>
          <AppText variant="bodyMuted">
            Connexion requise pour accepter ou refuser. Aperçu issu de la notification.
          </AppText>
        </View>
      ) : null}

      <View style={s.headerCard}>
        <View style={s.headerTop}>
          <View style={s.headerTitles}>
            <AppText variant="sectionTitle" style={s.patientTitle}>
              {patientName}
            </AppText>
            {expiresLabel ? (
              <AppText variant="bodyMuted" style={s.expMeta}>
                Exp: {expiresLabel}
              </AppText>
            ) : null}
          </View>
          <View
            style={[
              s.statusBadge,
              offerTerminal === "expired" && s.statusBadgeExpired,
              offerTerminal === "unavailable" && s.statusBadgeUnavailable,
            ]}
          >
            <AppText
              variant="label"
              style={[
                s.statusBadgeText,
                offerTerminal === "expired" && s.statusBadgeTextExpired,
                offerTerminal === "unavailable" && s.statusBadgeTextUnavailable,
              ]}
            >
              {statusLabel}
            </AppText>
          </View>
        </View>
        <AppText variant="bodyMuted">{institutionName}</AppText>
      </View>

      {uiState === "active" && offerActions.canRespond ? (
        <View style={s.actionsWrap}>
          {offerActions.hint ? (
            <AppText variant="bodyMuted" style={s.actionsHint}>
              {offerActions.hint}
            </AppText>
          ) : null}
          <View style={s.actions}>
            {offerActions.canValidate ? (
              <AppButton
                title={acceptMutation.isPending ? "Validation…" : offerActions.validateLabel}
                onPress={() => void onValidate()}
                disabled={acceptMutation.isPending || rejectMutation.isPending}
                style={s.actionBtn}
              />
            ) : null}
            {offerActions.canAcceptNow ? (
              <AppButton
                title={acceptMutation.isPending ? "Prise en charge…" : offerActions.acceptNowLabel}
                onPress={() => void onAcceptNow()}
                disabled={acceptMutation.isPending || rejectMutation.isPending}
                style={s.actionBtn}
              />
            ) : null}
            {offerActions.canPlan ? (
              <AppButton
                title={offerActions.planLabel}
                variant="secondary"
                onPress={() => setPlanModalOpen(true)}
                disabled={acceptMutation.isPending || rejectMutation.isPending}
                style={s.actionBtn}
              />
            ) : null}
            {offerActions.canReject ? (
              <AppButton
                title={rejectMutation.isPending ? "Refus…" : offerActions.rejectLabel}
                variant="secondary"
                onPress={() => void onReject()}
                disabled={acceptMutation.isPending || rejectMutation.isPending}
                style={s.actionBtn}
              />
            ) : null}
          </View>
        </View>
      ) : null}

      {uiState === "expired" ? (
        <InstitutionOfferStateNotice
          variant="expired"
          expiresLabel={expiresLabel}
          onPrimaryPress={goToOffers}
        />
      ) : null}

      {uiState === "unavailable" ? (
        <InstitutionOfferStateNotice variant="unavailable" onPrimaryPress={goToOffers} />
      ) : null}

      {uiState === "already_rejected" ? (
        <InstitutionOfferStateNotice variant="rejected" onPrimaryPress={goToOffers} />
      ) : null}

      {uiState === "cancelled" ? (
        <InstitutionOfferStateNotice variant="cancelled" onPrimaryPress={goToOffers} />
      ) : null}

      {uiState === "converted" ? (
        <InstitutionOfferStateNotice variant="converted" onPrimaryPress={goToOffers} />
      ) : null}

      <PlanOfferTimeModal
        visible={planModalOpen}
        offer={offer}
        pending={acceptMutation.isPending}
        onClose={() => setPlanModalOpen(false)}
        onConfirm={(id, iso) => void onPlanConfirm(id, iso)}
      />

      <View style={s.card}>
        <AppText variant="sectionTitle" style={s.sectionTitle}>
          Informations
        </AppText>
        <View style={s.summaryGrid}>
          <SummaryRow label="Passager" value={patientName} />
          {birthDate ? <SummaryRow label="Date de naissance" value={birthDate} /> : null}
          <SummaryRow label="Origine" value={institutionName} />
          <SummaryRow label="Horaire" value={scheduleLabel} />
          {missionType ? <SummaryRow label="Type" value={missionType} /> : null}
          {priceEstimate ? (
            <SummaryRow label={priceEstimate.label} value={priceEstimate.value} />
          ) : null}
        </View>
      </View>

      {routePoints.length > 0 ? (
        <View style={s.card}>
          <AppText variant="sectionTitle" style={s.sectionTitle}>
            Trajet
          </AppText>
          {routePoints.map((point, index) => (
            <View key={point.key} style={s.routeStop}>
              <View style={s.routeMarkerCol}>
                <View
                  style={[
                    s.routeDot,
                    index === 0 && s.routeDotStart,
                    index > 0 && index < routePoints.length - 1 && s.routeDotMid,
                    index === routePoints.length - 1 && s.routeDotEnd,
                  ]}
                />
                {index < routePoints.length - 1 ? <View style={s.routeConnector} /> : null}
              </View>
              <View style={s.routeBody}>
                <AppText variant="label" style={s.routeLabel}>
                  {point.label}
                  {point.timeLabel ? (
                    <AppText variant="label" style={s.routeTime}>
                      {" "}
                      · {point.timeLabel}
                    </AppText>
                  ) : null}
                </AppText>
                <AppText variant="bodyMuted" style={s.routeAddress}>
                  {point.address}
                </AppText>
                {point.details ? (
                  <AppText variant="bodyMuted" style={s.routeDetails}>
                    {point.details}
                  </AppText>
                ) : null}
              </View>
            </View>
          ))}
          {tripBadge ? (
            <View style={s.tripBadge}>
              <AppText variant="label" style={s.tripBadgeText}>
                {tripBadge}
              </AppText>
            </View>
          ) : null}
        </View>
      ) : null}

      {mobilityChips.length > 0 ? (
        <View style={s.card}>
          <AppText variant="sectionTitle" style={s.sectionTitle}>
            Besoins
          </AppText>
          <View style={s.chipRow}>
            {mobilityChips.map((chip) => (
              <View
                key={chip.key}
                style={[s.chip, chip.danger ? s.chipDanger : s.chipActive]}
              >
                <AppText
                  variant="label"
                  style={[s.chipText, chip.danger ? s.chipTextDanger : s.chipTextActive]}
                >
                  {chip.label}
                </AppText>
              </View>
            ))}
          </View>
        </View>
      ) : null}

      {notes ? (
        <View style={s.card}>
          <AppText variant="sectionTitle" style={s.sectionTitle}>
            Notes
          </AppText>
          <AppText variant="bodyMuted">{notes}</AppText>
        </View>
      ) : null}

      {uiState === "already_accepted" ? (
        <View style={s.card}>
          <AppText variant="body">Cette offre a déjà été acceptée.</AppText>
          {(bookingIdFromConflict ?? offer?.transport_request?.id) != null ? (
            <AppButton
              title="Voir la course"
              onPress={() =>
                bookingIdFromConflict != null
                  ? goToBooking(bookingIdFromConflict)
                  : router.push("/(app)/(company)/dashboard")
              }
            />
          ) : null}
        </View>
      ) : null}

      {uiState === "error" && detailQuery.error instanceof AxiosError ? (
        <AppText variant="error" style={s.info}>
          {parseInstitutionOfferApiError(detailQuery.error).message}
        </AppText>
      ) : null}

      {actionError ? (
        <AppText variant="error" style={s.info}>
          {actionError}
        </AppText>
      ) : null}
    </Screen>
  );
}

const s = StyleSheet.create({
  pageTitle: { marginBottom: 12, color: E.TEXT },
  banner: {
    backgroundColor: "rgba(255, 152, 0, 0.12)",
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
  },
  headerCard: {
    backgroundColor: E.CARD,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: E.BORDER,
    padding: 14,
    marginBottom: 12,
    gap: 6,
  },
  headerTop: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 10,
  },
  headerTitles: { flex: 1, gap: 2 },
  patientTitle: { color: E.TEXT },
  expMeta: { fontSize: 12 },
  statusBadge: {
    backgroundColor: "rgba(245, 158, 11, 0.15)",
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  statusBadgeExpired: {
    backgroundColor: "rgba(100, 116, 139, 0.15)",
  },
  statusBadgeUnavailable: {
    backgroundColor: "rgba(100, 116, 139, 0.15)",
  },
  statusBadgeText: { color: E.URGENT, fontSize: 11 },
  statusBadgeTextExpired: { color: E.TEXT_SEC },
  statusBadgeTextUnavailable: { color: E.TEXT_SEC },
  card: {
    backgroundColor: E.CARD,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: E.BORDER,
    padding: 14,
    marginBottom: 12,
    gap: 8,
  },
  sectionTitle: { color: E.TEXT, marginBottom: 4 },
  summaryGrid: { gap: 10 },
  summaryRow: { gap: 2 },
  summaryLabel: { color: E.TEXT_SEC, fontSize: 12 },
  summaryValue: { color: E.TEXT },
  routeStop: { flexDirection: "row", gap: 10 },
  routeMarkerCol: { width: 14, alignItems: "center" },
  routeDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: E.TEXT_MUTED,
    marginTop: 4,
  },
  routeDotStart: { backgroundColor: E.BRAND },
  routeDotMid: { backgroundColor: E.TEXT_SEC },
  routeDotEnd: { backgroundColor: E.BRAND_DARK },
  routeConnector: {
    flex: 1,
    width: 2,
    backgroundColor: E.BORDER,
    marginVertical: 2,
    minHeight: 16,
  },
  routeBody: { flex: 1, paddingBottom: 10 },
  routeLabel: { color: E.TEXT },
  routeTime: { color: E.TEXT_SEC, fontWeight: "400" },
  routeAddress: { marginTop: 2 },
  routeDetails: { marginTop: 2, fontSize: 13 },
  tripBadge: {
    alignSelf: "flex-start",
    backgroundColor: "rgba(0, 121, 107, 0.1)",
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 4,
    marginTop: 4,
  },
  tripBadgeText: { color: E.BRAND_DARK, fontSize: 12 },
  chipRow: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  chip: {
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderWidth: 1,
  },
  chipActive: {
    backgroundColor: "rgba(0, 121, 107, 0.1)",
    borderColor: "rgba(0, 121, 107, 0.25)",
  },
  chipDanger: {
    backgroundColor: "rgba(220, 53, 69, 0.08)",
    borderColor: "rgba(220, 53, 69, 0.25)",
  },
  chipText: { fontSize: 12 },
  chipTextActive: { color: E.BRAND_DARK },
  chipTextDanger: { color: E.DANGER },
  actionsWrap: { marginBottom: 12, gap: 8 },
  actionsHint: { fontSize: 13, lineHeight: 18 },
  actions: {
    flexDirection: "row",
    alignItems: "stretch",
    gap: 8,
    marginBottom: 12,
  },
  actionBtn: { flex: 1, minWidth: 0 },
  info: { marginTop: 12 },
});
