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
import { Screen, AppText, AppButton } from "../../../src/design/responsive";
import {
  useAcceptInstitutionOfferMutation,
  useInstitutionOfferDetailQuery,
  useRejectInstitutionOfferMutation,
} from "../../../src/features/company/hooks";
import { parseInstitutionOfferApiError } from "../../../src/features/company/api/institutionOffersApi";
import {
  canRespondToInstitutionOffer,
} from "../../../src/features/company/utils/institutionOfferResponse";
import { getOfferPushPreview } from "../../../src/features/company/push/companyPush";
import { E } from "../../../src/features/company/theme/enterpriseOpsTheme";

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
  if (!canRespondToInstitutionOffer(input.offer)) {
    if (status === "EXPIRED") return "expired";
    return "unavailable";
  }
  if (!input.online) return "offline_preview";
  return "active";
}

export default function InstitutionOfferDetailScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ offerId?: string; request?: string }>();
  const offerId = Number(params.offerId);
  const [online, setOnline] = useState(true);
  const [conflictCode, setConflictCode] = useState<string | undefined>();
  const [actionError, setActionError] = useState<string | null>(null);
  const [bookingIdFromConflict, setBookingIdFromConflict] = useState<number | null>(null);

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
  const uiState = resolveUiState({
    isLoading: detailQuery.isLoading,
    isError: detailQuery.isError,
    online,
    offer,
    conflictCode,
  });

  const institutionName =
    offer?.transport_request?.institution_name ?? preview?.institution_name ?? "Institution";
  const patientName = preview?.patient_name ?? "Patient";
  const scheduleLabel =
    preview?.scheduled_time_label ??
    offer?.transport_request?.scheduled_time ??
    "Horaire à confirmer";
  const pickup = offer?.transport_request?.pickup_location ?? "—";
  const dropoff = offer?.transport_request?.dropoff_location ?? "—";

  const onAccept = async () => {
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

  const goToBooking = (bookingId: number) => {
    router.push({
      pathname: "/(app)/(company)/ride-details",
      params: { rideId: String(bookingId) },
    });
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

      <View style={s.card}>
        <AppText variant="sectionTitle" style={s.title}>
          {institutionName}
        </AppText>
        <AppText variant="body">{patientName}</AppText>
        <AppText variant="label" style={s.meta}>
          {scheduleLabel}
        </AppText>
        {offer ? (
          <>
            <AppText variant="bodyMuted" style={s.route}>
              Départ : {pickup}
            </AppText>
            <AppText variant="bodyMuted" style={s.route}>
              Arrivée : {dropoff}
            </AppText>
          </>
        ) : null}
      </View>

      {uiState === "active" ? (
        <View style={s.actions}>
          <AppButton
            title={acceptMutation.isPending ? "Acceptation…" : "Accepter"}
            onPress={() => void onAccept()}
            disabled={acceptMutation.isPending || rejectMutation.isPending}
          />
          <AppButton
            title={rejectMutation.isPending ? "Refus…" : "Refuser"}
            variant="secondary"
            onPress={() => void onReject()}
            disabled={acceptMutation.isPending || rejectMutation.isPending}
          />
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

      {uiState === "already_rejected" ? (
        <AppText variant="bodyMuted" style={s.info}>
          Déjà refusée par votre entreprise.
        </AppText>
      ) : null}

      {uiState === "unavailable" ? (
        <AppText variant="bodyMuted" style={s.info}>
          Demande déjà prise par un autre transporteur.
        </AppText>
      ) : null}

      {uiState === "expired" ? (
        <View style={s.actions}>
          <AppText variant="bodyMuted" style={s.info}>
            Cette offre a expiré.
          </AppText>
          <AppButton title="Retour au dashboard" onPress={() => router.push("/(app)/(company)/dashboard")} />
        </View>
      ) : null}

      {uiState === "cancelled" ? (
        <AppText variant="bodyMuted" style={s.info}>
          La demande a été annulée par l&apos;institution.
        </AppText>
      ) : null}

      {uiState === "converted" ? (
        <AppText variant="bodyMuted" style={s.info}>
          Cette demande a déjà été convertie en course.
        </AppText>
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
  card: {
    backgroundColor: E.CARD,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: E.BORDER,
    padding: 14,
    marginBottom: 12,
    gap: 6,
  },
  title: { color: E.TEXT },
  meta: { color: E.BRAND, marginTop: 4 },
  route: { marginTop: 2 },
  actions: { gap: 10, marginTop: 8 },
  info: { marginTop: 12 },
});
