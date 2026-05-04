import { useMemo } from "react";
import { Redirect, useLocalSearchParams, useRouter } from "expo-router";
import { Pressable, StyleSheet, View } from "react-native";
import { PermissionGuard } from "../../../../src/core/guards";
import { useSession } from "../../../../src/core/sessionProvider";
import { bookingBelongsToActiveClient } from "../../../../src/features/client/accessControl";
import { useBookingDetailQuery, useClientProfileQuery } from "../../../../src/features/client/hooks";
import { getClientStatusUx } from "../../../../src/features/client/statusDictionary";
import {
  AppText,
  brandPrimary,
  brandSurfaceSoft,
  ResponsiveContainer,
  Screen,
  useAppViewport,
  useResponsiveTokens,
} from "../../../../src/design/responsive";

function parseBookingId(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

export default function ClientBookingDetailScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ bookingId?: string; pricingReason?: string; created?: string }>();
  const bookingId = parseBookingId(params.bookingId);
  const { activeContext } = useSession();

  const profileQuery = useClientProfileQuery();
  const bookingQuery = useBookingDetailQuery(bookingId);

  const t = useResponsiveTokens();
  const { horizontalPadding } = useAppViewport();

  const centeredStyle = useMemo(
    () => [
      styles.centered,
      {
        paddingHorizontal: horizontalPadding,
        gap: t.spacingMd,
        paddingVertical: t.spacingLg,
      },
    ],
    [horizontalPadding, t.spacingMd, t.spacingLg]
  );

  const pageStyle = useMemo(
    () => [
      styles.page,
      {
        paddingHorizontal: horizontalPadding,
        paddingTop: t.spacingSm + t.spacingXs,
        gap: t.spacingMd,
        paddingBottom: t.spacingLg,
      },
    ],
    [horizontalPadding, t.spacingSm, t.spacingXs, t.spacingMd, t.spacingLg]
  );

  if (!bookingId) {
    return (
      <Screen scroll backgroundColor={brandSurfaceSoft} contentContainerStyle={centeredStyle}>
        <ResponsiveContainer>
          <AppText variant="screenTitle">Identifiant réservation invalide.</AppText>
          <Pressable onPress={() => router.replace("/(app)/(client)/bookings")}>
            <AppText variant="body" style={styles.link}>
              Retour à la liste
            </AppText>
          </Pressable>
        </ResponsiveContainer>
      </Screen>
    );
  }

  if (!activeContext || activeContext.context_type !== "client") {
    return <Redirect href={"/(app)/unauthorized" as any} />;
  }

  const hasOwnership = bookingBelongsToActiveClient(bookingQuery.data, profileQuery.data);
  if (bookingQuery.data && profileQuery.data && !hasOwnership) {
    return <Redirect href={"/(app)/unauthorized" as any} />;
  }

  const paymentStatus = bookingQuery.data?.payment_status ?? "unknown";
  const paymentRequired = Boolean(bookingQuery.data?.payment_required);
  const canStartPayment = paymentRequired && paymentStatus !== "paid" && paymentStatus !== "pending_verification";
  const bookingStatusUx = getClientStatusUx(bookingQuery.data?.status);

  return (
    <PermissionGuard permission="booking:read:self">
      <Screen scroll backgroundColor={brandSurfaceSoft} withHorizontalPadding={false} contentContainerStyle={pageStyle}>
        <ResponsiveContainer>
          <AppText variant="screenTitle">Détail réservation</AppText>
          {bookingQuery.isLoading ? (
            <AppText variant="bodyMuted">Chargement du détail…</AppText>
          ) : null}
          {bookingQuery.isError ? (
            <AppText variant="error">
              Réservation introuvable ou inaccessible : {(bookingQuery.error as Error)?.message ?? "Erreur"}
            </AppText>
          ) : null}

          {bookingQuery.data ? (
            <View style={styles.card}>
              {params.created === "1" ? (
                <View style={styles.bannerOk}>
                  <AppText variant="caption" style={styles.bannerOkText}>
                    Réservation enregistrée avec succès.
                  </AppText>
                </View>
              ) : null}
              <AppText variant="sectionTitle" style={styles.cardTitle}>
                {bookingQuery.data.pickup_location ?? "Départ"}
                {" → "}
                {bookingQuery.data.dropoff_location ?? "Arrivée"}
              </AppText>
              <AppText variant="body">{bookingQuery.data.scheduled_time ?? "Date inconnue"}</AppText>
              <AppText variant="body">Statut : {bookingStatusUx.label}</AppText>
              <AppText variant="body">
                Transporteur : {bookingQuery.data.company_name ?? "Non attribué"}
              </AppText>
              <AppText variant="body">Paiement : {paymentStatus}</AppText>
              {bookingQuery.data.payment_amount ? (
                <AppText variant="body">
                  Montant : {bookingQuery.data.payment_amount} {bookingQuery.data.currency ?? "CHF"}
                </AppText>
              ) : null}
              {params.pricingReason ? (
                <View style={styles.bannerWarn}>
                  <AppText variant="caption" style={styles.bannerWarnText}>
                    Ajustement tarifaire : {params.pricingReason}
                  </AppText>
                </View>
              ) : null}
            </View>
          ) : null}

          {paymentRequired ? (
            <View style={styles.card}>
              <AppText variant="sectionTitle" style={styles.paySectionTitle}>
                Paiement en ligne
              </AppText>
              {paymentStatus === "pending_verification" ? (
                <AppText variant="body">Paiement en cours de confirmation…</AppText>
              ) : null}
              {paymentStatus === "failed" ? (
                <AppText variant="body">Paiement échoué ou annulé.</AppText>
              ) : null}
              {paymentStatus === "paid" ? <AppText variant="body">Paiement confirmé.</AppText> : null}
              {paymentStatus === "required" ? (
                <AppText variant="body">Paiement requis pour cette réservation.</AppText>
              ) : null}

              <Pressable
                onPress={() =>
                  router.push({
                    pathname: "/(app)/(client)/payment",
                    params: { bookingId: String(bookingId) },
                  })
                }
                disabled={!canStartPayment}
                style={({ pressed }) => [
                  styles.payBtn,
                  !canStartPayment && styles.payBtnDisabled,
                  pressed && canStartPayment && styles.payBtnPressed,
                ]}
              >
                <AppText variant="label" style={styles.payBtnText}>
                  {paymentStatus === "failed" ? "Réessayer le paiement" : "Payer maintenant"}
                </AppText>
              </Pressable>

              <Pressable onPress={() => void bookingQuery.refetch()} style={({ pressed }) => [styles.outlineBtn, pressed && { opacity: 0.88 }]}>
                <AppText variant="label" style={styles.outlineBtnText}>
                  Actualiser le statut
                </AppText>
              </Pressable>
            </View>
          ) : null}

          <Pressable onPress={() => router.replace("/(app)/(client)/bookings")}>
            <AppText variant="body" style={styles.link}>
              Retour aux réservations
            </AppText>
          </Pressable>
        </ResponsiveContainer>
      </Screen>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  centered: {
    flexGrow: 1,
    justifyContent: "center",
  },
  page: {
    flexGrow: 1,
  },
  card: {
    borderWidth: 1,
    borderColor: "#e2e8f0",
    borderRadius: 12,
    padding: 14,
    gap: 8,
    backgroundColor: "#fff",
  },
  cardTitle: {
    marginBottom: 2,
  },
  bannerOk: {
    backgroundColor: "#ecfdf3",
    borderRadius: 8,
    padding: 8,
    marginBottom: 4,
  },
  bannerOkText: {
    color: "#14532d",
  },
  bannerWarn: {
    backgroundColor: "#fff7ed",
    borderRadius: 8,
    padding: 8,
    marginTop: 4,
  },
  bannerWarnText: {
    color: "#9a3412",
  },
  paySectionTitle: {
    marginBottom: 4,
  },
  payBtn: {
    marginTop: 8,
    alignSelf: "flex-start",
    backgroundColor: brandPrimary,
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  payBtnDisabled: {
    opacity: 0.55,
  },
  payBtnPressed: {
    opacity: 0.9,
  },
  payBtnText: {
    color: "#fff",
  },
  outlineBtn: {
    alignSelf: "flex-start",
    marginTop: 10,
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 14,
    backgroundColor: "#fff",
  },
  outlineBtnText: {
    color: "#334155",
  },
  link: {
    color: brandPrimary,
    fontWeight: "600",
    marginTop: 4,
  },
});
