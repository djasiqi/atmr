import { useMemo } from "react";
import { useRouter } from "expo-router";
import { Pressable, StyleSheet, View } from "react-native";
import { PermissionGuard } from "../../../src/core/guards";
import { useClientBookingsQuery } from "../../../src/features/client/hooks";
import { useClientBottomContentPadding } from "../../../src/features/client/navigation/ClientFloatingAppBar";
import { getClientStatusUx } from "../../../src/features/client/statusDictionary";
import {
  AppText,
  brandPrimary,
  Screen,
  useAppViewport,
  useResponsiveTokens,
} from "../../../src/design/responsive";

export default function ClientBookingsScreen() {
  const router = useRouter();
  const t = useResponsiveTokens();
  const { horizontalPadding } = useAppViewport();
  const bottomPad = useClientBottomContentPadding();
  const bookingsQuery = useClientBookingsQuery();
  const now = Date.now();
  const upcoming = (bookingsQuery.data ?? []).filter((booking) => {
    const when = booking.scheduled_time ? Date.parse(booking.scheduled_time) : NaN;
    return Number.isFinite(when) ? when >= now : true;
  });
  const past = (bookingsQuery.data ?? []).filter((booking) => {
    const when = booking.scheduled_time ? Date.parse(booking.scheduled_time) : NaN;
    return Number.isFinite(when) ? when < now : false;
  });

  const scrollContentStyle = useMemo(
    () => [
      styles.scrollContent,
      {
        paddingHorizontal: horizontalPadding,
        paddingTop: t.spacingSm + t.spacingXs,
        gap: t.spacingSm + t.spacingXs,
        paddingBottom: t.spacingSm,
      },
    ],
    [horizontalPadding, t.spacingSm, t.spacingXs]
  );

  return (
    <PermissionGuard permission="booking:read:self">
      <Screen
        scroll
        backgroundColor="#f8fafc"
        withHorizontalPadding={false}
        includeSafeAreaInScrollBottomPadding={false}
        extraScrollBottomPadding={bottomPad}
        contentContainerStyle={scrollContentStyle}
      >
        <View style={styles.headerRow}>
          <AppText variant="screenTitle" style={styles.pageTitle}>
            Mes réservations
          </AppText>
          <View style={styles.headerActions}>
            <Pressable onPress={() => router.push("/(app)/(client)/booking/new")}>
              <AppText variant="body" style={styles.link}>
                Nouvelle
              </AppText>
            </Pressable>
            <Pressable onPress={() => void bookingsQuery.refetch()}>
              <AppText variant="body" style={styles.link}>
                Rafraîchir
              </AppText>
            </Pressable>
          </View>
        </View>

        {bookingsQuery.isLoading ? (
          <AppText variant="caption">Chargement…</AppText>
        ) : null}
        {bookingsQuery.isError ? (
          <AppText variant="error">
            Erreur de chargement : {(bookingsQuery.error as Error)?.message ?? "Erreur inconnue"}
          </AppText>
        ) : null}
        {!bookingsQuery.isLoading &&
        !bookingsQuery.isError &&
        (bookingsQuery.data?.length ?? 0) === 0 ? (
          <AppText variant="bodyMuted">Aucune réservation trouvée.</AppText>
        ) : null}

        <AppText variant="sectionTitle" style={styles.sectionTitle}>
          À venir
        </AppText>
        {upcoming.map((booking) => {
          const statusUx = getClientStatusUx(booking.status);
          return (
            <Pressable
              key={booking.id}
              onPress={() =>
                router.push({
                  pathname: "/(app)/(client)/booking/[bookingId]",
                  params: { bookingId: String(booking.id) },
                })
              }
              style={styles.card}
            >
              <AppText variant="body" style={styles.cardTitle}>
                {booking.pickup_location ?? "Départ"}
                {" → "}
                {booking.dropoff_location ?? "Arrivée"}
              </AppText>
              <AppText variant="caption">{booking.scheduled_time ?? "Date inconnue"}</AppText>
              <AppText variant="caption">Statut : {statusUx.label}</AppText>
            </Pressable>
          );
        })}
        {upcoming.length === 0 && !bookingsQuery.isLoading && !bookingsQuery.isError ? (
          <AppText variant="bodyMuted">Aucune réservation à venir.</AppText>
        ) : null}

        <AppText variant="sectionTitle" style={[styles.sectionTitle, styles.sectionPast]}>
          Passées
        </AppText>
        {past.slice(0, 10).map((booking) => {
          const statusUx = getClientStatusUx(booking.status);
          return (
            <Pressable
              key={`past-${booking.id}`}
              onPress={() =>
                router.push({
                  pathname: "/(app)/(client)/booking/[bookingId]",
                  params: { bookingId: String(booking.id) },
                })
              }
              style={[styles.card, styles.cardPast]}
            >
              <AppText variant="body" style={styles.cardTitle}>
                {booking.pickup_location ?? "Départ"}
                {" → "}
                {booking.dropoff_location ?? "Arrivée"}
              </AppText>
              <AppText variant="caption">{booking.scheduled_time ?? "Date inconnue"}</AppText>
              <AppText variant="caption">Statut : {statusUx.label}</AppText>
            </Pressable>
          );
        })}
      </Screen>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  scrollContent: {
    flexGrow: 1,
  },
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 12,
  },
  pageTitle: {
    flex: 1,
  },
  headerActions: {
    flexDirection: "row",
    gap: 14,
  },
  link: {
    color: brandPrimary,
    fontWeight: "600",
  },
  sectionTitle: {
    marginTop: 4,
  },
  sectionPast: {
    marginTop: 12,
  },
  card: {
    borderWidth: 1,
    borderColor: "#e2e8f0",
    borderRadius: 10,
    padding: 12,
    gap: 4,
    backgroundColor: "#fff",
  },
  cardPast: {
    opacity: 0.85,
  },
  cardTitle: {
    fontWeight: "600",
  },
});
