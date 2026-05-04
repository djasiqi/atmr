import React, { useCallback, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Platform,
  Pressable,
  RefreshControl,
  StyleSheet,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import { useRouter } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import {
  useClientBookingsQuery,
  useClientProfileQuery,
  usePrefetchClientDashboard,
} from "../../../src/features/client/hooks";
import { useClientBottomContentPadding } from "../../../src/features/client/navigation/ClientFloatingAppBar";
import type { Booking } from "../../../src/features/client/types";

import {
  AppButton,
  AppCard,
  AppNotice,
  AppStatusBadge,
  AppText,
  brandPrimary,
  Screen,
  useAppViewport,
  useResponsiveTokens,
} from "../../../src/design/responsive";

const ACCENT = brandPrimary;

function formatBookingWhen(value: string | null | undefined): string {
  const raw = String(value ?? "").trim();
  if (!raw) return "Date inconnue";
  const parsed = Date.parse(raw);
  if (!Number.isFinite(parsed)) return raw;
  return new Date(parsed).toLocaleString("fr-CH", {
    weekday: "short",
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function bookingStatusLabel(status: string | null | undefined): string {
  const s = String(status ?? "")
    .trim()
    .toLowerCase();
  const map: Record<string, string> = {
    pending: "En attente",
    draft: "Brouillon",
    requested: "Demandée",
    confirmed: "Confirmée",
    assigned: "Chauffeur assigné",
    accepted: "Acceptée",
    en_route: "En route",
    on_route: "En route",
    in_progress: "En cours",
    completed: "Terminée",
    cancelled: "Annulée",
    canceled: "Annulée",
  };
  return map[s] || (status ? status.replace(/_/g, " ") : "Inconnu");
}

async function hapticLight() {
  if (Platform.OS === "ios") {
    try {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    } catch {
      /* ignore */
    }
  }
}

function NextBookingCard({
  booking,
  onPressDetail,
}: {
  booking: Booking;
  onPressDetail: () => void;
}) {
  const t = useResponsiveTokens();
  const pickup = booking.pickup_location?.trim() || "Départ";
  const dropoff = booking.dropoff_location?.trim() || "Arrivée";
  const statusFr = bookingStatusLabel(booking.status);

  return (
    <AppCard variant="elevated" style={styles.nextCard}>
      <AppText variant="sectionTitle" style={styles.nextCardTitle}>
        Prochaine réservation
      </AppText>

      <View style={styles.routeBlock}>
        <View style={styles.routeLeg}>
          <View style={styles.routeRail}>
            <View style={[styles.routeDot, styles.routeDotPickup]} />
            <View style={styles.routeLine} />
          </View>
          <View
            style={[
              styles.routeLegText,
              {
                paddingLeft: t.spacingSm + 2,
                paddingBottom: t.spacingMd - 2,
              },
            ]}
          >
            <AppText variant="caption" style={styles.routeEyebrow}>
              Départ
            </AppText>
            <AppText variant="body" style={styles.routeAddr}>
              {pickup}
            </AppText>
          </View>
        </View>
        <View style={styles.routeLeg}>
          <View style={styles.routeRail}>
            <View style={[styles.routeDot, styles.routeDotDropoff]} />
          </View>
          <View
            style={[
              styles.routeLegText,
              {
                paddingLeft: t.spacingSm + 2,
                paddingBottom: t.spacingMd - 2,
              },
            ]}
          >
            <AppText variant="caption" style={[styles.routeEyebrow, { marginBottom: t.spacingXs }]}>
              Arrivée
            </AppText>
            <AppText variant="body" style={styles.routeAddr}>
              {dropoff}
            </AppText>
          </View>
        </View>
      </View>

      <View style={styles.metaRow}>
        <View style={styles.metaItem}>
          <AppText variant="caption" style={[styles.metaLabel, { marginBottom: t.spacingXs }]}>
            Date
          </AppText>
          <AppText variant="body" style={styles.metaValue}>
            {formatBookingWhen(booking.scheduled_time)}
          </AppText>
        </View>
        <AppStatusBadge status={booking.status} label={statusFr} />
      </View>

      <Pressable
        onPress={() => {
          void hapticLight();
          onPressDetail();
        }}
        style={({ pressed }) => [styles.detailLinkWrap, pressed && styles.pressedOpacity]}
        accessibilityRole="button"
        accessibilityLabel="Voir le détail de la réservation"
        android_ripple={{ color: "rgba(10, 143, 122, 0.15)", borderless: true }}
      >
        <AppText variant="body" style={styles.detailLink}>
          Voir le détail
        </AppText>
        <Ionicons name="chevron-forward" size={18} color={ACCENT} />
      </Pressable>
    </AppCard>
  );
}

export default function ClientHomeScreen() {
  const router = useRouter();
  const t = useResponsiveTokens();
  const { horizontalPadding } = useAppViewport();
  const bottomPad = useClientBottomContentPadding();
  const [refreshing, setRefreshing] = useState(false);
  usePrefetchClientDashboard();
  const profileQuery = useClientProfileQuery();
  const bookingsQuery = useClientBookingsQuery();
  const nextBooking = useMemo(
    () => (bookingsQuery.data ?? [])[0] ?? null,
    [bookingsQuery.data]
  );

  const firstName = profileQuery.data?.first_name?.trim();
  const greetingName = firstName || "client";

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await Promise.all([profileQuery.refetch(), bookingsQuery.refetch()]);
    } finally {
      setRefreshing(false);
    }
  }, [profileQuery, bookingsQuery]);

  const scrollContentStyle = useMemo(
    () => [
      styles.scrollContent,
      {
        paddingHorizontal: horizontalPadding,
        paddingTop: t.spacingSm + t.spacingXs,
        gap: t.spacingSm + t.spacingXs,
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
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={ACCENT}
            colors={[ACCENT]}
          />
        }
        contentContainerStyle={scrollContentStyle}
      >
        <AppText variant="caption" style={styles.kicker}>
          Espace client
        </AppText>
        <AppText variant="screenTitle" style={styles.title}>
          {profileQuery.isLoading ? "Bonjour" : `Bonjour ${greetingName}`}
        </AppText>
        <AppText variant="body" style={styles.subtitle}>
          Réservez en quelques étapes et suivez vos trajets en cours.
        </AppText>

        <AppButton
          title="Réserver un transport"
          variant="primary"
          onPress={() => {
            void hapticLight();
            router.push("/(app)/(client)/booking/new");
          }}
          style={[styles.cta, { marginTop: t.spacingXs }]}
          accessibilityLabel="Réserver un transport"
        />

        {bookingsQuery.isLoading ? (
          <View
            style={[
              styles.loadingRow,
              { gap: t.spacingSm + 2, paddingVertical: t.spacingSm },
            ]}
          >
            <ActivityIndicator color={ACCENT} />
            <AppText variant="caption">Chargement des courses…</AppText>
          </View>
        ) : null}

        {bookingsQuery.isError ? (
          <AppNotice variant="danger" title="Impossible de charger les courses" style={styles.messageCard}>
            {(bookingsQuery.error as Error)?.message ?? "Une erreur est survenue."}
          </AppNotice>
        ) : null}

        {!bookingsQuery.isLoading && !bookingsQuery.isError && !nextBooking ? (
          <AppCard variant="surface" style={styles.emptyCard}>
            <Ionicons name="calendar-outline" size={28} color="#94a3b8" />
            <AppText variant="sectionTitle" style={styles.emptyTitle}>
              Aucune réservation à venir
            </AppText>
            <AppText variant="bodyMuted" style={styles.emptyHint}>
              Votre prochaine course apparaîtra ici dès qu’elle sera planifiée.
            </AppText>
          </AppCard>
        ) : null}

        {nextBooking ? (
          <NextBookingCard
            booking={nextBooking}
            onPressDetail={() =>
              router.push({
                pathname: "/(app)/(client)/booking/[bookingId]",
                params: { bookingId: String(nextBooking.id) },
              })
            }
          />
        ) : null}
      </Screen>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  scrollContent: {
    flexGrow: 1,
  },
  kicker: {
    letterSpacing: 0.35,
    fontWeight: "600",
  },
  title: {
    letterSpacing: -0.5,
    marginTop: 2,
  },
  subtitle: {
    marginTop: 2,
  },
  cta: {
    alignSelf: "stretch",
  },
  loadingRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  messageCard: {
    marginTop: 4,
  },
  emptyCard: {
    alignItems: "center",
    gap: 8,
  },
  emptyTitle: {
    textAlign: "center",
  },
  emptyHint: {
    textAlign: "center",
  },
  nextCard: {
    gap: 12,
  },
  nextCardTitle: {
    letterSpacing: -0.2,
  },
  routeBlock: {
    gap: 2,
  },
  routeLeg: {
    flexDirection: "row",
    alignItems: "stretch",
    minHeight: 48,
  },
  routeRail: {
    width: 18,
    alignItems: "center",
    paddingTop: 4,
  },
  routeDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: "#fff",
  },
  routeDotPickup: {
    backgroundColor: "#059669",
    shadowColor: "#059669",
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.35,
    shadowRadius: 3,
    elevation: 2,
  },
  routeDotDropoff: {
    backgroundColor: "#64748b",
  },
  routeLine: {
    width: 2,
    flex: 1,
    minHeight: 18,
    marginVertical: 4,
    borderRadius: 1,
    backgroundColor: "#cbd5e1",
  },
  routeLegText: {
    flex: 1,
    minWidth: 0,
  },
  routeEyebrow: {
    fontWeight: "700",
    letterSpacing: 0.6,
    textTransform: "uppercase",
  },
  routeAddr: {
    fontWeight: "500",
  },
  metaRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "flex-end",
    justifyContent: "space-between",
    marginTop: 2,
    borderTopWidth: 1,
    borderTopColor: "#f1f5f9",
  },
  metaItem: {
    flex: 1,
    minWidth: 160,
  },
  metaLabel: {
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: 0.55,
    marginBottom: 4,
  },
  metaValue: {
    fontVariant: ["tabular-nums"],
    fontWeight: "600",
  },
  detailLinkWrap: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 4,
    paddingVertical: 8,
    marginHorizontal: -4,
  },
  detailLink: {
    color: ACCENT,
    fontWeight: "600",
  },
  pressedOpacity: {
    opacity: 0.75,
  },
});
