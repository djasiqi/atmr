import React, { useCallback, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Platform,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import { useRouter } from "expo-router";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { PermissionGuard } from "../../../src/core/guards";
import {
  useClientBookingsQuery,
  useClientProfileQuery,
  usePrefetchClientDashboard,
} from "../../../src/features/client/hooks";
import { useClientBottomContentPadding } from "../../../src/features/client/navigation/ClientFloatingAppBar";
import type { Booking } from "../../../src/features/client/types";

const ACCENT = "#0a7ea4";

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

/** Pastilles discrètes, alignées charte web / mobile (slate + accents sémantiques légers). */
function statusPillStyles(status: string | null | undefined) {
  const s = String(status ?? "")
    .trim()
    .toLowerCase();
  const base = {
    backgroundColor: "#f1f5f9",
    borderColor: "#e2e8f0",
    color: "#475569",
  };
  const variants: Record<string, { backgroundColor: string; borderColor: string; color: string }> = {
    pending: { backgroundColor: "#fffbeb", borderColor: "#fde68a", color: "#92400e" },
    requested: { backgroundColor: "#eff6ff", borderColor: "#bfdbfe", color: "#1e40af" },
    confirmed: { backgroundColor: "#ecfdf5", borderColor: "#bbf7d0", color: "#166534" },
    /* Export RN Web client home : pastille neutre pour « Chauffeur assigné » */
    assigned: base,
    accepted: base,
    en_route: { backgroundColor: "#ecfeff", borderColor: "#a5f3fc", color: "#0e7490" },
    on_route: { backgroundColor: "#ecfeff", borderColor: "#a5f3fc", color: "#0e7490" },
    in_progress: { backgroundColor: "#fffbeb", borderColor: "#fde68a", color: "#92400e" },
    completed: { backgroundColor: "#f0fdf4", borderColor: "#bbf7d0", color: "#166534" },
    cancelled: { backgroundColor: "#fef2f2", borderColor: "#fecaca", color: "#991b1b" },
    canceled: { backgroundColor: "#fef2f2", borderColor: "#fecaca", color: "#991b1b" },
  };
  return variants[s] ?? base;
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
  const pickup = booking.pickup_location?.trim() || "Départ";
  const dropoff = booking.dropoff_location?.trim() || "Arrivée";
  const statusFr = bookingStatusLabel(booking.status);
  const pillColors = statusPillStyles(booking.status);

  return (
    <View style={styles.nextCard}>
      <Text style={styles.nextCardTitle}>Prochaine réservation</Text>

      <View style={styles.routeBlock}>
        <View style={styles.routeLeg}>
          <View style={styles.routeRail}>
            <View style={[styles.routeDot, styles.routeDotPickup]} />
            <View style={styles.routeLine} />
          </View>
          <View style={styles.routeLegText}>
            <Text style={styles.routeEyebrow}>Départ</Text>
            <Text style={styles.routeAddr}>{pickup}</Text>
          </View>
        </View>
        <View style={styles.routeLeg}>
          <View style={styles.routeRail}>
            <View style={[styles.routeDot, styles.routeDotDropoff]} />
          </View>
          <View style={styles.routeLegText}>
            <Text style={styles.routeEyebrow}>Arrivée</Text>
            <Text style={styles.routeAddr}>{dropoff}</Text>
          </View>
        </View>
      </View>

      <View style={styles.metaRow}>
        <View style={styles.metaItem}>
          <Text style={styles.metaLabel}>Date</Text>
          <Text style={styles.metaValue}>{formatBookingWhen(booking.scheduled_time)}</Text>
        </View>
        <View
          style={[
            styles.statusPill,
            {
              backgroundColor: pillColors.backgroundColor,
              borderColor: pillColors.borderColor,
            },
          ]}
        >
          <Text style={[styles.statusPillText, { color: pillColors.color }]}>{statusFr}</Text>
        </View>
      </View>

      <Pressable
        onPress={() => {
          void hapticLight();
          onPressDetail();
        }}
        style={({ pressed }) => [styles.detailLinkWrap, pressed && styles.pressedOpacity]}
        accessibilityRole="button"
        accessibilityLabel="Voir le détail de la réservation"
        android_ripple={{ color: "rgba(10, 126, 164, 0.12)", borderless: true }}
      >
        <Text style={styles.detailLink}>Voir le détail</Text>
        <Ionicons name="chevron-forward" size={18} color={ACCENT} />
      </Pressable>
    </View>
  );
}

export default function ClientHomeScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
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

  return (
    <PermissionGuard permission="booking:read:self">
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={[
          styles.scrollContent,
          {
            paddingTop: Math.max(24, insets.top + 12),
            paddingBottom: Math.max(bottomPad, insets.bottom + 20),
          },
        ]}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={ACCENT}
            colors={[ACCENT]}
          />
        }
      >
        <Text style={styles.kicker}>Espace client</Text>
        <Text style={styles.title}>
          {profileQuery.isLoading ? "Bonjour" : `Bonjour ${greetingName}`}
        </Text>
        <Text style={styles.subtitle}>
          Réservez en quelques étapes et suivez vos trajets en cours.
        </Text>

        <Pressable
          onPress={() => {
            void hapticLight();
            router.push("/(app)/(client)/booking/new");
          }}
          style={({ pressed }) => [styles.cta, pressed && styles.ctaPressed]}
          accessibilityRole="button"
          accessibilityLabel="Réserver un transport"
          android_ripple={{ color: "rgba(255, 255, 255, 0.28)" }}
        >
          <Text style={styles.ctaText}>Réserver un transport</Text>
        </Pressable>

        {bookingsQuery.isLoading ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color={ACCENT} />
            <Text style={styles.loadingText}>Chargement des courses…</Text>
          </View>
        ) : null}

        {bookingsQuery.isError ? (
          <View style={styles.messageCard}>
            <Text style={styles.messageTitle}>Impossible de charger les courses</Text>
            <Text style={styles.messageBody}>
              {(bookingsQuery.error as Error)?.message ?? "Une erreur est survenue."}
            </Text>
          </View>
        ) : null}

        {!bookingsQuery.isLoading && !bookingsQuery.isError && !nextBooking ? (
          <View style={styles.emptyCard}>
            <Ionicons name="calendar-outline" size={28} color="#94a3b8" />
            <Text style={styles.emptyTitle}>Aucune réservation à venir</Text>
            <Text style={styles.emptyHint}>
              Votre prochaine course apparaîtra ici dès qu’elle sera planifiée.
            </Text>
          </View>
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
      </ScrollView>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flex: 1,
    backgroundColor: "#f8fafc",
  },
  scrollContent: {
    paddingHorizontal: 24,
    gap: 12,
  },
  kicker: {
    fontSize: 13,
    color: "#64748b",
    fontWeight: "600",
    letterSpacing: 0.35,
  },
  title: {
    fontSize: 24,
    fontWeight: "800",
    color: "#0f172a",
    letterSpacing: -0.5,
    marginTop: 2,
  },
  subtitle: {
    color: "#334155",
    fontSize: 15,
    lineHeight: 22,
    marginTop: 2,
  },
  cta: {
    borderRadius: 12,
    backgroundColor: ACCENT,
    paddingVertical: 14,
    paddingHorizontal: 16,
    marginTop: 4,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.12,
    shadowRadius: 10,
    elevation: 4,
  },
  ctaPressed: {
    opacity: 0.92,
    transform: [{ scale: 0.99 }],
  },
  ctaText: {
    color: "#fff",
    fontWeight: "800",
    fontSize: 16,
    textAlign: "center",
  },
  loadingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 8,
  },
  loadingText: {
    color: "#64748b",
    fontSize: 14,
  },
  messageCard: {
    borderWidth: 1,
    borderColor: "#fecaca",
    borderRadius: 12,
    padding: 14,
    backgroundColor: "#fef2f2",
    gap: 6,
  },
  messageTitle: {
    fontWeight: "700",
    color: "#991b1b",
    fontSize: 15,
  },
  messageBody: {
    color: "#7f1d1d",
    fontSize: 14,
    lineHeight: 20,
  },
  emptyCard: {
    borderWidth: 1,
    borderColor: "#e2e8f0",
    borderRadius: 12,
    padding: 20,
    backgroundColor: "#fff",
    alignItems: "center",
    gap: 8,
  },
  emptyTitle: {
    fontWeight: "700",
    color: "#0f172a",
    fontSize: 16,
    textAlign: "center",
  },
  emptyHint: {
    color: "#64748b",
    fontSize: 14,
    lineHeight: 20,
    textAlign: "center",
  },
  nextCard: {
    borderWidth: 1,
    borderColor: "#e2e8f0",
    borderRadius: 12,
    padding: 16,
    gap: 12,
    backgroundColor: "#fff",
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.07,
    shadowRadius: 12,
    elevation: 3,
  },
  nextCardTitle: {
    fontWeight: "700",
    fontSize: 16,
    color: "#0f172a",
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
    paddingLeft: 10,
    paddingBottom: 14,
    minWidth: 0,
  },
  routeEyebrow: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: "#94a3b8",
    marginBottom: 4,
  },
  routeAddr: {
    fontSize: 14,
    lineHeight: 20,
    color: "#334155",
    fontWeight: "500",
  },
  metaRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "flex-end",
    justifyContent: "space-between",
    gap: 12,
    paddingTop: 12,
    marginTop: 2,
    borderTopWidth: 1,
    borderTopColor: "#f1f5f9",
  },
  metaItem: {
    flex: 1,
    minWidth: 160,
  },
  metaLabel: {
    fontSize: 11,
    fontWeight: "600",
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: 0.55,
    marginBottom: 4,
  },
  metaValue: {
    fontSize: 14,
    fontWeight: "600",
    color: "#0f172a",
    fontVariant: ["tabular-nums"],
  },
  statusPill: {
    alignSelf: "flex-end",
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    borderWidth: 1,
  },
  statusPillText: {
    fontSize: 12,
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
    fontSize: 15,
  },
  pressedOpacity: {
    opacity: 0.75,
  },
});
