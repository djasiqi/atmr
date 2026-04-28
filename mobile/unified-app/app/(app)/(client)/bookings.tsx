import { useRouter } from "expo-router";
import { Pressable, ScrollView, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { PermissionGuard } from "../../../src/core/guards";
import { useClientBookingsQuery } from "../../../src/features/client/hooks";
import { useClientBottomContentPadding } from "../../../src/features/client/navigation/ClientFloatingAppBar";
import { getClientStatusUx } from "../../../src/features/client/statusDictionary";

export default function ClientBookingsScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
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

  return (
    <PermissionGuard permission="booking:read:self">
      <View
        style={{
          flex: 1,
          padding: 24,
          paddingTop: Math.max(24, insets.top + 8),
          paddingBottom: bottomPad,
          gap: 12,
        }}
      >
        <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
          <Text style={{ fontSize: 22, fontWeight: "700" }}>Mes réservations</Text>
          <View style={{ flexDirection: "row", gap: 14 }}>
            <Pressable onPress={() => router.push("/(app)/(client)/booking/new")}>
              <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Nouvelle</Text>
            </Pressable>
            <Pressable onPress={() => void bookingsQuery.refetch()}>
              <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Rafraîchir</Text>
            </Pressable>
          </View>
        </View>

        {bookingsQuery.isLoading ? <Text>Chargement...</Text> : null}
        {bookingsQuery.isError ? (
          <Text>
            Erreur de chargement: {(bookingsQuery.error as Error)?.message ?? "Erreur inconnue"}
          </Text>
        ) : null}
        {!bookingsQuery.isLoading &&
          !bookingsQuery.isError &&
          (bookingsQuery.data?.length ?? 0) === 0 ? (
          <Text>Aucune réservation trouvée.</Text>
        ) : null}

        <Text style={{ fontWeight: "700", color: "#0f172a" }}>A venir</Text>
        <ScrollView contentContainerStyle={{ gap: 10, paddingBottom: 16 }}>
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
                style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12, gap: 4 }}
              >
                <Text style={{ fontWeight: "600" }}>
                  {booking.pickup_location ?? "Départ"}
                  {" -> "}
                  {booking.dropoff_location ?? "Arrivée"}
                </Text>
                <Text>{booking.scheduled_time ?? "Date inconnue"}</Text>
                <Text>Statut: {statusUx.label}</Text>
              </Pressable>
            );
          })}
          {upcoming.length === 0 && !bookingsQuery.isLoading && !bookingsQuery.isError ? (
            <Text style={{ color: "#64748b" }}>Aucune reservation a venir.</Text>
          ) : null}
          <Text style={{ fontWeight: "700", color: "#0f172a", marginTop: 8 }}>Passees</Text>
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
                style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12, gap: 4, opacity: 0.85 }}
              >
                <Text style={{ fontWeight: "600" }}>
                  {booking.pickup_location ?? "Départ"}
                  {" -> "}
                  {booking.dropoff_location ?? "Arrivée"}
                </Text>
                <Text>{booking.scheduled_time ?? "Date inconnue"}</Text>
                <Text>Statut: {statusUx.label}</Text>
              </Pressable>
            );
          })}
        </ScrollView>
      </View>
    </PermissionGuard>
  );
}
