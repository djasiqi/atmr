import { Redirect, useLocalSearchParams, useRouter } from "expo-router";
import { Pressable, Text, View } from "react-native";
import { PermissionGuard } from "../../../../src/core/guards";
import { useSession } from "../../../../src/core/sessionProvider";
import { bookingBelongsToActiveClient } from "../../../../src/features/client/accessControl";
import { useBookingDetailQuery, useClientProfileQuery } from "../../../../src/features/client/hooks";
import { getClientStatusUx } from "../../../../src/features/client/statusDictionary";

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

  if (!bookingId) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center", padding: 24, gap: 12 }}>
        <Text>Identifiant réservation invalide.</Text>
        <Pressable onPress={() => router.replace("/(app)/(client)/bookings")}>
          <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Retour à la liste</Text>
        </Pressable>
      </View>
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
      <View style={{ flex: 1, padding: 24, gap: 12 }}>
        <Text style={{ fontSize: 22, fontWeight: "700" }}>Détail réservation</Text>
        {bookingQuery.isLoading ? <Text>Chargement du détail...</Text> : null}
        {bookingQuery.isError ? (
          <Text>
            Réservation introuvable ou inaccessible: {(bookingQuery.error as Error)?.message ?? "Erreur"}
          </Text>
        ) : null}

        {bookingQuery.data ? (
          <View style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12, gap: 6 }}>
            {params.created === "1" ? (
              <View style={{ backgroundColor: "#ecfdf3", borderRadius: 8, padding: 8 }}>
                <Text style={{ color: "#14532d" }}>Reservation enregistree avec succes.</Text>
              </View>
            ) : null}
            <Text style={{ fontWeight: "600" }}>
              {bookingQuery.data.pickup_location ?? "Départ"}
              {" -> "}
              {bookingQuery.data.dropoff_location ?? "Arrivée"}
            </Text>
            <Text>{bookingQuery.data.scheduled_time ?? "Date inconnue"}</Text>
            <Text>Statut: {bookingStatusUx.label}</Text>
            <Text>Transporteur: {bookingQuery.data.company_name ?? "Non attribué"}</Text>
            <Text>Paiement: {paymentStatus}</Text>
            {bookingQuery.data.payment_amount ? (
              <Text>
                Montant: {bookingQuery.data.payment_amount} {bookingQuery.data.currency ?? "CHF"}
              </Text>
            ) : null}
            {params.pricingReason ? (
              <View style={{ backgroundColor: "#fff7ed", borderRadius: 8, padding: 8 }}>
                <Text style={{ color: "#9a3412" }}>
                  Ajustement tarifaire: {params.pricingReason}
                </Text>
              </View>
            ) : null}
          </View>
        ) : null}

        {paymentRequired ? (
          <View
            style={{
              borderWidth: 1,
              borderColor: "#ddd",
              borderRadius: 10,
              padding: 12,
              gap: 8,
            }}
          >
            <Text style={{ fontWeight: "600" }}>Paiement en ligne</Text>
            {paymentStatus === "pending_verification" ? (
              <Text>Paiement en cours de confirmation...</Text>
            ) : null}
            {paymentStatus === "failed" ? <Text>Paiement échoué ou annulé.</Text> : null}
            {paymentStatus === "paid" ? <Text>Paiement confirmé.</Text> : null}
            {paymentStatus === "required" ? <Text>Paiement requis pour cette réservation.</Text> : null}

            <Pressable
              onPress={() =>
                router.push({
                  pathname: "/(app)/(client)/payment",
                  params: { bookingId: String(bookingId) },
                })
              }
              disabled={!canStartPayment}
              style={{
                opacity: canStartPayment ? 1 : 0.6,
                backgroundColor: "#0a7ea4",
                borderRadius: 8,
                paddingVertical: 10,
                paddingHorizontal: 14,
                alignSelf: "flex-start",
              }}
            >
              <Text style={{ color: "#fff", fontWeight: "600" }}>
                {paymentStatus === "failed" ? "Réessayer le paiement" : "Payer maintenant"}
              </Text>
            </Pressable>

            <Pressable
              onPress={() => void bookingQuery.refetch()}
              style={{
                borderWidth: 1,
                borderColor: "#cfcfcf",
                borderRadius: 8,
                paddingVertical: 8,
                paddingHorizontal: 12,
                alignSelf: "flex-start",
              }}
            >
              <Text style={{ fontWeight: "600" }}>Actualiser le statut</Text>
            </Pressable>
          </View>
        ) : null}

        <Pressable onPress={() => router.replace("/(app)/(client)/bookings")}>
          <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Retour aux réservations</Text>
        </Pressable>
      </View>
    </PermissionGuard>
  );
}
