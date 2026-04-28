import { useQuery } from '@tanstack/react-query';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Pressable, StyleSheet } from 'react-native';

import { InvalidRouteScreen } from '@/components/mobile/InvalidRouteScreen';
import { MobileListCard } from '@/components/mobile/MobileListCard';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { getBooking } from '@/services/clientApi';
import { queryKeys } from '@/services/queryKeys';

function parseBookingId(value: unknown): number | null {
  const num = Number(value);
  return Number.isInteger(num) && num > 0 ? num : null;
}

export default function BookingDetailsScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ bookingId?: string }>();
  const bookingId = parseBookingId(params.bookingId);

  const bookingQuery = useQuery({
    queryKey: queryKeys.booking(bookingId ?? 'invalid'),
    queryFn: () => getBooking(bookingId as number),
    enabled: bookingId !== null,
  });

  if (!bookingId) {
    return (
      <InvalidRouteScreen
        message="Identifiant de réservation manquant ou invalide."
        onPress={() => router.replace('/(client)/bookings')}
      />
    );
  }

  if (bookingQuery.isError) {
    return (
      <InvalidRouteScreen
        title="Réservation introuvable"
        message="La course demandée n'existe plus ou n'est plus accessible."
        onPress={() => router.replace('/(client)/bookings')}
      />
    );
  }

  const booking = bookingQuery.data;

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Détail course</ThemedText>
      {booking ? (
        <MobileListCard
          title={`${booking.pickup_address ?? 'Départ'} -> ${booking.destination_address ?? 'Arrivée'}`}
          subtitle={booking.scheduled_time ?? 'Date inconnue'}
          meta={`Transporteur: ${booking.company_name ?? 'Non attribué'}`}
          badge={booking.status ?? 'pending'}
        />
      ) : (
        <ThemedText style={styles.note}>Chargement…</ThemedText>
      )}

      <Pressable
        style={styles.payButton}
        onPress={() => router.push(`/(client)/payment?bookingId=${bookingId}`)}
      >
        <ThemedText type="defaultSemiBold" lightColor="#fff" darkColor="#fff">
          Payer en ligne
        </ThemedText>
      </Pressable>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    gap: 12,
  },
  note: {
    opacity: 0.75,
  },
  payButton: {
    marginTop: 12,
    alignSelf: 'flex-start',
    backgroundColor: '#0a7ea4',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 14,
  },
});
