import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { Pressable, StyleSheet } from 'react-native';

import { EmptyState } from '@/components/mobile/EmptyState';
import { MobileListCard } from '@/components/mobile/MobileListCard';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { getClientBookings } from '@/services/clientApi';
import { queryKeys } from '@/services/queryKeys';
import { selectNextBooking } from '@/utils/selectors';

export default function ClientDashboardScreen() {
  const router = useRouter();
  const bookingsQuery = useQuery({
    queryKey: queryKeys.bookings,
    queryFn: getClientBookings,
  });
  const nextBooking = selectNextBooking(bookingsQuery.data ?? []);

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Dashboard client</ThemedText>
      <ThemedText style={styles.note}>Vue rapide des prochaines courses.</ThemedText>

      {!nextBooking ? (
        <EmptyState
          title="Aucune course à venir"
          description="Créez ou attendez la prochaine réservation."
          actionLabel="Voir toutes les courses"
          onAction={() => router.push('/(client)/bookings')}
        />
      ) : (
        <MobileListCard
          title={`${nextBooking.pickup_address ?? 'Départ'} -> ${nextBooking.destination_address ?? 'Arrivée'}`}
          subtitle={nextBooking.scheduled_time ?? 'Date inconnue'}
          meta={`Statut: ${nextBooking.status ?? 'pending'}`}
          badge={nextBooking.payment_required ? 'Paiement requis' : 'OK'}
          onPress={() => router.push(`/(client)/booking/${nextBooking.id}`)}
        />
      )}

      <Pressable style={styles.linkButton} onPress={() => router.push('/(client)/bookings')}>
        <ThemedText type="defaultSemiBold">Ouvrir la liste des courses</ThemedText>
      </Pressable>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    gap: 12,
  },
  note: {
    opacity: 0.7,
  },
  linkButton: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
});
