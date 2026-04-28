import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { useMemo } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, View } from 'react-native';

import { EmptyState } from '@/components/mobile/EmptyState';
import { MobileListCard } from '@/components/mobile/MobileListCard';
import { NetworkRetryBanner } from '@/components/mobile/NetworkRetryBanner';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useAuth } from '@/hooks/useAuth';
import { getClientBookings } from '@/services/clientApi';
import { queryKeys } from '@/services/queryKeys';
import type { Booking } from '@/types/api';

function formatBookingWhen(value: string | null | undefined): string {
  const date = Date.parse(String(value ?? ''));
  if (!Number.isFinite(date)) return 'Date non renseignée';
  return new Date(date).toLocaleString('fr-CH', {
    weekday: 'short',
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default function ClientBookingsScreen() {
  const router = useRouter();
  const { role } = useAuth();
  const bookingsQuery = useQuery({
    queryKey: queryKeys.bookings,
    queryFn: getClientBookings,
    enabled: role === 'client',
  });

  const bookings = useMemo<Booking[]>(() => bookingsQuery.data ?? [], [bookingsQuery.data]);

  if (bookingsQuery.isLoading) {
    return (
      <ThemedView style={styles.centered}>
        <ActivityIndicator size="large" />
        <ThemedText style={styles.hint}>Chargement des courses…</ThemedText>
      </ThemedView>
    );
  }

  return (
    <ThemedView style={styles.container}>
      <View style={styles.headerRow}>
        <ThemedText type="title">Mes courses</ThemedText>
        <Pressable style={styles.refresh} onPress={() => void bookingsQuery.refetch()}>
          <ThemedText type="defaultSemiBold">Rafraîchir</ThemedText>
        </Pressable>
      </View>

      <NetworkRetryBanner
        showOnError={bookingsQuery.isError}
        onRetry={() => void bookingsQuery.refetch()}
      />

      {bookingsQuery.isError ? (
        <EmptyState
          title="Impossible de charger vos courses"
          description="Vérifiez la connexion puis réessayez."
          actionLabel="Réessayer"
          onAction={() => void bookingsQuery.refetch()}
        />
      ) : null}

      {!bookingsQuery.isError && bookings.length === 0 ? (
        <EmptyState
          title="Aucune course"
          description="Vos réservations apparaîtront ici."
          actionLabel="Actualiser"
          onAction={() => void bookingsQuery.refetch()}
        />
      ) : null}

      {!bookingsQuery.isError && bookings.length > 0 ? (
        <ScrollView contentContainerStyle={styles.list}>
          {bookings.map((booking) => (
            <MobileListCard
              key={booking.id}
              title={`${booking.pickup_address ?? 'Départ'} -> ${booking.destination_address ?? 'Arrivée'}`}
              subtitle={formatBookingWhen(booking.scheduled_time)}
              meta={`Transporteur: ${booking.company_name ?? 'Non attribué'}`}
              badge={booking.status ?? 'pending'}
              onPress={() => router.push(`/(client)/booking/${booking.id}`)}
            />
          ))}
        </ScrollView>
      ) : null}
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    gap: 12,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 10,
  },
  hint: {
    opacity: 0.8,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  refresh: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  list: {
    gap: 10,
    paddingBottom: 24,
  },
});
