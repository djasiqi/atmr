import { Pressable, StyleSheet } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';

import { EmptyState } from '@/components/mobile/EmptyState';
import { MobileListCard } from '@/components/mobile/MobileListCard';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useAuth } from '@/hooks/useAuth';
import { featureFlags } from '@/services/featureFlags';
import { listRequests } from '@/services/institutionApi';
import { useInstitutionPermissions } from '@/services/useInstitutionPermissions';
import { queryKeys } from '@/services/queryKeys';

export default function InstitutionDashboardScreen() {
  const router = useRouter();
  const { user, logout } = useAuth();
  const permissions = useInstitutionPermissions();
  const requestsQuery = useQuery({
    queryKey: queryKeys.institutionRequests({ page: 1, per_page: 20 }),
    queryFn: () => listRequests({ per_page: 20 }),
  });
  const openRequests = (requestsQuery.data?.items ?? []).filter((r) =>
    ['pending', 'urgent', 'in_progress'].includes(String(r.status ?? '').toLowerCase()),
  );
  const pendingOffers = (requestsQuery.data?.items ?? []).filter((r) => {
    const status = String(r.status ?? '').toUpperCase();
    return status === 'SENT' && !r.accepted_by_company;
  }).length;
  const first = openRequests[0];

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Institution</ThemedText>
      <ThemedText style={styles.muted}>
        {user?.username ?? user?.email ?? 'Session active'}
      </ThemedText>
      <ThemedText style={styles.note}>Vue des demandes en cours.</ThemedText>
      {first ? (
        <MobileListCard
          title={first.patient?.full_name ?? first.external_reference ?? `Demande #${first.id}`}
          subtitle={first.dropoff_location ?? first.pickup_location ?? 'Adresse inconnue'}
          meta={first.created_at ?? ''}
          badge={first.status ?? 'pending'}
        />
      ) : (
        <EmptyState title="Aucune demande en cours" description="Le tableau est à jour." />
      )}
      <ThemedText style={styles.note}>{openRequests.length} demande(s) active(s)</ThemedText>
      <ThemedText style={styles.noteStrong}>
        Offres en attente: {pendingOffers}
      </ThemedText>
      {permissions.canCreateRequest && featureFlags.institutionMobileRequestSendEnabled ? (
        <Pressable style={styles.quickAction} onPress={() => router.push('/(institution)/request-create')}>
          <ThemedText type="defaultSemiBold">Créer une demande</ThemedText>
        </Pressable>
      ) : null}
      <Pressable style={styles.logout} onPress={() => void logout()}>
        <ThemedText type="defaultSemiBold" lightColor="#fff" darkColor="#fff">
          Déconnexion
        </ThemedText>
      </Pressable>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    gap: 8,
  },
  muted: {
    opacity: 0.8,
  },
  note: {
    marginTop: 16,
    opacity: 0.7,
  },
  noteStrong: {
    marginTop: 8,
    fontWeight: '700',
  },
  quickAction: {
    marginTop: 10,
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#0a7ea4',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 14,
  },
  logout: {
    marginTop: 24,
    alignSelf: 'flex-start',
    backgroundColor: '#0a7ea4',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
});
