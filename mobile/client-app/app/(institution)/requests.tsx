import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, TextInput } from 'react-native';

import { EmptyState } from '@/components/mobile/EmptyState';
import { MobileListCard } from '@/components/mobile/MobileListCard';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { featureFlags } from '@/services/featureFlags';
import { listRequests } from '@/services/institutionApi';
import { queryKeys } from '@/services/queryKeys';

export default function InstitutionRequestsScreen() {
  const router = useRouter();
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [externalReference, setExternalReference] = useState('');
  const [page, setPage] = useState(1);

  const params = useMemo(
    () => ({
      status: featureFlags.institutionMobileRequestFiltersEnabled ? statusFilter || undefined : undefined,
      external_reference: featureFlags.institutionMobileRequestFiltersEnabled
        ? externalReference.trim() || undefined
        : undefined,
      page,
      per_page: 20,
    }),
    [externalReference, page, statusFilter],
  );

  const requestsQuery = useQuery({
    queryKey: queryKeys.institutionRequests(params),
    queryFn: () => listRequests(params),
  });

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Demandes</ThemedText>
      {featureFlags.institutionMobileRequestFiltersEnabled ? (
        <>
          <TextInput
            style={styles.input}
            placeholder="Référence externe"
            placeholderTextColor="#8b8b8b"
            value={externalReference}
            onChangeText={(next) => {
              setPage(1);
              setExternalReference(next);
            }}
          />
          <ThemedView style={styles.filters}>
            {[
              { key: '', label: 'Tous' },
              { key: 'DRAFT', label: 'Brouillon' },
              { key: 'SENT', label: 'Envoyées' },
              { key: 'CANCELLED', label: 'Annulées' },
              { key: 'CONVERTED', label: 'Converties' },
            ].map((option) => (
              <Pressable
                key={option.key || 'all'}
                style={[styles.filter, statusFilter === option.key ? styles.filterActive : undefined]}
                onPress={() => {
                  setPage(1);
                  setStatusFilter(option.key);
                }}
              >
                <ThemedText>{option.label}</ThemedText>
              </Pressable>
            ))}
          </ThemedView>
        </>
      ) : null}
      <Pressable style={styles.refresh} onPress={() => void requestsQuery.refetch()}>
        <ThemedText type="defaultSemiBold">Rafraîchir</ThemedText>
      </Pressable>

      {requestsQuery.isError ? (
        <EmptyState
          title="Impossible de charger les demandes"
          description="Vérifiez la connexion puis réessayez."
          actionLabel="Réessayer"
          onAction={() => void requestsQuery.refetch()}
        />
      ) : null}

      {!requestsQuery.isError && (requestsQuery.data?.items.length ?? 0) === 0 ? (
        <EmptyState title="Aucune demande" description="Les demandes apparaîtront ici." />
      ) : null}

      <ScrollView contentContainerStyle={styles.list}>
        {(requestsQuery.data?.items ?? []).map((request) => (
          <MobileListCard
            key={request.id}
            title={request.patient?.full_name ?? request.external_reference ?? `Demande #${request.id}`}
            subtitle={request.dropoff_location ?? request.pickup_location ?? 'Adresse non renseignée'}
            meta={request.scheduled_time ?? request.created_at ?? 'Date non renseignée'}
            badge={request.status ?? 'pending'}
            onPress={() => router.push(`/(institution)/request/${request.id}`)}
          />
        ))}
      </ScrollView>
      <ThemedView style={styles.pagination}>
        <Pressable
          style={[styles.pageButton, page <= 1 ? styles.disabled : undefined]}
          onPress={() => setPage((p) => Math.max(1, p - 1))}
          disabled={page <= 1}
        >
          <ThemedText>Précédent</ThemedText>
        </Pressable>
        <ThemedText>
          Page {requestsQuery.data?.page ?? page}
          {requestsQuery.data?.pages ? ` / ${requestsQuery.data.pages}` : ''}
        </ThemedText>
        <Pressable
          style={[
            styles.pageButton,
            (requestsQuery.data?.pages ? page >= requestsQuery.data.pages : false)
              ? styles.disabled
              : undefined,
          ]}
          onPress={() => setPage((p) => p + 1)}
          disabled={requestsQuery.data?.pages ? page >= requestsQuery.data.pages : false}
        >
          <ThemedText>Suivant</ThemedText>
        </Pressable>
      </ThemedView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    gap: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  filters: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  filter: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 20,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  filterActive: {
    borderColor: '#0a7ea4',
    backgroundColor: 'rgba(10,126,164,0.15)',
  },
  refresh: {
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  list: {
    gap: 10,
    paddingBottom: 20,
  },
  pagination: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  pageButton: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  disabled: {
    opacity: 0.4,
  },
});
