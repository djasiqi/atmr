import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Pressable, StyleSheet } from 'react-native';

import { EmptyState } from '@/components/mobile/EmptyState';
import { InvalidRouteScreen } from '@/components/mobile/InvalidRouteScreen';
import { MobileListCard } from '@/components/mobile/MobileListCard';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { getApiErrorMessage } from '@/services/api';
import { featureFlags } from '@/services/featureFlags';
import { cancelRequest, getRequest, sendRequest } from '@/services/institutionApi';
import { useInstitutionPermissions } from '@/services/useInstitutionPermissions';
import { queryKeys } from '@/services/queryKeys';

function parseRequestId(value: unknown): number | null {
  const n = Number(value);
  return Number.isInteger(n) && n > 0 ? n : null;
}

export default function InstitutionRequestDetailsScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const params = useLocalSearchParams<{ requestId?: string }>();
  const requestId = parseRequestId(params.requestId);
  const { canSendRequest } = useInstitutionPermissions();

  const requestQuery = useQuery({
    queryKey: queryKeys.institutionRequest(requestId ?? 'invalid'),
    queryFn: () => getRequest(requestId as number),
    enabled: requestId !== null,
  });

  const sendMutation = useMutation({
    mutationFn: sendRequest,
    onSuccess: async () => {
      if (!requestId) return;
      await queryClient.invalidateQueries({
        queryKey: queryKeys.institutionRequest(requestId),
      });
      await queryClient.invalidateQueries({
        queryKey: ['institution', 'requests'],
      });
    },
  });

  const cancelMutation = useMutation({
    mutationFn: cancelRequest,
    onSuccess: async () => {
      if (!requestId) return;
      await queryClient.invalidateQueries({
        queryKey: queryKeys.institutionRequest(requestId),
      });
      await queryClient.invalidateQueries({
        queryKey: ['institution', 'requests'],
      });
    },
  });

  if (!requestId) {
    return (
      <InvalidRouteScreen
        message="Identifiant de demande invalide."
        onPress={() => router.replace('/(institution)/requests')}
      />
    );
  }

  if (requestQuery.isError) {
    return (
      <InvalidRouteScreen
        title="Demande introuvable"
        onPress={() => router.replace('/(institution)/requests')}
      />
    );
  }

  const request = requestQuery.data;

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Détail demande</ThemedText>
      {request ? (
        <>
          <MobileListCard
            title={request.patient?.full_name ?? request.external_reference ?? `Demande #${request.id}`}
            subtitle={request.dropoff_location ?? request.pickup_location ?? 'Adresse inconnue'}
            meta={request.notes ?? request.created_at ?? ''}
            badge={request.status ?? 'pending'}
          />
          <ThemedText style={styles.line}>
            Départ: {request.pickup_location ?? 'Non renseigné'}
          </ThemedText>
          <ThemedText style={styles.line}>
            Destination: {request.dropoff_location ?? 'Non renseigné'}
          </ThemedText>
          <ThemedText style={styles.line}>
            Horaire: {request.scheduled_time ?? 'Non renseigné'}
          </ThemedText>
          <ThemedText style={styles.line}>
            Facturation: {request.billing_intent ?? 'patient'}
          </ThemedText>
          <ThemedText style={styles.line}>
            Mobilité: {JSON.stringify(request.mobility ?? {})}
          </ThemedText>
          <ThemedText style={styles.line}>
            Transporteur accepté: {request.accepted_by_company?.name ?? 'Aucun'}
          </ThemedText>

          {featureFlags.institutionMobileRequestSendEnabled
          && String(request.status ?? '').toUpperCase() === 'DRAFT'
          && canSendRequest ? (
            <Pressable
              style={[styles.action, sendMutation.isPending ? styles.actionDisabled : undefined]}
              onPress={() => sendMutation.mutate(request.id)}
              disabled={sendMutation.isPending}
            >
              <ThemedText type="defaultSemiBold" lightColor="#fff" darkColor="#fff">
                {sendMutation.isPending ? 'Envoi…' : 'Envoyer la demande'}
              </ThemedText>
            </Pressable>
          ) : null}

          {request.is_cancellable ? (
            <Pressable
              style={[styles.actionSecondary, cancelMutation.isPending ? styles.actionDisabled : undefined]}
              onPress={() => cancelMutation.mutate(request.id)}
              disabled={cancelMutation.isPending}
            >
              <ThemedText type="defaultSemiBold">Annuler la demande</ThemedText>
            </Pressable>
          ) : null}

          {sendMutation.isError ? (
            <EmptyState
              title="Échec d'envoi"
              description={getApiErrorMessage(sendMutation.error)}
              actionLabel="Réessayer"
              onAction={() => sendMutation.mutate(request.id)}
            />
          ) : null}
          {cancelMutation.isError ? (
            <EmptyState
              title="Échec d'annulation"
              description={getApiErrorMessage(cancelMutation.error)}
              actionLabel="Réessayer"
              onAction={() => cancelMutation.mutate(request.id)}
            />
          ) : null}
        </>
      ) : (
        <ThemedText style={styles.note}>Chargement…</ThemedText>
      )}
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
  line: {
    opacity: 0.85,
  },
  action: {
    marginTop: 8,
    backgroundColor: '#0a7ea4',
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 14,
    alignItems: 'center',
  },
  actionSecondary: {
    borderWidth: 1,
    borderColor: '#cc4545',
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 14,
    alignItems: 'center',
  },
  actionDisabled: {
    opacity: 0.6,
  },
});
