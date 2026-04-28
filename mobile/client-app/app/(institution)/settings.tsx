import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Pressable, StyleSheet, Switch } from 'react-native';

import { EmptyState } from '@/components/mobile/EmptyState';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useAuth } from '@/hooks/useAuth';
import { getApiErrorMessage } from '@/services/api';
import { featureFlags } from '@/services/featureFlags';
import { getInstitutionMe, getInstitutionSettings, updateInstitutionSettings } from '@/services/institutionApi';
import { queryKeys } from '@/services/queryKeys';
import { useInstitutionPermissions } from '@/services/useInstitutionPermissions';

export default function InstitutionSettingsScreen() {
  const { logout } = useAuth();
  const queryClient = useQueryClient();
  const permissions = useInstitutionPermissions();
  const meQuery = useQuery({
    queryKey: queryKeys.institutionMe,
    queryFn: getInstitutionMe,
  });
  const settingsQuery = useQuery({
    queryKey: queryKeys.institutionSettings,
    queryFn: getInstitutionSettings,
  });
  const settingsMutation = useMutation({
    mutationFn: updateInstitutionSettings,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: queryKeys.institutionSettings });
    },
  });

  const updateNotif = (key: 'notify_request_sent' | 'notify_offer_accepted' | 'notify_request_expired', value: boolean) => {
    if (!permissions.canEditNotifications) return;
    settingsMutation.mutate({ [key]: value });
  };

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Réglages</ThemedText>

      {meQuery.isError ? (
        <EmptyState
          title="Impossible de charger le profil institution"
          actionLabel="Réessayer"
          onAction={() => void meQuery.refetch()}
        />
      ) : null}

      {meQuery.data ? (
        <>
          <ThemedText style={styles.line}>Nom: {meQuery.data.name ?? 'Non renseigné'}</ThemedText>
          <ThemedText style={styles.line}>Rôle: {meQuery.data.institution_role ?? 'Non renseigné'}</ThemedText>
          <ThemedText style={styles.line}>
            Email: {meQuery.data.contact_email ?? 'Non renseigné'}
          </ThemedText>
          <ThemedText style={styles.line}>
            Téléphone: {meQuery.data.contact_phone ?? 'Non renseigné'}
          </ThemedText>
          <ThemedText style={styles.line}>
            Adresse: {meQuery.data.address ?? 'Non renseignée'}
          </ThemedText>
          <ThemedText style={styles.line}>
            Équipe curatelle: {meQuery.data.user?.first_name || meQuery.data.user?.last_name
              ? 'Visible via portail web (détail équipe)'
              : 'Non disponible'}
          </ThemedText>
        </>
      ) : null}

      {featureFlags.institutionMobileSettingsNotificationsEnabled ? (
        <>
          <ThemedText type="defaultSemiBold" style={styles.sectionTitle}>
            Notifications personnelles
          </ThemedText>
          {settingsQuery.isError ? (
            <EmptyState
              title="Impossible de charger les notifications"
              description="Les préférences restent gérables sur le portail web."
              actionLabel="Réessayer"
              onAction={() => void settingsQuery.refetch()}
            />
          ) : (
            <>
              <ThemedView style={styles.toggleRow}>
                <ThemedText>Demande envoyée</ThemedText>
                <Switch
                  value={Boolean(settingsQuery.data?.notify_request_sent)}
                  onValueChange={(value) => updateNotif('notify_request_sent', value)}
                  disabled={!permissions.canEditNotifications || settingsMutation.isPending}
                />
              </ThemedView>
              <ThemedView style={styles.toggleRow}>
                <ThemedText>Offre acceptée</ThemedText>
                <Switch
                  value={Boolean(settingsQuery.data?.notify_offer_accepted)}
                  onValueChange={(value) => updateNotif('notify_offer_accepted', value)}
                  disabled={!permissions.canEditNotifications || settingsMutation.isPending}
                />
              </ThemedView>
              <ThemedView style={styles.toggleRow}>
                <ThemedText>Demande expirée</ThemedText>
                <Switch
                  value={Boolean(settingsQuery.data?.notify_request_expired)}
                  onValueChange={(value) => updateNotif('notify_request_expired', value)}
                  disabled={!permissions.canEditNotifications || settingsMutation.isPending}
                />
              </ThemedView>
            </>
          )}
          {settingsMutation.isError ? (
            <ThemedText style={styles.error}>{getApiErrorMessage(settingsMutation.error)}</ThemedText>
          ) : null}
          {!permissions.canEditNotifications ? (
            <ThemedText style={styles.helper}>
              Seuls les rôles admin/billing peuvent modifier ces préférences.
            </ThemedText>
          ) : null}
        </>
      ) : null}

      <ThemedText type="defaultSemiBold" style={styles.sectionTitle}>
        Administration avancée
      </ThemedText>
      <ThemedText style={styles.helper}>
        Gestion des utilisateurs: disponible sur le portail web.
      </ThemedText>
      <ThemedText style={styles.helper}>
        Paramètres institution avancés et équipes: disponibles sur le portail web.
      </ThemedText>
      <ThemedText style={styles.helper}>
        Intégrations DPI et clés API: disponibles sur le portail web.
      </ThemedText>

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
    padding: 16,
    gap: 10,
  },
  line: {
    opacity: 0.85,
  },
  sectionTitle: {
    marginTop: 8,
  },
  toggleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  helper: {
    opacity: 0.75,
    fontSize: 13,
  },
  error: {
    color: '#cc4545',
  },
  logout: {
    marginTop: 16,
    alignSelf: 'flex-start',
    backgroundColor: '#0a7ea4',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 14,
  },
});
