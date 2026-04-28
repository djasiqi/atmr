import { Pressable, StyleSheet } from 'react-native';
import { useQuery } from '@tanstack/react-query';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useAuth } from '@/hooks/useAuth';
import { getClientProfile } from '@/services/clientApi';
import { queryKeys } from '@/services/queryKeys';

export default function ClientAccountScreen() {
  const { user, logout } = useAuth();
  const profileQuery = useQuery({
    queryKey: queryKeys.clientProfile,
    queryFn: getClientProfile,
    enabled: Boolean(user),
  });
  const profile = profileQuery.data;

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Compte</ThemedText>
      <ThemedText style={styles.line}>{profile?.email ?? user?.email ?? '—'}</ThemedText>
      <ThemedText style={styles.line}>{profile?.phone ?? 'Téléphone non renseigné'}</ThemedText>
      <ThemedText style={styles.line}>{profile?.address ?? 'Adresse non renseignée'}</ThemedText>
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
    gap: 12,
  },
  line: {
    opacity: 0.85,
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
