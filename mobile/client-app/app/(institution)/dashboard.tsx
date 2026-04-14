import { Pressable, StyleSheet } from 'react-native';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useAuth } from '@/hooks/useAuth';

export default function InstitutionDashboardScreen() {
  const { user, logout } = useAuth();

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Institution</ThemedText>
      <ThemedText style={styles.muted}>
        {user?.username ?? user?.email ?? 'Session active'}
      </ThemedText>
      <ThemedText style={styles.note}>
        Tableau de bord terrain (contenu métier à venir).
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
  logout: {
    marginTop: 24,
    alignSelf: 'flex-start',
    backgroundColor: '#0a7ea4',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
});
