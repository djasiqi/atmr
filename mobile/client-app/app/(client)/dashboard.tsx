import { StyleSheet } from 'react-native';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useAuth } from '@/hooks/useAuth';

export default function ClientDashboardScreen() {
  const { user } = useAuth();

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Client</ThemedText>
      <ThemedText style={styles.muted}>
        {user?.username ?? user?.email ?? 'Session active'}
      </ThemedText>
      <ThemedText style={styles.note}>
        Tableau de bord (contenu métier à venir selon le plan).
      </ThemedText>
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
});
