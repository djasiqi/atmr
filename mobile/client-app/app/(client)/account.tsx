import { Pressable, StyleSheet } from 'react-native';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useAuth } from '@/hooks/useAuth';

export default function ClientAccountScreen() {
  const { user, logout } = useAuth();

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Compte</ThemedText>
      <ThemedText style={styles.line}>{user?.email ?? user?.username ?? '—'}</ThemedText>
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
