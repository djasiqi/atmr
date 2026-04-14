import { useRouter } from 'expo-router';
import { useEffect } from 'react';
import { ActivityIndicator, Pressable, StyleSheet } from 'react-native';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useAuth } from '@/hooks/useAuth';

export default function IndexScreen() {
  const router = useRouter();
  const {
    status,
    role,
    bootstrapError,
    retryBootstrap,
    goToLoginAfterBootstrapFailure,
    logout,
  } = useAuth();

  useEffect(() => {
    if (status === 'bootstrapping' || status === 'bootstrap_failed') {
      return;
    }
    if (status === 'unauthenticated') {
      router.replace('/(auth)/login');
      return;
    }
    if (status === 'authenticated') {
      if (role === 'client') {
        router.replace('/(client)');
      } else if (role === 'institution') {
        router.replace('/(institution)');
      }
    }
  }, [status, role, router]);

  if (status === 'bootstrapping') {
    return (
      <ThemedView style={styles.centered}>
        <ActivityIndicator size="large" />
        <ThemedText style={styles.hint}>Chargement de la session…</ThemedText>
      </ThemedView>
    );
  }

  if (status === 'bootstrap_failed') {
    return (
      <ThemedView style={styles.centered}>
        <ThemedText type="subtitle" style={styles.title}>
          Connexion impossible
        </ThemedText>
        <ThemedText style={styles.message}>{bootstrapError ?? 'Erreur inconnue'}</ThemedText>
        <Pressable style={styles.button} onPress={() => void retryBootstrap()}>
          <ThemedText type="defaultSemiBold" lightColor="#fff" darkColor="#fff">
            Réessayer
          </ThemedText>
        </Pressable>
        <Pressable style={styles.link} onPress={goToLoginAfterBootstrapFailure}>
          <ThemedText type="link">Aller au login</ThemedText>
        </Pressable>
      </ThemedView>
    );
  }

  if (status === 'authenticated' && !role) {
    return (
      <ThemedView style={styles.centered}>
        <ThemedText type="subtitle" style={styles.title}>
          Rôle non pris en charge
        </ThemedText>
        <ThemedText style={styles.message}>
          Cette application est réservée aux comptes client et institution.
        </ThemedText>
        <Pressable style={styles.button} onPress={() => void logout()}>
          <ThemedText type="defaultSemiBold" lightColor="#fff" darkColor="#fff">
            Se déconnecter
          </ThemedText>
        </Pressable>
      </ThemedView>
    );
  }

  return (
    <ThemedView style={styles.centered}>
      <ActivityIndicator size="large" />
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    gap: 16,
  },
  hint: {
    marginTop: 12,
    opacity: 0.8,
  },
  title: {
    textAlign: 'center',
  },
  message: {
    textAlign: 'center',
    opacity: 0.85,
  },
  button: {
    marginTop: 8,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    backgroundColor: '#0a7ea4',
  },
  link: {
    marginTop: 4,
    padding: 8,
  },
});
