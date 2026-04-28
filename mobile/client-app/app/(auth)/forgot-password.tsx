import { useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, TextInput } from 'react-native';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { api, getApiErrorMessage } from '@/services/api';

export default function ForgotPasswordScreen() {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async () => {
    if (!email.trim()) {
      setError('Email requis.');
      return;
    }
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      await api.post('/auth/forgot-password', { email: email.trim() });
      setMessage('Si le compte existe, un email de réinitialisation a été envoyé.');
    } catch (e) {
      setError(getApiErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Mot de passe oublié</ThemedText>
      <ThemedText style={styles.helper}>
        Entrez votre email pour recevoir un lien de réinitialisation.
      </ThemedText>
      <TextInput
        style={styles.input}
        value={email}
        onChangeText={setEmail}
        keyboardType="email-address"
        autoCapitalize="none"
        placeholder="Email"
        placeholderTextColor="#888"
      />
      <Pressable style={styles.button} onPress={() => void onSubmit()} disabled={loading}>
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <ThemedText type="defaultSemiBold" lightColor="#fff" darkColor="#fff">
            Envoyer le lien
          </ThemedText>
        )}
      </Pressable>
      {message ? <ThemedText style={styles.success}>{message}</ThemedText> : null}
      {error ? <ThemedText style={styles.error}>{error}</ThemedText> : null}
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 24,
    gap: 12,
    justifyContent: 'center',
  },
  helper: {
    opacity: 0.8,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 16,
  },
  button: {
    marginTop: 8,
    backgroundColor: '#0a7ea4',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
  },
  success: {
    color: '#0a7a45',
  },
  error: {
    color: '#c00',
  },
});
