import { useMutation, useQueryClient } from '@tanstack/react-query';
import * as Linking from 'expo-linking';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet } from 'react-native';
import * as WebBrowser from 'expo-web-browser';

import { InvalidRouteScreen } from '@/components/mobile/InvalidRouteScreen';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { assertSaferpayCheckout, initializeSaferpayCheckout } from '@/services/paymentApi';
import { clearPendingPayment, getPendingPayment, setPendingPayment } from '@/services/pendingPayment';
import { queryKeys } from '@/services/queryKeys';

function parseId(value: unknown): number | null {
  const n = Number(value);
  return Number.isInteger(n) && n > 0 ? n : null;
}

export default function PaymentScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const params = useLocalSearchParams<{ bookingId?: string; paymentId?: string }>();
  const bookingId = parseId(params.bookingId);
  const paymentIdFromQuery = parseId(params.paymentId);
  const [assertMessage, setAssertMessage] = useState<string | null>(null);
  const [lastPaymentId, setLastPaymentId] = useState<number | null>(paymentIdFromQuery);

  useEffect(() => {
    let mounted = true;
    const hydratePending = async () => {
      if (!bookingId || paymentIdFromQuery) {
        return;
      }
      const pending = await getPendingPayment();
      if (mounted && pending?.bookingId === bookingId) {
        setLastPaymentId(pending.paymentId);
      }
    };
    void hydratePending();
    return () => {
      mounted = false;
    };
  }, [bookingId, paymentIdFromQuery]);

  useEffect(() => {
    if (paymentIdFromQuery) {
      setLastPaymentId(paymentIdFromQuery);
    }
  }, [paymentIdFromQuery]);

  const returnUrl = useMemo(() => {
    if (!bookingId) return '';
    return Linking.createURL('payment-return', {
      queryParams: { bookingId: String(bookingId) },
    });
  }, [bookingId]);

  const initializeMutation = useMutation({
    mutationFn: async () => {
      if (!bookingId) throw new Error('bookingId invalide');
      return initializeSaferpayCheckout(bookingId, returnUrl);
    },
    onSuccess: async (payload) => {
      const redirectUrl = payload.redirect_url;
      if (!redirectUrl) {
        throw new Error('redirect_url manquant');
      }
      if (payload.payment_id) {
        await setPendingPayment(bookingId as number, payload.payment_id);
        setLastPaymentId(payload.payment_id);
      }
      await WebBrowser.openBrowserAsync(redirectUrl);
    },
  });

  const assertMutation = useMutation({
    mutationFn: async () => {
      if (!bookingId) throw new Error('bookingId invalide');
      if (!lastPaymentId) throw new Error('paymentId manquant');
      return assertSaferpayCheckout(bookingId, lastPaymentId);
    },
    onSuccess: async (result) => {
      setAssertMessage(result.message ?? 'Paiement confirmé.');
      await clearPendingPayment();
      await queryClient.invalidateQueries({ queryKey: queryKeys.bookings });
      await queryClient.invalidateQueries({ queryKey: queryKeys.booking(bookingId as number) });
      router.replace(`/(client)/booking/${bookingId}`);
    },
  });

  if (!bookingId) {
    return (
      <InvalidRouteScreen
        message="Impossible de lancer le paiement: bookingId invalide."
        onPress={() => router.replace('/(client)/bookings')}
      />
    );
  }

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Paiement</ThemedText>
      <ThemedText style={styles.note}>Réservation #{bookingId}</ThemedText>

      <Pressable
        style={styles.button}
        disabled={initializeMutation.isPending}
        onPress={() => initializeMutation.mutate()}
      >
        {initializeMutation.isPending ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <ThemedText type="defaultSemiBold" lightColor="#fff" darkColor="#fff">
            Ouvrir le paiement sécurisé
          </ThemedText>
        )}
      </Pressable>

      <Pressable
        style={[styles.buttonSecondary, !lastPaymentId && styles.disabled]}
        disabled={!lastPaymentId || assertMutation.isPending}
        onPress={() => assertMutation.mutate()}
      >
        {assertMutation.isPending ? (
          <ActivityIndicator />
        ) : (
          <ThemedText type="defaultSemiBold">Finaliser (assert)</ThemedText>
        )}
      </Pressable>

      {lastPaymentId ? (
        <ThemedText style={styles.small}>paymentId détecté: {lastPaymentId}</ThemedText>
      ) : (
        <ThemedText style={styles.small}>
          Aucun paymentId détecté pour le moment. Revenez après le PSP.
        </ThemedText>
      )}

      {initializeMutation.error ? (
        <ThemedText style={styles.error}>{initializeMutation.error.message}</ThemedText>
      ) : null}
      {assertMutation.error ? (
        <ThemedText style={styles.error}>
          {assertMutation.error.message} (vous pouvez réessayer)
        </ThemedText>
      ) : null}
      {assertMessage ? <ThemedText style={styles.success}>{assertMessage}</ThemedText> : null}
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
    opacity: 0.8,
  },
  button: {
    backgroundColor: '#0a7ea4',
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 14,
    alignItems: 'center',
  },
  buttonSecondary: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 14,
    alignItems: 'center',
  },
  disabled: {
    opacity: 0.6,
  },
  small: {
    fontSize: 13,
    opacity: 0.7,
  },
  error: {
    color: '#c00',
  },
  success: {
    color: '#0a7a45',
  },
});
