import NetInfo from '@react-native-community/netinfo';
import { useEffect, useState } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/ThemedText';

type NetworkRetryBannerProps = {
  showOnError: boolean;
  onRetry?: () => void;
};

export function NetworkRetryBanner({ showOnError, onRetry }: NetworkRetryBannerProps) {
  const [isOffline, setIsOffline] = useState(false);

  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      setIsOffline(state.isConnected === false || state.isInternetReachable === false);
    });
    return () => unsubscribe();
  }, []);

  if (!showOnError || !isOffline) {
    return null;
  }

  return (
    <View style={styles.container}>
      <ThemedText style={styles.message}>
        Vous semblez hors ligne. Vérifiez la connexion puis réessayez.
      </ThemedText>
      {onRetry ? (
        <Pressable style={styles.button} onPress={onRetry}>
          <ThemedText type="defaultSemiBold" lightColor="#fff" darkColor="#fff">
            Réessayer
          </ThemedText>
        </Pressable>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderWidth: 1,
    borderColor: '#f0cc8b',
    backgroundColor: '#fff8ea',
    borderRadius: 10,
    padding: 12,
    gap: 8,
  },
  message: {
    color: '#7a4a00',
  },
  button: {
    alignSelf: 'flex-start',
    backgroundColor: '#b46d00',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
});
