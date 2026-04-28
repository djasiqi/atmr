import { Pressable, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/ThemedText';

type InvalidRouteScreenProps = {
  title?: string;
  message?: string;
  actionLabel?: string;
  onPress: () => void;
};

export function InvalidRouteScreen({
  title = 'Paramètre invalide',
  message = 'Le lien est incomplet ou la ressource est introuvable.',
  actionLabel = 'Retour à la liste',
  onPress,
}: InvalidRouteScreenProps) {
  return (
    <View style={styles.container}>
      <ThemedText type="subtitle" style={styles.title}>
        {title}
      </ThemedText>
      <ThemedText style={styles.message}>{message}</ThemedText>
      <Pressable style={styles.button} onPress={onPress}>
        <ThemedText type="defaultSemiBold" lightColor="#fff" darkColor="#fff">
          {actionLabel}
        </ThemedText>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    gap: 12,
  },
  title: {
    textAlign: 'center',
  },
  message: {
    textAlign: 'center',
    opacity: 0.8,
  },
  button: {
    marginTop: 8,
    borderRadius: 8,
    backgroundColor: '#0a7ea4',
    paddingVertical: 10,
    paddingHorizontal: 16,
  },
});
