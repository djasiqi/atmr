import { StyleSheet } from 'react-native';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';

export default function ClientBookingsScreen() {
  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Mes courses</ThemedText>
      <ThemedText style={styles.note}>Liste des réservations (phase 2).</ThemedText>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    gap: 8,
  },
  note: {
    opacity: 0.7,
  },
});
