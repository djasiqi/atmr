import { StyleSheet } from 'react-native';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';

export default function InstitutionRequestsScreen() {
  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Demandes</ThemedText>
      <ThemedText style={styles.note}>Liste (phase 3 du plan).</ThemedText>
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
