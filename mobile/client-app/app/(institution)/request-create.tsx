import { StyleSheet } from 'react-native';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';

export default function InstitutionRequestCreateScreen() {
  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Nouvelle demande</ThemedText>
      <ThemedText style={styles.note}>
        Formulaire minimal à brancher sur POST /institutions/requests (phase 3).
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
  note: {
    opacity: 0.7,
  },
});
