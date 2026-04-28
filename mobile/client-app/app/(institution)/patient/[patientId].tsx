import { useQuery } from '@tanstack/react-query';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { StyleSheet } from 'react-native';

import { InvalidRouteScreen } from '@/components/mobile/InvalidRouteScreen';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { featureFlags } from '@/services/featureFlags';
import { getPatient } from '@/services/institutionApi';
import { queryKeys } from '@/services/queryKeys';

function parsePatientId(value: unknown): number | null {
  const n = Number(value);
  return Number.isInteger(n) && n > 0 ? n : null;
}

export default function InstitutionPatientDetailsScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ patientId?: string }>();
  const patientId = parsePatientId(params.patientId);

  const patientQuery = useQuery({
    queryKey: queryKeys.institutionPatient(patientId ?? 'invalid'),
    queryFn: () => getPatient(patientId as number),
    enabled: featureFlags.institutionMobilePatientDetailEnabled && patientId !== null,
  });

  if (!featureFlags.institutionMobilePatientDetailEnabled) {
    return (
      <InvalidRouteScreen
        title="Vue non disponible"
        message="La fiche patient mobile est désactivée."
        onPress={() => router.replace('/(institution)/patients')}
      />
    );
  }

  if (!patientId) {
    return (
      <InvalidRouteScreen
        message="Identifiant patient invalide."
        onPress={() => router.replace('/(institution)/patients')}
      />
    );
  }

  if (patientQuery.isError) {
    return (
      <InvalidRouteScreen
        title="Patient introuvable"
        onPress={() => router.replace('/(institution)/patients')}
      />
    );
  }

  const patient = patientQuery.data;
  const fallbackName = `${patient?.first_name ?? ''} ${patient?.last_name ?? ''}`.trim();
  const displayName = patient?.full_name ?? (fallbackName || (patient ? `Patient #${patient.id}` : 'Patient'));

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Détail patient</ThemedText>
      {patient ? (
        <>
          <ThemedText style={styles.line}>
            Nom: {displayName}
          </ThemedText>
          <ThemedText style={styles.line}>Téléphone: {patient.phone ?? 'Non renseigné'}</ThemedText>
          <ThemedText style={styles.line}>Adresse: {patient.address ?? 'Non renseignée'}</ThemedText>
          <ThemedText style={styles.line}>Code porte: {patient.door_code ?? 'Non renseigné'}</ThemedText>
          <ThemedText style={styles.line}>Étage: {patient.floor ?? 'Non renseigné'}</ThemedText>
          <ThemedText style={styles.line}>
            Notes d&apos;accès: {patient.access_notes ?? 'Non renseignées'}
          </ThemedText>
          <ThemedText style={styles.line}>Genre: {patient.gender ?? 'Non renseigné'}</ThemedText>
          <ThemedText style={styles.line}>Naissance: {patient.dob ?? 'Non renseignée'}</ThemedText>
        </>
      ) : (
        <ThemedText style={styles.line}>Chargement…</ThemedText>
      )}
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    gap: 8,
  },
  line: {
    opacity: 0.85,
  },
});
