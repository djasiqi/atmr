import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Switch, TextInput } from 'react-native';

import { EmptyState } from '@/components/mobile/EmptyState';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { getApiErrorMessage } from '@/services/api';
import { featureFlags } from '@/services/featureFlags';
import { createRequest, getInstitutionSettings } from '@/services/institutionApi';
import { queryKeys } from '@/services/queryKeys';
import { useInstitutionPermissions } from '@/services/useInstitutionPermissions';

export default function InstitutionRequestCreateScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const permissions = useInstitutionPermissions();
  const [externalReference, setExternalReference] = useState('');
  const [scheduledTime, setScheduledTime] = useState('');
  const [pickupLocation, setPickupLocation] = useState('');
  const [dropoffLocation, setDropoffLocation] = useState('');
  const [pickupFloor, setPickupFloor] = useState('');
  const [pickupDoorCode, setPickupDoorCode] = useState('');
  const [dropoffFloor, setDropoffFloor] = useState('');
  const [dropoffDoorCode, setDropoffDoorCode] = useState('');
  const [floorElevatorInfo, setFloorElevatorInfo] = useState('');
  const [notes, setNotes] = useState('');
  const [billingIntent, setBillingIntent] = useState<'patient' | 'institution' | 'curator' | 'spc' | 'other'>('patient');
  const [wheelchair, setWheelchair] = useState(false);
  const [stretcher, setStretcher] = useState(false);
  const [oxygen, setOxygen] = useState(false);
  const [walking, setWalking] = useState(true);
  const [needsAssistance, setNeedsAssistance] = useState(false);

  const settingsQuery = useQuery({
    queryKey: queryKeys.institutionSettings,
    queryFn: getInstitutionSettings,
  });

  const effectiveBillingIntent = useMemo(() => {
    if (
      settingsQuery.data?.default_billing_intent
      && ['patient', 'institution', 'curator', 'spc', 'other'].includes(
        settingsQuery.data.default_billing_intent,
      )
    ) {
      return settingsQuery.data.default_billing_intent as
        | 'patient'
        | 'institution'
        | 'curator'
        | 'spc'
        | 'other';
    }
    return billingIntent;
  }, [billingIntent, settingsQuery.data?.default_billing_intent]);

  const createMutation = useMutation({
    mutationFn: createRequest,
    onSuccess: async (created) => {
      await queryClient.invalidateQueries({
        queryKey: ['institution', 'requests'],
      });
      await queryClient.invalidateQueries({
        queryKey: queryKeys.institutionRequest(created.id),
      });
      router.replace(`/(institution)/request/${created.id}`);
    },
  });

  const onSubmit = () => {
    createMutation.mutate({
      external_reference: externalReference.trim(),
      scheduled_time: scheduledTime.trim(),
      pickup_location: pickupLocation.trim(),
      dropoff_location: dropoffLocation.trim(),
      pickup_floor: pickupFloor.trim() || undefined,
      pickup_door_code: pickupDoorCode.trim() || undefined,
      dropoff_floor: dropoffFloor.trim() || undefined,
      dropoff_door_code: dropoffDoorCode.trim() || undefined,
      floor_elevator_info: floorElevatorInfo.trim() || undefined,
      billing_intent: featureFlags.institutionMobileBillingIntentEnabled
        ? (billingIntent || effectiveBillingIntent)
        : undefined,
      notes: notes.trim() || undefined,
      mobility: {
        wheelchair,
        stretcher,
        oxygen,
        walking,
        needs_assistance: needsAssistance,
      },
    });
  };

  const canSubmit = [
    externalReference.trim(),
    scheduledTime.trim(),
    pickupLocation.trim(),
    dropoffLocation.trim(),
  ].every(Boolean);

  if (!permissions.canCreateRequest) {
    return (
      <ThemedView style={styles.container}>
        <EmptyState
          title="Action non autorisée"
          description="Votre rôle ne permet pas de créer des demandes."
        />
      </ThemedView>
    );
  }

  return (
    <ThemedView style={styles.container}>
      <ScrollView contentContainerStyle={styles.form}>
        <ThemedText type="title">Nouvelle demande</ThemedText>
        <ThemedText style={styles.note}>
          Format ISO attendu pour l&apos;horaire: `YYYY-MM-DDTHH:mm:ss`.
        </ThemedText>

        <TextInput
          style={styles.input}
          placeholder="Référence externe *"
          placeholderTextColor="#8b8b8b"
          value={externalReference}
          onChangeText={setExternalReference}
        />
        <TextInput
          style={styles.input}
          placeholder="Date/heure prévue (ISO) *"
          placeholderTextColor="#8b8b8b"
          value={scheduledTime}
          onChangeText={setScheduledTime}
        />
        <TextInput
          style={styles.input}
          placeholder="Adresse départ *"
          placeholderTextColor="#8b8b8b"
          value={pickupLocation}
          onChangeText={setPickupLocation}
        />
        <TextInput
          style={styles.input}
          placeholder="Adresse destination *"
          placeholderTextColor="#8b8b8b"
          value={dropoffLocation}
          onChangeText={setDropoffLocation}
        />
        {featureFlags.institutionMobileFieldsetTerrainRequired ? (
          <>
            <TextInput
              style={styles.input}
              placeholder="Étage départ"
              placeholderTextColor="#8b8b8b"
              value={pickupFloor}
              onChangeText={setPickupFloor}
            />
            <TextInput
              style={styles.input}
              placeholder="Code porte départ"
              placeholderTextColor="#8b8b8b"
              value={pickupDoorCode}
              onChangeText={setPickupDoorCode}
            />
            <TextInput
              style={styles.input}
              placeholder="Étage destination"
              placeholderTextColor="#8b8b8b"
              value={dropoffFloor}
              onChangeText={setDropoffFloor}
            />
            <TextInput
              style={styles.input}
              placeholder="Code porte destination"
              placeholderTextColor="#8b8b8b"
              value={dropoffDoorCode}
              onChangeText={setDropoffDoorCode}
            />
            <TextInput
              style={styles.input}
              placeholder="Infos étage/ascenseur"
              placeholderTextColor="#8b8b8b"
              value={floorElevatorInfo}
              onChangeText={setFloorElevatorInfo}
            />
          </>
        ) : null}
        <TextInput
          style={styles.input}
          placeholder="Notes"
          placeholderTextColor="#8b8b8b"
          value={notes}
          onChangeText={setNotes}
        />

        {featureFlags.institutionMobileBillingIntentEnabled ? (
          <>
            <ThemedText type="defaultSemiBold" style={styles.sectionTitle}>
              Intent de facturation
            </ThemedText>
            <ThemedView style={styles.choiceRow}>
              {(['patient', 'institution', 'curator', 'spc', 'other'] as const).map((option) => (
                <Pressable
                  key={option}
                  style={[
                    styles.choice,
                    billingIntent === option ? styles.choiceActive : undefined,
                  ]}
                  onPress={() => setBillingIntent(option)}
                >
                  <ThemedText>{option}</ThemedText>
                </Pressable>
              ))}
            </ThemedView>
          </>
        ) : null}

        <ThemedText type="defaultSemiBold" style={styles.sectionTitle}>
          Mobilité
        </ThemedText>
        <ThemedView style={styles.switchRow}>
          <ThemedText>Fauteuil roulant</ThemedText>
          <Switch value={wheelchair} onValueChange={setWheelchair} />
        </ThemedView>
        <ThemedView style={styles.switchRow}>
          <ThemedText>Brancard</ThemedText>
          <Switch value={stretcher} onValueChange={setStretcher} />
        </ThemedView>
        <ThemedView style={styles.switchRow}>
          <ThemedText>Oxygène</ThemedText>
          <Switch value={oxygen} onValueChange={setOxygen} />
        </ThemedView>
        <ThemedView style={styles.switchRow}>
          <ThemedText>Marche</ThemedText>
          <Switch value={walking} onValueChange={setWalking} />
        </ThemedView>
        <ThemedView style={styles.switchRow}>
          <ThemedText>Besoin d&apos;assistance</ThemedText>
          <Switch value={needsAssistance} onValueChange={setNeedsAssistance} />
        </ThemedView>

        {createMutation.isError ? (
          <EmptyState
            title="Création impossible"
            description={getApiErrorMessage(createMutation.error)}
            actionLabel="Réessayer"
            onAction={onSubmit}
          />
        ) : null}

        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Créer la demande"
          onPress={onSubmit}
          style={[styles.cta, (!canSubmit || createMutation.isPending) ? styles.ctaDisabled : undefined]}
          disabled={!permissions.canCreateRequest || !canSubmit || createMutation.isPending}
        >
          <ThemedText type="defaultSemiBold" lightColor="#fff" darkColor="#fff">
            {createMutation.isPending ? 'Création…' : 'Créer la demande'}
          </ThemedText>
        </Pressable>
      </ScrollView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  form: {
    gap: 10,
    paddingBottom: 24,
  },
  note: {
    opacity: 0.7,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  sectionTitle: {
    marginTop: 8,
  },
  choiceRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  choice: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 10,
  },
  choiceActive: {
    borderColor: '#0a7ea4',
    backgroundColor: 'rgba(10,126,164,0.15)',
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  cta: {
    marginTop: 12,
    backgroundColor: '#0a7ea4',
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 14,
    alignItems: 'center',
  },
  ctaDisabled: {
    opacity: 0.6,
  },
});
