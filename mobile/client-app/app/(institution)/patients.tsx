import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { useRouter } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, TextInput } from 'react-native';

import { EmptyState } from '@/components/mobile/EmptyState';
import { MobileListCard } from '@/components/mobile/MobileListCard';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { getApiErrorMessage } from '@/services/api';
import { featureFlags } from '@/services/featureFlags';
import { createPatient, listPatients } from '@/services/institutionApi';
import { queryKeys } from '@/services/queryKeys';
import { useInstitutionPermissions } from '@/services/useInstitutionPermissions';

export default function InstitutionPatientsScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { canCreatePatient } = useInstitutionPermissions();
  const [searchInput, setSearchInput] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [searchUnsupported, setSearchUnsupported] = useState(false);
  const [page, setPage] = useState(1);
  const [showCreate, setShowCreate] = useState(false);
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [dob, setDob] = useState('');
  const [gender, setGender] = useState('');
  const [address, setAddress] = useState('');
  const [phone, setPhone] = useState('');
  const [doorCode, setDoorCode] = useState('');
  const [floor, setFloor] = useState('');
  const [accessNotes, setAccessNotes] = useState('');
  const [notes, setNotes] = useState('');

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(searchInput.trim()), 250);
    return () => clearTimeout(t);
  }, [searchInput]);

  useEffect(() => {
    setPage(1);
  }, [debouncedSearch]);

  const patientsQuery = useQuery({
    queryKey: queryKeys.institutionPatientsPage({
      query: debouncedSearch,
      page,
      per_page: 20,
    }),
    queryFn: async () => {
      try {
        return await listPatients(
          debouncedSearch ? { query: debouncedSearch, page, per_page: 20 } : { page, per_page: 20 },
        );
      } catch (error: unknown) {
        const status = (error as { response?: { status?: number } })?.response?.status;
        if (debouncedSearch && status === 400) {
          setSearchUnsupported(true);
          return listPatients({ page, per_page: 20 });
        }
        throw error;
      }
    },
  });

  const createMutation = useMutation({
    mutationFn: createPatient,
    onSuccess: async () => {
      setShowCreate(false);
      setFirstName('');
      setLastName('');
      setDob('');
      setGender('');
      setAddress('');
      setPhone('');
      setDoorCode('');
      setFloor('');
      setAccessNotes('');
      setNotes('');
      await queryClient.invalidateQueries({
        queryKey: ['institution', 'patients'],
      });
    },
  });

  const canCreate = canCreatePatient && firstName.trim() && lastName.trim();

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Patients</ThemedText>
      {canCreatePatient ? (
        <Pressable style={styles.createToggle} onPress={() => setShowCreate((v) => !v)}>
          <ThemedText type="defaultSemiBold">
            {showCreate ? 'Fermer le formulaire' : 'Nouveau patient'}
          </ThemedText>
        </Pressable>
      ) : null}
      {showCreate ? (
        <ThemedView style={styles.createPanel}>
          <TextInput
            style={styles.input}
            placeholder="Prénom *"
            placeholderTextColor="#8b8b8b"
            value={firstName}
            onChangeText={setFirstName}
          />
          <TextInput
            style={styles.input}
            placeholder="Nom *"
            placeholderTextColor="#8b8b8b"
            value={lastName}
            onChangeText={setLastName}
          />
          <TextInput
            style={styles.input}
            placeholder="Date de naissance (YYYY-MM-DD)"
            placeholderTextColor="#8b8b8b"
            value={dob}
            onChangeText={setDob}
          />
          <TextInput
            style={styles.input}
            placeholder="Genre"
            placeholderTextColor="#8b8b8b"
            value={gender}
            onChangeText={setGender}
          />
          <TextInput
            style={styles.input}
            placeholder="Adresse"
            placeholderTextColor="#8b8b8b"
            value={address}
            onChangeText={setAddress}
          />
          <TextInput
            style={styles.input}
            placeholder="Téléphone"
            placeholderTextColor="#8b8b8b"
            value={phone}
            onChangeText={setPhone}
          />
          <TextInput
            style={styles.input}
            placeholder="Code porte/interphone"
            placeholderTextColor="#8b8b8b"
            value={doorCode}
            onChangeText={setDoorCode}
          />
          <TextInput
            style={styles.input}
            placeholder="Étage"
            placeholderTextColor="#8b8b8b"
            value={floor}
            onChangeText={setFloor}
          />
          <TextInput
            style={styles.input}
            placeholder="Notes d'accès"
            placeholderTextColor="#8b8b8b"
            value={accessNotes}
            onChangeText={setAccessNotes}
          />
          <TextInput
            style={styles.input}
            placeholder="Notes"
            placeholderTextColor="#8b8b8b"
            value={notes}
            onChangeText={setNotes}
          />
          {createMutation.isError ? (
            <EmptyState
              title="Création impossible"
              description={getApiErrorMessage(createMutation.error)}
              actionLabel="Réessayer"
              onAction={() => createMutation.mutate({
                first_name: firstName.trim(),
                last_name: lastName.trim(),
                dob: dob.trim() || undefined,
                gender: gender.trim() || undefined,
                address: address.trim() || undefined,
                phone: phone.trim() || undefined,
                door_code: featureFlags.institutionMobileFieldsetTerrainRequired
                  ? doorCode.trim() || undefined
                  : undefined,
                floor: featureFlags.institutionMobileFieldsetTerrainRequired
                  ? floor.trim() || undefined
                  : undefined,
                access_notes: featureFlags.institutionMobileFieldsetTerrainRequired
                  ? accessNotes.trim() || undefined
                  : undefined,
                notes: notes.trim() || undefined,
              })}
            />
          ) : null}
          <Pressable
            style={[styles.createCta, (!canCreate || createMutation.isPending) ? styles.disabled : undefined]}
            disabled={!canCreate || createMutation.isPending}
            onPress={() => createMutation.mutate({
              first_name: firstName.trim(),
              last_name: lastName.trim(),
              dob: dob.trim() || undefined,
              gender: gender.trim() || undefined,
              address: address.trim() || undefined,
              phone: phone.trim() || undefined,
              door_code: featureFlags.institutionMobileFieldsetTerrainRequired
                ? doorCode.trim() || undefined
                : undefined,
              floor: featureFlags.institutionMobileFieldsetTerrainRequired
                ? floor.trim() || undefined
                : undefined,
              access_notes: featureFlags.institutionMobileFieldsetTerrainRequired
                ? accessNotes.trim() || undefined
                : undefined,
              notes: notes.trim() || undefined,
            })}
          >
            <ThemedText type="defaultSemiBold" lightColor="#fff" darkColor="#fff">
              {createMutation.isPending ? 'Création…' : 'Créer le patient'}
            </ThemedText>
          </Pressable>
        </ThemedView>
      ) : null}
      <TextInput
        style={styles.input}
        placeholder="Rechercher un patient"
        placeholderTextColor="#8b8b8b"
        editable={!searchUnsupported}
        value={searchInput}
        onChangeText={setSearchInput}
      />
      {searchUnsupported ? (
        <ThemedText style={styles.helper}>
          La recherche n'est pas disponible sur cette API. Liste complète affichée.
        </ThemedText>
      ) : null}

      {patientsQuery.isError ? (
        <EmptyState
          title="Impossible de charger les patients"
          actionLabel="Réessayer"
          onAction={() => void patientsQuery.refetch()}
        />
      ) : null}

      {!patientsQuery.isError && (patientsQuery.data?.items.length ?? 0) === 0 ? (
        <EmptyState title="Aucun patient" description="Aucun résultat pour cette recherche." />
      ) : null}

      <ScrollView contentContainerStyle={styles.list}>
        {(patientsQuery.data?.items ?? []).map((patient) => (
          <MobileListCard
            key={patient.id}
            title={
              (patient.full_name
              ?? `${patient.first_name ?? ''} ${patient.last_name ?? ''}`.trim())
              || `Patient #${patient.id}`
            }
            subtitle={patient.phone ?? 'Téléphone non renseigné'}
            meta={patient.access_notes ?? patient.email ?? 'Accès non renseigné'}
            onPress={
              featureFlags.institutionMobilePatientDetailEnabled
                ? () => router.push(`/(institution)/patient/${patient.id}`)
                : undefined
            }
          />
        ))}
      </ScrollView>
      <ThemedView style={styles.pagination}>
        <Pressable
          style={[styles.pageButton, page <= 1 ? styles.disabled : undefined]}
          disabled={page <= 1}
          onPress={() => setPage((p) => Math.max(1, p - 1))}
        >
          <ThemedText>Précédent</ThemedText>
        </Pressable>
        <ThemedText>
          Page {patientsQuery.data?.page ?? page}
          {patientsQuery.data?.pages ? ` / ${patientsQuery.data.pages}` : ''}
        </ThemedText>
        <Pressable
          style={[
            styles.pageButton,
            (patientsQuery.data?.pages ? page >= patientsQuery.data.pages : false)
              ? styles.disabled
              : undefined,
          ]}
          disabled={patientsQuery.data?.pages ? page >= patientsQuery.data.pages : false}
          onPress={() => setPage((p) => p + 1)}
        >
          <ThemedText>Suivant</ThemedText>
        </Pressable>
      </ThemedView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    gap: 10,
  },
  createToggle: {
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#0a7ea4',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  createPanel: {
    gap: 8,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  helper: {
    fontSize: 13,
    opacity: 0.75,
  },
  list: {
    gap: 10,
    paddingBottom: 20,
  },
  createCta: {
    marginTop: 4,
    alignItems: 'center',
    backgroundColor: '#0a7ea4',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 12,
  },
  pagination: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  pageButton: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  disabled: {
    opacity: 0.45,
  },
});
