// hooks/useInstitutionData.js
/**
 * ÉTAPE 6: React Query hooks pour le portail Institution
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import institutionService from '../services/institutionService';

// ============================================================================
// Query Keys (centralisés pour invalidation cohérente)
// ============================================================================

export const institutionQueryKeys = {
  all: ['institution'],
  me: () => [...institutionQueryKeys.all, 'me'],
  requests: () => [...institutionQueryKeys.all, 'requests'],
  requestsList: (filters) => [...institutionQueryKeys.requests(), 'list', filters],
  requestDetail: (id) => [...institutionQueryKeys.requests(), 'detail', id],
  patients: () => [...institutionQueryKeys.all, 'patients'],
  patientsList: (filters) => [...institutionQueryKeys.patients(), 'list', filters],
  patientDetail: (id) => [...institutionQueryKeys.patients(), 'detail', id],
  patientSyncStatus: (id) => [...institutionQueryKeys.patients(), 'sync-status', id],
  patientMatches: (id) => [...institutionQueryKeys.patients(), 'matches', id],
  patientIdentity: (id) => [...institutionQueryKeys.patients(), 'identity', id],
  patientSuggestions: (id) => [...institutionQueryKeys.patients(), 'suggestions', id],
  settings: () => [...institutionQueryKeys.all, 'settings'],
  preferences: () => [...institutionQueryKeys.all, 'preferences'],
  eligibleCompanies: () => [...institutionQueryKeys.all, 'eligible-companies'],
  apiKeys: () => [...institutionQueryKeys.all, 'api-keys'],
  users: () => [...institutionQueryKeys.all, 'users'],
  notifications: () => [...institutionQueryKeys.all, 'notifications'],
  permissionRequests: () => [...institutionQueryKeys.all, 'permission-requests'],
  teams: () => [...institutionQueryKeys.all, 'teams'],
};

// ============================================================================
// Institution Info
// ============================================================================

export function useInstitutionMe() {
  return useQuery({
    queryKey: institutionQueryKeys.me(),
    queryFn: institutionService.getMe,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useUpdateInstitution() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.updateInstitution,
    onSuccess: (data) => {
      // Mettre à jour le cache immédiatement avec les données retournées par le serveur
      if (data) {
        queryClient.setQueryData(institutionQueryKeys.me(), data);
      }
      // Puis invalider pour forcer un refetch en arrière-plan
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.me() });
    },
  });
}

// ============================================================================
// Requests
// ============================================================================

export function useInstitutionRequests(filters = {}) {
  return useQuery({
    queryKey: institutionQueryKeys.requestsList(filters),
    queryFn: () => institutionService.listRequests(filters),
    staleTime: 30 * 1000, // 30 secondes
  });
}

export function useInstitutionRequest(requestId) {
  return useQuery({
    queryKey: institutionQueryKeys.requestDetail(requestId),
    queryFn: () => institutionService.getRequest(requestId),
    enabled: !!requestId,
  });
}

export function useCreateRequest() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: institutionService.createRequest,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    },
  });
}

export function useUpdateRequest() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ requestId, data }) => institutionService.updateRequest(requestId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(variables.requestId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    },
  });
}

export function useSendRequest() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ requestId, options }) => institutionService.sendRequest(requestId, options),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(variables.requestId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    },
  });
}

export function useCancelRequest() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ requestId, reason }) => institutionService.cancelRequest(requestId, reason),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(variables.requestId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    },
  });
}

// ============================================================================
// Patients
// ============================================================================

export function useInstitutionPatients(filters = {}) {
  return useQuery({
    queryKey: institutionQueryKeys.patientsList(filters),
    queryFn: () => institutionService.listPatients(filters),
    staleTime: 60 * 1000, // 1 minute
  });
}

export function useInstitutionPatient(patientId) {
  return useQuery({
    queryKey: institutionQueryKeys.patientDetail(patientId),
    queryFn: () => institutionService.getPatient(patientId),
    enabled: !!patientId,
  });
}

export function useCreatePatient() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: institutionService.createPatient,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patients() });
    },
  });
}

export function useUpdatePatient() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ patientId, data }) => institutionService.updatePatient(patientId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientDetail(variables.patientId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patients() });
    },
  });
}

// ============================================================================
// Settings (billing, notifications, timeouts)
// ============================================================================

export function useInstitutionSettings() {
  return useQuery({
    queryKey: institutionQueryKeys.settings(),
    queryFn: institutionService.getInstitutionSettings,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useUpdateInstitutionSettings() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.updateInstitutionSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.settings() });
      // Invalider aussi 'me' car les champs billing de l'institution ont pu changer
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.me() });
    },
  });
}

// ============================================================================
// Transport Preferences
// ============================================================================

export function useTransportPreferences() {
  return useQuery({
    queryKey: institutionQueryKeys.preferences(),
    queryFn: institutionService.getTransportPreferences,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useEligibleCompanies() {
  return useQuery({
    queryKey: institutionQueryKeys.eligibleCompanies(),
    queryFn: institutionService.getEligibleCompanies,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useUpdateTransportPreferences() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: institutionService.updateTransportPreferences,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.preferences() });
    },
  });
}

// ============================================================================
// API Keys
// ============================================================================

export function useApiKeys() {
  return useQuery({
    queryKey: institutionQueryKeys.apiKeys(),
    queryFn: institutionService.listApiKeys,
    staleTime: 60 * 1000, // 1 minute
  });
}

export function useCreateApiKey() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: institutionService.createApiKey,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.apiKeys() });
    },
  });
}

export function useRevokeApiKey() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: institutionService.revokeApiKey,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.apiKeys() });
    },
  });
}

// ============================================================================
// Billing
// ============================================================================

export function useUpdateRequestBilling() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ requestId, data }) => institutionService.updateRequestBilling(requestId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(variables.requestId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    },
  });
}

export function useUpdateBookingBilling() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ bookingId, data }) => institutionService.updateBookingBilling(bookingId, data),
    onSuccess: () => {
      // Invalider toutes les requêtes de détail pour rafraîchir le booking_summary
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    },
  });
}

// ============================================================================
// Users Management
// ============================================================================

export function useInstitutionUsers() {
  return useQuery({
    queryKey: institutionQueryKeys.users(),
    queryFn: institutionService.listInstitutionUsers,
    staleTime: 2 * 60 * 1000,
  });
}

export function useInviteInstitutionUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.inviteInstitutionUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.users() });
    },
  });
}

export function useUpdateUserRole() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.updateUserRole,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.users() });
    },
  });
}

export function useRemoveInstitutionUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.removeInstitutionUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.users() });
    },
  });
}

export function useResendInvite() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.resendInvite,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.users() });
    },
  });
}

export function useDisableInstitutionUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.disableInstitutionUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.users() });
    },
  });
}

// ============================================================================
// Notifications
// ============================================================================

export function useInstitutionNotifications(options = {}) {
  return useQuery({
    queryKey: institutionQueryKeys.notifications(),
    queryFn: () => institutionService.getNotifications({ limit: 30 }),
    staleTime: 30 * 1000, // 30 secondes
    refetchInterval: 60 * 1000, // Refetch toutes les 60 secondes en fallback
    ...options,
  });
}

export function useMarkNotificationRead() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.markNotificationRead,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.notifications() });
    },
  });
}

export function useMarkAllNotificationsRead() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.markAllNotificationsRead,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.notifications() });
    },
  });
}

// ============================================================================
// Mon profil (personnel)
// ============================================================================

export function useUpdateMyProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.updateMyProfile,
    onSuccess: () => {
      // Invalider le cache /me pour rafraîchir les infos utilisateur partout
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.me() });
    },
  });
}

// ============================================================================
// Demandes de droits
// ============================================================================

export function usePermissionRequests() {
  return useQuery({
    queryKey: institutionQueryKeys.permissionRequests(),
    queryFn: institutionService.listPermissionRequests,
    staleTime: 30 * 1000,
  });
}

export function useCreatePermissionRequest() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.createPermissionRequest,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.permissionRequests() });
    },
  });
}

export function useResolvePermissionRequest() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ requestId, action }) =>
      institutionService.resolvePermissionRequest(requestId, action),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.permissionRequests() });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.users() });
    },
  });
}

// ============================================================================
// Curator Teams (curatelle)
// ============================================================================

export function useInstitutionTeams() {
  return useQuery({
    queryKey: institutionQueryKeys.teams(),
    queryFn: institutionService.listTeams,
    staleTime: 2 * 60 * 1000,
  });
}

export function useCreateTeam() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: institutionService.createTeam,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.teams() });
    },
  });
}

export function useUpdateTeam() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ teamId, data }) => institutionService.updateTeam(teamId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.teams() });
    },
  });
}

export function useDeleteTeam() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: institutionService.deleteTeam,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.teams() });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patients() });
    },
  });
}

export function useAddTeamMember() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ teamId, userId }) => institutionService.addTeamMember(teamId, userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.teams() });
    },
  });
}

export function useRemoveTeamMember() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ teamId, userId }) => institutionService.removeTeamMember(teamId, userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.teams() });
    },
  });
}

export function useAssignPatientTeam() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ patientId, teamId }) => institutionService.assignPatientTeam(patientId, teamId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.teams() });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patients() });
    },
  });
}

// ============================================================================
// Patient Identity / Sync / Matching
// ============================================================================

export function usePatientSyncStatus(patientId) {
  return useQuery({
    queryKey: institutionQueryKeys.patientSyncStatus(patientId),
    queryFn: () => institutionService.getPatientSyncStatus(patientId),
    enabled: !!patientId,
    staleTime: 30 * 1000,
  });
}

export function usePatientMatches(patientId) {
  return useQuery({
    queryKey: institutionQueryKeys.patientMatches(patientId),
    queryFn: () => institutionService.getPatientMatches(patientId),
    enabled: !!patientId,
    staleTime: 60 * 1000,
  });
}

export function usePatientIdentity(patientId) {
  return useQuery({
    queryKey: institutionQueryKeys.patientIdentity(patientId),
    queryFn: () => institutionService.getPatientIdentity(patientId),
    enabled: !!patientId,
    staleTime: 60 * 1000,
  });
}

export function useConfirmPatientMatch() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ patientId, identityId }) =>
      institutionService.confirmPatientMatch(patientId, identityId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientMatches(variables.patientId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientSyncStatus(variables.patientId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientIdentity(variables.patientId) });
    },
  });
}

export function useRejectPatientMatch() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ patientId, identityId }) =>
      institutionService.rejectPatientMatch(patientId, identityId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientMatches(variables.patientId) });
    },
  });
}

export function useDetachPatientIdentity() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ patientId, reason }) =>
      institutionService.detachPatientIdentity(patientId, reason),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientSyncStatus(variables.patientId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientIdentity(variables.patientId) });
    },
  });
}

// ============================================================================
// Patient Link Suggestions
// ============================================================================

export function usePatientSuggestions(patientId) {
  return useQuery({
    queryKey: institutionQueryKeys.patientSuggestions(patientId),
    queryFn: () => institutionService.getPatientSuggestions(patientId),
    enabled: !!patientId,
    staleTime: 30 * 1000,
  });
}

export function useConfirmPatientSuggestion() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ patientId, suggestionId }) =>
      institutionService.confirmPatientSuggestion(patientId, suggestionId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientSuggestions(variables.patientId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientSyncStatus(variables.patientId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientIdentity(variables.patientId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientMatches(variables.patientId) });
    },
  });
}

export function useRejectPatientSuggestion() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ patientId, suggestionId }) =>
      institutionService.rejectPatientSuggestion(patientId, suggestionId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.patientSuggestions(variables.patientId) });
    },
  });
}
