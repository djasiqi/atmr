// hooks/useInstitutionData.js
/**
 * ÉTAPE 6: React Query hooks pour le portail Institution
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import institutionService from '../services/institutionService';
import institutionBillingControlService from '../services/institutionBillingControlService';
import {
  applyOptimisticControlStatus,
  applyOptimisticPayerMutation,
  mergeBookingControlFromMutation,
} from '../utils/institutionBillingControlUi';
import {
  markBillingPayerRequestEnd,
  markBillingPayerRequestStart,
  markBillingValidateRequestEnd,
  markBillingValidateRequestStart,
} from '../utils/billingControlValidatePerf';
import {
  applyOperationalBookingPatchToRequest,
  upsertInstitutionRequestInLists,
} from '../utils/institutionRequestCache';
import {
  ELIGIBLE_CARRIERS_STALE_MS,
  ELIGIBLE_CARRIERS_TIMEOUT_MS,
  shouldRetryEligibleCarriers,
} from '../utils/institutionEligibleCarriers';

// ============================================================================
// Query Keys (centralisés pour invalidation cohérente)
// ============================================================================

export const institutionQueryKeys = {
  all: ['institution'],
  me: () => [...institutionQueryKeys.all, 'me'],
  requests: () => [...institutionQueryKeys.all, 'requests'],
  requestsList: (filters) => [...institutionQueryKeys.requests(), 'list', filters],
  requestDetail: (id) => [...institutionQueryKeys.requests(), 'detail', id],
  requestTimeline: (id) => [...institutionQueryKeys.requests(), 'timeline', id],
  bookingTimeline: (id) => [...institutionQueryKeys.requests(), 'booking-timeline', id],
  patientTransportHistory: (id) => [...institutionQueryKeys.patients(), 'transport-history', id],
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
  pendingActivation: () => [...institutionQueryKeys.all, 'users', 'pending-activation'],
  notifications: () => [...institutionQueryKeys.all, 'notifications'],
  permissionRequests: () => [...institutionQueryKeys.all, 'permission-requests'],
  teams: () => [...institutionQueryKeys.all, 'teams'],
  billingControl: () => [...institutionQueryKeys.all, 'billing-control'],
  billingControlList: (filters) => [...institutionQueryKeys.billingControl(), 'list', filters],
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

function markInstitutionRequestsStale(queryClient, requestId) {
  queryClient.invalidateQueries({
    queryKey: institutionQueryKeys.requests(),
    refetchType: 'none',
  });
  if (requestId != null) {
    queryClient.invalidateQueries({
      queryKey: institutionQueryKeys.requestDetail(requestId),
      refetchType: 'none',
    });
  }
}

function writeInstitutionRequestCache(queryClient, request) {
  if (!request?.id) return;
  queryClient.setQueryData(institutionQueryKeys.requestDetail(request.id), request);
  queryClient.setQueriesData(
    { queryKey: [...institutionQueryKeys.requests(), 'list'] },
    (old) => upsertInstitutionRequestInLists(old, request),
  );
}

export function useCreateRequest() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: institutionService.createRequest,
    onSuccess: (created) => {
      writeInstitutionRequestCache(queryClient, created);
      markInstitutionRequestsStale(queryClient, created?.id);
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
    onSuccess: (sent, variables) => {
      writeInstitutionRequestCache(queryClient, sent);
      markInstitutionRequestsStale(queryClient, variables?.requestId || sent?.id);
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

export function useAssignExternalCarrier() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ requestId, data }) => institutionService.assignExternalCarrier(requestId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(variables.requestId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestTimeline(variables.requestId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    },
  });
}

export function useCompleteExternalMission() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ requestId, data }) => institutionService.completeExternalMission(requestId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestDetail(variables.requestId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requestTimeline(variables.requestId) });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    },
  });
}

// ============================================================================
// Patients
// ============================================================================

export function useInstitutionPatients(filters = {}) {
  const { fetchAll = false, ...queryFilters } = filters;
  return useQuery({
    queryKey: institutionQueryKeys.patientsList({ fetchAll, ...queryFilters }),
    queryFn: () =>
      fetchAll
        ? institutionService.listAllPatients(queryFilters)
        : institutionService.listPatients(queryFilters),
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

export function useEligibleCompanies({ enabled = true } = {}) {
  return useQuery({
    queryKey: institutionQueryKeys.eligibleCompanies(),
    queryFn: ({ signal }) => institutionService.getEligibleCompanies({
      signal,
      timeout: ELIGIBLE_CARRIERS_TIMEOUT_MS,
    }),
    staleTime: ELIGIBLE_CARRIERS_STALE_MS,
    gcTime: 30 * 60 * 1000,
    enabled,
    retry: shouldRetryEligibleCarriers,
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

export function usePatchInstitutionBooking() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ bookingId, data }) => institutionService.patchInstitutionBooking(bookingId, data),
    onSuccess: (data, vars) => {
      if (vars.requestId) {
        queryClient.setQueryData(
          institutionQueryKeys.requestDetail(vars.requestId),
          (old) => applyOperationalBookingPatchToRequest(old, vars.data, data),
        );
        const patched = queryClient.getQueryData(
          institutionQueryKeys.requestDetail(vars.requestId),
        );
        if (patched) {
          queryClient.setQueriesData(
            { queryKey: [...institutionQueryKeys.requests(), 'list'] },
            (old) => upsertInstitutionRequestInLists(old, patched),
          );
        }
        queryClient.invalidateQueries({
          queryKey: institutionQueryKeys.requestDetail(vars.requestId),
        });
        queryClient.invalidateQueries({
          queryKey: institutionQueryKeys.requestTimeline(vars.requestId),
        });
      }
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
    },
    onError: (err, vars) => {
      const unchanged = err?.response?.data?.error === 'Aucun champ modifié.';
      if (unchanged && vars.requestId) {
        queryClient.invalidateQueries({
          queryKey: institutionQueryKeys.requestDetail(vars.requestId),
        });
        queryClient.invalidateQueries({
          queryKey: institutionQueryKeys.requestTimeline(vars.requestId),
        });
      }
    },
  });
}

export function useCancelInstitutionBooking() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ bookingId, data }) => institutionService.cancelInstitutionBooking(bookingId, data),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.requests() });
      if (variables?.requestId) {
        queryClient.invalidateQueries({
          queryKey: institutionQueryKeys.requestDetail(variables.requestId),
        });
        queryClient.invalidateQueries({
          queryKey: institutionQueryKeys.requestTimeline(variables.requestId),
        });
      }
    },
  });
}

export function useBookingChangeEvents(bookingId, enabled = true) {
  return useQuery({
    queryKey: [...institutionQueryKeys.requests(), 'change-events', bookingId],
    queryFn: () => institutionService.fetchBookingChangeEvents(bookingId),
    enabled: Boolean(bookingId) && enabled,
    staleTime: 30 * 1000,
  });
}

export function useRequestTimeline(requestId, enabled = true) {
  return useQuery({
    queryKey: institutionQueryKeys.requestTimeline(requestId),
    queryFn: () => institutionService.getRequestTimeline(requestId),
    enabled: Boolean(requestId) && enabled,
    staleTime: 30 * 1000,
  });
}

export function useBookingTimeline(bookingId, enabled = true) {
  return useQuery({
    queryKey: institutionQueryKeys.bookingTimeline(bookingId),
    queryFn: () => institutionService.getBookingTimeline(bookingId),
    enabled: Boolean(bookingId) && enabled,
    staleTime: 30 * 1000,
  });
}

export function usePatientTransportHistory(patientId, enabled = true) {
  return useQuery({
    queryKey: institutionQueryKeys.patientTransportHistory(patientId),
    queryFn: () => institutionService.getPatientTransportHistory(patientId),
    enabled: Boolean(patientId) && enabled,
    staleTime: 60 * 1000,
  });
}

export function useReleaseBookingForRedispatch() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ bookingId, ...data }) =>
      institutionService.releaseBookingForRedispatch(bookingId, data),
    onSuccess: () => {
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

export function useUpdateInstitutionUserProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.updateUserProfile,
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
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.pendingActivation() });
    },
  });
}

export function usePendingActivationUsers() {
  return useQuery({
    queryKey: institutionQueryKeys.pendingActivation(),
    queryFn: institutionService.getPendingActivationUsers,
    staleTime: 60 * 1000,
  });
}

export function useResetInstitutionUserPassword() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: institutionService.resetInstitutionUserPassword,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.users() });
      queryClient.invalidateQueries({ queryKey: institutionQueryKeys.pendingActivation() });
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

// ============================================================================
// Billing Control (INSTITUTION-07)
// ============================================================================

export function useBillingControlBookings(filters = {}, enabled = true) {
  return useQuery({
    queryKey: institutionQueryKeys.billingControlList(filters),
    queryFn: () => institutionBillingControlService.listBillingControlBookings(filters),
    enabled,
    staleTime: 15 * 1000,
    retry: (failureCount, error) => {
      if (error?.response?.status === 403) return false;
      return failureCount < 2;
    },
  });
}

function invalidateBillingControlQueries(queryClient) {
  queryClient.invalidateQueries({ queryKey: institutionQueryKeys.billingControl() });
}

function restoreBillingControlSnapshots(queryClient, snapshots) {
  snapshots?.forEach(([key, data]) => {
    queryClient.setQueryData(key, data);
  });
}

function invalidateBillingControlWhenIdle(queryClient, mutationKey) {
  const inflight = queryClient.isMutating({ mutationKey });
  if (inflight === 0) {
    invalidateBillingControlQueries(queryClient);
  }
}

function useBillingControlStatusMutation({
  mutationKey,
  requestFn,
  nextStatus,
  controlPatchFromVariables,
}) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey,
    mutationFn: async (variables) => {
      const { bookingId, data, clickAt } = variables;
      const requestStartAt = markBillingValidateRequestStart(bookingId);
      try {
        const result = await requestFn(bookingId, data);
        markBillingValidateRequestEnd(bookingId, { clickAt, requestStartAt, ok: true });
        return result;
      } catch (error) {
        markBillingValidateRequestEnd(bookingId, { clickAt, requestStartAt, ok: false });
        throw error;
      }
    },
    onMutate: async (variables) => {
      const { bookingId } = variables;
      await queryClient.cancelQueries({ queryKey: institutionQueryKeys.billingControl() });
      const snapshots = queryClient.getQueriesData({
        queryKey: institutionQueryKeys.billingControl(),
      });
      const controlPatch = controlPatchFromVariables?.(variables) || {};
      queryClient.setQueriesData(
        { queryKey: institutionQueryKeys.billingControl() },
        (old) => applyOptimisticControlStatus(old, bookingId, nextStatus, controlPatch),
      );
      return { snapshots };
    },
    onError: (_error, _variables, context) => {
      restoreBillingControlSnapshots(queryClient, context?.snapshots);
    },
    onSuccess: (result, { bookingId }) => {
      if (result?.control) {
        queryClient.setQueriesData(
          { queryKey: institutionQueryKeys.billingControl() },
          (old) => mergeBookingControlFromMutation(old, bookingId, result.control),
        );
      }
    },
    onSettled: () => {
      invalidateBillingControlWhenIdle(queryClient, mutationKey);
    },
  });
}

const VALIDATE_MUTATION_KEY = ['institution', 'billing-control', 'validate'];
const REOPEN_MUTATION_KEY = ['institution', 'billing-control', 'reopen'];
const ANOMALY_MUTATION_KEY = ['institution', 'billing-control', 'anomaly'];

export function useValidateBillingControlBooking() {
  return useBillingControlStatusMutation({
    mutationKey: VALIDATE_MUTATION_KEY,
    requestFn: institutionBillingControlService.validateBillingControlBooking,
    nextStatus: 'validated',
  });
}

export function useMarkBillingControlAnomaly() {
  return useBillingControlStatusMutation({
    mutationKey: ANOMALY_MUTATION_KEY,
    requestFn: institutionBillingControlService.markBillingControlAnomaly,
    nextStatus: 'anomaly',
    controlPatchFromVariables: ({ data } = {}) => ({
      anomaly_reason: data?.comment
        ? `${data.anomaly_reason_code || 'OTHER'}: ${data.comment}`
        : null,
    }),
  });
}

export function useReopenBillingControlBooking() {
  return useBillingControlStatusMutation({
    mutationKey: REOPEN_MUTATION_KEY,
    requestFn: institutionBillingControlService.reopenBillingControlBooking,
    nextStatus: 'pending_review',
    controlPatchFromVariables: () => ({
      validated_at: null,
      validated_by_display_name: null,
      anomaly_reason: null,
    }),
  });
}

const PAYER_MUTATION_KEY = ['institution', 'billing-control', 'payer'];
const latestPayerGeneration = new Map();

export function useChangeBillingControlPayer() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationKey: PAYER_MUTATION_KEY,
    mutationFn: async ({ bookingId, data, clickAt, signal, generation }) => {
      if (generation != null) {
        latestPayerGeneration.set(String(bookingId), generation);
      }
      const requestStartAt = markBillingPayerRequestStart(bookingId);
      try {
        const result = await institutionBillingControlService.changeBillingControlPayer(
          bookingId,
          data,
          signal ? { signal } : {},
        );
        markBillingPayerRequestEnd(bookingId, { clickAt, requestStartAt, ok: true });
        return result;
      } catch (error) {
        markBillingPayerRequestEnd(bookingId, { clickAt, requestStartAt, ok: false });
        throw error;
      }
    },
    onMutate: async ({ bookingId, payerType }) => {
      await queryClient.cancelQueries({ queryKey: institutionQueryKeys.billingControl() });
      const snapshots = queryClient.getQueriesData({
        queryKey: institutionQueryKeys.billingControl(),
      });
      queryClient.setQueriesData(
        { queryKey: institutionQueryKeys.billingControl() },
        (old) => applyOptimisticPayerMutation(old, bookingId, payerType),
      );
      return { snapshots };
    },
    onError: (_error, variables, context) => {
      if (
        variables?.generation != null
        && latestPayerGeneration.get(String(variables.bookingId)) !== variables.generation
      ) {
        return;
      }
      restoreBillingControlSnapshots(queryClient, context?.snapshots);
    },
    onSettled: (_result, _error, variables) => {
      if (
        variables?.generation != null
        && latestPayerGeneration.get(String(variables.bookingId)) !== variables.generation
      ) {
        return;
      }
      invalidateBillingControlWhenIdle(queryClient, PAYER_MUTATION_KEY);
    },
  });
}
