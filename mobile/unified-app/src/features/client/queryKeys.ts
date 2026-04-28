import { QueryClient } from "@tanstack/react-query";

export const clientQueryKeys = {
  dashboard: (contextId: string) => ["client-dashboard", contextId] as const,
  profile: (contextId: string) => ["client-profile", contextId] as const,
  bookings: (contextId: string, filters: string = "default") =>
    ["client-bookings", contextId, filters] as const,
  bookingDetail: (contextId: string, bookingId: number) =>
    ["booking-detail", contextId, bookingId] as const,
  bookingDraft: (contextId: string) => ["booking-draft", contextId] as const,
  bookingPricingPreview: (contextId: string, fingerprint: string) =>
    ["booking-pricing-preview", contextId, fingerprint] as const,
};

export function invalidateClientQueries(
  queryClient: QueryClient,
  contextId: string,
  bookingId?: number
) {
  queryClient.invalidateQueries({ queryKey: clientQueryKeys.dashboard(contextId) });
  queryClient.invalidateQueries({ queryKey: clientQueryKeys.profile(contextId) });
  queryClient.invalidateQueries({ queryKey: clientQueryKeys.bookings(contextId) });
  if (bookingId) {
    queryClient.invalidateQueries({ queryKey: clientQueryKeys.bookingDetail(contextId, bookingId) });
  }
}
