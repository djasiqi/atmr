import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { useSession } from "../../core/sessionProvider";
import { getBookingDetail, getClientBookings, getClientMe } from "./api";
import { clientQueryKeys } from "./queryKeys";

export function useActiveClientContextId(): string | null {
  const { activeContext } = useSession();
  if (!activeContext || activeContext.context_type !== "client") return null;
  return activeContext.context_id;
}

export function useClientProfileQuery() {
  const contextId = useActiveClientContextId();
  return useQuery({
    queryKey: contextId ? clientQueryKeys.profile(contextId) : ["client-profile", "disabled"],
    queryFn: getClientMe,
    enabled: Boolean(contextId),
  });
}

export function useClientBookingsQuery(options?: { limit?: number; filterKey?: string }) {
  const contextId = useActiveClientContextId();
  const filterKey = options?.filterKey ?? "default";
  return useQuery({
    queryKey: contextId
      ? clientQueryKeys.bookings(contextId, filterKey)
      : ["client-bookings", "disabled", filterKey],
    queryFn: () => getClientBookings({ limit: options?.limit }),
    enabled: Boolean(contextId),
  });
}

export function useBookingDetailQuery(bookingId: number | null) {
  const contextId = useActiveClientContextId();
  return useQuery({
    queryKey:
      contextId && bookingId
        ? clientQueryKeys.bookingDetail(contextId, bookingId)
        : ["booking-detail", "disabled", bookingId ?? "none"],
    queryFn: () => getBookingDetail(bookingId as number),
    enabled: Boolean(contextId && bookingId),
  });
}

export function usePrefetchClientDashboard() {
  const contextId = useActiveClientContextId();
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!contextId) return;
    void queryClient.prefetchQuery({
      queryKey: clientQueryKeys.profile(contextId),
      queryFn: getClientMe,
    });
    void queryClient.prefetchQuery({
      queryKey: clientQueryKeys.bookings(contextId, "dashboard-limit-1"),
      queryFn: () => getClientBookings({ limit: 1 }),
    });
  }, [contextId, queryClient]);
}
