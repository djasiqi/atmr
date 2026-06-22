import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { contextScopedKey } from "../../../core/cache/contextCache";
import { contextRealtimeRouter } from "../../../core/realtime/contextRealtimeRouter";
import { companyQueryKeys } from "../companyQueryKeys";

type InstitutionOfferRealtimeEvent = {
  event_type?: string;
  is_relaunch?: boolean;
  offer_id?: number;
  transport_request_id?: number;
};

function isInstitutionOfferEvent(event: unknown): event is InstitutionOfferRealtimeEvent {
  if (!event || typeof event !== "object") return false;
  const type = (event as InstitutionOfferRealtimeEvent).event_type;
  return (
    type === "institution_offer_updated" ||
    type === "new_company_notification"
  );
}

export async function invalidateInstitutionOfferQueries(
  queryClient: ReturnType<typeof useQueryClient>,
  contextId: string,
  offerId?: number
): Promise<void> {
  await queryClient.invalidateQueries({
    queryKey: contextScopedKey(contextId, [
      ...companyQueryKeys.institutionOffers(contextId, "PENDING"),
    ] as unknown[]),
  });
  await queryClient.invalidateQueries({
    queryKey: contextScopedKey(contextId, [...companyQueryKeys.inbox(contextId)] as unknown[]),
  });
  if (offerId != null) {
    await queryClient.invalidateQueries({
      queryKey: contextScopedKey(contextId, [
        ...companyQueryKeys.institutionOfferDetail(contextId, offerId),
      ] as unknown[]),
    });
  }
}

/** Rafraîchit inbox + liste offres institution sur relance / nouvelle notification. */
export function useInstitutionOffersRealtimeListener(contextId: string | null): void {
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!contextId) return undefined;

    return contextRealtimeRouter.subscribe(contextId, (event) => {
      if (!isInstitutionOfferEvent(event)) return;

      if (event.event_type === "new_company_notification") {
        const meta = (event as { metadata?: { event_type?: string } }).metadata;
        const inboxType = meta?.event_type;
        if (inboxType && inboxType !== "new_request" && inboxType !== "request_updated") {
          return;
        }
      }

      const offerId =
        typeof event.offer_id === "number"
          ? event.offer_id
          : Number((event as { metadata?: { offer_id?: number } }).metadata?.offer_id);

      void invalidateInstitutionOfferQueries(
        queryClient,
        contextId,
        Number.isFinite(offerId) ? offerId : undefined
      );
    });
  }, [contextId, queryClient]);
}
