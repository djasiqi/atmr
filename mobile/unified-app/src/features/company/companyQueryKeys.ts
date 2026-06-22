function parseCompanyIdFromContextId(contextId: string): string | null {
  if (!contextId.startsWith("company:")) return null;
  const candidate = contextId.slice("company:".length).trim();
  return candidate.length > 0 ? candidate : null;
}

export function companyContextScope(contextId: string) {
  return {
    context_type: "company" as const,
    context_id: contextId,
    company_id: parseCompanyIdFromContextId(contextId),
  };
}

export const companyQueryKeys = {
  root: ["company", "dispatch"] as const,
  missions: (contextId: string, date: string, search: string, status: string) =>
    [
      ...companyQueryKeys.root,
      "missions",
      companyContextScope(contextId),
      date,
      search,
      status,
    ] as const,
  /** Aligné vue web : GET `/company_dispatch/delays`. */
  dispatchDelays: (contextId: string, date: string) =>
    [...companyQueryKeys.root, "dispatch-delays", companyContextScope(contextId), date] as const,
  dashboard: (contextId: string) =>
    [...companyQueryKeys.root, "dashboard", companyContextScope(contextId)] as const,
  optimizer: (contextId: string) =>
    [...companyQueryKeys.root, "optimizer", companyContextScope(contextId)] as const,
  driversLocations: (contextId: string) =>
    [
      ...companyQueryKeys.root,
      "drivers",
      "locations",
      companyContextScope(contextId),
    ] as const,
  rideDetails: (contextId: string, missionId: number) =>
    [
      ...companyQueryKeys.root,
      "ride-details",
      companyContextScope(contextId),
      missionId,
    ] as const,
  institutionOffers: (contextId: string, status?: string) =>
    [
      ...companyQueryKeys.root,
      "institution-offers",
      companyContextScope(contextId),
      status ?? "all",
    ] as const,
  institutionOfferDetail: (contextId: string, offerId: number) =>
    [
      ...companyQueryKeys.root,
      "institution-offer",
      companyContextScope(contextId),
      offerId,
    ] as const,
  inbox: (contextId: string) =>
    [...companyQueryKeys.root, "inbox", companyContextScope(contextId)] as const,
} as const;

export type CompanyQueryKey =
  | ReturnType<typeof companyQueryKeys.missions>
  | ReturnType<typeof companyQueryKeys.dispatchDelays>
  | ReturnType<typeof companyQueryKeys.dashboard>
  | ReturnType<typeof companyQueryKeys.optimizer>
  | ReturnType<typeof companyQueryKeys.driversLocations>
  | ReturnType<typeof companyQueryKeys.rideDetails>
  | ReturnType<typeof companyQueryKeys.inbox>;
