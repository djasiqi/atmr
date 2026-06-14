import type { CompanyDispatchMission } from "../api/contracts";

export type BookingSourceView = {
  type?: string | null;
  id?: number | string | null;
  code?: string | null;
  name?: string | null;
};

export type BookingIdentityView = {
  passengerLabel: string;
  source: BookingSourceView;
  requester: { id: number | string | null; name: string } | null;
  ownership: { owner_company_id?: number | null; owner_company_name?: string | null } | null;
  execution: { executing_company_id?: number | null; executing_company_name?: string | null } | null;
  upstream: BookingSourceView | null;
};

const DEFAULT_PASSENGER = "Non spécifié";

type IdentityPayload = {
  passenger?: { name?: string | null };
  source?: BookingSourceView | null;
  requester?: { id?: number | string | null; name?: string | null } | null;
  ownership?: BookingIdentityView["ownership"];
  execution?: BookingIdentityView["execution"];
  upstream?: BookingSourceView | null;
};

type MissionLike = CompanyDispatchMission & {
  identity?: IdentityPayload | null;
  search_index?: string[] | null;
  client?: { name?: string | null; institution_name?: string | null } | null;
};

function legacySource(mission: MissionLike): BookingSourceView {
  const institutionName = mission.client?.institution_name?.trim();
  if (institutionName) {
    return { type: "institution", id: null, code: null, name: institutionName };
  }
  if (mission.partner_company_name?.trim()) {
    return {
      type: "partner_company",
      id: null,
      code: null,
      name: mission.partner_company_name.trim(),
    };
  }
  return { type: "legacy", id: null, code: null, name: null };
}

function normalizeSource(
  source: IdentityPayload["source"] | null | undefined,
  fallbackType = "legacy"
): BookingSourceView | null {
  if (!source) return null;
  return {
    type: source.type || fallbackType,
    id: source.id ?? null,
    code: source.code ?? null,
    name: source.name ?? null,
  };
}

export function buildIdentityFromMission(mission: MissionLike | null | undefined): BookingIdentityView {
  if (!mission) {
    return {
      passengerLabel: DEFAULT_PASSENGER,
      source: { type: "legacy", id: null, code: null, name: null },
      requester: null,
      ownership: null,
      execution: null,
      upstream: null,
    };
  }

  const identity = mission.identity;
  if (identity?.passenger?.name || identity?.source) {
    return {
      passengerLabel: identity.passenger?.name?.trim() || mission.client_name?.trim() || DEFAULT_PASSENGER,
      source: normalizeSource(identity.source) || legacySource(mission),
      requester: identity.requester?.name
        ? { id: identity.requester.id ?? null, name: identity.requester.name }
        : null,
      ownership: identity.ownership ?? null,
      execution: identity.execution ?? null,
      upstream: normalizeSource(identity.upstream),
    };
  }

  const passenger = mission.client_name?.trim() || mission.client?.name?.trim() || DEFAULT_PASSENGER;
  return {
    passengerLabel: passenger,
    source: legacySource(mission),
    requester: null,
    ownership: null,
    execution: null,
    upstream: null,
  };
}

export function matchesMissionSearchIndex(mission: MissionLike, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const index = mission.search_index;
  if (Array.isArray(index) && index.length > 0) {
    return index.some((token) => String(token).toLowerCase().includes(q));
  }
  const identity = buildIdentityFromMission(mission);
  const haystack = [
    identity.passengerLabel,
    identity.source.name,
    identity.upstream?.name,
    identity.requester?.name,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(q);
}
