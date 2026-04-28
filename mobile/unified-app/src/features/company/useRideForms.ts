import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createCompanyRide,
  searchCompanyAddresses,
  searchCompanyClients,
  updateCompanyRide,
} from "./api/companyApi";
import {
  createStandardCompanyClient,
  createStandardCompanyClientStay,
  fetchStandardCompanyBillingParties,
  linkStandardCompanyClientBillingParty,
} from "./api/companyStandardApi";
import { useActiveCompanyContextId } from "./hooks";
import { QUERY_STALE_TIME_MS } from "../../core/queryStaleTimes";
import { contextScopedKey } from "../../core/cache/contextCache";
import { companyQueryKeys } from "./companyQueryKeys";

export type RideFormOption = { id: number; label: string };
export type RideAddressOption = {
  id: number;
  label: string;
  placeId: string | null;
  latitude: number | null;
  longitude: number | null;
};

function parseId(input: unknown): number | null {
  if (typeof input === "number" && Number.isFinite(input)) return input;
  if (typeof input === "string" && input.trim()) {
    const parsed = Number.parseInt(input, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function parseOptions(payload: unknown, idKeys: string[], labelKeys: string[]): RideFormOption[] {
  if (!payload || typeof payload !== "object") return [];
  const source = payload as Record<string, unknown>;
  const candidates = [source.items, source.results, source.data, source.clients, source.addresses];
  const rows = candidates.find((entry) => Array.isArray(entry));
  if (!Array.isArray(rows)) return [];
  return rows
    .map((row) => {
      if (!row || typeof row !== "object") return null;
      const raw = row as Record<string, unknown>;
      const id = parseId(idKeys.map((key) => raw[key]).find((value) => value != null));
      if (id == null) return null;
      const label =
        labelKeys.map((key) => raw[key]).find((value) => typeof value === "string" && value.trim()) ?? `#${id}`;
      return { id, label: String(label) };
    })
    .filter((value): value is RideFormOption => value !== null);
}

function parseAddressOptions(payload: unknown): RideAddressOption[] {
  if (!payload || typeof payload !== "object") return [];
  const source = payload as Record<string, unknown>;
  const candidates = [source.items, source.results, source.data, source.clients, source.addresses];
  const rows = candidates.find((entry) => Array.isArray(entry));
  if (!Array.isArray(rows)) return [];
  return rows
    .map((row) => {
      if (!row || typeof row !== "object") return null;
      const raw = row as Record<string, unknown>;
      const id = parseId(raw.id ?? raw.place_id ?? raw.placeId);
      if (id == null) return null;
      const label =
        (typeof raw.label === "string" && raw.label.trim()) ||
        (typeof raw.description === "string" && raw.description.trim()) ||
        (typeof raw.address === "string" && raw.address.trim()) ||
        `#${id}`;
      const latitudeCandidate =
        raw.lat ?? raw.latitude ?? (raw.location as Record<string, unknown> | undefined)?.lat;
      const longitudeCandidate =
        raw.lon ??
        raw.lng ??
        raw.longitude ??
        (raw.location as Record<string, unknown> | undefined)?.lng ??
        (raw.location as Record<string, unknown> | undefined)?.lon;
      const latitude =
        typeof latitudeCandidate === "number"
          ? latitudeCandidate
          : typeof latitudeCandidate === "string"
            ? Number.parseFloat(latitudeCandidate)
            : NaN;
      const longitude =
        typeof longitudeCandidate === "number"
          ? longitudeCandidate
          : typeof longitudeCandidate === "string"
            ? Number.parseFloat(longitudeCandidate)
            : NaN;
      return {
        id,
        label: String(label),
        placeId:
          typeof raw.place_id === "string"
            ? raw.place_id
            : typeof raw.placeId === "string"
              ? raw.placeId
              : null,
        latitude: Number.isFinite(latitude) ? latitude : null,
        longitude: Number.isFinite(longitude) ? longitude : null,
      };
    })
    .filter((value): value is RideAddressOption => value !== null);
}

export function useCompanyClientSearch(query: string) {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(
          contextId,
          [...companyQueryKeys.root, "ride-form", "clients", contextId, query] as unknown[]
        )
      : ["company", "ride-form", "clients", "disabled"],
    enabled: Boolean(contextId) && query.trim().length > 1,
    queryFn: async () => {
      const payload = await searchCompanyClients({ contextId: contextId as string, q: query.trim() });
      return parseOptions(payload, ["client_id", "id"], ["full_name", "name", "label"]);
    },
    staleTime: QUERY_STALE_TIME_MS.default,
  });
}

export function useCompanyAddressSearch(query: string) {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(
          contextId,
          [...companyQueryKeys.root, "ride-form", "addresses", contextId, query] as unknown[]
        )
      : ["company", "ride-form", "addresses", "disabled"],
    enabled: Boolean(contextId) && query.trim().length > 2,
    queryFn: async () => {
      const payload = await searchCompanyAddresses({ contextId: contextId as string, q: query.trim() });
      return parseAddressOptions(payload);
    },
    staleTime: QUERY_STALE_TIME_MS.default,
  });
}

export function useRideCreate() {
  const contextId = useActiveCompanyContextId();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: Record<string, unknown>) =>
      createCompanyRide({ contextId: contextId as string, payload }),
    onSuccess: async () => {
      if (!contextId) return;
      await queryClient.invalidateQueries({
        queryKey: contextScopedKey(
          contextId,
          [...companyQueryKeys.root, "missions"] as unknown[]
        ),
        exact: false,
      });
    },
  });
}

export function useRideEdit() {
  const contextId = useActiveCompanyContextId();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: { missionId: number; payload: Record<string, unknown> }) =>
      updateCompanyRide({ contextId: contextId as string, missionId: params.missionId, payload: params.payload }),
    onSuccess: async (_, params) => {
      if (!contextId) return;
      await queryClient.invalidateQueries({
        queryKey: contextScopedKey(
          contextId,
          [...companyQueryKeys.rideDetails(contextId, params.missionId)] as unknown[]
        ),
      });
      await queryClient.invalidateQueries({
        queryKey: contextScopedKey(contextId, [...companyQueryKeys.root, "missions"] as unknown[]),
        exact: false,
      });
    },
  });
}

export function useClientCreate() {
  const contextId = useActiveCompanyContextId();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: Record<string, unknown>) => {
      const firstNameRaw =
        (typeof payload.first_name === "string" && payload.first_name.trim()) ||
        (typeof payload.name === "string" && payload.name.trim().split(" ")[0]) ||
        "Client";
      const lastNameRaw =
        (typeof payload.last_name === "string" && payload.last_name.trim()) ||
        (typeof payload.name === "string"
          ? payload.name.trim().split(" ").slice(1).join(" ").trim() || "Company"
          : "Company");
      const gender =
        payload.gender === "male" || payload.gender === "female" ? payload.gender : "female";
      const createdClient = await createStandardCompanyClient({
        contextId: contextId as string,
        payload: {
          first_name: firstNameRaw,
          last_name: lastNameRaw,
          gender,
          phone: typeof payload.phone === "string" ? payload.phone : null,
          email: typeof payload.email === "string" ? payload.email : null,
        },
      });

      const responsePayload = createdClient as Record<string, unknown>;
      const candidateClientId = Number(
        responsePayload.client_id ??
          responsePayload.id ??
          (responsePayload.data as Record<string, unknown> | undefined)?.id
      );
      const clientId = Number.isFinite(candidateClientId) ? candidateClientId : null;

      const stayStartDate =
        typeof payload.stay_start_date === "string" ? payload.stay_start_date.trim() : "";
      const stayCompanyId = Number(payload.stay_company_id);
      if (clientId && stayStartDate.length > 0 && Number.isFinite(stayCompanyId)) {
        await createStandardCompanyClientStay({
          contextId: contextId as string,
          clientId,
          payload: {
            company_id: stayCompanyId,
            start_date: stayStartDate,
            end_date: typeof payload.stay_end_date === "string" ? payload.stay_end_date : null,
            notes: typeof payload.stay_notes === "string" ? payload.stay_notes : null,
          },
        });
      }

      const billingPartyId = Number(payload.billing_party_id);
      if (clientId && Number.isFinite(billingPartyId)) {
        await linkStandardCompanyClientBillingParty({
          contextId: contextId as string,
          clientId,
          payload: {
            billing_party_id: billingPartyId,
            is_default: true,
          },
        });
      }

      return createdClient;
    },
    onSuccess: async () => {
      if (!contextId) return;
      await queryClient.invalidateQueries({
        queryKey: contextScopedKey(
          contextId,
          [...companyQueryKeys.root, "ride-form", "clients"] as unknown[]
        ),
        exact: false,
      });
    },
  });
}

export function useCompanyBillingPartiesQuery() {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(
          contextId,
          [...companyQueryKeys.root, "ride-form", "billing-parties", contextId] as unknown[]
        )
      : ["company", "ride-form", "billing-parties", "disabled"],
    enabled: Boolean(contextId),
    queryFn: async () => fetchStandardCompanyBillingParties({ contextId: contextId as string }),
    staleTime: QUERY_STALE_TIME_MS.companySlow,
  });
}

export function useDefaultScheduledAt() {
  return useMemo(() => new Date(Date.now() + 30 * 60 * 1000).toISOString().slice(0, 16), []);
}

export function useRideFormState() {
  const defaultScheduledAt = useDefaultScheduledAt();
  const [clientId, setClientId] = useState<number | null>(null);
  const [pickup, setPickupValue] = useState("");
  const [dropoff, setDropoffValue] = useState("");
  const [pickupAddress, setPickupAddress] = useState<RideAddressOption | null>(null);
  const [dropoffAddress, setDropoffAddress] = useState<RideAddressOption | null>(null);
  const [scheduledAt, setScheduledAt] = useState(defaultScheduledAt);
  const [recurrence, setRecurrence] = useState<"none" | "daily" | "weekly">("none");
  const [notes, setNotes] = useState("");

  const setPickup = (value: string) => {
    setPickupValue(value);
    setPickupAddress(null);
  };

  const setDropoff = (value: string) => {
    setDropoffValue(value);
    setDropoffAddress(null);
  };

  const selectPickupAddress = (address: RideAddressOption) => {
    setPickupValue(address.label);
    setPickupAddress(address);
  };

  const selectDropoffAddress = (address: RideAddressOption) => {
    setDropoffValue(address.label);
    setDropoffAddress(address);
  };

  const reset = () => {
    setClientId(null);
    setPickupValue("");
    setDropoffValue("");
    setPickupAddress(null);
    setDropoffAddress(null);
    setScheduledAt(defaultScheduledAt);
    setRecurrence("none");
    setNotes("");
  };

  return {
    clientId,
    setClientId,
    pickup,
    setPickup,
    dropoff,
    setDropoff,
    pickupAddress,
    setPickupAddress,
    dropoffAddress,
    setDropoffAddress,
    selectPickupAddress,
    selectDropoffAddress,
    scheduledAt,
    setScheduledAt,
    recurrence,
    setRecurrence,
    notes,
    setNotes,
    reset,
  };
}
