import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createCompanyRide,
  getCompanyBillingSettings,
  getCompanyClients,
  getCompanyClientDetail,
  searchCompanyAddresses,
  searchCompanyClients,
  simulateCompanyPricing,
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
export type RideClientOption = RideFormOption & {
  pickupAddressCandidate: RideAddressOption | null;
  phone: string | null;
  pickupAccessNotes: string;
  dropoffAccessNotes: string;
  notesMedical: string;
  establishment: string;
  hospitalService: string;
  doctorName: string;
  wheelchairClient: boolean;
  wheelchairProvide: boolean;
  preferentialRate: number | null;
};
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

function asNonEmptyString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function extractDeepAddressCandidate(input: unknown): string {
  const seen = new Set<unknown>();
  const queue: unknown[] = [input];
  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || typeof current !== "object" || seen.has(current)) continue;
    seen.add(current);
    const raw = current as Record<string, unknown>;
    const directCandidates = [
      raw.domicile_address_line1,
      raw.address_line1,
      raw.billing_address,
      raw.address,
      raw.home_address,
      raw.full_address,
      raw.display_name,
      raw.label,
      raw.description,
      raw.location_label,
    ];
    for (const candidate of directCandidates) {
      const text = asNonEmptyString(candidate);
      if (text.length >= 6) return text;
    }
    for (const value of Object.values(raw)) {
      if (value && typeof value === "object") queue.push(value);
    }
  }
  return "";
}

function parseClientAddressCandidate(raw: Record<string, unknown>, fallbackId: number): RideAddressOption | null {
  const domicile = raw.domicile as Record<string, unknown> | undefined;
  const user = raw.user as Record<string, unknown> | undefined;
  const rootDomicileLabel = [
    typeof raw.domicile_address_line1 === "string" ? raw.domicile_address_line1.trim() : "",
    [
      typeof raw.domicile_zip === "string" ? raw.domicile_zip.trim() : "",
      typeof raw.domicile_city === "string" ? raw.domicile_city.trim() : "",
    ]
      .filter(Boolean)
      .join(" "),
  ]
    .filter(Boolean)
    .join(", ");
  const rootAddressLabel = [
    typeof raw.address_line1 === "string" ? raw.address_line1.trim() : "",
    [
      typeof raw.zip === "string" ? raw.zip.trim() : "",
      typeof raw.city === "string" ? raw.city.trim() : "",
    ]
      .filter(Boolean)
      .join(" "),
  ]
    .filter(Boolean)
    .join(", ");
  const domicileStructuredLabel =
    typeof domicile?.address_line1 === "string" && domicile.address_line1.trim()
      ? [
          domicile.address_line1.trim(),
          [
            typeof domicile.zip === "string" ? domicile.zip.trim() : "",
            typeof domicile.city === "string" ? domicile.city.trim() : "",
          ]
            .filter(Boolean)
            .join(" "),
        ]
          .filter(Boolean)
          .join(", ")
      : "";
  const billingLabel =
    typeof raw.billing_address === "string" && raw.billing_address.trim() ? raw.billing_address.trim() : "";
  const userAddressLabel =
    typeof user?.address === "string" && user.address.trim() ? user.address.trim() : "";
  const nestedAddressCandidate =
    (raw.pickup_address as Record<string, unknown> | undefined) ??
    (raw.pickupAddress as Record<string, unknown> | undefined) ??
    (raw.default_pickup_address as Record<string, unknown> | undefined) ??
    (raw.defaultPickupAddress as Record<string, unknown> | undefined) ??
    (raw.home_address as Record<string, unknown> | undefined) ??
    (raw.homeAddress as Record<string, unknown> | undefined) ??
    (domicile as Record<string, unknown> | undefined) ??
    (user as Record<string, unknown> | undefined) ??
    (raw.address as Record<string, unknown> | undefined);
  const labelCandidate =
    (typeof raw.pickup_address_label === "string" && raw.pickup_address_label.trim()) ||
    (typeof raw.pickupAddressLabel === "string" && raw.pickupAddressLabel.trim()) ||
    (typeof raw.default_pickup_address === "string" && raw.default_pickup_address.trim()) ||
    (typeof raw.defaultPickupAddress === "string" && raw.defaultPickupAddress.trim()) ||
    (typeof raw.home_address === "string" && raw.home_address.trim()) ||
    (typeof raw.homeAddress === "string" && raw.homeAddress.trim()) ||
    rootDomicileLabel ||
    rootAddressLabel ||
    domicileStructuredLabel ||
    billingLabel ||
    userAddressLabel ||
    (typeof domicile?.address === "string" && domicile.address.trim()
      ? [
          domicile.address.trim(),
          typeof domicile.zip === "string" ? domicile.zip.trim() : "",
          typeof domicile.city === "string" ? domicile.city.trim() : "",
        ]
          .filter(Boolean)
          .join(", ")
      : "") ||
    (typeof raw.address === "string" && raw.address.trim()) ||
    (typeof raw.full_address === "string" && raw.full_address.trim()) ||
    (typeof raw.fullAddress === "string" && raw.fullAddress.trim()) ||
    (typeof nestedAddressCandidate?.label === "string" && nestedAddressCandidate.label.trim()) ||
    (typeof nestedAddressCandidate?.address === "string" && nestedAddressCandidate.address.trim()) ||
    (typeof nestedAddressCandidate?.description === "string" && nestedAddressCandidate.description.trim()) ||
    extractDeepAddressCandidate(raw) ||
    "";
  if (!labelCandidate) return null;
  const placeIdCandidate = raw.place_id ?? raw.placeId ?? nestedAddressCandidate?.place_id ?? nestedAddressCandidate?.placeId;
  const latitudeCandidate =
    raw.pickup_lat ??
    raw.pickupLat ??
    raw.lat ??
    raw.latitude ??
    raw.domicile_lat ??
    raw.domicileLat ??
    raw.billing_lat ??
    raw.billingLat ??
    domicile?.lat ??
    domicile?.latitude ??
    nestedAddressCandidate?.lat ??
    nestedAddressCandidate?.latitude;
  const longitudeCandidate =
    raw.pickup_lon ??
    raw.pickupLon ??
    raw.lon ??
    raw.lng ??
    raw.longitude ??
    raw.domicile_lon ??
    raw.domicileLon ??
    raw.billing_lon ??
    raw.billingLon ??
    domicile?.lon ??
    domicile?.lng ??
    domicile?.longitude ??
    nestedAddressCandidate?.lon ??
    nestedAddressCandidate?.lng ??
    nestedAddressCandidate?.longitude;
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
    id: parseId(raw.pickup_address_id ?? raw.pickupAddressId ?? nestedAddressCandidate?.id) ?? fallbackId,
    label: labelCandidate,
    placeId: typeof placeIdCandidate === "string" ? placeIdCandidate : null,
    latitude: Number.isFinite(latitude) ? latitude : null,
    longitude: Number.isFinite(longitude) ? longitude : null,
  };
}

function parseClientOptions(payload: unknown): RideClientOption[] {
  const rows = Array.isArray(payload)
    ? payload
    : payload && typeof payload === "object"
      ? [payload.items, payload.results, payload.data, payload.clients].find((entry) => Array.isArray(entry))
      : null;
  if (!Array.isArray(rows)) return [];
  return rows
    .map((row) => {
      if (!row || typeof row !== "object") return null;
      const raw = row as Record<string, unknown>;
      const id = parseId(raw.client_id ?? raw.id);
      if (id == null) return null;
      const label =
        (typeof raw.full_name === "string" && raw.full_name.trim()) ||
        (typeof raw.name === "string" && raw.name.trim()) ||
        (typeof raw.label === "string" && raw.label.trim()) ||
        `#${id}`;
      return {
        id,
        label,
        pickupAddressCandidate: parseClientAddressCandidate(raw, id),
        phone:
          typeof raw.phone === "string" && raw.phone.trim()
            ? raw.phone.trim()
            : typeof raw.contact_phone === "string" && raw.contact_phone.trim()
              ? raw.contact_phone.trim()
              : null,
        pickupAccessNotes:
          (typeof raw.pickup_access_notes === "string" && raw.pickup_access_notes.trim()) ||
          (typeof raw.access_notes === "string" && raw.access_notes.trim()) ||
          "",
        dropoffAccessNotes:
          (typeof raw.dropoff_access_notes === "string" && raw.dropoff_access_notes.trim()) || "",
        notesMedical:
          (typeof raw.notes_medical === "string" && raw.notes_medical.trim()) ||
          (typeof raw.medical_notes === "string" && raw.medical_notes.trim()) ||
          "",
        establishment:
          (typeof raw.medical_facility === "string" && raw.medical_facility.trim()) ||
          (typeof raw.establishment === "string" && raw.establishment.trim()) ||
          "",
        hospitalService:
          (typeof raw.hospital_service === "string" && raw.hospital_service.trim()) ||
          (typeof raw.service === "string" && raw.service.trim()) ||
          "",
        doctorName:
          (typeof raw.doctor_name === "string" && raw.doctor_name.trim()) ||
          (typeof raw.doctor === "string" && raw.doctor.trim()) ||
          "",
        wheelchairClient: Boolean(raw.wheelchair_client_has),
        wheelchairProvide: Boolean(raw.wheelchair_need),
        preferentialRate:
          typeof raw.preferential_rate === "number"
            ? raw.preferential_rate
            : typeof raw.preferential_rate === "string"
              ? Number.parseFloat(raw.preferential_rate)
              : null,
      };
    })
    .filter((value): value is RideClientOption => value !== null);
}

export type RideClientDetail = {
  phone: string | null;
  pickupAddressCandidate: RideAddressOption | null;
  pickupAccessNotes: string;
  dropoffAccessNotes: string;
  notesMedical: string;
  establishment: string;
  hospitalService: string;
  doctorName: string;
  wheelchairClient: boolean;
  wheelchairProvide: boolean;
  preferentialRate: number | null;
  hasActiveStay: boolean;
  clinicName: string;
  clinicAddress: RideAddressOption | null;
  clinicService: string;
  clinicRoom: string;
  clinicFloor: string;
  clinicBillingPartyId: number | null;
};

function parseClientDetail(payload: unknown): RideClientDetail | null {
  if (!payload || typeof payload !== "object") return null;
  const raw = payload as Record<string, unknown>;
  const detail =
    (raw.item as Record<string, unknown> | undefined) ??
    (raw.data as Record<string, unknown> | undefined) ??
    (raw.client as Record<string, unknown> | undefined) ??
    raw;
  const pickupAddressCandidate = parseClientAddressCandidate(detail, parseId(detail.id) ?? 0);
  const preferentialRateRaw =
    detail.preferential_rate ??
    (detail.stay as Record<string, unknown> | undefined)?.preferential_rate ??
    (detail.clinic as Record<string, unknown> | undefined)?.preferential_rate;
  const preferentialRate =
    typeof preferentialRateRaw === "number"
      ? preferentialRateRaw
      : typeof preferentialRateRaw === "string"
        ? Number.parseFloat(preferentialRateRaw)
        : null;
  const stay =
    (detail.active_stay as Record<string, unknown> | undefined) ??
    (detail.activeStay as Record<string, unknown> | undefined) ??
    (detail.stay as Record<string, unknown> | undefined) ??
    null;
  const clinic =
    (stay?.clinic as Record<string, unknown> | undefined) ??
    (detail.clinic as Record<string, unknown> | undefined) ??
    null;
  const clinicName =
    (typeof clinic?.name === "string" && clinic.name.trim()) ||
    (typeof detail.medical_facility === "string" && detail.medical_facility.trim()) ||
    "";
  const clinicAddressLine1 =
    (typeof clinic?.domicile_address_line1 === "string" && clinic.domicile_address_line1.trim()) ||
    (typeof clinic?.address === "string" && clinic.address.trim()) ||
    (typeof clinic?.domicile_address === "string" && clinic.domicile_address.trim()) ||
    "";
  const clinicZip = typeof clinic?.domicile_zip === "string" ? clinic.domicile_zip.trim() : "";
  const clinicCity = typeof clinic?.domicile_city === "string" ? clinic.domicile_city.trim() : "";
  const clinicAddressLabel = [clinicAddressLine1, [clinicZip, clinicCity].filter(Boolean).join(" ").trim()]
    .filter(Boolean)
    .join(", ");
  const clinicLatCandidate = clinic?.latitude ?? clinic?.lat;
  const clinicLonCandidate = clinic?.longitude ?? clinic?.lon ?? clinic?.lng;
  const clinicLat =
    typeof clinicLatCandidate === "number"
      ? clinicLatCandidate
      : typeof clinicLatCandidate === "string"
        ? Number.parseFloat(clinicLatCandidate)
        : NaN;
  const clinicLon =
    typeof clinicLonCandidate === "number"
      ? clinicLonCandidate
      : typeof clinicLonCandidate === "string"
        ? Number.parseFloat(clinicLonCandidate)
        : NaN;
  const clinicBillingPartyId = parseId(
    stay?.billing_party_id ??
      clinic?.billing_party_id ??
      (stay?.billing_party as Record<string, unknown> | undefined)?.id
  );
  const hasActiveStay =
    Boolean(stay) &&
    raw.active_stay !== null &&
    raw.activeStay !== null;
  return {
    phone:
      (typeof detail.phone === "string" && detail.phone.trim()) ||
      (typeof detail.contact_phone === "string" && detail.contact_phone.trim()) ||
      null,
    pickupAddressCandidate,
    pickupAccessNotes:
      (typeof detail.pickup_access_notes === "string" && detail.pickup_access_notes.trim()) ||
      (typeof detail.access_notes === "string" && detail.access_notes.trim()) ||
      "",
    dropoffAccessNotes:
      (typeof detail.dropoff_access_notes === "string" && detail.dropoff_access_notes.trim()) || "",
    notesMedical:
      (typeof detail.notes_medical === "string" && detail.notes_medical.trim()) ||
      (typeof detail.medical_notes === "string" && detail.medical_notes.trim()) ||
      "",
    establishment:
      (typeof detail.medical_facility === "string" && detail.medical_facility.trim()) ||
      (typeof detail.establishment === "string" && detail.establishment.trim()) ||
      "",
    hospitalService:
      (typeof detail.hospital_service === "string" && detail.hospital_service.trim()) ||
      (typeof detail.service === "string" && detail.service.trim()) ||
      "",
    doctorName:
      (typeof detail.doctor_name === "string" && detail.doctor_name.trim()) ||
      (typeof detail.doctor === "string" && detail.doctor.trim()) ||
      "",
    wheelchairClient: Boolean(detail.wheelchair_client_has),
    wheelchairProvide: Boolean(detail.wheelchair_need),
    preferentialRate: Number.isFinite(preferentialRate) ? preferentialRate : null,
    hasActiveStay,
    clinicName,
    clinicAddress:
      clinicAddressLabel || clinicName
        ? {
            id: parseId(clinic?.id) ?? -1,
            label: clinicAddressLabel || clinicName,
            placeId: null,
            latitude: Number.isFinite(clinicLat) ? clinicLat : null,
            longitude: Number.isFinite(clinicLon) ? clinicLon : null,
          }
        : null,
    clinicService:
      (typeof stay?.hospital_service === "string" && stay.hospital_service.trim()) ||
      (typeof stay?.service === "string" && stay.service.trim()) ||
      "",
    clinicRoom:
      (typeof stay?.room === "string" && stay.room.trim()) ||
      (typeof stay?.chamber === "string" && stay.chamber.trim()) ||
      "",
    clinicFloor:
      (typeof stay?.floor === "string" && stay.floor.trim()) ||
      (typeof stay?.level === "string" && stay.level.trim()) ||
      "",
    clinicBillingPartyId: clinicBillingPartyId ?? null,
  };
}

export function __parseClientDetailForTests(payload: unknown): RideClientDetail | null {
  return parseClientDetail(payload);
}

function parseAddressOptions(payload: unknown): RideAddressOption[] {
  const rows = Array.isArray(payload)
    ? payload
    : payload && typeof payload === "object"
      ? [
          payload.items,
          payload.results,
          payload.data,
          payload.clients,
          payload.addresses,
          payload.features,
          payload.predictions,
          payload.suggestions,
        ].find((entry) =>
          Array.isArray(entry)
        )
      : null;
  if (!Array.isArray(rows)) return [];
  const computeFallbackId = (raw: Record<string, unknown>, index: number): number => {
    const idTextCandidates = [
      raw.place_id,
      raw.placeId,
      raw.photon_id,
      raw.osm_id,
      raw.id,
      (raw.properties as Record<string, unknown> | undefined)?.osm_id,
      (raw.properties as Record<string, unknown> | undefined)?.id,
    ];
    const firstText = idTextCandidates.find((entry) => typeof entry === "string" && entry.trim()) as
      | string
      | undefined;
    if (firstText) {
      let hash = 0;
      for (let i = 0; i < firstText.length; i += 1) {
        hash = (hash * 31 + firstText.charCodeAt(i)) | 0;
      }
      return Math.abs(hash) || index + 1;
    }
    return index + 1;
  };
  return rows
    .map((row, index) => {
      if (!row || typeof row !== "object") return null;
      const raw = row as Record<string, unknown>;
      const properties = raw.properties as Record<string, unknown> | undefined;
      const id =
        parseId(raw.id ?? raw.place_id ?? raw.placeId ?? raw.photon_id ?? raw.osm_id ?? properties?.id) ??
        computeFallbackId(raw, index);
      const label =
        (typeof raw.label === "string" && raw.label.trim()) ||
        (typeof raw.description === "string" && raw.description.trim()) ||
        (typeof raw.address === "string" && raw.address.trim()) ||
        (typeof raw.display_name === "string" && raw.display_name.trim()) ||
        (typeof properties?.label === "string" && properties.label.trim()) ||
        (typeof properties?.name === "string" && properties.name.trim()) ||
        (typeof properties?.city === "string" && properties.city.trim()) ||
        `#${id}`;
      const geometry = raw.geometry as Record<string, unknown> | undefined;
      const geometryCoords = Array.isArray(geometry?.coordinates) ? geometry?.coordinates : null;
      const latitudeCandidate =
        raw.lat ??
        raw.latitude ??
        properties?.lat ??
        (raw.location as Record<string, unknown> | undefined)?.lat ??
        (geometryCoords && geometryCoords.length > 1 ? geometryCoords[1] : null);
      const longitudeCandidate =
        raw.lon ??
        raw.lng ??
        raw.longitude ??
        properties?.lon ??
        properties?.lng ??
        (raw.location as Record<string, unknown> | undefined)?.lng ??
        (raw.location as Record<string, unknown> | undefined)?.lon ??
        (geometryCoords && geometryCoords.length > 0 ? geometryCoords[0] : null);
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
              : typeof properties?.place_id === "string"
                ? properties.place_id
              : null,
        latitude: Number.isFinite(latitude) ? latitude : null,
        longitude: Number.isFinite(longitude) ? longitude : null,
      };
    })
    .filter((value): value is RideAddressOption => value !== null);
}

export function __parseAddressOptionsForTests(payload: unknown): RideAddressOption[] {
  return parseAddressOptions(payload);
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
      return parseClientOptions(payload);
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

export function useCompanyClientDetail(clientId: number | null) {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey:
      contextId && clientId
        ? contextScopedKey(
            contextId,
            [...companyQueryKeys.root, "ride-form", "client-detail", contextId, clientId] as unknown[]
          )
        : ["company", "ride-form", "client-detail", "disabled"],
    enabled: Boolean(contextId) && clientId != null,
    queryFn: async () => {
      const payload = await getCompanyClientDetail({
        contextId: contextId as string,
        clientId: clientId as number,
      });
      const parsed = parseClientDetail(payload);
      if (parsed?.pickupAddressCandidate) return parsed;
      const listPayload = await getCompanyClients({
        contextId: contextId as string,
        limit: 1000,
      });
      const rows = Array.isArray((listPayload as Record<string, unknown>)?.data)
        ? ((listPayload as Record<string, unknown>).data as unknown[])
        : Array.isArray((listPayload as Record<string, unknown>)?.items)
          ? ((listPayload as Record<string, unknown>).items as unknown[])
          : Array.isArray(listPayload)
            ? (listPayload as unknown[])
            : [];
      const fallbackRow = rows.find((row) => {
        if (!row || typeof row !== "object") return false;
        const candidateId = parseId((row as Record<string, unknown>).id ?? (row as Record<string, unknown>).client_id);
        return candidateId === clientId;
      });
      if (!fallbackRow) return parsed;
      const enriched = parseClientDetail(fallbackRow);
      if (!parsed) return enriched;
      return {
        ...parsed,
        pickupAddressCandidate: parsed.pickupAddressCandidate ?? enriched?.pickupAddressCandidate ?? null,
        phone: parsed.phone ?? enriched?.phone ?? null,
      };
    },
    staleTime: QUERY_STALE_TIME_MS.default,
  });
}

export type CompanyBillingPricingContext = {
  pricingProfileId: number | null;
  pricingProfileVersionId: number | null;
};

function parseCompanyBillingPricingContext(payload: unknown): CompanyBillingPricingContext {
  if (!payload || typeof payload !== "object") {
    return { pricingProfileId: null, pricingProfileVersionId: null };
  }
  const raw = payload as Record<string, unknown>;
  const profileId = parseId(raw.active_pricing_profile_id ?? raw.pricing_profile_id);
  const versionId = parseId(
    raw.active_pricing_profile_version_id ?? raw.pricing_profile_version_id
  );
  return {
    pricingProfileId: profileId,
    pricingProfileVersionId: versionId,
  };
}

export function useCompanyBillingPricingContext() {
  const contextId = useActiveCompanyContextId();
  return useQuery({
    queryKey: contextId
      ? contextScopedKey(
          contextId,
          [...companyQueryKeys.root, "ride-form", "billing-pricing-context", contextId] as unknown[]
        )
      : ["company", "ride-form", "billing-pricing-context", "disabled"],
    enabled: Boolean(contextId),
    queryFn: async () => {
      const payload = await getCompanyBillingSettings({ contextId: contextId as string });
      return parseCompanyBillingPricingContext(payload);
    },
    staleTime: QUERY_STALE_TIME_MS.companySlow,
  });
}

export function useCompanyPricingSimulation() {
  const contextId = useActiveCompanyContextId();
  return useMutation({
    mutationFn: async (payload: Record<string, unknown>) =>
      simulateCompanyPricing({ contextId: contextId as string, payload }),
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

/** Format attendu par le backend : `YYYY-MM-DDTHH:mm:ss` (sans fuseau). */
export function normalizeScheduledTimeIso(raw: string): string {
  const t = raw.trim();
  if (!t) return t;
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(t)) {
    return t.includes(".") ? t.split(".")[0] : t.replace(/Z$/i, "");
  }
  const short = /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})$/.exec(t);
  if (short) return `${short[1]}:00`;
  return t;
}

export function useDefaultScheduledAt() {
  return useMemo(() => "", []);
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
  const [internalNotes, setInternalNotes] = useState("");
  const [isRoundTrip, setIsRoundTrip] = useState(false);
  const [amountInput, setAmountInput] = useState("");
  const [notesMedical, setNotesMedical] = useState("");
  const [establishment, setEstablishment] = useState("");
  const [hospitalService, setHospitalService] = useState("");
  const [doctorName, setDoctorName] = useState("");
  const [pickupAccessNotes, setPickupAccessNotes] = useState("");
  const [dropoffAccessNotes, setDropoffAccessNotes] = useState("");
  const [wheelchairClient, setWheelchairClient] = useState(false);
  const [wheelchairProvide, setWheelchairProvide] = useState(false);
  const [isMaterialDelivery, setIsMaterialDelivery] = useState(false);
  const [deliveryDescription, setDeliveryDescription] = useState("");
  const [returnScheduledAt, setReturnScheduledAt] = useState("");

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

  const swapAddresses = () => {
    const tmpPickup = pickup;
    const tmpDropoff = dropoff;
    const tmpPickupAddress = pickupAddress;
    const tmpDropoffAddress = dropoffAddress;
    setPickupValue(tmpDropoff);
    setDropoffValue(tmpPickup);
    setPickupAddress(tmpDropoffAddress);
    setDropoffAddress(tmpPickupAddress);
  };

  const reset = () => {
    setClientId(null);
    setPickupValue("");
    setDropoffValue("");
    setPickupAddress(null);
    setDropoffAddress(null);
    setScheduledAt(defaultScheduledAt);
    setRecurrence("none");
    setInternalNotes("");
    setIsRoundTrip(false);
    setAmountInput("");
    setNotesMedical("");
    setEstablishment("");
    setHospitalService("");
    setDoctorName("");
    setPickupAccessNotes("");
    setDropoffAccessNotes("");
    setWheelchairClient(false);
    setWheelchairProvide(false);
    setIsMaterialDelivery(false);
    setDeliveryDescription("");
    setReturnScheduledAt("");
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
    swapAddresses,
    scheduledAt,
    setScheduledAt,
    recurrence,
    setRecurrence,
    internalNotes,
    setInternalNotes,
    isRoundTrip,
    setIsRoundTrip,
    returnScheduledAt,
    setReturnScheduledAt,
    amountInput,
    setAmountInput,
    notesMedical,
    setNotesMedical,
    establishment,
    setEstablishment,
    hospitalService,
    setHospitalService,
    doctorName,
    setDoctorName,
    pickupAccessNotes,
    setPickupAccessNotes,
    dropoffAccessNotes,
    setDropoffAccessNotes,
    wheelchairClient,
    setWheelchairClient,
    wheelchairProvide,
    setWheelchairProvide,
    isMaterialDelivery,
    setIsMaterialDelivery,
    deliveryDescription,
    setDeliveryDescription,
    reset,
  };
}
