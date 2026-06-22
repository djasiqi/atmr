import dayjs from "dayjs";
import "dayjs/locale/fr";
import type { InstitutionRequestOffer } from "../api/institutionOffersApi";
import { resolveInstitutionOfferTerminalState } from "./institutionOfferResponse";
import type { InstitutionOfferPushPreview } from "../push/companyPush";

dayjs.locale("fr");

export type InstitutionTransportRequestDetail = NonNullable<
  InstitutionRequestOffer["transport_request"]
>;

export type InstitutionRoutePoint = {
  key: string;
  label: string;
  address: string;
  timeLabel?: string;
  details?: string;
};

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

export function formatWallClockTime(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

export function formatWallClockDateShort(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = dayjs(iso);
  return d.isValid() ? d.format("D MMM") : null;
}

export function formatInstantDateTimeCH(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return `${pad2(d.getDate())}.${pad2(d.getMonth() + 1)}.${d.getFullYear()} ${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

export function formatBirthDateCH(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = dayjs(iso);
  return d.isValid() ? d.format("DD.MM.YYYY") : null;
}

export function resolveInstitutionPatientName(
  req: InstitutionTransportRequestDetail | null | undefined,
  preview?: InstitutionOfferPushPreview
): string {
  const fromApi = req?.patient_name?.trim();
  if (fromApi) return fromApi;
  const patient = req?.patient;
  if (patient?.first_name || patient?.last_name) {
    return `${patient.first_name ?? ""} ${patient.last_name ?? ""}`.trim();
  }
  const fromPreview = preview?.patient_name?.trim();
  if (fromPreview) return fromPreview;
  return "—";
}

function formatSchedulePartLabel(label: string, time: string): string {
  return `${label} ${time}`;
}

/** Horaire principal affiché (aligné web InstitutionOfferDetailPanel). */
export function buildInstitutionScheduleLabel(
  req: InstitutionTransportRequestDetail | null | undefined,
  preview?: InstitutionOfferPushPreview
): string {
  if (preview?.scheduled_time_label?.trim()) {
    const date =
      formatWallClockDateShort(preview.mission_date) ??
      formatWallClockDateShort(req?.mission_date ?? req?.scheduled_time);
    return date
      ? `${date} · ${preview.scheduled_time_label.trim()}`
      : preview.scheduled_time_label.trim();
  }

  if (!req) return "Horaire à confirmer";

  const parts: string[] = [];
  const legs = [...(req.legs ?? [])].sort(
    (a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0)
  );

  if (req.pickup_time_confirmed && req.scheduled_time && req.scheduled_time_type !== "arrival") {
    const t = formatWallClockTime(req.scheduled_time);
    if (t) parts.push(formatSchedulePartLabel("Départ", t));
  }

  legs.forEach((leg, index) => {
    const isReturn = Boolean(req.return_to_institution) && index === legs.length - 1;
    if (leg.time_confirmed === false || !leg.scheduled_time) {
      if (isReturn) parts.push("Retour à définir");
      return;
    }
    const t = formatWallClockTime(leg.scheduled_time);
    if (!t) return;
    parts.push(formatSchedulePartLabel(isReturn ? "Retour" : "RDV", t));
  });

  if (
    parts.length === 0 &&
    req.scheduled_time_type === "arrival" &&
    req.scheduled_time
  ) {
    const t = formatWallClockTime(req.scheduled_time);
    if (t) parts.push(formatSchedulePartLabel("RDV", t));
  }

  if (parts.length === 0 && req.next_confirmed_time) {
    const t = formatWallClockTime(req.next_confirmed_time);
    const label = req.scheduled_time_type === "arrival" ? "RDV" : "Départ";
    if (t) parts.push(formatSchedulePartLabel(label, t));
  }

  const dateLabel = formatWallClockDateShort(req.mission_date ?? req.scheduled_time);
  const timeLabel = parts.length > 0 ? parts.join(" · ") : "Horaire à définir";
  return dateLabel ? `${dateLabel} · ${timeLabel}` : timeLabel;
}

function formatRouteStopTime(
  kind: "start" | "destination" | "return",
  req: InstitutionTransportRequestDetail,
  leg?: InstitutionTransportRequestDetail["legs"] extends (infer L)[] | undefined ? L : never
): string | undefined {
  if (kind === "start") {
    if (req.pickup_time_confirmed && req.scheduled_time && req.scheduled_time_type !== "arrival") {
      const t = formatWallClockTime(req.scheduled_time);
      return t ? `Départ ${t}` : "Départ · À définir";
    }
    return "À définir";
  }
  if (!leg) return undefined;
  if (leg.time_confirmed === false || !leg.scheduled_time) {
    return kind === "return" ? "Départ · À définir" : "RDV · À définir";
  }
  const t = formatWallClockTime(leg.scheduled_time);
  if (!t) return undefined;
  return kind === "return" ? `Départ ${t}` : `RDV ${t}`;
}

export function buildInstitutionRoutePoints(
  req: InstitutionTransportRequestDetail | null | undefined
): InstitutionRoutePoint[] {
  if (!req) return [];

  const legs = [...(req.legs ?? [])].sort(
    (a, b) => (a.sequence_index ?? 0) - (b.sequence_index ?? 0)
  );

  if (legs.length > 0) {
    const points: InstitutionRoutePoint[] = [
      {
        key: "start",
        label: "Départ",
        address: legs[0].pickup_location ?? req.pickup_location ?? "—",
        timeLabel: formatRouteStopTime("start", req),
      },
    ];
    legs.forEach((leg, index) => {
      const isReturn = Boolean(req.return_to_institution) && index === legs.length - 1;
      const details = [leg.dropoff_establishment, leg.dropoff_service, leg.dropoff_doctor]
        .filter((v): v is string => Boolean(v && String(v).trim()))
        .join(" · ");
      points.push({
        key: `leg-${index}`,
        label: isReturn ? "Retour" : `Destination ${index + 1}`,
        address: leg.dropoff_location ?? "—",
        timeLabel: formatRouteStopTime(isReturn ? "return" : "destination", req, leg),
        details: details || undefined,
      });
    });
    return points;
  }

  return [
    {
      key: "start",
      label: "Départ",
      address: req.pickup_location ?? "—",
      timeLabel: formatRouteStopTime("start", req),
    },
    {
      key: "dest",
      label: "Destination 1",
      address: req.dropoff_location ?? "—",
      timeLabel:
        req.scheduled_time_type === "arrival" && req.scheduled_time
          ? formatRouteStopTime("destination", req, {
              scheduled_time: req.scheduled_time,
              time_confirmed: true,
            })
          : undefined,
    },
  ];
}

function shortRouteStopLabel(address: string, details?: string): string {
  const fromDetails = details?.split(" · ")[0]?.trim();
  if (fromDetails) return fromDetails;
  if (!address.trim() || address === "—") return "—";
  return address.split(",")[0].trim();
}

/** Parcours compact pour listes (aligné web InstitutionOffersTable). */
export function buildInstitutionRouteSummaryShort(
  req: InstitutionTransportRequestDetail | null | undefined,
  options?: { includeStopTimes?: boolean }
): string {
  const includeStopTimes = options?.includeStopTimes ?? false;
  const points = buildInstitutionRoutePoints(req);
  if (points.length === 0) return "—";
  return points
    .map((point) => {
      const name =
        point.label === "Retour" && point.key !== "start"
          ? "Retour institution"
          : shortRouteStopLabel(point.address, point.details);
      if (!includeStopTimes) return name;
      const time = point.timeLabel?.replace(" · À définir", "").trim();
      if (time && !time.includes("À définir")) {
        return `${name} (${time})`;
      }
      return name;
    })
    .join(" → ");
}

function splitScheduleLabel(schedule: string): { dateLabel: string | null; detailLabel: string } {
  const trimmed = schedule.trim();
  if (!trimmed) return { dateLabel: null, detailLabel: "Horaire à confirmer" };
  const sep = trimmed.indexOf(" · ");
  if (sep <= 0) return { dateLabel: null, detailLabel: trimmed };
  return {
    dateLabel: trimmed.slice(0, sep).trim() || null,
    detailLabel: trimmed.slice(sep + 3).trim() || trimmed,
  };
}

function extractPrimaryClock(detail: string): string | null {
  const match = detail.match(/\b(\d{2}:\d{2})\b/);
  return match?.[1] ?? null;
}

function extractScheduleExtras(detail: string, primaryTime: string | null): string | null {
  const parts = detail
    .split(" · ")
    .map((part) => part.trim())
    .filter(Boolean);
  const extras = parts.filter((part) => {
    if (primaryTime && part.includes(primaryTime) && /^(Départ|RDV|Retour)\s/.test(part)) {
      return false;
    }
    return true;
  });
  return extras.length > 0 ? extras.join(" · ") : null;
}

export type InstitutionOfferListPreview = {
  title: string;
  institutionLabel: string | null;
  schedule: string;
  scheduleDate: string | null;
  scheduleDetail: string;
  primaryTime: string | null;
  scheduleExtras: string | null;
  route: string;
  tripBadge: string | null;
};

/** Titres et sous-titres pour la liste mobile des offres institution. */
export function buildInstitutionOfferListPreview(
  req: InstitutionTransportRequestDetail | null | undefined
): InstitutionOfferListPreview {
  const patientName = resolveInstitutionPatientName(req);
  const institutionName = req?.institution_name?.trim() ?? null;
  const title =
    patientName !== "—" ? patientName : institutionName ?? "Demande institution";
  const institutionLabel =
    patientName !== "—" && institutionName ? institutionName : null;
  const schedule = buildInstitutionScheduleLabel(req);
  const { dateLabel, detailLabel } = splitScheduleLabel(schedule);
  const primaryTime = extractPrimaryClock(detailLabel);
  const routePoints = buildInstitutionRoutePoints(req);
  return {
    title,
    institutionLabel,
    schedule,
    scheduleDate: dateLabel,
    scheduleDetail: detailLabel,
    primaryTime,
    scheduleExtras: extractScheduleExtras(detailLabel, primaryTime),
    route: buildInstitutionRouteSummaryShort(req),
    tripBadge: buildInstitutionTripBadge(req, routePoints),
  };
}

export function buildInstitutionTripBadge(
  req: InstitutionTransportRequestDetail | null | undefined,
  routePoints: InstitutionRoutePoint[]
): string | null {
  if (!req) return null;
  if (req.return_to_institution) {
    return `A/R institution — ${Math.max(routePoints.length - 1, 1)} trajet(s)`;
  }
  if (req.multi_stop || routePoints.length > 2) {
    return `${routePoints.length - 1} destination(s)`;
  }
  if (req.is_round_trip) return "Aller-retour";
  return "Aller simple";
}

export type InstitutionMobilityChip = {
  key: string;
  label: string;
  danger?: boolean;
};

export function buildInstitutionMobilityChips(
  req: InstitutionTransportRequestDetail | null | undefined
): InstitutionMobilityChip[] {
  if (!req) return [];
  const mob = req.mobility ?? {};
  const chips: InstitutionMobilityChip[] = [];
  if (req.requires_wheelchair || mob.wheelchair) {
    chips.push({ key: "wheelchair", label: "Fauteuil" });
  }
  if (mob.vehicle_wheelchair) {
    chips.push({ key: "vehicle_wheelchair", label: "Prendre chaise" });
  }
  if (req.requires_assistance || mob.needs_assistance) {
    chips.push({ key: "assistance", label: "Assistance" });
  }
  if (req.requires_stretcher || mob.stretcher) {
    chips.push({ key: "stretcher", label: "Brancard" });
  }
  if (req.requires_oxygen || mob.oxygen) {
    chips.push({ key: "oxygen", label: "O₂", danger: true });
  }
  return chips;
}

export function formatMissionTypeLabel(value: string | null | undefined): string | null {
  if (!value) return null;
  if (value === "patient_transport") return "Transport patient";
  if (value === "material_delivery") return "Livraison matériel";
  return value.replace(/_/g, " ");
}

export function formatPriceEstimateLabel(
  offer: InstitutionRequestOffer | null | undefined
): { label: string; value: string } | null {
  const est = offer?.price_estimate;
  if (!est || est.amount == null) return null;
  const amount = Number(est.amount);
  if (!Number.isFinite(amount) || amount <= 0) {
    return { label: "Tarif estimé", value: "À définir à l'acceptation" };
  }
  const intent = String(offer?.transport_request?.billing_intent ?? "patient").toLowerCase();
  let label = "Tarif estimé";
  if (est.source === "preferential" && intent === "institution") {
    label = "Tarif préférentiel";
  } else if (est.source === "company_profile" || est.source === "profile") {
    label = "Tarif estimé (profil tarifaire)";
  }
  return {
    label,
    value: `${amount.toFixed(2)} ${est.currency ?? "CHF"}`,
  };
}

export function buildOfferStatusLabel(offer: InstitutionRequestOffer | null | undefined): string {
  switch (resolveInstitutionOfferTerminalState(offer)) {
    case "active":
      return "En attente";
    case "expired":
      return "Expiré";
    case "accepted":
      return "Acceptée";
    case "rejected":
      return "Refusée";
    case "unavailable":
    default:
      return "Indisponible";
  }
}
