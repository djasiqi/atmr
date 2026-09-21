import type { DriverMission, DriverMissionStatus } from "../types";

const SWISS_TZ = "Europe/Zurich";

/** Flèche trajet — échappement Unicode pour éviter la corruption « â†' » en source. */
export const MISSION_ROUTE_ARROW = " \u2192 ";

/** Convention typographique FR : prénom(s) + NOM en majuscules. */
export function formatClientHeaderName(raw: string): string {
  const t = raw.trim().replace(/\s+/g, " ");
  if (!t) return t;
  if (/^mission\s*#\s*\d+$/i.test(t)) return t;
  const parts = t.split(" ");
  if (parts.length === 1) return parts[0]!;
  const family = parts[parts.length - 1]!.toUpperCase();
  return `${parts.slice(0, -1).join(" ")} ${family}`;
}

export function getMissionClientDisplayName(mission: DriverMission): string {
  const direct = typeof mission.client_name === "string" ? mission.client_name.trim() : "";
  if (direct.length > 0) return formatClientHeaderName(direct);
  const nest = mission.client as { full_name?: unknown } | null | undefined;
  const full =
    nest?.full_name != null && String(nest.full_name).trim().length > 0
      ? String(nest.full_name).trim()
      : "";
  if (full.length > 0) return formatClientHeaderName(full);
  return `Mission #${mission.id}`;
}

function readClientGender(mission: DriverMission): string | null {
  const nest = mission.client as { gender?: unknown } | null | undefined;
  const raw = nest?.gender ?? (mission as Record<string, unknown>).client_gender;
  if (raw == null) return null;
  const g = String(raw).trim().toUpperCase();
  return g.length > 0 ? g : null;
}

/** Civilité pour messages équipe — `null` si genre inconnu. */
export function getClientCivilityTitle(mission: DriverMission): "Madame" | "Monsieur" | null {
  const g = readClientGender(mission);
  if (!g || g === "AUTRE" || g === "OTHER") return null;
  if (g === "HOMME" || g === "MALE" || g === "M") return "Monsieur";
  if (g === "FEMME" || g === "FEMALE" || g === "F") return "Madame";
  return null;
}

/** Nom de famille en majuscules (ex. BRONNIMANN). */
export function getClientFamilyName(mission: DriverMission): string {
  const nest = mission.client as { last_name?: unknown } | null | undefined;
  const last =
    nest?.last_name != null && String(nest.last_name).trim().length > 0
      ? String(nest.last_name).trim()
      : "";
  if (last.length > 0) return last.toUpperCase();
  const display = getMissionClientDisplayName(mission);
  const parts = display.trim().split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return parts[parts.length - 1]!.toUpperCase();
  if (parts.length === 1) return parts[0]!.toUpperCase();
  return "CLIENT";
}

/** Formule polie : « Madame BRONNIMANN » ou « BRONNIMANN » si genre inconnu. */
export function getClientFormalAddress(mission: DriverMission): string {
  const family = getClientFamilyName(mission);
  const civility = getClientCivilityTitle(mission);
  return civility ? `${civility} ${family}` : family;
}

export function getBadgeStatusLabel(statusKey: DriverMissionStatus): string {
  switch (statusKey) {
    case "ASSIGNED":
      return "Assignée";
    case "EN_ROUTE":
      return "En route";
    case "ARRIVED":
      return "Arrivé";
    case "IN_PROGRESS":
      return "En cours";
    case "COMPLETED":
      return "Terminée";
    case "CANCELLED":
      return "Annulée";
    case "REASSIGNED":
      return "Réassignée";
    case "NO_SHOW":
      return "Absent";
    case "FAILED":
      return "Échec";
    default:
      return "À venir";
  }
}

/** Date+heure planifiée — ex. `mar. 19.05.2026 • 10:50`. */
export function getScheduledWhenDisplay(mission: DriverMission): string | null {
  const raw =
    typeof mission.scheduled_time === "string" && mission.scheduled_time.length > 0
      ? mission.scheduled_time
      : typeof mission.scheduled_at === "string"
        ? (mission.scheduled_at as string)
        : null;
  if (!raw) return null;
  const d = new Date(raw);
  if (!Number.isFinite(d.getTime())) return null;
  const datePart = d.toLocaleDateString("fr-CH", {
    timeZone: SWISS_TZ,
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
  const timePart = d.toLocaleTimeString("fr-CH", {
    timeZone: SWISS_TZ,
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${datePart} • ${timePart}`;
}

/**
 * Date de naissance client → `dd/MM/yyyy`.
 * Accepte `mission.client.birth_date` ou champs plats API.
 */
export function getClientBirthDateDisplay(mission: DriverMission): string | null {
  const rawMission = mission as Record<string, unknown>;
  const nest = mission.client as { birth_date?: unknown } | null | undefined;
  const candidates = [nest?.birth_date, rawMission.client_birth_date, rawMission.birth_date];
  for (const raw of candidates) {
    if (raw == null) continue;
    const s = String(raw).trim();
    if (!s) continue;
    const d = new Date(s);
    if (!Number.isFinite(d.getTime())) continue;
    return d.toLocaleDateString("fr-FR", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
    });
  }
  return null;
}

export function isMaterialDeliveryMission(mission: DriverMission | null | undefined): boolean {
  if (!mission) return false;
  const raw = mission as Record<string, unknown>;
  return String(raw.mission_type || "").trim().toLowerCase() === "material_delivery";
}

export const MISSION_DELIVERY_BADGE = "LIVRAISON";
export const MISSING_DELIVERY_DESCRIPTION = "Description de livraison non renseignée";
export const DELIVERY_BENEFICIARY_LABEL = "Bénéficiaire";
export const DELIVERY_CARGO_LABEL = "À transporter";

export function getDeliveryDescription(mission: DriverMission | null | undefined): string | null {
  if (!mission) return null;
  const raw = mission as Record<string, unknown>;
  const desc = typeof raw.delivery_description === "string" ? raw.delivery_description.trim() : "";
  return desc.length > 0 ? desc : null;
}

export function formatDeliveryDescriptionDisplay(mission: DriverMission | null | undefined): string {
  return getDeliveryDescription(mission) || MISSING_DELIVERY_DESCRIPTION;
}

export function formatMissionTypeLabel(value: string | null | undefined): string {
  const key = String(value || "").trim().toLowerCase();
  if (key === "material_delivery") return "Livraison";
  if (key === "patient_transport" || !key) return "Transport patient";
  return key.replace(/_/g, " ");
}
