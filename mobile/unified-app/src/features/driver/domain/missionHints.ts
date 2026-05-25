/**
 * Hints contextuels mission — port direct de
 * `mobile/operations-app/src/domain/missionHints.ts`.
 *
 * - Aller (domicile → hôpital) : `PickupHints` = domicile, `DropoffHints` = hôpital
 * - Retour (hôpital → domicile) : `PickupHints` = hôpital, `DropoffHints` = domicile
 * - Unknown : pas d'inversion, infos génériques
 *
 * Le `MissionCard` choisit `getPickupHints` (statut ASSIGNED / EN_ROUTE / ARRIVED)
 * vs `getDropoffHints` (IN_PROGRESS) pour adapter les infos affichées au moment
 * critique : préparation prise en charge vs arrivée à destination.
 */

import type { Ionicons } from "@expo/vector-icons";

export type IoniconsName = keyof typeof Ionicons.glyphMap;

export type TripDirection = "outbound" | "return" | "unknown";

export interface HintItem {
  icon: IoniconsName;
  label: string;
  value: string;
  priority: number;
}

/** Champs utilisés par les hints — sous-ensemble compatible `DriverMission`. */
export interface MissionHintLike {
  is_return?: boolean | number | null;
  pickup_location?: string | null;
  dropoff_location?: string | null;
  medical_facility?: string | null;
  doctor_name?: string | null;
  hospital_service?: string | null;
  notes_medical?: string | null;
  notes?: string | null;
  pickup_access_notes?: string | null;
  dropoff_access_notes?: string | null;
  pickup_floor?: string | null;
  pickup_door_code?: string | null;
  dropoff_floor?: string | null;
  dropoff_door_code?: string | null;
  wheelchair?: boolean | string | number | null;
  wheelchair_client_has?: boolean | string | number | null;
  wheelchair_need?: boolean | string | number | null;
  client?: {
    door_code?: string | null;
    floor?: string | null;
    access_notes?: string | null;
    contact_phone?: string | null;
  } | null;
}

/**
 * Valeurs « placeholder » côté backend ou saisies par l'utilisateur que l'on
 * traite comme vides (le hint correspondant ne s'affiche pas). Couvre les
 * variantes FR/accentuées + ponctuation classique + « Aucune note » et apparentés.
 */
const EMPTY_VALUES = new Set([
  "",
  "non spécifié",
  "non specifié",
  "non spécifiée",
  "non specifiee",
  "non renseigné",
  "non renseigne",
  "non renseignée",
  "non renseignee",
  "—",
  "-",
  "—",
  "n/a",
  "na",
  "aucune note",
  "aucunes notes",
  "aucune",
  "aucun",
  "pas de note",
  "pas de notes",
  "sans note",
  "sans notes",
  "rien",
  "ras",
  "r.a.s.",
  "r.a.s",
  "null",
  "undefined",
]);
const MAX_HINT_ITEMS = 5;

function isMeaningful(value: string | undefined | null): boolean {
  const v = (value ?? "").trim().toLowerCase();
  if (v.length === 0) return false;
  if (EMPTY_VALUES.has(v)) return false;
  return true;
}

function pushHint(
  out: HintItem[],
  icon: IoniconsName,
  label: string,
  value: string | undefined | null,
  priority: number
): void {
  if (!isMeaningful(value)) return;
  out.push({ icon, label, value: (value ?? "").trim(), priority });
}

function firstMeaningful(
  ...values: Array<string | undefined | null>
): string | undefined | null {
  return values.find((value) => isMeaningful(value)) ?? null;
}

/** Détermine aller vs retour. Non-bloquant : si indécidable => unknown. */
export function inferTripDirection(mission: MissionHintLike | null | undefined): TripDirection {
  if (!mission) return "unknown";
  const ir = mission.is_return;
  if (ir === true || (typeof ir === "number" && ir === 1)) return "return";
  if (ir === false || (typeof ir === "number" && ir === 0)) return "outbound";
  const pickup = (mission.pickup_location ?? "").trim().toLowerCase();
  const dropoff = (mission.dropoff_location ?? "").trim().toLowerCase();
  const hasMedicalAtDropoff =
    isMeaningful(mission.medical_facility) || isMeaningful(mission.hospital_service);
  if (hasMedicalAtDropoff && pickup.length > 0 && dropoff.length > 0) {
    const looksLikeHospital = (s: string) =>
      /hôpital|hopital|hospital|clinique|centre médical|chuv|hug|réseau/i.test(s) ||
      /\b(bât|bat|bloc|entrée|service)\b/i.test(s);
    if (looksLikeHospital(dropoff) && !looksLikeHospital(pickup)) return "outbound";
    if (looksLikeHospital(pickup) && !looksLikeHospital(dropoff)) return "return";
  }
  return "unknown";
}

// Ordre d'affichage demandé par les opérations : sur un point hôpital,
// les infos métier (Établissement → Service → Médecin) doivent apparaître
// AVANT la mobilité ; sur un point domicile, les infos d'accès (Code porte
// → Étage → Notes → Contact) AVANT la mobilité. Les notes/instructions
// arrivent en dernier puisqu'elles sont parfois longues.
const HOSPITAL_ESTABLISHMENT_PRIORITY = 10;
const HOSPITAL_SERVICE_PRIORITY = 11;
const HOSPITAL_DOCTOR_PRIORITY = 12;
const HOME_DOOR_CODE_PRIORITY = 10;
const HOME_FLOOR_PRIORITY = 11;
const HOME_ACCESS_NOTES_PRIORITY = 12;
const HOME_CONTACT_PRIORITY = 13;
const MOBILITY_PRIORITY_FIRST = 20;
const MOBILITY_PRIORITY_SECOND = 20.5;
const NOTES_MEDICAL_PRIORITY = 30;
const ACCESS_NOTES_PRIORITY = 31;

/** Normalise (`true` / `"true"` / `1`) en booléen strict. */
function toBool(v: unknown): boolean {
  if (v === true || v === 1) return true;
  if (v === "true" || v === "1") return true;
  return false;
}

function pushMobilityHints(out: HintItem[], mission: MissionHintLike): void {
  if (toBool(mission.wheelchair_client_has) || toBool(mission.wheelchair))
    out.push({
      icon: "accessibility-outline",
      label: "Mobilité",
      value: "Client en chaise roulante",
      priority: MOBILITY_PRIORITY_FIRST,
    });
  if (toBool(mission.wheelchair_need))
    out.push({
      icon: "medkit-outline",
      label: "Mobilité",
      value: "Prendre une chaise roulante",
      priority: MOBILITY_PRIORITY_SECOND,
    });
}

/** Infos pour accéder au point de départ (où récupérer le client). */
export function getPickupHints(mission: MissionHintLike | null | undefined): HintItem[] {
  if (!mission) return [];
  const dir = inferTripDirection(mission);
  const out: HintItem[] = [];
  const c = mission.client;

  if (dir === "outbound") {
    pushHint(
      out,
      "key-outline",
      "Code porte",
      firstMeaningful(mission.pickup_door_code, c?.door_code),
      HOME_DOOR_CODE_PRIORITY
    );
    pushHint(
      out,
      "business-outline",
      "Étage",
      firstMeaningful(mission.pickup_floor, c?.floor),
      HOME_FLOOR_PRIORITY
    );
    pushHint(
      out,
      "document-text-outline",
      "Notes d'accès",
      firstMeaningful(mission.pickup_access_notes, c?.access_notes),
      HOME_ACCESS_NOTES_PRIORITY
    );
    pushHint(out, "call-outline", "Contact", c?.contact_phone, HOME_CONTACT_PRIORITY);
    pushMobilityHints(out, mission);
  } else if (dir === "return") {
    pushHint(out, "location-outline", "Établissement", mission.medical_facility, HOSPITAL_ESTABLISHMENT_PRIORITY);
    pushHint(out, "business-outline", "Service / Bâtiment", mission.hospital_service, HOSPITAL_SERVICE_PRIORITY);
    pushHint(out, "person-outline", "Médecin", mission.doctor_name, HOSPITAL_DOCTOR_PRIORITY);
    pushHint(out, "business-outline", "Étage / Secteur", mission.pickup_floor, HOME_FLOOR_PRIORITY);
    pushMobilityHints(out, mission);
    pushHint(
      out,
      "document-text-outline",
      "Notes sortie",
      firstMeaningful(mission.pickup_access_notes, mission.notes_medical),
      NOTES_MEDICAL_PRIORITY
    );
  } else {
    pushHint(out, "key-outline", "Code porte", mission.pickup_door_code, HOME_DOOR_CODE_PRIORITY);
    pushHint(out, "business-outline", "Étage", mission.pickup_floor, HOME_FLOOR_PRIORITY);
    pushHint(out, "document-text-outline", "Instructions accès", mission.pickup_access_notes, ACCESS_NOTES_PRIORITY);
    pushMobilityHints(out, mission);
  }

  out.sort((a, b) => a.priority - b.priority);
  return out.slice(0, MAX_HINT_ITEMS);
}

/** Infos pour accéder au point d'arrivée (destination). */
export function getDropoffHints(mission: MissionHintLike | null | undefined): HintItem[] {
  if (!mission) return [];
  const dir = inferTripDirection(mission);
  const out: HintItem[] = [];
  const c = mission.client;

  if (dir === "return") {
    pushHint(
      out,
      "key-outline",
      "Code porte",
      firstMeaningful(mission.dropoff_door_code, c?.door_code),
      HOME_DOOR_CODE_PRIORITY
    );
    pushHint(
      out,
      "business-outline",
      "Étage",
      firstMeaningful(mission.dropoff_floor, c?.floor),
      HOME_FLOOR_PRIORITY
    );
    pushHint(
      out,
      "document-text-outline",
      "Notes d'accès",
      firstMeaningful(mission.dropoff_access_notes, c?.access_notes),
      HOME_ACCESS_NOTES_PRIORITY
    );
    pushHint(out, "call-outline", "Contact", c?.contact_phone, HOME_CONTACT_PRIORITY);
    pushMobilityHints(out, mission);
  } else {
    pushHint(out, "location-outline", "Établissement", mission.medical_facility, HOSPITAL_ESTABLISHMENT_PRIORITY);
    pushHint(out, "business-outline", "Service / Bâtiment", mission.hospital_service, HOSPITAL_SERVICE_PRIORITY);
    pushHint(out, "person-outline", "Médecin", mission.doctor_name, HOSPITAL_DOCTOR_PRIORITY);
    pushHint(out, "business-outline", "Étage / Secteur", mission.dropoff_floor, HOME_FLOOR_PRIORITY);
    pushMobilityHints(out, mission);
    pushHint(out, "document-text-outline", "Notes médicales", mission.notes_medical, NOTES_MEDICAL_PRIORITY);
    pushHint(out, "document-text-outline", "Instructions accès", mission.dropoff_access_notes, ACCESS_NOTES_PRIORITY);
  }

  out.sort((a, b) => a.priority - b.priority);
  return out.slice(0, MAX_HINT_ITEMS);
}
