/**
 * Module de mapping des infos mission pour chauffeurs : tri pickup vs dropoff + aller/retour.
 * - Aller (domicile -> hôpital) : PickupHints = domicile, DropoffHints = hôpital
 * - Retour (hôpital -> domicile) : PickupHints = hôpital, DropoffHints = domicile
 * - Unknown : pas d'inversion, affichage générique (pickup = infos premier point, dropoff = infos second point)
 */

export type TripDirection = "outbound" | "return" | "unknown";

export interface HintItem {
  icon: string;
  label: string;
  value: string;
  priority: number;
}

/** Mission-like : champs utilisés par les hints (compatible Booking sans importer api). */
export interface MissionLike {
  is_return?: boolean;
  pickup_location?: string | null;
  dropoff_location?: string | null;
  medical_facility?: string | null;
  doctor_name?: string | null;
  hospital_service?: string | null;
  notes_medical?: string | null;
  notes?: string | null;
  /** Instructions d'accès au point de prise en charge (ex: restaurant, hôtel) */
  pickup_access_notes?: string | null;
  /** Instructions d'accès à la destination */
  dropoff_access_notes?: string | null;
  wheelchair?: boolean;
  wheelchair_client_has?: boolean;
  wheelchair_need?: boolean;
  client?: {
    door_code?: string | null;
    floor?: string | null;
    access_notes?: string | null;
    contact_phone?: string | null;
  } | null;
}

const EMPTY_VALUES = ["", "non spécifié", "non specifié", "—", "-", "n/a", "na"];
const MAX_HINT_ITEMS = 5;

function isMeaningful(value: string | undefined | null): boolean {
  const v = (value ?? "").trim().toLowerCase();
  return v.length > 0 && !EMPTY_VALUES.includes(v);
}

function pushHint(
  out: HintItem[],
  icon: string,
  label: string,
  value: string | undefined | null,
  priority: number
): void {
  if (!isMeaningful(value)) return;
  out.push({ icon, label, value: (value ?? "").trim(), priority });
}

/** Détermine aller vs retour. Non-bloquant : si indécidable => unknown. */
export function inferTripDirection(mission: MissionLike | null): TripDirection {
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

/** Infos pour accéder au point de départ (où récupérer le client). */
export function getPickupHints(mission: MissionLike | null): HintItem[] {
  if (!mission) return [];
  const dir = inferTripDirection(mission);
  const out: HintItem[] = [];
  const c = mission.client;

  if (dir === "outbound") {
    // Pickup = domicile → infos domicile (code, étage, notes)
    pushHint(out, "key-outline", "Code porte", c?.door_code, 1);
    pushHint(out, "business-outline", "Étage", c?.floor, 2);
    pushHint(out, "document-text-outline", "Notes d'accès", c?.access_notes, 3);
    pushHint(out, "call-outline", "Contact", c?.contact_phone, 4);
    if (mission.wheelchair_client_has || mission.wheelchair)
      out.push({ icon: "accessibility-outline", label: "Mobilité", value: "Client en chaise roulante", priority: 5 });
    if (mission.wheelchair_need)
      out.push({ icon: "medkit-outline", label: "Mobilité", value: "Prendre une chaise roulante", priority: 6 });
  } else if (dir === "return") {
    // Pickup = hôpital → infos médicales
    pushHint(out, "business-outline", "Service / Bâtiment", mission.hospital_service, 10);
    pushHint(out, "location-outline", "Établissement", mission.medical_facility, 11);
    pushHint(out, "person-outline", "Médecin", mission.doctor_name, 12);
    pushHint(out, "document-text-outline", "Notes sortie", mission.notes_medical, 13);
  } else {
    // unknown (ex: restaurant → HUG) → uniquement pickup_access_notes si renseigné ; pas de fallback domicile (prudence)
    if (isMeaningful(mission.pickup_access_notes)) {
      pushHint(out, "document-text-outline", "Instructions accès", mission.pickup_access_notes, 1);
    }
    if (mission.wheelchair_client_has || mission.wheelchair)
      out.push({ icon: "accessibility-outline", label: "Mobilité", value: "Client en chaise roulante", priority: 5 });
    if (mission.wheelchair_need)
      out.push({ icon: "medkit-outline", label: "Mobilité", value: "Prendre une chaise roulante", priority: 6 });
  }

  out.sort((a, b) => a.priority - b.priority);
  return out.slice(0, MAX_HINT_ITEMS);
}

/** Infos pour accéder au point d'arrivée (destination). */
export function getDropoffHints(mission: MissionLike | null): HintItem[] {
  if (!mission) return [];
  const dir = inferTripDirection(mission);
  const out: HintItem[] = [];
  const c = mission.client;

  if (dir === "return") {
    // Dropoff = domicile → infos domicile
    pushHint(out, "key-outline", "Code porte", c?.door_code, 1);
    pushHint(out, "business-outline", "Étage", c?.floor, 2);
    pushHint(out, "document-text-outline", "Notes d'accès", c?.access_notes, 3);
    pushHint(out, "call-outline", "Contact", c?.contact_phone, 4);
  } else {
    // Dropoff = hôpital (ou autre) → infos médicales + instructions accès destination si présentes
    pushHint(out, "business-outline", "Service / Bâtiment", mission.hospital_service, 20);
    pushHint(out, "location-outline", "Établissement", mission.medical_facility, 21);
    pushHint(out, "person-outline", "Médecin", mission.doctor_name, 22);
    pushHint(out, "document-text-outline", "Notes médicales", mission.notes_medical, 23);
    pushHint(out, "document-text-outline", "Instructions accès", mission.dropoff_access_notes, 24);
  }

  out.sort((a, b) => a.priority - b.priority);
  return out.slice(0, MAX_HINT_ITEMS);
}
