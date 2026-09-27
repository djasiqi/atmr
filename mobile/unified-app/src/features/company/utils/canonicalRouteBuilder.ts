/**
 * Parcours canonique saisi sur mobile.
 * L'ordre du tableau est la seule position : `position` n'existe qu'au moment
 * de construire le DTO. Ce module ne déclenche aucun envoi.
 */

export type MissionType = "patient_transport" | "material_delivery";
export type RouteStepKind = "pickup" | "destination" | "return";
export type DestinationKind = "medical" | "other";

export type RouteStepDraft = {
  kind: RouteStepKind;
  location: string;
  latitude: number | null;
  longitude: number | null;
  arrivalAt: string | null;
  departureAt: string | null;
  destinationKind: DestinationKind | null;
  accessNotes: string;
  establishment: string;
  service: string;
  doctor: string;
};

export type CanonicalRouteDraft = {
  missionType: MissionType;
  missionDate: string;
  isUrgent: boolean;
  routeSteps: RouteStepDraft[];
};

export type CanonicalApiStep = {
  position: number;
  kind: RouteStepKind;
  location: string;
  latitude: number | null;
  longitude: number | null;
  arrival_at: string | null;
  departure_at: string | null;
  destination_kind: DestinationKind | null;
  establishment: string | null;
  service: string | null;
  doctor: string | null;
  access_notes: string | null;
};

export type TimeFieldMode = {
  arrival: boolean;
  departure: boolean;
};

export type LegacyRideProjection = {
  pickup: string;
  dropoff: string;
  isRoundTrip: boolean;
  scheduledAt: string;
  returnScheduledAt: string;
};

const NAIVE_ISO = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?/;

function emptyStep(kind: RouteStepKind, destinationKind: DestinationKind | null): RouteStepDraft {
  return {
    kind,
    location: "",
    latitude: null,
    longitude: null,
    arrivalAt: null,
    departureAt: null,
    destinationKind,
    accessNotes: "",
    establishment: "",
    service: "",
    doctor: "",
  };
}

export function createRouteDraft(input?: {
  missionDate?: string;
  missionType?: MissionType;
  isUrgent?: boolean;
  pickupLocation?: string;
  pickupLatitude?: number | null;
  pickupLongitude?: number | null;
  pickupDepartureAt?: string | null;
  destinationLocation?: string;
  destinationLatitude?: number | null;
  destinationLongitude?: number | null;
  destinationArrivalAt?: string | null;
  isRoundTrip?: boolean;
  returnArrivalAt?: string | null;
}): CanonicalRouteDraft {
  const missionType = input?.missionType ?? "patient_transport";
  const destinationKind: DestinationKind | null =
    missionType === "patient_transport" ? "medical" : null;
  const pickup = emptyStep("pickup", null);
  pickup.location = input?.pickupLocation?.trim() ?? "";
  pickup.latitude = input?.pickupLatitude ?? null;
  pickup.longitude = input?.pickupLongitude ?? null;
  pickup.departureAt = normalizeNaiveIso(input?.pickupDepartureAt ?? null);
  const destination = emptyStep("destination", destinationKind);
  destination.location = input?.destinationLocation?.trim() ?? "";
  destination.latitude = input?.destinationLatitude ?? null;
  destination.longitude = input?.destinationLongitude ?? null;
  destination.arrivalAt = normalizeNaiveIso(input?.destinationArrivalAt ?? null);
  const routeSteps = [pickup, destination];
  const draft: CanonicalRouteDraft = {
    missionType,
    missionDate: input?.missionDate?.trim() || "2026-01-01",
    isUrgent: Boolean(input?.isUrgent),
    routeSteps,
  };
  if (!input?.isRoundTrip) return draft;
  return setRoundTrip(
    {
      ...draft,
      routeSteps: routeSteps.map((step) =>
        step.kind === "destination"
          ? { ...step, departureAt: step.departureAt }
          : step,
      ),
    },
    true,
    normalizeNaiveIso(input.returnArrivalAt ?? null),
  );
}

export function destinationIndexes(steps: readonly RouteStepDraft[]): number[] {
  return steps.flatMap((step, index) => (step.kind === "destination" ? [index] : []));
}

export function hasReturn(steps: readonly RouteStepDraft[]): boolean {
  return steps.length > 0 && steps[steps.length - 1]?.kind === "return";
}

/** Champs horaires imposés par la place de l'étape. L'utilisateur ne les choisit pas. */
export function timeFieldsForStep(steps: readonly RouteStepDraft[], index: number): TimeFieldMode {
  const step = steps[index];
  if (!step || step.kind === "pickup") return { arrival: false, departure: true };
  if (step.kind === "return") return { arrival: true, departure: false };
  const followed = index < steps.length - 1;
  return { arrival: true, departure: followed };
}

/** Aucun sélecteur d'arrivée : l'horaire saisi est toujours un départ vers l'étape suivante. */
export function arrivalClockVisible(_steps: readonly RouteStepDraft[], _index: number): boolean {
  return false;
}

/** Heure et jour seulement s'il reste une étape après celle-ci. */
export function departureClockVisible(steps: readonly RouteStepDraft[], index: number): boolean {
  const step = steps[index];
  if (!step || step.kind === "return") return false;
  return index < steps.length - 1;
}

export function normalizeNaiveIso(value: string | null | undefined): string | null {
  if (value == null) return null;
  const match = NAIVE_ISO.exec(value.trim());
  if (!match) return null;
  return `${match[1]}T${match[2]}:${match[3]}:${match[4] ?? "00"}`;
}

export function clockFromIso(value: string | null | undefined): string {
  const iso = normalizeNaiveIso(value);
  if (!iso) return "";
  return iso.slice(11, 16);
}

export function isOtherMissionDay(missionDate: string, value: string | null | undefined): boolean {
  const iso = normalizeNaiveIso(value);
  if (!iso) return false;
  return iso.slice(0, 10) !== missionDate;
}

/** Heure seule le jour de mission, ou date complète si un autre jour est choisi. */
export function formatStepClock(
  missionDate: string,
  value: string | null | undefined,
  otherDayOpen: boolean,
): string {
  const iso = normalizeNaiveIso(value);
  if (!iso) return "";
  const clock = iso.slice(11, 16);
  if (!otherDayOpen && !isOtherMissionDay(missionDate, iso)) return clock;
  const [year, month, day] = iso.slice(0, 10).split("-");
  return `${day}.${month}.${year}  ${clock}`;
}

/** Ramène l'heure sur la date de mission. Conserve l'heure, y compris après minuit. */
export function snapClockToMissionDate(missionDate: string, value: string | null | undefined): string | null {
  const iso = normalizeNaiveIso(value);
  if (!iso) return null;
  return `${missionDate}T${iso.slice(11, 19)}`;
}

function copyReturnFromPickup(steps: RouteStepDraft[]): RouteStepDraft[] {
  const pickup = steps[0];
  if (!pickup || !hasReturn(steps)) return steps;
  return steps.map((step, index) =>
    index === steps.length - 1 && step.kind === "return"
      ? {
          ...step,
          location: pickup.location,
          latitude: pickup.latitude,
          longitude: pickup.longitude,
          destinationKind: null,
          departureAt: null,
        }
      : step,
  );
}

export function setMissionType(draft: CanonicalRouteDraft, missionType: MissionType): CanonicalRouteDraft {
  if (draft.missionType === missionType) return draft;
  return {
    ...draft,
    missionType,
    routeSteps: draft.routeSteps.map((step) => {
      if (step.kind !== "destination") return step;
      if (missionType === "material_delivery") return { ...step, destinationKind: null };
      return { ...step, destinationKind: step.destinationKind ?? "medical" };
    }),
  };
}

export function setUrgent(draft: CanonicalRouteDraft, isUrgent: boolean): CanonicalRouteDraft {
  if (draft.isUrgent === isUrgent) return draft;
  return { ...draft, isUrgent };
}

export function setMissionDate(draft: CanonicalRouteDraft, missionDate: string): CanonicalRouteDraft {
  return { ...draft, missionDate };
}

export function setStepLocation(
  draft: CanonicalRouteDraft,
  index: number,
  location: string,
  latitude: number | null = null,
  longitude: number | null = null,
): CanonicalRouteDraft {
  const routeSteps = draft.routeSteps.map((step, stepIndex) =>
    stepIndex === index ? { ...step, location: location.trim(), latitude, longitude } : step,
  );
  return { ...draft, routeSteps: copyReturnFromPickup(routeSteps) };
}

export function setStepDateTime(
  draft: CanonicalRouteDraft,
  index: number,
  field: "arrivalAt" | "departureAt",
  value: string | null,
): CanonicalRouteDraft {
  const mode = timeFieldsForStep(draft.routeSteps, index);
  const allowed = field === "arrivalAt" ? mode.arrival : mode.departure;
  if (!allowed) return draft;
  const nextValue = normalizeNaiveIso(value);
  const routeSteps = draft.routeSteps.map((step, stepIndex) => {
    if (stepIndex !== index) return step;
    return field === "arrivalAt" ? { ...step, arrivalAt: nextValue } : { ...step, departureAt: nextValue };
  });
  return { ...draft, routeSteps };
}

export function setDestinationKind(
  draft: CanonicalRouteDraft,
  index: number,
  destinationKind: DestinationKind,
): CanonicalRouteDraft {
  if (draft.missionType !== "patient_transport") return draft;
  const step = draft.routeSteps[index];
  if (!step || step.kind !== "destination") return draft;
  const routeSteps = draft.routeSteps.map((item, stepIndex) =>
    stepIndex === index ? { ...item, destinationKind } : item,
  );
  return { ...draft, routeSteps };
}

/** Un seul choix « Médical / Autre lieu » s'applique à toutes les destinations. */
export function setAllDestinationsKind(
  draft: CanonicalRouteDraft,
  destinationKind: DestinationKind,
): CanonicalRouteDraft {
  if (draft.missionType !== "patient_transport") return draft;
  const routeSteps = draft.routeSteps.map((step) =>
    step.kind === "destination" ? { ...step, destinationKind } : step,
  );
  return { ...draft, routeSteps };
}

export function uniformDestinationKind(draft: CanonicalRouteDraft): DestinationKind | null {
  const kinds = draft.routeSteps
    .filter((step) => step.kind === "destination")
    .map((step) => step.destinationKind);
  if (kinds.length === 0) return "medical";
  const first = kinds[0] ?? null;
  return kinds.every((kind) => kind === first) ? first : null;
}

export type StepClinicalPatch = Partial<
  Pick<RouteStepDraft, "accessNotes" | "establishment" | "service" | "doctor">
>;

export function patchStepDetails(
  draft: CanonicalRouteDraft,
  index: number,
  patch: StepClinicalPatch,
): CanonicalRouteDraft {
  const step = draft.routeSteps[index];
  if (!step) return draft;
  const routeSteps = draft.routeSteps.map((item, stepIndex) =>
    stepIndex === index ? { ...item, ...patch } : item,
  );
  return { ...draft, routeSteps };
}

/** Établissement obligatoire, et service ou médecin, pour chaque destination médicale. */
export function clinicalSubmitErrors(draft: CanonicalRouteDraft): string[] {
  if (draft.missionType === "material_delivery") return [];
  const destinations = draft.routeSteps.filter((step) => step.kind === "destination");
  const several = destinations.length > 1;
  const errors: string[] = [];
  destinations.forEach((step, index) => {
    if (step.destinationKind !== "medical") return;
    const number = index + 1;
    if (!step.establishment.trim()) {
      errors.push(
        several
          ? `Veuillez indiquer l'établissement de la destination ${number}`
          : "Veuillez indiquer l'établissement",
      );
    }
    if (!step.service.trim() && !step.doctor.trim()) {
      errors.push(
        several
          ? `Veuillez indiquer le service ou le médecin de la destination ${number}`
          : "Veuillez indiquer le service ou le médecin",
      );
    }
  });
  return errors;
}

export function addDestination(draft: CanonicalRouteDraft): CanonicalRouteDraft {
  const inherited = draft.routeSteps.find((step) => step.kind === "destination")?.destinationKind;
  const kind: DestinationKind | null =
    draft.missionType === "patient_transport" ? inherited ?? "medical" : null;
  const step = emptyStep("destination", kind);
  const routeSteps = [...draft.routeSteps];
  const insertAt = hasReturn(routeSteps) ? routeSteps.length - 1 : routeSteps.length;
  routeSteps.splice(insertAt, 0, step);
  return { ...draft, routeSteps };
}

export function removeDestination(draft: CanonicalRouteDraft, destinationOrdinal: number): CanonicalRouteDraft {
  const indexes = destinationIndexes(draft.routeSteps);
  if (indexes.length <= 1) return draft;
  const index = indexes[destinationOrdinal];
  if (index == null) return draft;
  const routeSteps = draft.routeSteps.filter((_, stepIndex) => stepIndex !== index);
  return { ...draft, routeSteps: copyReturnFromPickup(routeSteps) };
}

export function moveDestination(
  draft: CanonicalRouteDraft,
  destinationOrdinal: number,
  direction: -1 | 1,
): CanonicalRouteDraft {
  const indexes = destinationIndexes(draft.routeSteps);
  const targetOrdinal = destinationOrdinal + direction;
  if (targetOrdinal < 0 || targetOrdinal >= indexes.length) return draft;
  const from = indexes[destinationOrdinal];
  const to = indexes[targetOrdinal];
  if (from == null || to == null) return draft;
  const routeSteps = [...draft.routeSteps];
  const [moved] = routeSteps.splice(from, 1);
  if (!moved) return draft;
  routeSteps.splice(to, 0, moved);
  return { ...draft, routeSteps: copyReturnFromPickup(routeSteps) };
}

export function setRoundTrip(
  draft: CanonicalRouteDraft,
  enabled: boolean,
  arrivalAt: string | null = null,
): CanonicalRouteDraft {
  const withoutReturn = draft.routeSteps.filter((step) => step.kind !== "return");
  if (!enabled) return { ...draft, routeSteps: withoutReturn };
  if (hasReturn(draft.routeSteps)) {
    return {
      ...draft,
      routeSteps: copyReturnFromPickup(
        draft.routeSteps.map((step, index) =>
          index === draft.routeSteps.length - 1
            ? { ...step, arrivalAt: arrivalAt ?? step.arrivalAt, departureAt: null }
            : step,
        ),
      ),
    };
  }
  const pickup = withoutReturn[0];
  const ret = emptyStep("return", null);
  ret.location = pickup?.location ?? "";
  ret.latitude = pickup?.latitude ?? null;
  ret.longitude = pickup?.longitude ?? null;
  ret.arrivalAt = normalizeNaiveIso(arrivalAt);
  return { ...draft, routeSteps: [...withoutReturn, ret] };
}

function optionalText(value: string): string | null {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function apiClock(step: RouteStepDraft, field: "arrivalAt" | "departureAt", enabled: boolean): string | null {
  if (!enabled) return null;
  return normalizeNaiveIso(step[field]);
}

export function buildCanonicalRouteSteps(draft: CanonicalRouteDraft): CanonicalApiStep[] {
  const pickup = draft.routeSteps[0];
  return draft.routeSteps.map((step, index) => {
    const mode = timeFieldsForStep(draft.routeSteps, index);
    const isReturn = step.kind === "return";
    const location = isReturn ? pickup?.location ?? step.location : step.location;
    const latitude = isReturn ? pickup?.latitude ?? null : step.latitude;
    const longitude = isReturn ? pickup?.longitude ?? null : step.longitude;
    const destinationKind =
      step.kind === "destination" && draft.missionType === "patient_transport"
        ? step.destinationKind
        : null;
    return {
      position: index,
      kind: step.kind,
      location,
      latitude,
      longitude,
      arrival_at: apiClock(step, "arrivalAt", mode.arrival),
      departure_at: apiClock(step, "departureAt", mode.departure),
      destination_kind: destinationKind,
      establishment: step.kind === "destination" ? optionalText(step.establishment) : null,
      service: step.kind === "destination" ? optionalText(step.service) : null,
      doctor: step.kind === "destination" ? optionalText(step.doctor) : null,
      access_notes: optionalText(step.accessNotes),
    };
  });
}

export function validateCanonicalRoute(draft: CanonicalRouteDraft): string[] {
  const errors: string[] = [];
  const steps = buildCanonicalRouteSteps(draft);
  if (steps.length < 2 || steps[0]?.kind !== "pickup") {
    errors.push("Le départ est unique et en première position.");
  }
  if (!steps.some((step) => step.kind === "destination")) {
    errors.push("Le parcours contient au moins une destination.");
  }
  const returnCount = steps.filter((step) => step.kind === "return").length;
  if (returnCount > 1 || (returnCount === 1 && steps[steps.length - 1]?.kind !== "return")) {
    errors.push("Le retour est unique et en dernière position.");
  }
  steps.forEach((step, index) => {
    if (!step.location.trim()) errors.push(`L'étape ${index + 1} n'a pas d'adresse.`);
    if (step.kind === "pickup" && !step.departure_at) {
      errors.push("Le départ a une heure de départ.");
    }
    if (step.kind === "destination" && arrivalClockVisible(draft.routeSteps, index) && !step.arrival_at) {
      errors.push("Une destination a une heure d'arrivée.");
    }
    if (step.kind === "return" && arrivalClockVisible(draft.routeSteps, index) && !step.arrival_at) {
      errors.push("Le retour a une heure d'arrivée.");
    }
    if (
      step.kind === "destination" &&
      draft.missionType === "patient_transport" &&
      step.destination_kind !== "medical" &&
      step.destination_kind !== "other"
    ) {
      errors.push("Le type de lieu est obligatoire pour un transport de personne.");
    }
    if (step.arrival_at && step.departure_at && step.departure_at < step.arrival_at) {
      errors.push("L'heure de départ d'une étape précède son arrivée.");
    }
  });
  let previous: string | null = null;
  for (const step of steps) {
    const marker = step.arrival_at ?? step.departure_at;
    if (previous && step.arrival_at && step.arrival_at < previous) {
      errors.push("Les étapes ne sont pas dans l'ordre chronologique.");
      break;
    }
    if (step.departure_at) previous = step.departure_at;
    else if (marker) previous = marker;
  }
  return errors;
}

export function projectLegacyRideFields(draft: CanonicalRouteDraft): LegacyRideProjection {
  const pickup = draft.routeSteps[0];
  const destination = draft.routeSteps.find((step) => step.kind === "destination");
  const ret = hasReturn(draft.routeSteps) ? draft.routeSteps[draft.routeSteps.length - 1] : null;
  return {
    pickup: pickup?.location ?? "",
    dropoff: destination?.location ?? "",
    isRoundTrip: ret != null,
    scheduledAt: normalizeNaiveIso(pickup?.departureAt) ?? "",
    returnScheduledAt: normalizeNaiveIso(ret?.arrivalAt) ?? "",
  };
}

/** Le POST actuel ne sait envoyer qu'un aller, ou un aller-retour simple. */
export function isLegacySubmitShape(draft: CanonicalRouteDraft): boolean {
  return destinationIndexes(draft.routeSteps).length === 1;
}
