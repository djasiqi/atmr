import { describe, expect, it } from "@jest/globals";
import { buildRideCreatePayload } from "../components/rides/rideCreateHelpers";
import {
  addDestination,
  buildCanonicalRouteSteps,
  createRouteDraft,
  isLegacySubmitShape,
  moveDestination,
  projectLegacyRideFields,
  removeDestination,
  clinicalSubmitErrors,
  setAllDestinationsKind,
  setDestinationKind,
  setMissionType,
  uniformDestinationKind,
  setRoundTrip,
  setStepDateTime,
  setStepLocation,
  snapClockToMissionDate,
  arrivalClockVisible,
  departureClockVisible,
  timeFieldsForStep,
  validateCanonicalRoute,
} from "./canonicalRouteBuilder";

const MISSION_DATE = "2026-09-28";

function filledOneWay() {
  return createRouteDraft({
    missionDate: MISSION_DATE,
    pickupLocation: "EMS Les Marronniers",
    pickupDepartureAt: "2026-09-28T09:00",
    destinationLocation: "HUG",
    destinationArrivalAt: "2026-09-28T09:30",
  });
}

describe("buildCanonicalRouteSteps", () => {
  it("A → B produit pickup puis destination", () => {
    const steps = buildCanonicalRouteSteps(filledOneWay());
    expect(steps.map((step) => step.kind)).toEqual(["pickup", "destination"]);
    expect(steps.map((step) => step.position)).toEqual([0, 1]);
    expect(steps[0]).toMatchObject({
      arrival_at: null,
      departure_at: "2026-09-28T09:00:00",
      destination_kind: null,
    });
    expect(steps[1]).toMatchObject({
      location: "HUG",
      arrival_at: "2026-09-28T09:30:00",
      departure_at: null,
      destination_kind: "medical",
    });
    expect(validateCanonicalRoute(filledOneWay())).toEqual([]);
  });

  it("A → B → A produit pickup, destination et return", () => {
    const draft = setStepDateTime(
      setRoundTrip(filledOneWay(), true, "2026-09-28T12:30"),
      1,
      "departureAt",
      "2026-09-28T10:00",
    );
    const steps = buildCanonicalRouteSteps(draft);
    expect(steps.map((step) => step.kind)).toEqual(["pickup", "destination", "return"]);
    expect(steps[2]).toMatchObject({
      position: 2,
      location: "EMS Les Marronniers",
      arrival_at: "2026-09-28T12:30:00",
      departure_at: null,
      destination_kind: null,
    });
    expect(steps[1]?.departure_at).toBe("2026-09-28T10:00:00");
    expect(validateCanonicalRoute(draft)).toEqual([]);
  });

  it("A → B → C produit deux destinations sans retour", () => {
    let draft = addDestination(filledOneWay());
    draft = setStepLocation(draft, 2, "Clinique La Colline");
    draft = setStepDateTime(draft, 1, "departureAt", "2026-09-28T10:00");
    draft = setStepDateTime(draft, 2, "arrivalAt", "2026-09-28T11:00");
    const steps = buildCanonicalRouteSteps(draft);
    expect(steps.map((step) => step.kind)).toEqual(["pickup", "destination", "destination"]);
    expect(steps.map((step) => step.position)).toEqual([0, 1, 2]);
    expect(steps[1]?.departure_at).toBe("2026-09-28T10:00:00");
    expect(steps[2]?.departure_at).toBeNull();
    expect(isLegacySubmitShape(draft)).toBe(false);
  });

  it("A → B → C → A numérote pickup, deux destinations et le retour", () => {
    let draft = setRoundTrip(addDestination(filledOneWay()), true, "2026-09-28T12:30");
    draft = setStepLocation(draft, 2, "Clinique La Colline");
    draft = setStepDateTime(draft, 1, "departureAt", "2026-09-28T10:00");
    draft = setStepDateTime(draft, 2, "arrivalAt", "2026-09-28T11:00");
    draft = setStepDateTime(draft, 2, "departureAt", "2026-09-28T11:30");
    const steps = buildCanonicalRouteSteps(draft);
    expect(steps.map((step) => `${step.position} ${step.kind}`)).toEqual([
      "0 pickup",
      "1 destination",
      "2 destination",
      "3 return",
    ]);
    expect(steps[3]?.location).toBe("EMS Les Marronniers");
    expect(validateCanonicalRoute(draft)).toEqual([]);
  });
});

describe("réorganisation du parcours", () => {
  function threeStops() {
    let draft = setRoundTrip(addDestination(filledOneWay()), true, "2026-09-28T12:30");
    draft = setStepLocation(draft, 1, "HUG");
    draft = setStepLocation(draft, 2, "Clinique La Colline");
    return draft;
  }

  it("ajoute une destination avant le retour", () => {
    const draft = addDestination(setRoundTrip(filledOneWay(), true));
    expect(draft.routeSteps.map((step) => step.kind)).toEqual([
      "pickup",
      "destination",
      "destination",
      "return",
    ]);
    expect(draft.routeSteps.some((step) => "position" in step)).toBe(false);
  });

  it("supprime une destination et conserve le retour en dernier", () => {
    const draft = removeDestination(threeStops(), 0);
    expect(draft.routeSteps.map((step) => step.location)).toEqual([
      "EMS Les Marronniers",
      "Clinique La Colline",
      "EMS Les Marronniers",
    ]);
  });

  it("refuse de supprimer la dernière destination", () => {
    const draft = removeDestination(filledOneWay(), 0);
    expect(draft.routeSteps).toHaveLength(2);
  });

  it("monte et descend une destination sans déplacer le retour", () => {
    const down = moveDestination(threeStops(), 0, 1);
    expect(down.routeSteps.map((step) => step.location)).toEqual([
      "EMS Les Marronniers",
      "Clinique La Colline",
      "HUG",
      "EMS Les Marronniers",
    ]);
    const up = moveDestination(down, 1, -1);
    expect(up.routeSteps.map((step) => step.kind)).toEqual([
      "pickup",
      "destination",
      "destination",
      "return",
    ]);
    expect(buildCanonicalRouteSteps(up).map((step) => step.position)).toEqual([0, 1, 2, 3]);
    expect(moveDestination(up, 0, -1)).toBe(up);
  });

  it("le retour reprend toujours le pickup", () => {
    const draft = setStepLocation(setRoundTrip(filledOneWay(), true, "2026-09-28T12:30"), 0, "Nouveau départ", 46.2, 6.1);
    const ret = draft.routeSteps[draft.routeSteps.length - 1];
    expect(ret).toMatchObject({
      kind: "return",
      location: "Nouveau départ",
      latitude: 46.2,
      longitude: 6.1,
    });
    expect(buildCanonicalRouteSteps(draft).at(-1)?.location).toBe("Nouveau départ");
  });
});

describe("horaires et type de lieu", () => {
  it("impose les champs selon la position", () => {
    const withReturn = setRoundTrip(filledOneWay(), true);
    expect(timeFieldsForStep(withReturn.routeSteps, 0)).toEqual({ arrival: false, departure: true });
    expect(timeFieldsForStep(withReturn.routeSteps, 1)).toEqual({ arrival: true, departure: true });
    expect(timeFieldsForStep(withReturn.routeSteps, 2)).toEqual({ arrival: true, departure: false });
    const oneWay = filledOneWay();
    expect(timeFieldsForStep(oneWay.routeSteps, 1)).toEqual({ arrival: true, departure: false });
  });

  it("seul le premier départ exige une heure", () => {
    const draft = addDestination(filledOneWay());
    expect(validateCanonicalRoute(draft).join(" ")).not.toContain("heure de départ");
    const withoutPickupTime = {
      ...filledOneWay(),
      routeSteps: filledOneWay().routeSteps.map((step) =>
        step.kind === "pickup" ? { ...step, departureAt: null } : step,
      ),
    };
    expect(validateCanonicalRoute(withoutPickupTime).join(" ")).toContain("heure de départ");
  });

  it("la destination finale n'exige que l'arrivée", () => {
    const draft = filledOneWay();
    expect(buildCanonicalRouteSteps(draft)[1]?.departure_at).toBeNull();
    expect(validateCanonicalRoute(draft)).toEqual([]);
  });

  it("n'affiche un horaire que s'il reste une étape après", () => {
    const oneWay = filledOneWay();
    expect(arrivalClockVisible(oneWay.routeSteps, 1)).toBe(false);
    expect(departureClockVisible(oneWay.routeSteps, 0)).toBe(true);
    expect(departureClockVisible(oneWay.routeSteps, 1)).toBe(false);
    const withoutArrival = {
      ...oneWay,
      routeSteps: oneWay.routeSteps.map((step) =>
        step.kind === "destination" ? { ...step, arrivalAt: null } : step,
      ),
    };
    expect(validateCanonicalRoute(withoutArrival)).toEqual([]);

    const withSecond = addDestination(oneWay);
    expect(departureClockVisible(withSecond.routeSteps, 1)).toBe(true);
    expect(departureClockVisible(withSecond.routeSteps, 2)).toBe(false);
    expect(arrivalClockVisible(withSecond.routeSteps, 1)).toBe(false);
    expect(arrivalClockVisible(withSecond.routeSteps, 2)).toBe(false);

    const withReturn = setRoundTrip(oneWay, true);
    expect(departureClockVisible(withReturn.routeSteps, 1)).toBe(true);
    expect(departureClockVisible(withReturn.routeSteps, 2)).toBe(false);
    expect(arrivalClockVisible(withReturn.routeSteps, 2)).toBe(false);
    expect(validateCanonicalRoute(withReturn).join(" ")).not.toContain("heure d'arrivée");
  });

  it("conserve un passage après minuit sur un autre jour", () => {
    const picked = snapClockToMissionDate(MISSION_DATE, "2026-09-29T00:30");
    expect(picked).toBe("2026-09-28T00:30:00");
    const kept = setStepDateTime(filledOneWay(), 1, "arrivalAt", "2026-09-29T00:30");
    expect(buildCanonicalRouteSteps(kept)[1]?.arrival_at).toBe("2026-09-29T00:30:00");
    expect(validateCanonicalRoute(kept)).toEqual([]);
    const tooEarly = setStepDateTime(filledOneWay(), 1, "arrivalAt", "2026-09-28T08:00");
    expect(validateCanonicalRoute(tooEarly).join(" ")).toContain("chronologique");
  });

  it("exige destination_kind pour un transport, et le retire pour une livraison", () => {
    const patient = setDestinationKind(filledOneWay(), 1, "other");
    expect(buildCanonicalRouteSteps(patient)[1]?.destination_kind).toBe("other");
    const missing = {
      ...patient,
      routeSteps: patient.routeSteps.map((step) =>
        step.kind === "destination" ? { ...step, destinationKind: null } : step,
      ),
    };
    expect(validateCanonicalRoute(missing).join(" ")).toContain("type de lieu");
    const delivery = setMissionType(patient, "material_delivery");
    expect(buildCanonicalRouteSteps(delivery)[1]?.destination_kind).toBeNull();
    expect(validateCanonicalRoute(delivery).join(" ")).not.toContain("type de lieu");
  });

  it("exige l'établissement et le service ou le médecin sur une destination médicale", () => {
    const draft = filledOneWay();
    expect(clinicalSubmitErrors(draft)[0]).toBe("Veuillez indiquer l'établissement");
    const withPlace = {
      ...draft,
      routeSteps: draft.routeSteps.map((step) =>
        step.kind === "destination" ? { ...step, establishment: "HUG", service: "Urgences" } : step,
      ),
    };
    expect(clinicalSubmitErrors(withPlace)).toEqual([]);
    const other = setAllDestinationsKind(draft, "other");
    expect(clinicalSubmitErrors(other)).toEqual([]);
  });

  it("applique le type de lieu à toutes les destinations et le reprend à l'ajout", () => {
    const twoStops = addDestination(filledOneWay());
    const other = setAllDestinationsKind(twoStops, "other");
    expect(uniformDestinationKind(other)).toBe("other");
    expect(other.routeSteps.filter((step) => step.kind === "destination").map((step) => step.destinationKind)).toEqual([
      "other",
      "other",
    ]);
    const third = addDestination(other);
    expect(third.routeSteps.filter((step) => step.kind === "destination").map((step) => step.destinationKind)).toEqual([
      "other",
      "other",
      "other",
    ]);
  });
});

describe("soumission legacy inchangée", () => {
  it("projette A → B → C → A sur le premier trajet aller-retour", () => {
    let draft = setRoundTrip(addDestination(filledOneWay()), true, "2026-09-28T12:30");
    draft = setStepLocation(draft, 2, "Clinique La Colline");
    expect(projectLegacyRideFields(draft)).toEqual({
      pickup: "EMS Les Marronniers",
      dropoff: "HUG",
      isRoundTrip: true,
      scheduledAt: "2026-09-28T09:00:00",
      returnScheduledAt: "2026-09-28T12:30:00",
    });
    expect(isLegacySubmitShape(draft)).toBe(false);
    expect(isLegacySubmitShape(filledOneWay())).toBe(true);
  });

  it("le payload de création actuel ne contient pas route_steps", () => {
    const payload = buildRideCreatePayload({
      structuredPayloadEnabled: false,
      clientId: 1,
      pickup: "EMS Les Marronniers",
      dropoff: "HUG",
      pickupAddress: null,
      dropoffAddress: null,
      scheduledTime: "2026-09-28T09:00:00",
      isRoundTrip: true,
      recurrence: "none",
      notesMedical: "",
      establishment: "",
      hospitalService: "",
      doctorName: "",
      pickupAccessNotes: "",
      dropoffAccessNotes: "",
      wheelchairClient: false,
      wheelchairProvide: false,
      internalNotes: "",
      notesMax: 500,
      amountInput: "40.00",
      amountSource: "manual",
      pricingProfileId: null,
      pricingProfileVersionId: null,
      isMaterialDelivery: false,
      deliveryDescription: "",
      returnScheduledAt: "2026-09-28T12:30:00",
      billToPatient: false,
      hasActiveStay: false,
      clinicBillingPartyId: null,
      recurrenceLimitMode: "count",
      recurrenceOccurrences: 10,
      recurrenceEndDate: "",
      recurrenceDays: [],
      recurrenceIntervalWeeks: 2,
    });
    expect(payload).not.toHaveProperty("route_steps");
    expect(payload.pickup_location).toBe("EMS Les Marronniers");
    expect(payload.dropoff_location).toBe("HUG");
    expect(payload.is_round_trip).toBe(true);
  });
});
