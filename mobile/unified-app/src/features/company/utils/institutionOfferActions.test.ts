import { resolveInstitutionOfferActions } from "./institutionOfferActions";

const NOW = new Date("2026-06-22T22:21:00+02:00");

function pendingOffer(req: Record<string, unknown>) {
  return {
    id: 1,
    status: "PENDING",
    can_respond: true,
    expires_at: "2030-01-01T12:00:00Z",
    transport_request: req,
  };
}

describe("resolveInstitutionOfferActions", () => {
  it("masque toute action si offre expirée (urgente incluse)", () => {
    const actions = resolveInstitutionOfferActions(
      {
        id: 1,
        status: "PENDING",
        can_respond: false,
        expires_at: "2020-01-01T12:00:00Z",
        transport_request: { is_urgent: true },
      },
      NOW
    );
    expect(actions.canRespond).toBe(false);
    expect(actions.canAcceptNow).toBe(false);
    expect(actions.canPlan).toBe(false);
  });

  it("cas Khalid : Planifier + Refuser uniquement", () => {
    const actions = resolveInstitutionOfferActions(
      pendingOffer({
        pickup_time_confirmed: false,
        scheduled_time: "2026-06-22T20:00:00",
        scheduled_time_type: "arrival",
        appointment_time_confirmed: true,
        is_urgent: false,
        legs: [
          {
            sequence_index: 0,
            scheduled_time: "2026-06-22T20:00:00",
            time_confirmed: true,
          },
        ],
      }),
      NOW
    );
    expect(actions.canValidate).toBe(false);
    expect(actions.canAcceptNow).toBe(false);
    expect(actions.canPlan).toBe(true);
    expect(actions.canReject).toBe(true);
    expect(actions.hint).toContain("rendez-vous");
  });

  it("cas 1 : départ confirmé futur — Valider + Planifier + Refuser", () => {
    const actions = resolveInstitutionOfferActions(
      pendingOffer({
        pickup_time_confirmed: true,
        scheduled_time: "2026-06-22T23:00:00",
        scheduled_time_type: "departure",
        is_urgent: false,
      }),
      NOW
    );
    expect(actions.canValidate).toBe(true);
    expect(actions.canPlan).toBe(true);
    expect(actions.canAcceptNow).toBe(false);
    expect(actions.canReject).toBe(true);
  });

  it("cas 3 : urgent — Départ immédiat + Planifier + Refuser", () => {
    const actions = resolveInstitutionOfferActions(
      pendingOffer({
        pickup_time_confirmed: false,
        is_urgent: true,
      }),
      NOW
    );
    expect(actions.canValidate).toBe(false);
    expect(actions.canAcceptNow).toBe(true);
    expect(actions.canPlan).toBe(true);
    expect(actions.canReject).toBe(true);
  });
});
