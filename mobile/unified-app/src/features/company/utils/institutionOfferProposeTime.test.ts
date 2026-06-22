import {
  computeDefaultProposedDate,
  datetimeLocalValueFromDate,
  formatOutboundRouteLabel,
  formatProposedPickupIso,
  isoFromDatetimeLocalValue,
} from "./institutionOfferProposeTime";
import type { InstitutionTransportRequestSummary } from "../api/institutionOffersApi";

function sampleRequest(): InstitutionTransportRequestSummary {
  return {
    institution_name: "Clinique Les Hauts d'Anières",
    scheduled_time_type: "arrival",
    pickup_location: "Chemin des Courbes 9, 1247, Anières",
    dropoff_location: "HUG, Genève",
    legs: [
      {
        sequence_index: 0,
        pickup_location: "Chemin des Courbes 9, 1247, Anières",
        dropoff_location: "HUG, Genève",
        scheduled_time: "2026-06-22T18:00:00Z",
        time_confirmed: true,
      },
    ],
  };
}

describe("institutionOfferProposeTime", () => {
  it("formate le trajet aller", () => {
    expect(formatOutboundRouteLabel(sampleRequest())).toContain("Anières");
    expect(formatOutboundRouteLabel(sampleRequest())).toContain("HUG");
  });

  it("décale l'horaire proposé quand le RDV est une arrivée", () => {
    const req = sampleRequest();
    const withTravel = computeDefaultProposedDate(req, 30);
    const withoutTravel = computeDefaultProposedDate(req, null);
    expect(withTravel).not.toBeNull();
    expect(withoutTravel).not.toBeNull();
    if (withTravel && withoutTravel) {
      expect(withTravel.getTime()).toBeLessThan(withoutTravel.getTime());
    }
  });

  it("produit un ISO naïf pour l'API accept", () => {
    const iso = formatProposedPickupIso(new Date("2026-06-22T17:30:00"));
    expect(iso).toMatch(/^2026-06-22T\d{2}:\d{2}:00$/);
  });

  it("convertit datetime-local ↔ ISO naïf", () => {
    expect(isoFromDatetimeLocalValue("2026-06-22T19:31")).toBe("2026-06-22T19:31:00");
    const local = datetimeLocalValueFromDate(new Date("2026-06-22T17:30:00Z"));
    expect(local).toMatch(/^2026-06-22T\d{2}:\d{2}$/);
  });
});
