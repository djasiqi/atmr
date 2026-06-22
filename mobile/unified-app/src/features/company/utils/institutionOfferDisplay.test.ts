import {
  buildInstitutionMobilityChips,
  buildInstitutionOfferListPreview,
  buildInstitutionRoutePoints,
  buildInstitutionRouteSummaryShort,
  buildInstitutionScheduleLabel,
  buildOfferStatusLabel,
  resolveInstitutionPatientName,
} from "./institutionOfferDisplay";
import type { InstitutionRequestOffer } from "../api/institutionOffersApi";

function sampleOffer(): InstitutionRequestOffer {
  return {
    id: 1,
    status: "PENDING",
    can_respond: true,
    expires_at: "2026-06-22T15:16:00Z",
    price_estimate: { amount: 80, currency: "CHF", source: "preferential" },
    transport_request: {
      institution_name: "Clinique Les Hauts d'Anières",
      patient_name: "Khalid ALAOUI",
      patient: { first_name: "Khalid", last_name: "ALAOUI", dob: "1974-04-13" },
      mission_type: "patient_transport",
      mission_date: "2026-06-22",
      scheduled_time_type: "arrival",
      return_to_institution: true,
      is_round_trip: true,
      billing_intent: "institution",
      requires_wheelchair: true,
      pickup_location: "Chemin des Courbes 9, 1247, Anières",
      dropoff_location: "HUG, Rue Gabrielle-Perret-Gentil 4, 1205, Genève",
      legs: [
        {
          sequence_index: 0,
          pickup_location: "Chemin des Courbes 9, 1247, Anières",
          dropoff_location:
            "Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4, 1205, Genève",
          dropoff_establishment: "Hôpitaux Universitaires de Genève (HUG)",
          scheduled_time: "2026-06-22T18:00:00Z",
          time_confirmed: true,
        },
        {
          sequence_index: 1,
          pickup_location:
            "Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4, 1205, Genève",
          dropoff_location: "Chemin des Courbes 9, 1247, Anières",
          scheduled_time: null,
          time_confirmed: false,
        },
      ],
    },
  };
}

describe("institutionOfferDisplay", () => {
  it("résout le nom patient depuis l'API (pas seulement le preview push)", () => {
    const req = sampleOffer().transport_request;
    expect(resolveInstitutionPatientName(req, undefined)).toBe("Khalid ALAOUI");
    expect(resolveInstitutionPatientName(undefined, { patient_name: "Preview" })).toBe(
      "Preview"
    );
  });

  it("affiche RDV 20:00 et le parcours aller-retour", () => {
    const req = sampleOffer().transport_request;
    const schedule = buildInstitutionScheduleLabel(req);
    expect(schedule).toContain("22 juin");
    expect(schedule).toMatch(/RDV \d{2}:\d{2}/);

    const points = buildInstitutionRoutePoints(req);
    expect(points).toHaveLength(3);
    expect(points[0].label).toBe("Départ");
    expect(points[1].label).toBe("Destination 1");
    expect(points[1].timeLabel).toMatch(/^RDV \d{2}:\d{2}$/);
    expect(points[2].label).toBe("Retour");
    expect(points[2].timeLabel).toBe("Départ · À définir");
  });

  it("détecte le besoin fauteuil", () => {
    const chips = buildInstitutionMobilityChips(sampleOffer().transport_request);
    expect(chips.some((c) => c.label === "Fauteuil")).toBe(true);
  });

  it("affiche Expiré quand le délai est dépassé mais le statut reste PENDING", () => {
    expect(
      buildOfferStatusLabel({
        id: 1,
        status: "PENDING",
        can_respond: false,
        expires_at: "2020-01-01T12:00:00Z",
      })
    ).toBe("Expiré");
  });

  it("formate la liste mobile sans ISO brut ni adresse seule", () => {
    const req = {
      institution_name: "Clinique Les Hauts d'Anières",
      patient_name: "Marie Dupont",
      mission_date: "2026-06-23",
      scheduled_time_type: "arrival",
      scheduled_time: "2026-06-23T11:30:00",
      pickup_location: "Chemin des Courbes 9, 1247, Anières",
      dropoff_location: "HUG, Rue Gabrielle-Perret-Gentil 4, 1205, Genève",
    };
    const preview = buildInstitutionOfferListPreview(req);
    expect(preview.title).toBe("Marie Dupont");
    expect(preview.institutionLabel).toBe("Clinique Les Hauts d'Anières");
    expect(preview.schedule).toContain("23 juin");
    expect(preview.schedule).toMatch(/RDV \d{2}:\d{2}/);
    expect(preview.schedule).not.toContain("T11:30:00");
    expect(preview.route).toContain("→");
    expect(preview.route).not.toBe(req.pickup_location);
    expect(preview.route).not.toMatch(/\(\s*(Départ|RDV)/);
    expect(preview.primaryTime).toMatch(/^\d{2}:\d{2}$/);
    expect(preview.scheduleDate).toBe("23 juin");
  });

  it("résume le parcours avec établissements courts", () => {
    const req = sampleOffer().transport_request;
    const summary = buildInstitutionRouteSummaryShort(req);
    expect(summary).toContain("Hôpitaux Universitaires de Genève (HUG)");
    expect(summary).toContain("→");
  });
});
