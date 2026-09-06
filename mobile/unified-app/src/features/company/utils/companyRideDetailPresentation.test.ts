import {
  buildRideBillingSummary,
  buildRideDetailInfoRows,
  buildRideTimeline,
  formatRideCurrency,
  formatRideOriginLine,
  readRidePassengerBirthDate,
} from "./companyRideDetailPresentation";
import { buildIdentityFromMission } from "./bookingIdentity";
import type { CompanyDispatchMission } from "../api/contracts";

describe("companyRideDetailPresentation", () => {
  const baseMission = {
    mission_id: 31773,
    client_name: "Jean-Marie GODET",
    status: "completed",
    amount: 45,
    scheduled_at: "2026-06-28T14:27:00",
    pickup_label: "Ruelle de la Mère-Elise, 5, 1252, Meinier",
    dropoff_label: "HUG, Rue Gabrielle-Perret-Gentil 4, 1205, Genève",
    medical_facility: "Hôpitaux Universitaires de Genève (HUG)",
    identity: {
      passenger: { name: "Jean-Marie GODET", birth_date: "1946-12-28", gender: "MALE" },
      source: { type: "company_client", name: "Emmenez Moi", code: "C24205" },
      ownership: { owner_company_name: "Emmenez Moi" },
      execution: { executing_company_name: "Emmenez Moi" },
    },
    origin_channel: "Portefeuille propre",
    billed_to_type: "patient",
    created_at: "2026-06-28T11:57:00",
    picked_up_at: "2026-06-28T13:20:00",
    completed_at: "2026-06-28T14:17:00",
  } as Record<string, unknown>;

  it("formate le montant en CHF", () => {
    expect(formatRideCurrency(45)).toBe("45.00 CHF");
  });

  it("construit la ligne origine comme sur le web", () => {
    const identity = buildIdentityFromMission(baseMission as CompanyDispatchMission);
    expect(formatRideOriginLine(identity, baseMission)).toBe(
      "Portefeuille · Portefeuille propre · Emmenez Moi (C24205)"
    );
  });

  it("expose la date de naissance", () => {
    expect(readRidePassengerBirthDate(baseMission)).toBe("28.12.1946");
  });

  it("construit les lignes informations clés", () => {
    const identity = buildIdentityFromMission(baseMission as CompanyDispatchMission);
    const billing = buildRideBillingSummary(baseMission, null);
    const rows = buildRideDetailInfoRows(baseMission, identity, {
      statusLabel: "Terminée",
      scheduledIso: "2026-06-28T14:27:00",
      driverDisplay: "Emmenez Moi",
      billingSummary: billing,
    });
    expect(rows.some((r) => r.label === "Montant" && r.value === "45.00 CHF")).toBe(true);
    expect(rows.some((r) => r.label === "Date de naissance")).toBe(true);
    expect(rows.some((r) => r.label === "Passager" && r.value.startsWith("M."))).toBe(true);
  });

  it("en attente serveur : skeleton téléphone, pas de facture vide inventée", () => {
    const listOnly = {
      mission_id: 1,
      client_name: "Sonia BAUER",
      status: "assigned",
    } as Record<string, unknown>;
    const identity = buildIdentityFromMission(listOnly as CompanyDispatchMission);
    const rows = buildRideDetailInfoRows(listOnly, identity, {
      statusLabel: "Assignée",
      scheduledIso: null,
      driverDisplay: "Karim",
      billingSummary: buildRideBillingSummary(listOnly, null),
      awaitingServer: true,
    });
    expect(rows.find((r) => r.label === "Téléphone")?.pending).toBe(true);
    expect(rows.some((r) => r.label === "Facturation")).toBe(false);
  });

  it("construit l'historique opérationnel", () => {
    const timeline = buildRideTimeline(baseMission, "Emmenez Moi");
    expect(timeline[0]?.event).toContain("terminée");
    expect(timeline.some((e) => e.event.includes("Prise en charge"))).toBe(true);
  });
});
