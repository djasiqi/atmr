import { describe, expect, it } from "@jest/globals";
import type { CompanyDispatchMission } from "../api/contracts";
import { resolveRideDetailView } from "./resolveRideDetailView";
import { buildRideBillingSummary, buildRideDetailInfoRows } from "./companyRideDetailPresentation";
import { buildIdentityFromMission } from "./bookingIdentity";

const snapshot: CompanyDispatchMission = {
  mission_id: 42,
  status: "assigned",
  client_name: "Sonia BAUER",
  scheduled_at: "2026-09-05T15:30:00+02:00",
  pickup_label: "Gare",
  dropoff_label: "Hôpital",
  driver_name: "Karim",
  driver_id: 7,
};

describe("resolveRideDetailView", () => {
  it("affiche le snapshot tant que le GET n’est pas là", () => {
    const view = resolveRideDetailView({ serverData: null, snapshot });
    expect(view.source).toBe("snapshot");
    expect(view.awaitingServer).toBe(true);
    expect(view.data?.client_name).toBe("Sonia BAUER");
    expect(view.data?.driver_name).toBe("Karim");
  });

  it("le serveur remplace immédiatement une valeur obsolète (chauffeur)", () => {
    const view = resolveRideDetailView({
      serverData: { mission_id: 42, driver_name: "Léa", status: "assigned" },
      snapshot,
    });
    expect(view.source).toBe("server");
    expect(view.awaitingServer).toBe(false);
    expect(view.data?.driver_name).toBe("Léa");
  });

  it("n’invente pas Téléphone / facture vides depuis le snapshot", () => {
    const view = resolveRideDetailView({ serverData: null, snapshot });
    const data = view.data as Record<string, unknown>;
    const identity = buildIdentityFromMission(snapshot);
    const rows = buildRideDetailInfoRows(data, identity, {
      statusLabel: "Assignée",
      scheduledIso: snapshot.scheduled_at ?? null,
      driverDisplay: "Karim",
      billingSummary: buildRideBillingSummary(data, null),
      awaitingServer: true,
    });
    expect(rows.some((row) => row.label === "Téléphone" && row.pending)).toBe(true);
    expect(rows.some((row) => row.label === "Téléphone" && row.value === "—")).toBe(false);
    expect(rows.some((row) => row.label === "Facturation")).toBe(false);
  });
});
