import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";
import { enrichFleetDrivers } from "../../components/maps/fleetMapLogic";
import type { DashboardOpsFeedProps } from "../../components/dashboard/DashboardOpsFeed";

export type CockpitOperationalCountsInput = {
  missions: CompanyDispatchMission[];
  drivers: CompanyDriverLiveLocation[];
  opsFeed: Pick<DashboardOpsFeedProps, "delayedMission" | "stats">;
  organizationName?: string | null;
};

export type CockpitOperationalCounts = {
  delayedCount: number;
  urgentCount: number;
  criticalEtaCount: number;
};

function parseStatDelayed(opsFeed: CockpitOperationalCountsInput["opsFeed"]): number {
  const raw = opsFeed.stats.find((s) => s.key === "delayed")?.value ?? "0";
  const n = Number.parseInt(String(raw).replace(/\D/g, ""), 10);
  return Number.isFinite(n) ? n : 0;
}

/** Sépare urgent (incidents / urgence) et delayed (retards) — jamais aliasés. */
export function computeCockpitOperationalCounts(
  input: CockpitOperationalCountsInput
): CockpitOperationalCounts {
  const delayedFromStats = parseStatDelayed(input.opsFeed);
  const delayedCount =
    delayedFromStats > 0 ? delayedFromStats : input.opsFeed.delayedMission ? 1 : 0;

  const enriched = enrichFleetDrivers(
    input.drivers,
    input.missions,
    input.organizationName ?? null
  );

  let urgentCount = 0;
  for (const d of enriched) {
    if (d.enrichment.operationalStatus === "incident") {
      urgentCount += 1;
    }
  }

  for (const m of input.missions) {
    if (m.status === "completed" || m.status === "cancelled") continue;
    if (m.driver_type === "EMERGENCY") {
      urgentCount += 1;
    }
  }

  const criticalEtaCount = input.missions.filter(
    (m) =>
      m.status !== "completed" &&
      m.status !== "cancelled" &&
      (m.assignment_pickup_delay_minutes ?? 0) >= 15
  ).length;

  return { delayedCount, urgentCount, criticalEtaCount };
}
