import type { CompanyDispatchMission } from "../api/contracts";

export type RideDetailViewSource = "server" | "snapshot" | "none";

export type RideDetailView = {
  source: RideDetailViewSource;
  data: Record<string, unknown> | null;
  awaitingServer: boolean;
};

/**
 * Le serveur est autoritaire dès qu’il est là.
 * Le snapshot Courses n’est qu’un affichage temporaire — jamais une source de vérité.
 */
export function resolveRideDetailView(args: {
  serverData: Record<string, unknown> | null | undefined;
  snapshot: CompanyDispatchMission | null;
}): RideDetailView {
  if (args.serverData) {
    return { source: "server", data: args.serverData, awaitingServer: false };
  }
  if (args.snapshot) {
    return {
      source: "snapshot",
      data: { ...args.snapshot } as Record<string, unknown>,
      awaitingServer: true,
    };
  }
  return { source: "none", data: null, awaitingServer: true };
}
