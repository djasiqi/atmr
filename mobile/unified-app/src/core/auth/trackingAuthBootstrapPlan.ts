/**
 * Plan d'hydratation tracking auth au bootstrap (cold start).
 * Sépare « lease déjà driver_active » de « mémoire tracking déjà hydratée ».
 */
import type { TrackingContextLease } from "../../features/driver/services/trackingContextLease";

export type TrackingAuthBootstrapPlan =
  | {
      action: "republish_from_lease";
      driverId: number;
      trackingIdentityId: string;
      sessionGenerationId: number;
    }
  | { action: "acquire_and_publish"; driverId: number }
  | {
      action: "none";
      reason:
        | "not_driver_context"
        | "not_authenticated"
        | "driver_id_unresolvable"
        | "lease_driver_mismatch";
    };

/**
 * Décide si le bootstrap doit republier / acquérir l'auth tracking.
 * Ne mute rien — exécution côté sessionProvider.
 */
export function planTrackingAuthPublishOnBootstrap(input: {
  isAuthenticated: boolean;
  contextType: string | null | undefined;
  contextDriverId: number | null;
  lease: TrackingContextLease;
}): TrackingAuthBootstrapPlan {
  if (!input.isAuthenticated) {
    return { action: "none", reason: "not_authenticated" };
  }
  if (input.contextType !== "driver") {
    return { action: "none", reason: "not_driver_context" };
  }
  if (
    input.contextDriverId == null ||
    !Number.isFinite(input.contextDriverId)
  ) {
    return { action: "none", reason: "driver_id_unresolvable" };
  }

  if (input.lease.state === "driver_active") {
    if (input.lease.driverId !== input.contextDriverId) {
      return { action: "none", reason: "lease_driver_mismatch" };
    }
    return {
      action: "republish_from_lease",
      driverId: input.lease.driverId,
      trackingIdentityId: input.lease.trackingIdentityId,
      sessionGenerationId: input.lease.sessionGenerationId,
    };
  }

  return { action: "acquire_and_publish", driverId: input.contextDriverId };
}
