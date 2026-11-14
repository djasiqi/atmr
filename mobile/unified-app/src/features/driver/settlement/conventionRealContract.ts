/**
 * Contrat mobile — convention Client réelle pendant un transport.
 * Signature à l'écran ≠ certification juridique. Différé réel fermé.
 */

export type ConventionRealStatus =
  | "draft"
  | "awaiting_signature"
  | "signed_pending_validation"
  | "applicable"
  | "expired"
  | "revoked"
  | "replaced";

export type ConventionSignPayload = {
  conventionId: number;
  acknowledged: true;
  signerKind: "client" | "representative";
  signerName: string;
  documentSha256Presented: string;
  trace: Array<{ x: number; y: number }>;
  idempotencyKey: string;
};

export function canCollectSignatureDuringTransport(status: ConventionRealStatus | null): boolean {
  return status === "awaiting_signature";
}

export function deferredBillingAuthorizedFromGate(gate: {
  authorizes_deferred_billing?: boolean;
  legally_effective?: boolean;
}): boolean {
  return gate.authorizes_deferred_billing === true && gate.legally_effective === true;
}

export function buildSignPayload(input: {
  conventionId: number;
  signerKind: "client" | "representative";
  signerName: string;
  documentSha256: string;
  trace: Array<{ x: number; y: number }>;
  idempotencyKey: string;
}): ConventionSignPayload | { error: string } {
  if (input.signerName.trim().length < 2) {
    return { error: "signataire_nom_absent" };
  }
  if (!input.documentSha256 || input.documentSha256.length < 16) {
    return { error: "version_presentee_differente" };
  }
  if (!input.trace.length) {
    return { error: "trace_vide" };
  }
  if (input.idempotencyKey.trim().length < 8) {
    return { error: "cle_absente" };
  }
  return {
    conventionId: input.conventionId,
    acknowledged: true,
    signerKind: input.signerKind,
    signerName: input.signerName.trim(),
    documentSha256Presented: input.documentSha256,
    trace: input.trace,
    idempotencyKey: input.idempotencyKey.trim(),
  };
}
