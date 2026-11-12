export type SignatureOffer = "wizard" | "already" | "blocked" | "document";

export const SIGNATURE_STEP_LABELS = [
  "Identité",
  "Tarifs",
  "Conditions",
  "Signature",
  "Document",
  "Transmission",
] as const;

export type SignatureStep = "identity" | "tariffs" | "conditions" | "signature" | "document" | "transmission";

/**
 * Décide si le chauffeur ouvre le parcours, le document ou un blocage.
 * Un mandat déjà applicable ne redemande pas de signature.
 */
export function signatureOffer(kind: string | undefined, status: string | undefined): SignatureOffer {
  if (kind === "applicable_demonstration" || status === "demo_valid_simulated") return "already";
  if (
    kind === "expired" ||
    kind === "amendment_required" ||
    kind === "other_company" ||
    kind === "not_concerned"
  ) {
    return "blocked";
  }
  if (status === "demo_specimen_signed" || kind === "specimen") return "document";
  return "wizard";
}

export function traceAcceptable(points: { x: number; y: number }[]): boolean {
  return (
    points.length > 0 &&
    points.length <= 400 &&
    points.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
  );
}

export function representativeComplete(name: string, quality: string, relation: string): boolean {
  return [name, quality, relation].every((value) => value.trim().length >= 2);
}

export type SignatureRequest = {
  trace: { x: number; y: number }[];
  signer: "patient" | "representative";
  representativeName?: string;
  representativeQuality?: string;
  representativeRelation?: string;
  tariffVersion?: string | null;
  versionNumber?: number | null;
  idempotencyKey: string;
};

/** Charge utile du parcours existant. La version affichée est celle qui sera signée. */
export function signatureRequest(input: SignatureRequest): Record<string, unknown> {
  const representative = input.signer === "representative";
  return {
    trace: input.trace,
    signer: input.signer,
    representative_name: representative ? (input.representativeName || "").trim() : null,
    representative_quality: representative ? (input.representativeQuality || "").trim() : null,
    representative_relation: representative ? (input.representativeRelation || "").trim() : null,
    acknowledged: true,
    tariff_version_presented: input.tariffVersion ?? null,
    version_number: input.versionNumber ?? null,
    idempotency_key: input.idempotencyKey,
  };
}
