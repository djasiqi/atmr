export const CLIENT_SURFACE_CONTRACT_VERSIONS = {
  statusDictionaryVersion: "1.0.0",
  pricingContractVersion: "1.0.0",
  canonicalAddressContractVersion: "1.0.0",
  previewContractVersion: "1.0.0",
  missionStatusVersion: "1.0.0",
  missionSnapshotVersion: "1.0.0",
  driverSocketContractVersion: "1.0.0",
  driverTrackingContractVersion: "1.0.0",
} as const;

export function logContractMismatchEvent(
  contract:
    | "status"
    | "pricing"
    | "canonical_address"
    | "preview"
    | "mission_status"
    | "mission_snapshot"
    | "driver_socket"
    | "driver_tracking",
  expected: string,
  received: string | undefined
) {
  console.warn("[status_dictionary_mismatch_event]", {
    contract,
    expected,
    received: received ?? null,
  });
}

