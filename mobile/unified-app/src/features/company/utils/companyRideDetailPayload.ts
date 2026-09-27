/** Extrait le booking ouvert depuis la réponse détail mobile.

Le hook ne lit que `summary`. Les champs mission doivent y être, pas sur l'enveloppe.
*/
export function unwrapCompanyRideDetail(
  detailPayload: unknown,
  rideId: number
): Record<string, unknown> | null {
  if (!detailPayload || typeof detailPayload !== "object") return null;
  const payload = detailPayload as Record<string, unknown>;
  const directCandidate =
    payload.summary ?? payload.data ?? payload.item ?? payload.ride ?? payload.mission;
  if (directCandidate && typeof directCandidate === "object") {
    const directMission = directCandidate as Record<string, unknown>;
    const directMissionId = Number.parseInt(
      String(directMission.mission_id ?? directMission.booking_id ?? directMission.id ?? "NaN"),
      10
    );
    if (Number.isFinite(directMissionId) && directMissionId === rideId) {
      return directMission;
    }
  }
  const rowsCandidate =
    (Array.isArray(payload.items) && payload.items) ||
    (Array.isArray(payload.missions) && payload.missions) ||
    (Array.isArray(payload.data) && payload.data) ||
    [];
  if (Array.isArray(rowsCandidate)) {
    const match = rowsCandidate.find((entry) => {
      if (!entry || typeof entry !== "object") return false;
      const row = entry as Record<string, unknown>;
      const missionId = Number.parseInt(
        String(row.mission_id ?? row.booking_id ?? row.id ?? "NaN"),
        10
      );
      return Number.isFinite(missionId) && missionId === rideId;
    });
    if (match && typeof match === "object") return match as Record<string, unknown>;
  }
  return null;
}
