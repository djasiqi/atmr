/** Codes d'erreur API chauffeur (alignés backend `constants/driver_api_errors.py`). */
export const BOOKING_ASSIGNED_TO_OTHER_DRIVER = "BOOKING_ASSIGNED_TO_OTHER_DRIVER";
export const BOOKING_COMPANY_FORBIDDEN = "BOOKING_COMPANY_FORBIDDEN";

export function isBookingAssignedToOtherDriver(
  body: unknown
): body is { code: string } {
  return (
    typeof body === "object" &&
    body !== null &&
    (body as { code?: string }).code === BOOKING_ASSIGNED_TO_OTHER_DRIVER
  );
}
