import { Booking, ClientProfile } from "./types";

export function bookingBelongsToActiveClient(
  booking: Booking | null | undefined,
  profile: ClientProfile | null | undefined
): boolean {
  if (!booking || !profile) return true;
  const bookingClientId = booking.client?.id;
  const profileClientId = profile.id;
  if (typeof bookingClientId === "number" && typeof profileClientId === "number") {
    return bookingClientId === profileClientId;
  }
  return true;
}
