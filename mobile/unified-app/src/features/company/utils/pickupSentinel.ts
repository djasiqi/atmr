import dayjs from "dayjs";

/** Heure « non définie » (T00:00:00) — aligné sur operations `isPickupSentinel`. */
export function isPickupSentinel(pickupAt: string | null | undefined): boolean {
  if (pickupAt == null || pickupAt === "") return true;
  const d = dayjs(pickupAt);
  if (!d.isValid()) return true;
  const m = pickupAt.match(/T(\d{2}):(\d{2}):(\d{2})/);
  if (m && m[1] === "00" && m[2] === "00" && m[3] === "00") return true;
  return false;
}
