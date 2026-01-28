import dayjs, { Dayjs } from "dayjs";

/**
 * Calcule l'heure "urgent" = now + minutes (aligné backend now_local Europe/Zurich).
 * Utilisé pour affichage (toast) et tests. L'API /urgent envoie extra_delay_minutes,
 * c'est le backend qui fait now_local + delta.
 *
 * @param now - Référence "maintenant" (dayjs)
 * @param minutes - Délai en minutes (défaut 15)
 * @returns ISO sans Z (YYYY-MM-DDTHH:mm:ss), interprété local par le backend
 */
export function computeUrgentDatetime(
  now: Dayjs = dayjs(),
  minutes: number = 15
): string {
  return now.add(minutes, "minute").format("YYYY-MM-DDTHH:mm:ss");
}

/**
 * Valeur sentinelle 00:00 : pickup_at = 00:00:00 = "heure non définie".
 * pickup_at != 00:00 = "course déjà planifiée". Urgent autorisé uniquement si sentinelle.
 * Aucun parsing ne doit transformer 00:00 en null ; aucun affichage ne doit
 * montrer 00:00 comme une vraie heure utilisateur.
 *
 * On vérifie la partie temps dans la chaîne (T00:00:00) pour éviter les effets
 * de timezone : "2026-01-28T00:00:00.000Z" reste sentinelle (UTC minuit).
 *
 * @param pickupAt - ISO string ou null/undefined (pickup_at / scheduled_time)
 * @returns true si pas d'heure définie (null/undefined ou time 00:00:00)
 */
export function isPickupSentinel(
  pickupAt: string | null | undefined
): boolean {
  if (pickupAt == null || pickupAt === "") return true;
  const d = dayjs(pickupAt);
  if (!d.isValid()) return true;
  const m = pickupAt.match(/T(\d{2}):(\d{2}):(\d{2})/);
  if (m && m[1] === "00" && m[2] === "00" && m[3] === "00") return true;
  return false;
}
