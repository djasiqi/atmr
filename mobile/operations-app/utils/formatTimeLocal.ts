/**
 * Formatage heure/date en heure locale, sans affichage de fuseau horaire.
 * Utilise l'heure du périphérique (locale) pour l'affichage.
 * Backend ou mobile : on affiche toujours l'heure locale de l'utilisateur.
 */

/**
 * Heure au format HH:mm (ex: "08:30")
 */
export function formatTimeLocal(isoOrDate: string | Date): string {
  const d = typeof isoOrDate === "string" ? new Date(isoOrDate) : isoOrDate;
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/**
 * Date au format court (ex: "12.03.2026" ou selon locale)
 */
export function formatDateLocal(isoOrDate: string | Date): string {
  const d = typeof isoOrDate === "string" ? new Date(isoOrDate) : isoOrDate;
  return d.toLocaleDateString([], { day: "2-digit", month: "2-digit", year: "numeric" });
}

/**
 * Date + heure (ex: "12.03.2026 08:30")
 */
export function formatDateTimeLocal(isoOrDate: string | Date): string {
  const d = typeof isoOrDate === "string" ? new Date(isoOrDate) : isoOrDate;
  return `${formatDateLocal(d)} ${formatTimeLocal(d)}`;
}
