/**
 * ✅ P0-1: Utilitaires pour normaliser les statuts de course.
 * 
 * Le backend utilise des statuts en UPPERCASE (EN_ROUTE, IN_PROGRESS, etc.)
 * Cette fonction normalise les statuts pour correspondre au backend.
 */

export type BookingStatus =
  | "PENDING"
  | "ASSIGNED"
  | "EN_ROUTE"
  | "IN_PROGRESS"
  | "COMPLETED"
  | "RETURN_COMPLETED"
  | "CANCELED";

/**
 * Normalise un statut de course en uppercase pour correspondre au backend.
 * Gère la compatibilité avec les anciennes valeurs en lowercase.
 */
export function normalizeBookingStatus(
  status: string | undefined | null
): BookingStatus | string {
  if (!status) return status || "";
  const upper = status.toUpperCase();
  // Vérifier si c'est un statut valide
  const validStatuses: BookingStatus[] = [
    "PENDING",
    "ASSIGNED",
    "EN_ROUTE",
    "IN_PROGRESS",
    "COMPLETED",
    "RETURN_COMPLETED",
    "CANCELED",
  ];
  if (validStatuses.includes(upper as BookingStatus)) {
    return upper as BookingStatus;
  }
  // Retourner tel quel si ce n'est pas un statut de course (ex: "assigned", "pending")
  return status;
}

/**
 * Vérifie si un statut correspond à une course complétée.
 */
export function isCompletedStatus(status: string | undefined | null): boolean {
  const normalized = normalizeBookingStatus(status);
  return normalized === "COMPLETED" || normalized === "RETURN_COMPLETED";
}

/**
 * Vérifie si un statut correspond à une course annulée.
 */
export function isCanceledStatus(status: string | undefined | null): boolean {
  const normalized = normalizeBookingStatus(status);
  return normalized === "CANCELED";
}

