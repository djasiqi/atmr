import type { CompanyDataFreshness } from "./companyRealtimeState";

/** Sous-titre discret pour la pill LIVE quand le socket est OK mais peu d’événements métier. */
export function formatCompanyActivityHint(
  lastEventAt: string | null,
  dataFreshness: CompanyDataFreshness
): string | null {
  if (dataFreshness === "fresh") return null;
  if (!lastEventAt) return "Aucune activité récente";

  const ageMs = Date.now() - Date.parse(lastEventAt);
  if (!Number.isFinite(ageMs) || ageMs < 0) return "Aucune activité récente";

  const minutes = Math.max(1, Math.round(ageMs / 60_000));
  if (dataFreshness === "idle") {
    if (minutes < 2) return "Activité calme";
    return `Dernière activité il y a ${minutes} min`;
  }
  return `Données anciennes · ${minutes} min`;
}
