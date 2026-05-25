import { E } from "../theme/enterpriseOpsTheme";

/** Couleur pastille flux WS (chip mode `EnterpriseHeader`). */
export function realtimeSocketDotColor(status: string): string {
  const s = status.toLowerCase().trim();
  if (s === "healthy") return E.BRAND;
  if (s === "connecting" || s === "reconnecting") return E.URGENT;
  return E.DANGER;
}

/** Libellé accessibilité pour la pastille flux. */
export function realtimeStatusA11yLabel(status: string): string {
  const s = status.toLowerCase().trim();
  if (s === "healthy") return "Flux temps réel connecté";
  if (s === "connecting") return "Flux temps réel connexion en cours";
  if (s === "reconnecting") return "Flux temps réel reconnexion";
  if (s === "reconnecting") return "Flux temps réel en reconnexion";
  if (s === "failed") return "Flux temps réel indisponible";
  if (s === "idle") return "Flux temps réel déconnecté";
  return `Flux temps réel ${status}`;
}
