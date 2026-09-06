export type CockpitConnectivityCopy = {
  title: string;
  body: string;
};

const TEMPS_REEL: CockpitConnectivityCopy = {
  title: "Temps réel indisponible",
  body: "Données issues du dernier chargement. Vérifiez la connexion réseau.",
};

const AUCUN_GPS: CockpitConnectivityCopy = {
  title: "Aucun GPS récent",
  body: "Aucune position fraîche. Vérifiez l'app chauffeur et le GPS.",
};

/**
 * Un seul bandeau sous la barre ops — même sémantique que le bandeau carte,
 * sans second bloc « Temps réel » qui se superpose.
 */
export function resolveCockpitConnectivityBanner(input: {
  showNoGps: boolean;
  socketConnected: boolean;
  realtimeOffline: boolean;
}): CockpitConnectivityCopy | null {
  if (input.showNoGps) {
    return input.socketConnected ? AUCUN_GPS : TEMPS_REEL;
  }
  if (input.realtimeOffline) {
    return TEMPS_REEL;
  }
  return null;
}
