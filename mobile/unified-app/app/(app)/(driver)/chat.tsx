import { Redirect } from "expo-router";

/** Rétrocompatibilité : l’onglet historique « chat » redirige vers le hub Messages. */
export default function DriverChatRedirectScreen() {
  return <Redirect href="/(app)/(driver)/messages" />;
}
