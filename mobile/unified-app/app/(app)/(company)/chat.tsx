import { Redirect } from "expo-router";

/** Rétrocompatibilité : l’onglet « Chat » ouvre le hub Messages entreprise. */
export default function CompanyChatRedirectScreen() {
  return <Redirect href="/(app)/(company)/messages" />;
}
