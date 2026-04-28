import { Redirect } from "expo-router";

/**
 * Ancienne étape 2 (coordonnées) : fusionnée dans `step-1.tsx` pour raccourcir le parcours invité.
 * Redirection conservée pour les liens / historiques profonds.
 */
export default function PublicPreRequestStepTwoRedirect() {
  return <Redirect href="/(public)/pre-request/step-1" />;
}
