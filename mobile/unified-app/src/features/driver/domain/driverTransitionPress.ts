/** Délai après une transition confirmée localement avant d'accepter l'étape suivante. */
export const TRANSITION_QUIET_MS = 2500;

export type TransitionPressInput = {
  target: string | null;
  pending: boolean;
  now: number;
  quietUntil: number;
};

/**
 * Mémorise la cible au début du geste.
 * Un appui commencé pendant une requête ou pendant le délai de silence est ignoré.
 */
export function beginTransitionPress(input: TransitionPressInput): string | null {
  if (input.pending || input.now < input.quietUntil || !input.target) return null;
  return input.target;
}

/**
 * Déclenche uniquement la cible capturée, si elle est encore l'action affichée.
 * Un rejeu ou un changement d'écran vers l'étape suivante ne part pas.
 */
export function commitTransitionPress(
  input: TransitionPressInput & { captured: string | null }
): { target: string | null; quietUntil: number } {
  if (
    !input.captured ||
    input.pending ||
    input.captured !== input.target ||
    input.now < input.quietUntil
  ) {
    return { target: null, quietUntil: input.quietUntil };
  }
  return { target: input.captured, quietUntil: input.now + TRANSITION_QUIET_MS };
}
