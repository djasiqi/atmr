/**
 * Pipeline création + envoi LIRIE : ne jamais recréer si le brouillon existe.
 */

export const SEND_RETRY_TOAST = (
  "La demande a été enregistrée, mais l'envoi aux transporteurs a échoué. "
  + 'Réessayez — elle ne sera pas recréée.'
);

export function resolveCreatedRequestId(existingId, createResult) {
  const fromExisting = Number(existingId);
  if (Number.isFinite(fromExisting) && fromExisting > 0) return fromExisting;
  const fromCreate = Number(createResult?.id);
  if (Number.isFinite(fromCreate) && fromCreate > 0) return fromCreate;
  return null;
}

export function institutionSubmitBusy(phase, mutations = {}) {
  return Boolean(
    phase
    || mutations.createPending
    || mutations.sendPending
    || mutations.assignPending,
  );
}

export function institutionSubmitButtonLabel({
  phase,
  busy,
  isLirieSendMode,
  isDraftMode,
  hasCreatedRequest,
}) {
  if (isLirieSendMode && busy) return 'Envoi…';
  if (isLirieSendMode && hasCreatedRequest) return "Réessayer l'envoi LIRIE";
  if (isDraftMode) return 'Créer le brouillon';
  if (isLirieSendMode) return 'Envoyer aux transporteurs LIRIE';
  return 'Enregistrer';
}
