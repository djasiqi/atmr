/** Message utilisateur pour les échecs d'envoi hub (company + driver). */

export function mapHubSendError(error: unknown): string {
  if (error && typeof error === "object") {
    const status = (error as { response?: { status?: number } }).response?.status;
    if (status === 403) return "Envoi refusé (permissions).";
    if (status === 429) return "Trop de messages. Réessayez dans un instant.";
    if (status && status >= 500) return "Serveur indisponible. Réessayez.";
  }
  return "Échec de l'envoi. Réessayez.";
}
