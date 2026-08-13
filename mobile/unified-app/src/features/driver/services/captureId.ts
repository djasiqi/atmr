/** Identité stable d'un fix GPS physique — distincte de l'ACK / location_event_id. */

export function createCaptureId(): string {
  const cryptoObj = globalThis.crypto as Crypto | undefined;
  if (cryptoObj && typeof cryptoObj.randomUUID === "function") {
    return cryptoObj.randomUUID();
  }
  const rand = Math.random().toString(36).slice(2, 12);
  return `cap_${Date.now().toString(36)}_${rand}`;
}
