import type { SharedChatMessage } from "./types";

type DriverLike = { timestamp: string };

export function toEpoch(s: string | null | undefined): number {
  if (!s) return 0;
  const v = Date.parse(s);
  return Number.isFinite(v) ? v : 0;
}

function isCompanyChatInbound(message: SharedChatMessage): boolean {
  return message.senderRole?.toUpperCase() !== "COMPANY";
}

/**
 * Côté entreprise : premier message entrant (non l’équipe) compté « non lu »
 * d’après la même règle que le badge (contenu texte requis, ou pièce / vocal).
 * Retourne l’index dans `messages` (tri chrono **croissant**), ou -1.
 */
export function getCompanyFirstUnreadIndex(
  messages: SharedChatMessage[],
  lastReadAt: string | null
): number {
  const lastReadEpoch = toEpoch(lastReadAt);
  for (let i = 0; i < messages.length; i += 1) {
    const m = messages[i]!;
    if (!isCompanyChatInbound(m) || m.content?.trim() === "") continue;
    if (lastReadEpoch === 0) return i;
    if (toEpoch(m.timestamp) > lastReadEpoch) return i;
  }
  return -1;
}

/**
 * Côté chauffeur (ou toute file sans filtrage d’émetteur) : non lu = horodatage après last read.
 * Si aucune dernière lecture, le premier non lu est le message le plus **ancien** (index 0).
 */
export function getDriverFirstUnreadIndex<T extends DriverLike>(messages: T[], lastReadAt: string | null): number {
  if (messages.length === 0) return -1;
  if (!lastReadAt || !lastReadAt.trim()) return 0;
  const lr = toEpoch(lastReadAt);
  for (let i = 0; i < messages.length; i += 1) {
    if (toEpoch(messages[i]!.timestamp) > lr) return i;
  }
  return -1;
}

/**
 * Ancre de scroll : premier message entrant (company) / premier hors lastRead (driver) non lu,
 * ou bas de liste (dernier message) s’il n’y a aucun non lu.
 */
export function getChatListInitialScroll(
  options: { kind: "company"; messages: SharedChatMessage[]; lastReadAt: string | null } | { kind: "driver"; messages: DriverLike[]; lastReadAt: string | null }
): { type: "last" } | { type: "index"; index: number } {
  if (options.kind === "company") {
    const { messages, lastReadAt } = options;
    if (messages.length === 0) return { type: "last" };
    const i = getCompanyFirstUnreadIndex(messages, lastReadAt);
    return i >= 0 ? { type: "index", index: i } : { type: "last" };
  }
  const { messages, lastReadAt } = options;
  if (messages.length === 0) return { type: "last" };
  const i = getDriverFirstUnreadIndex(messages, lastReadAt);
  return i >= 0 ? { type: "index", index: i } : { type: "last" };
}
