import type { HubChatMessage } from "./types";

/** Identifiant client (optimistic) pour réconcilier avec l'id serveur. */
export function resolveHubMessageLocalId(message: HubChatMessage): string | null {
  const localId = message._localId;
  if (typeof localId === "string" && localId.length > 0) return localId;
  const id = String(message.id);
  if (id.startsWith("local-")) return id;
  return null;
}

function isPendingLocalMessage(message: HubChatMessage): boolean {
  return String(message.id).startsWith("local-");
}

function sortByTimestamp(messages: HubChatMessage[]): HubChatMessage[] {
  return [...messages].sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));
}

/** Ajoute ou remplace un message en retirant l'optimistic associé (_localId). */
export function upsertHubChatMessage(
  previous: HubChatMessage[],
  incoming: HubChatMessage,
  options?: { replaceLocalId?: string | null }
): HubChatMessage[] {
  const replaceLocalId =
    options?.replaceLocalId ?? resolveHubMessageLocalId(incoming) ?? null;

  const filtered = replaceLocalId
    ? previous.filter((message) => {
        const messageLocalId = resolveHubMessageLocalId(message);
        return messageLocalId !== replaceLocalId && String(message.id) !== replaceLocalId;
      })
    : previous;

  const map = new Map(filtered.map((message) => [String(message.id), message]));
  map.set(String(incoming.id), incoming);
  return sortByTimestamp([...map.values()]);
}

/** Fusionne plusieurs listes (HTTP + live) en dédoublonnant optimistic ↔ serveur. */
export function mergeHubChatMessageLists(...lists: HubChatMessage[][]): HubChatMessage[] {
  const result = new Map<string, HubChatMessage>();
  const localIdToResultKey = new Map<string, string>();

  for (const message of lists.flat()) {
    const localId = resolveHubMessageLocalId(message);
    const idKey = String(message.id);
    const pending = isPendingLocalMessage(message);

    if (localId && !pending) {
      const pendingKey = localIdToResultKey.get(localId);
      if (pendingKey) {
        result.delete(pendingKey);
      }
      result.set(idKey, message);
      localIdToResultKey.set(localId, idKey);
      continue;
    }

    if (localId && pending) {
      const existingKey = localIdToResultKey.get(localId);
      if (existingKey && !existingKey.startsWith("local-")) {
        continue;
      }
      result.set(idKey, message);
      localIdToResultKey.set(localId, idKey);
      continue;
    }

    result.set(idKey, message);
  }

  return sortByTimestamp([...result.values()]);
}
