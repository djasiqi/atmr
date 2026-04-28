type SessionJournalEvent = {
  type: string;
  at: string;
  contextId?: string | null;
  payload?: Record<string, unknown>;
};

const STORAGE_KEY = "driver_session_journal_v1";
const MAX_EVENTS = 200;
let inMemoryEvents: SessionJournalEvent[] = [];

function clamp(events: SessionJournalEvent[]): SessionJournalEvent[] {
  if (events.length <= MAX_EVENTS) return events;
  return events.slice(events.length - MAX_EVENTS);
}

async function readStorage(): Promise<SessionJournalEvent[]> {
  try {
    const storage = await import("@react-native-async-storage/async-storage");
    const raw = await storage.default.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as SessionJournalEvent[]) : [];
  } catch {
    return [];
  }
}

async function writeStorage(events: SessionJournalEvent[]): Promise<void> {
  try {
    const storage = await import("@react-native-async-storage/async-storage");
    await storage.default.setItem(STORAGE_KEY, JSON.stringify(clamp(events)));
  } catch {
    // best effort
  }
}

export async function appendSessionJournalEvent(
  type: string,
  payload?: Record<string, unknown>,
  contextId?: string | null
): Promise<void> {
  const next: SessionJournalEvent = {
    type,
    at: new Date().toISOString(),
    contextId: contextId ?? null,
    payload: payload ?? undefined,
  };
  inMemoryEvents = clamp([...inMemoryEvents, next]);
  const persisted = await readStorage();
  await writeStorage([...persisted, next]);
}

export function getSessionJournalSnapshot(): SessionJournalEvent[] {
  return [...inMemoryEvents];
}

export async function hydrateSessionJournal(): Promise<void> {
  inMemoryEvents = clamp(await readStorage());
}

export async function clearSessionJournal(): Promise<void> {
  inMemoryEvents = [];
  try {
    const storage = await import("@react-native-async-storage/async-storage");
    await storage.default.removeItem(STORAGE_KEY);
  } catch {
    // best effort
  }
}

export function buildSessionDiagHeader(): string {
  const latest = inMemoryEvents[inMemoryEvents.length - 1];
  if (!latest) return "unified_driver_runtime:none";
  return `unified_driver_runtime:${latest.type}:${latest.at}`;
}
