import { apiClient } from "../../../core/api/client";
import { emitPerfKpi } from "../../../core/observability/perfKpi";
import type { EmergencyIssueType, HubChatMessage, MessageHubThread } from "./types";

function normalizeHubMessage(entry: unknown): HubChatMessage | null {
  if (!entry || typeof entry !== "object") return null;
  const raw = entry as Record<string, unknown>;
  const timestamp =
    typeof raw.timestamp === "string" && raw.timestamp.length > 0
      ? raw.timestamp
      : new Date().toISOString();
  const id = raw.id;
  return {
    id: typeof id === "string" || typeof id === "number" ? id : `${timestamp}`,
    sender_id:
      typeof raw.sender_id === "string" || typeof raw.sender_id === "number"
        ? raw.sender_id
        : null,
    content: typeof raw.content === "string" ? raw.content : "",
    sender_role: typeof raw.sender_role === "string" ? raw.sender_role : undefined,
    sender_name: typeof raw.sender_name === "string" ? raw.sender_name : null,
    timestamp,
    image_url: typeof raw.image_url === "string" ? raw.image_url : null,
    pdf_url: typeof raw.pdf_url === "string" ? raw.pdf_url : null,
    pdf_filename: typeof raw.pdf_filename === "string" ? raw.pdf_filename : null,
    audio_url: typeof raw.audio_url === "string" ? raw.audio_url : null,
    thread_id: typeof raw.thread_id === "string" ? raw.thread_id : null,
    conversation_id:
      typeof raw.conversation_id === "number"
        ? raw.conversation_id
        : typeof raw.conversation_id === "string"
          ? Number.parseInt(raw.conversation_id, 10) || null
          : null,
    receiver_id:
      typeof raw.receiver_id === "number" || typeof raw.receiver_id === "string"
        ? raw.receiver_id
        : null,
    booking_id: typeof raw.booking_id === "number" ? raw.booking_id : null,
    message_type: typeof raw.message_type === "string" ? raw.message_type : undefined,
    priority:
      raw.priority === "urgent" || raw.priority === "important" || raw.priority === "normal"
        ? raw.priority
        : "normal",
    is_read: raw.is_read === true,
    acked_at: typeof raw.acked_at === "string" ? raw.acked_at : null,
  };
}

export type HubColleagueRow = {
  peer_user_id: number;
  driver_id?: number | null;
  title: string;
  thread_id: string;
};

export type HubColleaguesResponse = {
  colleagues: HubColleagueRow[];
  team_member_count: number | null;
};

const RESOLVE_CONVERSATION_TTL_MS = 60_000;
const resolveConversationCache = new Map<string, { conversationId: number | null; updatedAtMs: number }>();

export async function fetchHubColleagues(companyId: number): Promise<HubColleaguesResponse> {
  const { data } = await apiClient.get(`/messages/${companyId}/hub/colleagues`);
  const payload = (data ?? {}) as Record<string, unknown>;
  const rows = Array.isArray(payload.colleagues) ? payload.colleagues : [];
  const rawCount = payload.team_member_count;
  const team_member_count =
    typeof rawCount === "number" && Number.isFinite(rawCount)
      ? Math.max(0, Math.round(rawCount))
      : null;
  const colleagues = rows
    .map((row) => {
      if (!row || typeof row !== "object") return null;
      const raw = row as Record<string, unknown>;
      const peer =
        typeof raw.peer_user_id === "number"
          ? raw.peer_user_id
          : Number.parseInt(String(raw.peer_user_id ?? ""), 10);
      if (!Number.isFinite(peer)) return null;
      const title = typeof raw.title === "string" ? raw.title : `Collègue #${peer}`;
      const thread_id =
        typeof raw.thread_id === "string" ? raw.thread_id : `direct:${peer}`;
      return {
        peer_user_id: peer,
        driver_id: typeof raw.driver_id === "number" ? raw.driver_id : null,
        title,
        thread_id,
      } satisfies HubColleagueRow;
    })
    .filter((r): r is HubColleagueRow => r != null);
  return { colleagues, team_member_count };
}

function mapInboxRow(row: Record<string, unknown>): MessageHubThread | null {
  const threadId =
    typeof row.thread_id === "string"
      ? row.thread_id
      : row.conversation_id != null
        ? String(row.conversation_id)
        : null;
  if (!threadId) return null;
  const sectionRaw = typeof row.section === "string" ? row.section : "dispatch";
  let section = sectionRaw;
  if (section === "company") section = "dispatch";
  if (section === "groups") section = "team";
  return {
    thread_id: threadId,
    conversation_id:
      typeof row.conversation_id === "number" ? row.conversation_id : null,
    section,
    title: typeof row.title === "string" ? row.title : "Conversation",
    subtitle: typeof row.subtitle === "string" ? row.subtitle : null,
    booking_id: typeof row.booking_id === "number" ? row.booking_id : null,
    status: typeof row.status === "string" ? row.status : null,
    unread_count: typeof row.unread_count === "number" ? row.unread_count : 0,
    priority:
      row.priority === "urgent" || row.priority === "important"
        ? row.priority
        : "normal",
    last_message_preview:
      typeof row.last_message_preview === "string" ? row.last_message_preview : null,
    last_message_at:
      typeof row.last_message_at === "string" ? row.last_message_at : null,
    last_message_from_self: row.last_message_from_self === true,
  };
}

export async function fetchConversationsInbox(): Promise<{
  threads: MessageHubThread[];
  unread_total: number;
  sections?: Record<string, MessageHubThread[]>;
}> {
  const { data } = await apiClient.get("/conversations/inbox");
  const payload = (data ?? {}) as Record<string, unknown>;
  const sectionsRaw = payload.sections as Record<string, unknown[]> | undefined;
  const threads: MessageHubThread[] = [];
  if (sectionsRaw && typeof sectionsRaw === "object") {
    Object.values(sectionsRaw).forEach((list) => {
      if (!Array.isArray(list)) return;
      list.forEach((row) => {
        const mapped = mapInboxRow(row as Record<string, unknown>);
        if (mapped) threads.push(mapped);
      });
    });
  }
  const unread_total =
    typeof payload.unread_total === "number" ? payload.unread_total : 0;
  return { threads, unread_total, sections: sectionsRaw as Record<string, MessageHubThread[]> };
}

export async function resolveConversationId(
  companyId: number,
  threadId: string
): Promise<number | null> {
  const cacheKey = `${companyId}:${threadId}`;
  const cached = resolveConversationCache.get(cacheKey);
  if (cached && Date.now() - cached.updatedAtMs < RESOLVE_CONVERSATION_TTL_MS) {
    return cached.conversationId;
  }
  try {
    const { data } = await apiClient.get("/conversations/resolve", {
      params: { thread_id: threadId },
    });
    const payload = (data ?? {}) as Record<string, unknown>;
    const conversationId = typeof payload.conversation_id === "number" ? payload.conversation_id : null;
    resolveConversationCache.set(cacheKey, {
      conversationId,
      updatedAtMs: Date.now(),
    });
    return conversationId;
  } catch {
    resolveConversationCache.set(cacheKey, {
      conversationId: null,
      updatedAtMs: Date.now(),
    });
    return null;
  }
}

export async function fetchMessageHubThreads(companyId: number): Promise<{
  threads: MessageHubThread[];
  unread_total: number;
}> {
  try {
    const inbox = await fetchConversationsInbox();
    if (inbox.threads.length > 0) {
      return { threads: inbox.threads, unread_total: inbox.unread_total };
    }
  } catch {
    // fallback hub
  }
  const { data } = await apiClient.get(`/messages/${companyId}/hub/threads`);
  const payload = (data ?? {}) as Record<string, unknown>;
  const rawThreads = Array.isArray(payload.threads) ? payload.threads : [];
  const threads = rawThreads
    .map((row) =>
      typeof row === "object" && row
        ? mapInboxRow(row as Record<string, unknown>)
        : null
    )
    .filter((t): t is MessageHubThread => t != null);
  const unread_total =
    typeof payload.unread_total === "number" ? payload.unread_total : 0;
  return { threads, unread_total };
}

export async function fetchConversationMessages(
  conversationId: number,
  options?: { before?: string; limit?: number }
): Promise<HubChatMessage[]> {
  const { data } = await apiClient.get(
    `/conversations/${conversationId}/messages`,
    { params: { before: options?.before, limit: options?.limit ?? 40 } }
  );
  const payload = (data ?? {}) as Record<string, unknown>;
  const rows = Array.isArray(payload.messages) ? payload.messages : [];
  const normalizeStartedAt = Date.now();
  const normalized = rows.map(normalizeHubMessage).filter((m): m is HubChatMessage => m != null);
  emitPerfKpi("perf.thread_runtime", {
    source: "thread.messages.normalize",
    phase: "perf.thread.normalize_done",
    path: "conversation",
    conversation_id: conversationId,
    row_count: rows.length,
    message_count: normalized.length,
    duration_ms: Date.now() - normalizeStartedAt,
  });
  return normalized;
}

export async function fetchThreadMessages(
  companyId: number,
  threadId: string,
  options?: { before?: string; limit?: number }
): Promise<HubChatMessage[]> {
  try {
    const { data } = await apiClient.get(
      `/messages/${companyId}/hub/threads/${encodeURIComponent(threadId)}/messages`,
      { params: { before: options?.before, limit: options?.limit ?? 40 } }
    );
    const payload = (data ?? {}) as Record<string, unknown>;
    const rows = Array.isArray(payload.messages) ? payload.messages : [];
    const normalizeStartedAt = Date.now();
    const normalized = rows.map(normalizeHubMessage).filter((m): m is HubChatMessage => m != null);
    emitPerfKpi("perf.thread_runtime", {
      source: "thread.messages.normalize",
      phase: "perf.thread.normalize_done",
      path: "hub_thread",
      company_id: companyId,
      thread_id: threadId,
      row_count: rows.length,
      message_count: normalized.length,
      duration_ms: Date.now() - normalizeStartedAt,
    });
    return normalized;
  } catch (error: unknown) {
    const status = (error as { response?: { status?: number } })?.response?.status;
    if (status === 403 || status === 404) {
      return [];
    }
    throw error;
  }
}

export async function sendHubMessage(
  companyId: number,
  threadId: string,
  body: Record<string, unknown>
): Promise<HubChatMessage> {
  const { data } = await apiClient.post(
    `/messages/${companyId}/hub/threads/${encodeURIComponent(threadId)}/messages`,
    body
  );
  const payload = (data ?? {}) as Record<string, unknown>;
  const raw =
    payload.message && typeof payload.message === "object"
      ? payload.message
      : payload;
  const normalized = normalizeHubMessage(raw);
  if (!normalized) {
    throw new Error("Réponse serveur invalide après envoi du message.");
  }
  return normalized;
}

export async function markThreadRead(companyId: number, threadId: string): Promise<void> {
  try {
    await apiClient.post(`/messages/${companyId}/hub/read`, { thread_id: threadId });
  } catch {
    /* Non bloquant : permissions, CSRF ou company_id transitoire au switch contexte */
  }
}

export async function ackHubMessage(companyId: number, messageId: number): Promise<void> {
  try {
    await apiClient.post(`/messages/${companyId}/hub/ack/${messageId}`);
  } catch {
    /* Ack optionnel — ne pas faire échouer le fil */
  }
}

export async function fetchHubUnreadCount(companyId: number): Promise<number> {
  const { data } = await apiClient.get(`/messages/${companyId}/hub/unread-count`);
  const payload = (data ?? {}) as Record<string, unknown>;
  return typeof payload.unread_count === "number" ? payload.unread_count : 0;
}

export async function reportMessageHubEmergency(body: {
  issue_type: EmergencyIssueType;
  booking_id?: number | null;
  latitude?: number | null;
  longitude?: number | null;
  note?: string | null;
}): Promise<Record<string, unknown>> {
  const { data } = await apiClient.post("/conversations/emergency", body);
  return (data ?? {}) as Record<string, unknown>;
}

export async function searchMessageHub(
  companyId: number,
  query: string
): Promise<HubChatMessage[]> {
  const { data } = await apiClient.get(`/messages/${companyId}`, {
    params: { limit: 100 },
  });
  const rows = Array.isArray(data) ? data : [];
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return rows
    .map(normalizeHubMessage)
    .filter((m): m is HubChatMessage => {
      if (!m) return false;
      return (
        m.content.toLowerCase().includes(q) ||
        (m.sender_name ?? "").toLowerCase().includes(q)
      );
    });
}
