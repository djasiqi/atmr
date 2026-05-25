import { resolveMediaUrl } from "../../../core/api/mediaUrl";
import type { HubChatMessage } from "../../driver/messages/types";

export function mapCompanyRealtimeToHubMessage(event: unknown): HubChatMessage | null {
  if (!event || typeof event !== "object") return null;
  const payload = event as Record<string, unknown>;
  const messageCandidate =
    (payload.message as Record<string, unknown> | undefined) ??
    (payload.data as Record<string, unknown> | undefined) ??
    payload;
  if (!messageCandidate || typeof messageCandidate !== "object") return null;
  const raw = messageCandidate as Record<string, unknown>;
  const content = typeof raw.content === "string" ? raw.content : "";
  const imageUrl = resolveMediaUrl(
    typeof raw.image_url === "string"
      ? raw.image_url
      : typeof raw.image === "string"
        ? raw.image
        : null
  );
  const pdfUrl = resolveMediaUrl(
    typeof raw.pdf_url === "string" ? raw.pdf_url : typeof raw.pdf === "string" ? raw.pdf : null
  );
  const audioUrl = resolveMediaUrl(
    typeof raw.audio_url === "string"
      ? raw.audio_url
      : typeof raw.audio === "string"
        ? raw.audio
        : null
  );
  if (!content.trim() && !imageUrl && !pdfUrl && !audioUrl) return null;

  const timestamp =
    (typeof raw.created_at === "string" && raw.created_at) ||
    (typeof raw.timestamp === "string" && raw.timestamp) ||
    new Date().toISOString();
  const messageIdCandidate = raw.id ?? payload.event_id ?? payload.message_id;
  const id =
    typeof messageIdCandidate === "number" || typeof messageIdCandidate === "string"
      ? messageIdCandidate
      : `${timestamp}-${Math.random().toString(36).slice(2, 8)}`;

  const threadFromPayload =
    typeof raw.thread_id === "string"
      ? raw.thread_id
      : typeof payload.thread_id === "string"
        ? payload.thread_id
        : null;
  const bookingId =
    typeof raw.booking_id === "number"
      ? raw.booking_id
      : typeof payload.booking_id === "number"
        ? payload.booking_id
        : null;

  return {
    id,
    sender_id:
      typeof raw.sender_id === "number" || typeof raw.sender_id === "string"
        ? raw.sender_id
        : null,
    content,
    sender_role:
      (typeof raw.sender_type === "string" && raw.sender_type) ||
      (typeof raw.sender_role === "string" && raw.sender_role) ||
      undefined,
    sender_name:
      (typeof raw.sender_label === "string" && raw.sender_label) ||
      (typeof raw.sender_name === "string" && raw.sender_name) ||
      null,
    timestamp,
    image_url: imageUrl,
    pdf_url: pdfUrl,
    pdf_filename: typeof raw.pdf_filename === "string" ? raw.pdf_filename : null,
    audio_url: audioUrl,
    thread_id:
      threadFromPayload ??
      (bookingId != null ? `mission:${bookingId}` : null),
    booking_id: bookingId,
    conversation_id:
      typeof raw.conversation_id === "number"
        ? raw.conversation_id
        : typeof raw.conversation_id === "string"
          ? Number.parseInt(raw.conversation_id, 10)
          : null,
    message_type: typeof raw.message_type === "string" ? raw.message_type : "text",
    priority:
      raw.priority === "urgent" || raw.priority === "important" ? raw.priority : "normal",
    _localId:
      typeof raw._localId === "string" && raw._localId.length > 0
        ? raw._localId
        : typeof raw.client_message_id === "string" && raw.client_message_id.length > 0
          ? raw.client_message_id
          : null,
  };
}
