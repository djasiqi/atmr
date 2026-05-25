/**
 * Contrat API / socket — Hub Messages chauffeur (V1).
 *
 * REST (JWT chauffeur) :
 * - GET  /api/v1/messages/:companyId/hub/threads
 * - GET  /api/v1/messages/:companyId/hub/threads/:threadId/messages
 * - POST /api/v1/messages/:companyId/hub/threads/:threadId/messages (envoi texte / pièces)
 * - POST /api/v1/messages/:companyId/hub/read { thread_id }
 * - POST /api/v1/messages/:companyId/hub/ack/:messageId
 * - GET  /api/v1/messages/:companyId/hub/unread-count
 * - POST /api/v1/messages/:companyId/hub/emergency { issue_type, booking_id?, note? }
 *
 * Socket (émission chauffeur) :
 * - team_chat_message { content, thread_id, booking_id, message_type?, priority?, _localId, image_url?, pdf_url?, audio_url? }
 * - team_chat_typing { surface, sender_name? }
 *
 * Socket (réception) :
 * - team_chat_message (payload enrichi + thread_id, message_type, priority)
 * - team_chat_typing { sender_name, user_id }
 *
 * Présence UI : connected | slow | offline (dérivé realtimeManager).
 */

export const MESSAGE_HUB_THREAD_DISPATCH = "dispatch";
export const MESSAGE_HUB_THREAD_TEAM = "team";
export const MESSAGE_HUB_THREAD_SUPPORT = "support";
export const MESSAGE_HUB_THREAD_MISSION_PREFIX = "mission:";
export const MESSAGE_HUB_THREAD_DIRECT_PREFIX = "direct:";

export function directThreadId(peerUserId: number): string {
  return `${MESSAGE_HUB_THREAD_DIRECT_PREFIX}${peerUserId}`;
}

export function parseDirectThreadId(threadId: string): number | null {
  if (!threadId.startsWith(MESSAGE_HUB_THREAD_DIRECT_PREFIX)) return null;
  const parsed = Number.parseInt(threadId.slice(MESSAGE_HUB_THREAD_DIRECT_PREFIX.length), 10);
  return Number.isFinite(parsed) ? parsed : null;
}
