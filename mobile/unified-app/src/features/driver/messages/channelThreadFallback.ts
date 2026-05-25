import type { MessageHubThread } from "./types";

/** Fil minimal quand l’inbox n’a pas encore chargé le thread. */
export function buildFallbackHubThread(threadId: string): MessageHubThread {
  if (threadId === "team") {
    return {
      thread_id: "team",
      section: "team",
      title: "Équipe chauffeurs",
      subtitle: "Canal groupe",
      unread_count: 0,
      priority: "normal",
    };
  }
  if (threadId === "dispatch") {
    return {
      thread_id: "dispatch",
      section: "dispatch",
      title: "Dispatch",
      subtitle: "Exploitation & régulation",
      unread_count: 0,
      priority: "normal",
    };
  }
  if (threadId === "support") {
    return {
      thread_id: "support",
      section: "support",
      title: "Support LIRIE",
      subtitle: "Assistance LIRIE",
      unread_count: 0,
      priority: "normal",
    };
  }
  if (threadId.startsWith("direct:")) {
    return {
      thread_id: threadId,
      section: "colleagues",
      title: "Collègue",
      subtitle: "Message direct",
      unread_count: 0,
      priority: "normal",
    };
  }
  if (threadId.startsWith("company_driver:")) {
    return {
      thread_id: threadId,
      section: "dispatch",
      title: "Exploitation",
      subtitle: "Conversation privée",
      unread_count: 0,
      priority: "normal",
    };
  }
  if (threadId.startsWith("mission:")) {
    const bookingId = Number.parseInt(threadId.slice("mission:".length), 10);
    return {
      thread_id: threadId,
      section: "mission_active",
      title: "Mission",
      subtitle: null,
      booking_id: Number.isFinite(bookingId) ? bookingId : undefined,
      unread_count: 0,
      priority: "normal",
    };
  }
  return {
    thread_id: threadId,
    section: "dispatch",
    title: "Conversation",
    subtitle: null,
    unread_count: 0,
    priority: "normal",
  };
}
