export type MessagePriority = "normal" | "important" | "urgent";

export type MessageHubSection =
  | "mission_active"
  | "urgent"
  | "team"
  | "dispatch"
  | "colleagues"
  | "support"
  | "archives";

export type MessageHubThread = {
  thread_id: string;
  conversation_id?: number | null;
  section: MessageHubSection | string;
  title: string;
  subtitle?: string | null;
  peer_user_id?: number | null;
  booking_id?: number | null;
  status?: string | null;
  scheduled_time?: string | null;
  pickup_location?: string | null;
  dropoff_location?: string | null;
  unread_count: number;
  priority: MessagePriority;
  last_message_preview?: string | null;
  last_message_at?: string | null;
  last_message_from_self?: boolean;
};

export type HubChatMessage = {
  id: number | string;
  sender_id?: number | string | null;
  content: string;
  sender_role?: string;
  sender_name?: string | null;
  timestamp: string;
  image_url?: string | null;
  pdf_url?: string | null;
  pdf_filename?: string | null;
  audio_url?: string | null;
  thread_id?: string | null;
  booking_id?: number | null;
  message_type?: string;
  priority?: MessagePriority;
  is_read?: boolean;
  acked_at?: string | null;
  receiver_id?: number | string | null;
  conversation_id?: number | null;
  /** Id client pour réconcilier optimistic ↔ serveur (socket _localId). */
  _localId?: string | null;
};

export type SyncPresenceStatus = "connected" | "slow" | "offline";

export type EmergencyIssueType =
  | "patient_absent"
  | "retard_important"
  | "panne_vehicule"
  | "incident"
  | "besoin_assistance";

export const QUICK_REPLY_TEMPLATES = [
  { id: "arrive", label: "J'arrive", content: "J'arrive" },
  { id: "onboard", label: "Client à bord", content: "Client à bord" },
  { id: "delay5", label: "Retard 5 min", content: "Retard 5 min" },
  { id: "onsite", label: "Je suis sur place", content: "Je suis sur place" },
  { id: "done", label: "Mission terminée", content: "Mission terminée" },
] as const;
