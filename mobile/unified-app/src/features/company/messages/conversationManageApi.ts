import { apiClient } from "../../../core/api/client";

export type ConversationParticipantRow = {
  id: number;
  user_id: number;
  driver_id?: number | null;
  participant_role: string;
  display_name: string;
  role_label?: string;
  is_admin?: boolean;
  last_activity_at?: string | null;
  can_remove?: boolean;
};

export type AvailableDriverRow = {
  driver_id: number;
  user_id: number;
  display_name: string;
};

export type ConversationAttachmentRow = {
  id: string;
  message_id: number;
  kind: "photo" | "document" | "audio";
  url: string;
  label: string;
  timestamp?: string | null;
};

export type ChannelManagePermissions = {
  add_participants: boolean;
  send_files: boolean;
  reply: boolean;
  edit_channel: boolean;
  delete_messages: boolean;
};

export type ChannelHistoryEntry = {
  at?: string | null;
  label: string;
  type?: string;
};

export type ChannelManageInfo = {
  id: number;
  title: string;
  description: string;
  channel_type_label: string;
  legacy_thread_id?: string | null;
  created_at?: string | null;
  created_by_name: string;
  participant_count: number;
  attachment_count: number;
};

export type ChannelManagePayload = {
  channel: ChannelManageInfo;
  participants: ConversationParticipantRow[];
  available_drivers: AvailableDriverRow[];
  attachments_preview: ConversationAttachmentRow[];
  attachments_all?: ConversationAttachmentRow[];
  attachment_counts: {
    all: number;
    photo: number;
    document: number;
    audio: number;
  };
  permissions: ChannelManagePermissions;
  history: ChannelHistoryEntry[];
  can_manage: boolean;
};

function normalizeManagePayload(raw: Record<string, unknown>): ChannelManagePayload {
  const channelRaw = (raw.channel ?? {}) as Record<string, unknown>;
  const countsRaw = (raw.attachment_counts ?? {}) as Record<string, unknown>;
  const permsRaw = (raw.permissions ?? {}) as Record<string, unknown>;
  return {
    channel: {
      id: Number(channelRaw.id ?? 0),
      title: String(channelRaw.title ?? "Dispatch"),
      description: String(channelRaw.description ?? ""),
      channel_type_label: String(channelRaw.channel_type_label ?? "Canal privé"),
      legacy_thread_id:
        typeof channelRaw.legacy_thread_id === "string" ? channelRaw.legacy_thread_id : null,
      created_at: typeof channelRaw.created_at === "string" ? channelRaw.created_at : null,
      created_by_name: String(channelRaw.created_by_name ?? "—"),
      participant_count: Number(channelRaw.participant_count ?? 0),
      attachment_count: Number(channelRaw.attachment_count ?? 0),
    },
    participants: Array.isArray(raw.participants)
      ? (raw.participants as ConversationParticipantRow[])
      : [],
    available_drivers: Array.isArray(raw.available_drivers)
      ? (raw.available_drivers as AvailableDriverRow[])
      : [],
    attachments_preview: Array.isArray(raw.attachments_preview)
      ? (raw.attachments_preview as ConversationAttachmentRow[])
      : [],
    attachments_all: Array.isArray(raw.attachments_all)
      ? (raw.attachments_all as ConversationAttachmentRow[])
      : Array.isArray(raw.attachments_preview)
        ? (raw.attachments_preview as ConversationAttachmentRow[])
        : [],
    attachment_counts: {
      all: Number(countsRaw.all ?? 0),
      photo: Number(countsRaw.photo ?? 0),
      document: Number(countsRaw.document ?? 0),
      audio: Number(countsRaw.audio ?? 0),
    },
    permissions: {
      add_participants: permsRaw.add_participants === true,
      send_files: permsRaw.send_files !== false,
      reply: permsRaw.reply !== false,
      edit_channel: permsRaw.edit_channel === true,
      delete_messages: permsRaw.delete_messages === true,
    },
    history: Array.isArray(raw.history) ? (raw.history as ChannelHistoryEntry[]) : [],
    can_manage: raw.can_manage === true,
  };
}

export async function fetchChannelManageDetail(
  conversationId: number
): Promise<ChannelManagePayload> {
  const { data } = await apiClient.get(`/conversations/${conversationId}/manage`);
  return normalizeManagePayload((data ?? {}) as Record<string, unknown>);
}

export async function updateChannelManageDetail(
  conversationId: number,
  body: { title?: string; description?: string }
): Promise<ChannelManagePayload> {
  const { data } = await apiClient.patch(`/conversations/${conversationId}/manage`, body);
  return normalizeManagePayload((data ?? {}) as Record<string, unknown>);
}

export async function addConversationParticipant(
  conversationId: number,
  driverId: number
): Promise<ConversationParticipantRow> {
  const { data } = await apiClient.post(`/conversations/${conversationId}/participants`, {
    driver_id: driverId,
  });
  const payload = (data ?? {}) as Record<string, unknown>;
  return payload.participant as ConversationParticipantRow;
}

export async function removeConversationParticipant(
  conversationId: number,
  userId: number
): Promise<void> {
  await apiClient.delete(`/conversations/${conversationId}/participants/${userId}`);
}

export async function clearChannelHistory(
  conversationId: number
): Promise<ChannelManagePayload> {
  const { data } = await apiClient.post(
    `/conversations/${conversationId}/manage/clear-history`
  );
  return normalizeManagePayload((data ?? {}) as Record<string, unknown>);
}

export async function fetchConversationAttachments(
  conversationId: number,
  limit = 80
): Promise<ConversationAttachmentRow[]> {
  const { data } = await apiClient.get(`/conversations/${conversationId}/attachments`, {
    params: { limit },
  });
  const payload = (data ?? {}) as Record<string, unknown>;
  return Array.isArray(payload.attachments)
    ? (payload.attachments as ConversationAttachmentRow[])
    : [];
}
