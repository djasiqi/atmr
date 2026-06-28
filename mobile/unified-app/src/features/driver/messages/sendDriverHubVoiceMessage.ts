import { realtimeManager } from "../../../core/realtime/realtimeManager";
import { uploadChatAttachment } from "../../chat/services/chatMediaUpload";
import { sendHubMessage } from "./api";
import { MESSAGE_HUB_THREAD_TEAM } from "./contracts";

export type SendDriverHubVoiceMessageOptions = {
  companyId: number;
  threadId?: string;
};

/**
 * Upload un enregistrement local et le publie sur le hub (socket si possible, REST sinon).
 */
export async function sendDriverHubVoiceMessage(
  localUri: string,
  options: SendDriverHubVoiceMessageOptions
): Promise<void> {
  const threadId = options.threadId ?? MESSAGE_HUB_THREAD_TEAM;
  const localId = `local-voice-${Date.now()}`;
  const publicUrl = await uploadChatAttachment({ uri: localUri });
  const outbound: Record<string, unknown> = {
    content: "Message vocal",
    audio_url: publicUrl,
    _localId: localId,
    thread_id: threadId,
    booking_id: null,
    message_type: "text",
    priority: "normal",
  };

  if (realtimeManager.isDriverSocketReady()) {
    const sent = realtimeManager.emitTeamChatMessage(outbound);
    if (sent) return;
  }

  await sendHubMessage(options.companyId, threadId, outbound);
}
