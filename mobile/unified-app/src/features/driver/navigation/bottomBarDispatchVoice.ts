import { MESSAGE_HUB_THREAD_DISPATCH } from "../messages/contracts";
import { sendDriverHubVoiceMessage } from "../messages/sendDriverHubVoiceMessage";

/** Origine figée du micro central — jamais un chat déjà ouvert. */
export const BOTTOM_BAR_MIC_SOURCE = "bottom_bar_micro" as const;

export type BottomBarDispatchVoiceTarget = {
  source: typeof BOTTOM_BAR_MIC_SOURCE;
  channelType: "dispatch";
  channelId: typeof MESSAGE_HUB_THREAD_DISPATCH;
  messageType: "audio";
};

/**
 * Cible unique du micro barre du bas.
 * Aucun argument : impossible d’hériter de lastOpenedChannel / team / currentChat.
 */
export function resolveBottomBarDispatchVoiceTarget(): BottomBarDispatchVoiceTarget {
  return {
    source: BOTTOM_BAR_MIC_SOURCE,
    channelType: "dispatch",
    channelId: MESSAGE_HUB_THREAD_DISPATCH,
    messageType: "audio",
  };
}

/** Micro barre du bas : toujours le canal Dispatch canonique. */
export async function sendBottomBarDispatchVoiceMessage(
  localUri: string,
  options: { companyId: number }
): Promise<void> {
  const target = resolveBottomBarDispatchVoiceTarget();
  await sendDriverHubVoiceMessage(localUri, {
    companyId: options.companyId,
    threadId: target.channelId,
    source: target.source,
  });
}
