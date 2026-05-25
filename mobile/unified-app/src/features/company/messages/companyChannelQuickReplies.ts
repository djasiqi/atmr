import type { ChannelQuickRepliesMode } from "../../driver/messages/channelQuickReplies";

/** Suggestions rapides — exploitation mobile (pas d’équipe / support / DM). */
export function resolveCompanyQuickRepliesMode(
  threadId: string,
  bookingId?: number | null
): ChannelQuickRepliesMode {
  if (threadId === "team" || threadId === "support" || threadId.startsWith("direct:")) {
    return "off";
  }
  if (
    threadId === "dispatch" ||
    bookingId != null ||
    threadId.startsWith("mission:") ||
    threadId.startsWith("company_driver:")
  ) {
    return "standard";
  }
  return "off";
}
