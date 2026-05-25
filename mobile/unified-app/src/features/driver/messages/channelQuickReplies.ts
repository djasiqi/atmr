/** Détermine quelles suggestions rapides afficher selon le canal. */
export type ChannelQuickRepliesMode = "team-mission" | "standard" | "off";

export function resolveChannelQuickRepliesMode(
  threadId: string,
  bookingId?: number | null
): ChannelQuickRepliesMode {
  if (threadId === "support") return "off";
  if (threadId.startsWith("direct:")) return "off";
  if (threadId.startsWith("company_driver:")) return "standard";
  if (threadId === "team") return "team-mission";
  if (threadId === "dispatch" || bookingId != null || threadId.startsWith("mission:")) {
    return "standard";
  }
  return "off";
}
