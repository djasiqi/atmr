import { resolveChannelQuickRepliesMode } from "./channelQuickReplies";

describe("resolveChannelQuickRepliesMode", () => {
  it("équipe → suggestions mission", () => {
    expect(resolveChannelQuickRepliesMode("team")).toBe("team-mission");
  });

  it("dispatch et mission → modèles standard", () => {
    expect(resolveChannelQuickRepliesMode("dispatch")).toBe("standard");
    expect(resolveChannelQuickRepliesMode("mission:42", 42)).toBe("standard");
  });

  it("support, DM et exploitation privée → aucune", () => {
    expect(resolveChannelQuickRepliesMode("support")).toBe("off");
    expect(resolveChannelQuickRepliesMode("direct:7")).toBe("off");
    expect(resolveChannelQuickRepliesMode("company_driver:3")).toBe("standard");
  });
});
