import { describe, expect, it } from "@jest/globals";
import { uploadChatAttachment } from "../../chat/services/chatMediaUpload";
import { sendHubMessage } from "../messages/api";
import { MESSAGE_HUB_THREAD_DISPATCH, MESSAGE_HUB_THREAD_TEAM } from "../messages/contracts";
import {
  BOTTOM_BAR_MIC_SOURCE,
  resolveBottomBarDispatchVoiceTarget,
  sendBottomBarDispatchVoiceMessage,
} from "./bottomBarDispatchVoice";

jest.mock("../../chat/services/chatMediaUpload", () => ({
  uploadChatAttachment: jest.fn(),
}));

jest.mock("../messages/api", () => ({
  sendHubMessage: jest.fn(),
}));

describe("resolveBottomBarDispatchVoiceTarget", () => {
  it("cible toujours Dispatch, jamais Équipe", () => {
    const a = resolveBottomBarDispatchVoiceTarget();
    const b = resolveBottomBarDispatchVoiceTarget();
    expect(a).toEqual({
      source: BOTTOM_BAR_MIC_SOURCE,
      channelType: "dispatch",
      channelId: MESSAGE_HUB_THREAD_DISPATCH,
      messageType: "audio",
    });
    expect(a.channelId).not.toBe(MESSAGE_HUB_THREAD_TEAM);
    expect(a).toEqual(b);
  });
});

describe("sendBottomBarDispatchVoiceMessage", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("publie toujours dans Dispatch avec source bottom_bar_micro", async () => {
    jest.mocked(uploadChatAttachment).mockResolvedValue("https://cdn.example/audio.m4a");
    jest.mocked(sendHubMessage).mockResolvedValue({ id: 1 } as never);

    await sendBottomBarDispatchVoiceMessage("file:///voice.m4a", { companyId: 42 });

    expect(sendHubMessage).toHaveBeenCalledWith(
      42,
      "dispatch",
      expect.objectContaining({
        thread_id: "dispatch",
        message_type: "audio",
        source: "bottom_bar_micro",
        audio_url: "https://cdn.example/audio.m4a",
      })
    );
    expect(sendHubMessage).not.toHaveBeenCalledWith(42, "team", expect.anything());
  });
});
