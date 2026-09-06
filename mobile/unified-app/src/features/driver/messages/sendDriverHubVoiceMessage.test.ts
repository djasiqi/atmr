import { uploadChatAttachment } from "../../chat/services/chatMediaUpload";
import { sendHubMessage } from "./api";
import { sendDriverHubVoiceMessage } from "./sendDriverHubVoiceMessage";

jest.mock("../../chat/services/chatMediaUpload", () => ({
  uploadChatAttachment: jest.fn(),
}));

jest.mock("./api", () => ({
  sendHubMessage: jest.fn(),
}));

describe("sendDriverHubVoiceMessage", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("refuse un threadId vide (pas de repli Équipe)", async () => {
    await expect(
      sendDriverHubVoiceMessage("file:///voice.m4a", { companyId: 42, threadId: "  " })
    ).rejects.toThrow(/threadId/);
    expect(sendHubMessage).not.toHaveBeenCalled();
  });

  it("respecte le threadId demandé", async () => {
    jest.mocked(uploadChatAttachment).mockResolvedValue("https://cdn.example/audio.m4a");
    jest.mocked(sendHubMessage).mockResolvedValue({ id: 1 } as never);

    await sendDriverHubVoiceMessage("file:///voice.m4a", {
      companyId: 42,
      threadId: "dispatch",
    });

    expect(sendHubMessage).toHaveBeenCalledWith(
      42,
      "dispatch",
      expect.objectContaining({
        audio_url: "https://cdn.example/audio.m4a",
        message_type: "audio",
      })
    );
  });
});
