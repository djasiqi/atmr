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

  it("upload puis envoie via REST avec audio_url (canal équipe)", async () => {
    jest.mocked(uploadChatAttachment).mockResolvedValue("https://cdn.example/audio.m4a");
    jest.mocked(sendHubMessage).mockResolvedValue({ id: 1 } as never);

    await sendDriverHubVoiceMessage("file:///voice.m4a", { companyId: 42 });

    expect(uploadChatAttachment).toHaveBeenCalledWith({ uri: "file:///voice.m4a" });
    expect(sendHubMessage).toHaveBeenCalledWith(
      42,
      "team",
      expect.objectContaining({
        audio_url: "https://cdn.example/audio.m4a",
        content: "Message vocal",
        message_type: "audio",
      })
    );
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
      })
    );
  });
});
