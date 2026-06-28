import { realtimeManager } from "../../../core/realtime/realtimeManager";
import { uploadChatAttachment } from "../../chat/services/chatMediaUpload";
import { sendHubMessage } from "./api";
import { sendDriverHubVoiceMessage } from "./sendDriverHubVoiceMessage";

jest.mock("../../../core/realtime/realtimeManager", () => ({
  realtimeManager: {
    isDriverSocketReady: jest.fn(),
    emitTeamChatMessage: jest.fn(),
  },
}));

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

  it("émet via socket quand le socket chauffeur est prêt", async () => {
    jest.mocked(uploadChatAttachment).mockResolvedValue("https://cdn.example/audio.m4a");
    jest.mocked(realtimeManager.isDriverSocketReady).mockReturnValue(true);
    jest.mocked(realtimeManager.emitTeamChatMessage).mockReturnValue(true);

    await sendDriverHubVoiceMessage("file:///voice.m4a", { companyId: 42 });

    expect(realtimeManager.emitTeamChatMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        thread_id: "team",
        audio_url: "https://cdn.example/audio.m4a",
        content: "Message vocal",
      })
    );
    expect(sendHubMessage).not.toHaveBeenCalled();
  });

  it("retombe sur REST si le socket n'est pas prêt", async () => {
    jest.mocked(uploadChatAttachment).mockResolvedValue("https://cdn.example/audio.m4a");
    jest.mocked(realtimeManager.isDriverSocketReady).mockReturnValue(false);
    jest.mocked(sendHubMessage).mockResolvedValue({ id: 1 } as never);

    await sendDriverHubVoiceMessage("file:///voice.m4a", { companyId: 42, threadId: "dispatch" });

    expect(sendHubMessage).toHaveBeenCalledWith(
      42,
      "dispatch",
      expect.objectContaining({
        audio_url: "https://cdn.example/audio.m4a",
      })
    );
  });
});
