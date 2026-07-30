import { isNumericMessageId, isPrivateChatUploadUrl } from "./chatAudioUrl";

describe("chatAudioUrl helpers", () => {
  it("détecte les uploads chat privés", () => {
    expect(isPrivateChatUploadUrl("https://api.lirie.ch/uploads/chat/a.m4a")).toBe(true);
    expect(isPrivateChatUploadUrl("/uploads/chat/voice.m4a")).toBe(true);
    expect(isPrivateChatUploadUrl("uploads/chat/voice.m4a")).toBe(true);
    expect(isPrivateChatUploadUrl("https://api.lirie.ch/uploads/company_logos/x.png")).toBe(
      false
    );
    expect(isPrivateChatUploadUrl("file:///tmp/x.m4a")).toBe(false);
  });

  it("accepte uniquement des ids message numériques", () => {
    expect(isNumericMessageId(42)).toBe(true);
    expect(isNumericMessageId("99")).toBe(true);
    expect(isNumericMessageId("local-voice-1")).toBe(false);
    expect(isNumericMessageId(null)).toBe(false);
  });
});
