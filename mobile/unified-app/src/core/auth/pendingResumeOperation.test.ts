import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  clearPendingResumeOperation,
  ensurePendingResumeOperation,
  readPendingResumeOperation,
} from "./pendingResumeOperation";

describe("pendingResumeOperation", () => {
  beforeEach(async () => {
    await AsyncStorage.clear();
  });

  it("réutilise le même operationId pour la même session/génération", async () => {
    const first = await ensurePendingResumeOperation({
      sessionId: "sess-1",
      sourceCredentialGeneration: 6,
    });
    const second = await ensurePendingResumeOperation({
      sessionId: "sess-1",
      sourceCredentialGeneration: 6,
    });
    expect(second.operationId).toBe(first.operationId);
    expect(first.operationId.startsWith("res-")).toBe(true);
  });

  it("crée un nouvel operationId si credential_generation change", async () => {
    const first = await ensurePendingResumeOperation({
      sessionId: "sess-1",
      sourceCredentialGeneration: null,
    });
    const second = await ensurePendingResumeOperation({
      sessionId: "sess-1",
      sourceCredentialGeneration: 7,
    });
    expect(second.operationId).not.toBe(first.operationId);
  });

  it("ne copie jamais refresh_generation dans credential_generation (migration absente)", async () => {
    const op = await ensurePendingResumeOperation({
      sessionId: "sess-39",
      sourceCredentialGeneration: null,
    });
    expect(op.sourceCredentialGeneration).toBeNull();
    const stored = await readPendingResumeOperation();
    expect(stored?.sourceCredentialGeneration).toBeNull();
    await clearPendingResumeOperation();
    expect(await readPendingResumeOperation()).toBeNull();
  });
});
