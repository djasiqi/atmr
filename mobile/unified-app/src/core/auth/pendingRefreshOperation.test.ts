import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  ensurePendingRefreshOperation,
} from "./pendingRefreshOperation";

describe("pendingRefreshOperation", () => {
  beforeEach(async () => {
    await AsyncStorage.clear();
  });

  it("réutilise le même operationId pour la même session/génération", async () => {
    const first = await ensurePendingRefreshOperation({
      sessionId: "sess-1",
      sourceRefreshGeneration: 3,
    });
    const second = await ensurePendingRefreshOperation({
      sessionId: "sess-1",
      sourceRefreshGeneration: 3,
    });
    expect(second.operationId).toBe(first.operationId);
    expect(first.operationId.startsWith("ref-")).toBe(true);
  });

  it("crée un nouvel operationId pour une autre génération", async () => {
    const first = await ensurePendingRefreshOperation({
      sessionId: "sess-1",
      sourceRefreshGeneration: 3,
    });
    const second = await ensurePendingRefreshOperation({
      sessionId: "sess-1",
      sourceRefreshGeneration: 4,
    });
    expect(second.operationId).not.toBe(first.operationId);
    expect(second.operationId.startsWith("ref-")).toBe(true);
  });
});
