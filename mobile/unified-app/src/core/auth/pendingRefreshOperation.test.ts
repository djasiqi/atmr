/**
 * P0-3 — PendingRefreshOperation après session-resume / bump de génération.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  clearPendingRefreshOperation,
  ensurePendingRefreshOperation,
  readPendingRefreshOperation,
  writePendingRefreshOperation,
} from "./pendingRefreshOperation";

describe("pendingRefreshOperation P0-3", () => {
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
  });

  it("C: clear après resume invalide l'ancienne PendingRefreshOperation", async () => {
    await writePendingRefreshOperation({
      operationId: "ref-old-gen",
      sessionId: "sess-1",
      sourceRefreshGeneration: 3,
      createdAt: new Date().toISOString(),
    });
    expect(await readPendingRefreshOperation()).not.toBeNull();

    // Même séquence que sessionResumeRequest après apply réussi (P0-3).
    await clearPendingRefreshOperation();
    expect(await readPendingRefreshOperation()).toBeNull();

    const next = await ensurePendingRefreshOperation({
      sessionId: "sess-1",
      sourceRefreshGeneration: 4,
    });
    expect(next.operationId).not.toBe("ref-old-gen");
    expect(next.sourceRefreshGeneration).toBe(4);
  });
});
