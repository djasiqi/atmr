/**
 * Tests de validation forensic "refresh_token missing".
 * Critère : sur un run reproduisant l'erreur, on sait dire si le token
 * n'a jamais été écrit, ou a été effacé, ou est illisible.
 *
 * Les tests simulent les flux et vérifient que les logs DEBUG_AUTH
 * (ou leurs équivalents via spy) permettent de trancher.
 */
import * as SecureStore from "expo-secure-store";
import { secureStorage } from "../storage";

jest.mock("expo-secure-store", () => ({
  setItemAsync: jest.fn(),
  getItemAsync: jest.fn(),
  deleteItemAsync: jest.fn(),
}));

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  multiRemove: jest.fn(),
}));

// Capture des appels à debugAuthLog pour vérifier la séquence
const debugAuthLogCalls: Array<{ phase: string; payload: Record<string, unknown> }> = [];
jest.mock("@/services/authDebug", () => ({
  isDebugAuthEnabled: () => true,
  debugAuthLog: (phase: string, payload: Record<string, unknown>) => {
    debugAuthLogCalls.push({ phase, payload });
  },
}));

describe("Forensic enterprise refresh_token – diagnostic", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    debugAuthLogCalls.length = 0;
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(null);
    (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);
    (SecureStore.deleteItemAsync as jest.Mock).mockResolvedValue(undefined);
  });

  it("E2E diagnostic 'never written': getEnterpriseRefreshToken null sans aucun write → logs permettent de conclure 'jamais écrit'", async () => {
    // Simuler un boot où personne n'a jamais appelé setEnterpriseRefreshToken
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(null);

    const got = await secureStorage.getEnterpriseRefreshToken();

    expect(got).toBeNull();
    // Avec DEBUG_AUTH=1, ent_refresh_read est appelé depuis getEnterpriseRefreshToken
    const readCalls = debugAuthLogCalls.filter((c) => c.phase === "ent_refresh_read");
    expect(readCalls.length).toBeGreaterThanOrEqual(1);
    expect(readCalls[0].payload.present).toBe(0);
    expect(readCalls[0].payload.len).toBe(0);
    // Aucun ent_refresh_write dans ce run => diagnostic "token jamais écrit"
    const writeCalls = debugAuthLogCalls.filter((c) => c.phase === "ent_refresh_write");
    expect(writeCalls.length).toBe(0);
  });

  it("E2E diagnostic 'erased': write puis clear puis read null → logs permettent de conclure 'effacé'", async () => {
    // 1. Write
    await secureStorage.setEnterpriseRefreshToken("refreshtoken-value");
    const writeCalls = debugAuthLogCalls.filter((c) => c.phase === "ent_refresh_write");
    expect(writeCalls.length).toBe(1);
    expect((writeCalls[0].payload.len as number)).toBeGreaterThan(0);

    debugAuthLogCalls.length = 0;

    // 2. Clear (simule clearEnterpriseTokens)
    await secureStorage.clearEnterpriseTokens();

    // 3. Read
    const got = await secureStorage.getEnterpriseRefreshToken();
    expect(got).toBeNull();

    const readCalls = debugAuthLogCalls.filter((c) => c.phase === "ent_refresh_read");
    expect(readCalls.length).toBeGreaterThanOrEqual(1);
    expect(readCalls[0].payload.present).toBe(0);
    // On a vu un write avec len>0 avant, puis read present=0 => diagnostic "effacé"
  });
});
