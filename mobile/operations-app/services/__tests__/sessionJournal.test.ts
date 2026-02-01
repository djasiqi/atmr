/**
 * P0.2/P0.3/P0.4 – Tests du journal de session et X-Session-Diag
 * - pushSessionEvent / getLastSessionEvent
 * - getSessionDiagHeaderValue (format EVENT|ts, suffixe S:ONLINE/RECONN/OFF)
 * - subscribeSessionJournal
 */

import {
  pushSessionEvent,
  getLastSessionEvent,
  getSessionDiagHeaderValue,
  setConnectionStateSuffix,
  getConnectionStateSuffix,
  subscribeSessionJournal,
  _testingReset,
  SESSION_JOURNAL_KEYS,
} from "../sessionJournal";

jest.mock("@react-native-async-storage/async-storage", () => ({
  getItem: jest.fn(),
  setItem: jest.fn().mockResolvedValue(undefined),
  removeItem: jest.fn(),
  multiRemove: jest.fn(),
}));

describe("sessionJournal", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    _testingReset();
    if (typeof localStorage !== "undefined") {
      localStorage.clear();
    }
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe("pushSessionEvent / getLastSessionEvent", () => {
    it("enregistre un événement et le retourne via getLastSessionEvent", () => {
      pushSessionEvent("REFRESH_START");
      const last = getLastSessionEvent();
      expect(last).not.toBeNull();
      expect(last!.event).toBe("REFRESH_START");
      expect(typeof last!.at).toBe("number");
    });

    it("REFRESH_WAIT est un événement valide (P0.2)", () => {
      pushSessionEvent("REFRESH_WAIT");
      const last = getLastSessionEvent();
      expect(last!.event).toBe("REFRESH_WAIT");
    });

    it("FOREGROUND_RESYNC_START / SUCCESS / FAIL sont valides (P0.4)", () => {
      pushSessionEvent("FOREGROUND_RESYNC_START");
      expect(getLastSessionEvent()!.event).toBe("FOREGROUND_RESYNC_START");
      pushSessionEvent("FOREGROUND_RESYNC_SUCCESS");
      expect(getLastSessionEvent()!.event).toBe("FOREGROUND_RESYNC_SUCCESS");
      pushSessionEvent("FOREGROUND_RESYNC_FAIL");
      expect(getLastSessionEvent()!.event).toBe("FOREGROUND_RESYNC_FAIL");
    });
  });

  describe("getSessionDiagHeaderValue", () => {
    it("retourne null quand aucun événement", () => {
      expect(getSessionDiagHeaderValue()).toBeNull();
    });

    it("retourne format EVENT|ts après pushSessionEvent", () => {
      pushSessionEvent("REFRESH_SUCCESS");
      const v = getSessionDiagHeaderValue();
      expect(v).not.toBeNull();
      expect(v).toMatch(/^REFRESH_SUCCESS\|[0-9]+$/);
    });

    it("inclut le suffixe S:ONLINE/RECONN/OFF quand setConnectionStateSuffix est défini (P0.3)", () => {
      pushSessionEvent("SOCKET_CONNECTED");
      expect(getSessionDiagHeaderValue()).toMatch(/^SOCKET_CONNECTED\|[0-9]+$/);
      setConnectionStateSuffix("ONLINE");
      expect(getSessionDiagHeaderValue()).toMatch(/^SOCKET_CONNECTED\|[0-9]+\|S:ONLINE$/);
      setConnectionStateSuffix("RECONN");
      expect(getSessionDiagHeaderValue()).toMatch(/^SOCKET_CONNECTED\|[0-9]+\|S:RECONN$/);
      setConnectionStateSuffix("OFF");
      expect(getSessionDiagHeaderValue()).toMatch(/^SOCKET_CONNECTED\|[0-9]+\|S:OFF$/);
      setConnectionStateSuffix(null);
      expect(getSessionDiagHeaderValue()).toMatch(/^SOCKET_CONNECTED\|[0-9]+$/);
    });
  });

  describe("setConnectionStateSuffix / getConnectionStateSuffix", () => {
    it("getConnectionStateSuffix retourne la valeur définie", () => {
      expect(getConnectionStateSuffix()).toBeNull();
      setConnectionStateSuffix("ONLINE");
      expect(getConnectionStateSuffix()).toBe("ONLINE");
      setConnectionStateSuffix("RECONN");
      expect(getConnectionStateSuffix()).toBe("RECONN");
      setConnectionStateSuffix("OFF");
      expect(getConnectionStateSuffix()).toBe("OFF");
      setConnectionStateSuffix(null);
      expect(getConnectionStateSuffix()).toBeNull();
    });
  });

  describe("subscribeSessionJournal", () => {
    it("notifie le listener à chaque pushSessionEvent", () => {
      const fn = jest.fn();
      const unsub = subscribeSessionJournal(fn);
      pushSessionEvent("API_401");
      expect(fn).toHaveBeenCalledWith("API_401", expect.any(Number));
      pushSessionEvent("REFRESH_START");
      expect(fn).toHaveBeenCalledWith("REFRESH_START", expect.any(Number));
      unsub();
      pushSessionEvent("REFRESH_SUCCESS");
      expect(fn).toHaveBeenCalledTimes(2);
    });
  });
});
