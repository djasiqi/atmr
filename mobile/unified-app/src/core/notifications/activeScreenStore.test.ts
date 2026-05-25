import { beforeEach, describe, expect, it } from "@jest/globals";
import {
  clearActiveMissionScreen,
  clearActiveThreadScreen,
  getActiveScreenState,
  setActiveMissionScreen,
  setActiveThreadScreen,
} from "./activeScreenStore";

describe("activeScreenStore", () => {
  beforeEach(() => {
    clearActiveMissionScreen(-1);
    clearActiveThreadScreen("");
  });

  it("tracks mission focus and clears on blur", () => {
    setActiveMissionScreen(42);
    expect(getActiveScreenState().currentMissionId).toBe(42);
    clearActiveMissionScreen(42);
    expect(getActiveScreenState().currentMissionId).toBeNull();
  });

  it("tracks thread focus and clears on blur", () => {
    setActiveThreadScreen("thread-9", "company");
    expect(getActiveScreenState().currentThreadId).toBe("thread-9");
    clearActiveThreadScreen("thread-9");
    expect(getActiveScreenState().currentThreadId).toBeNull();
  });
});
