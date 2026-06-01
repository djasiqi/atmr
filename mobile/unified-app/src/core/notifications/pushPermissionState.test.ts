import { beforeEach, describe, expect, it } from "@jest/globals";

import {
  clearPushPermissionDenied,
  getPushPermissionDenied,
  setPushPermissionDenied,
  subscribePushPermissionDenied,
} from "./pushPermissionState";

describe("pushPermissionState", () => {
  beforeEach(() => {
    clearPushPermissionDenied();
  });

  it("notifies subscribers when permission denied toggles", () => {
    const seen: boolean[] = [];
    const unsubscribe = subscribePushPermissionDenied(() => {
      seen.push(getPushPermissionDenied());
    });

    setPushPermissionDenied(true);
    setPushPermissionDenied(false);

    expect(seen).toEqual([true, false]);
    unsubscribe();
  });
});
