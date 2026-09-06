import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import {
  getOptimizerStatus,
  runCompanyDispatch,
  runCompanyOptimizer,
  switchCompanyDispatchMode,
} from "./companyApi";
import { DispatchFeatureDisabledError } from "../dispatch/dispatchModeLock";

const mockGet = jest.fn<(...args: unknown[]) => unknown>();
const mockPost = jest.fn<(...args: unknown[]) => unknown>();
const mockPut = jest.fn<(...args: unknown[]) => unknown>();

jest.mock("../../../core/api/client", () => ({
  apiClient: {
    get: (...args: unknown[]) => mockGet(...args),
    post: (...args: unknown[]) => mockPost(...args),
    put: (...args: unknown[]) => mockPut(...args),
  },
}));

jest.mock("../telemetry/companyTelemetry", () => ({
  emitCompanyDispatchTelemetry: jest.fn(),
}));

describe("dispatch engine guard (LOCK)", () => {
  beforeEach(() => {
    mockGet.mockReset();
    mockPost.mockReset();
    mockPut.mockReset();
  });

  it("refuse runCompanyDispatch sans HTTP", async () => {
    await expect(runCompanyDispatch({ contextId: "company:42", date: "2026-09-05" })).rejects.toThrow(
      DispatchFeatureDisabledError
    );
    expect(mockPost).not.toHaveBeenCalled();
  });

  it("refuse runCompanyOptimizer sans HTTP", async () => {
    await expect(runCompanyOptimizer({ contextId: "company:42", date: "2026-09-05" })).rejects.toThrow(
      DispatchFeatureDisabledError
    );
    expect(mockPost).not.toHaveBeenCalled();
  });

  it("refuse getOptimizerStatus sans HTTP", async () => {
    await expect(getOptimizerStatus({ contextId: "company:42" })).rejects.toThrow(
      DispatchFeatureDisabledError
    );
    expect(mockGet).not.toHaveBeenCalled();
  });

  it("refuse le switch vers semi-auto / auto sans HTTP", async () => {
    await expect(
      switchCompanyDispatchMode({ contextId: "company:42", mode: "semi_auto" })
    ).rejects.toThrow(DispatchFeatureDisabledError);
    await expect(
      switchCompanyDispatchMode({ contextId: "company:42", mode: "fully_auto" })
    ).rejects.toThrow(DispatchFeatureDisabledError);
    expect(mockPut).not.toHaveBeenCalled();
    expect(mockPost).not.toHaveBeenCalled();
  });
});
