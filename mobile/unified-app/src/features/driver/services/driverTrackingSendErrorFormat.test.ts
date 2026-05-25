import { describe, expect, it } from "@jest/globals";
import { AxiosError } from "axios";
import { formatTrackingSendError } from "./driverTrackingSendErrorFormat";

describe("formatTrackingSendError", () => {
  it("classifie une erreur API normalisée (circuit)", () => {
    const meta = formatTrackingSendError({
      status: null,
      code: "HTTP_CIRCUIT_BREAKER_OPEN",
      message: "Circuit breaker HTTP ouvert (tracking_http)",
    });
    expect(meta.error_class).toBe("circuit_open");
    expect(meta.api_error_code).toBe("HTTP_CIRCUIT_BREAKER_OPEN");
  });

  it("classifie une erreur HTTP normalisée", () => {
    const meta = formatTrackingSendError({
      status: 503,
      code: "SERVICE_UNAVAILABLE",
      message: "Service indisponible",
    });
    expect(meta.error_class).toBe("http");
    expect(meta.http_status).toBe(503);
  });

  it("masque les URL dans le message", () => {
    const err = new AxiosError(
      "Request failed",
      "ERR_NETWORK",
      { baseURL: "https://api.example.com", url: "/x" } as any,
      undefined,
      undefined
    );
    const meta = formatTrackingSendError(err);
    expect(meta.error_message).not.toMatch(/https:\/\//);
    expect(meta.error_class).toBe("network");
  });
});
