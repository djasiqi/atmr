import { describe, expect, it } from "@jest/globals";
import { shouldIgnoreNotification } from "./shouldIgnoreNotification";

describe("shouldIgnoreNotification", () => {
  const driverContext = {
    contextType: "driver",
    userId: 42,
    companyId: 9,
  };

  const companyContext = {
    contextType: "company",
    userId: 11,
    companyId: 9,
  };

  it("ignores invalid payload for driver context", () => {
    const result = shouldIgnoreNotification(null, driverContext);
    expect(result).toEqual({ ignore: true, reason: "invalid_payload" });
  });

  it("ignores when no active context (fail-closed)", () => {
    const result = shouldIgnoreNotification(
      { recipient_role: "driver" },
      { contextType: null, userId: 42, companyId: 9 }
    );
    expect(result).toEqual({ ignore: true, reason: "no_active_context" });
  });

  it("ignores recipient role mismatch for driver", () => {
    const result = shouldIgnoreNotification({ recipient_role: "company" }, driverContext);
    expect(result).toEqual({ ignore: true, reason: "recipient_role_mismatch" });
  });

  it("ignores self actor updates for driver", () => {
    const result = shouldIgnoreNotification(
      { actor_role: "driver", actor_id: "42", recipient_role: "driver" },
      driverContext
    );
    expect(result).toEqual({ ignore: true, reason: "self_actor" });
  });

  it("ignores company mismatch when payload includes company", () => {
    const result = shouldIgnoreNotification(
      { recipient_role: "driver", company_id: "11" },
      driverContext
    );
    expect(result).toEqual({ ignore: true, reason: "company_mismatch" });
  });

  it("keeps valid driver payload", () => {
    const result = shouldIgnoreNotification(
      { recipient_role: "driver", actor_role: "dispatcher", actor_id: "11", company_id: 9 },
      driverContext
    );
    expect(result).toEqual({ ignore: false });
  });

  it("ignores driver-targeted payload in company context", () => {
    const result = shouldIgnoreNotification({ recipient_role: "driver", company_id: 9 }, companyContext);
    expect(result).toEqual({ ignore: true, reason: "recipient_role_mismatch" });
  });

  it("ignores company mismatch in company context", () => {
    const result = shouldIgnoreNotification(
      { recipient_role: "company", company_id: "11" },
      companyContext
    );
    expect(result).toEqual({ ignore: true, reason: "company_mismatch" });
  });

  it("keeps valid company payload", () => {
    const result = shouldIgnoreNotification(
      { recipient_role: "company", company_id: 9 },
      companyContext
    );
    expect(result).toEqual({ ignore: false });
  });
});
