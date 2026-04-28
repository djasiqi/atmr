import { describe, expect, it } from "@jest/globals";
import { derivePaymentStatusFromBooking } from "./paymentStatus";

describe("payment status mapping", () => {
  it("maps awaiting_client_payment to required", () => {
    const status = derivePaymentStatusFromBooking({
      id: 1,
      status: "awaiting_client_payment",
    });
    expect(status).toBe("required");
  });

  it("maps online payment completed to paid", () => {
    const status = derivePaymentStatusFromBooking({
      id: 1,
      status: "pending",
      online_payment: { status: "completed" },
    });
    expect(status).toBe("paid");
  });

  it("maps pending online payment session to pending_verification", () => {
    const status = derivePaymentStatusFromBooking({
      id: 1,
      status: "pending",
      online_payment: { status: "pending", has_pending_session: true },
    });
    expect(status).toBe("pending_verification");
  });
});
