import { describe, expect, it } from "@jest/globals";
import { isValidPaymentReturnPayload, parsePaymentReturnLink } from "./paymentDeepLink";

describe("payment deep link parser", () => {
  it("parses valid payment-return URL payload", () => {
    const parsed = parsePaymentReturnLink(
      "lirie://payment-return?bookingId=12&paymentId=99&outcome=success"
    );
    expect(parsed.isPaymentReturn).toBe(true);
    expect(parsed.bookingId).toBe(12);
    expect(parsed.paymentId).toBe(99);
    expect(parsed.outcome).toBe("success");
    expect(isValidPaymentReturnPayload(parsed)).toBe(true);
  });

  it("rejects malformed payload", () => {
    const parsed = parsePaymentReturnLink("lirie://payment-return?paymentId=99");
    expect(isValidPaymentReturnPayload(parsed)).toBe(false);
  });
});
