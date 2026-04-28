import React from "react";
import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { act, create } from "react-test-renderer";
import PaymentReturnScreen from "../../app/payment-return";

const mockUseLocalSearchParams = jest.fn() as jest.Mock<any>;
const mockRedirect = jest.fn() as jest.Mock<any>;

jest.mock("expo-router", () => ({
  useLocalSearchParams: () => mockUseLocalSearchParams(),
  Redirect: (props: { href: unknown }) => {
    mockRedirect(props);
    return null;
  },
}));

describe("payment-return route", () => {
  beforeEach(() => {
    mockUseLocalSearchParams.mockReset();
    mockRedirect.mockReset();
  });

  it("redirects to invalid-link fallback when bookingId is missing", async () => {
    mockUseLocalSearchParams.mockReturnValue({ paymentId: "p-1", outcome: "success" });
    await act(async () => {
      create(<PaymentReturnScreen />);
      await Promise.resolve();
    });
    expect(mockRedirect).toHaveBeenCalledWith(
      expect.objectContaining({
        href: "/(public)/fallback/invalid-link?reason=payment_booking_missing",
      })
    );
  });

  it("redirects to client payment route when params are valid", async () => {
    mockUseLocalSearchParams.mockReturnValue({
      bookingId: "b-100",
      paymentId: "pay-9",
      outcome: "success",
    });
    await act(async () => {
      create(<PaymentReturnScreen />);
      await Promise.resolve();
    });
    expect(mockRedirect).toHaveBeenCalledWith(
      expect.objectContaining({
        href: expect.objectContaining({
          pathname: "/(app)/(client)/payment",
          params: expect.objectContaining({
            bookingId: "b-100",
            paymentId: "pay-9",
          }),
        }),
      })
    );
  });
});
