import { Redirect, useLocalSearchParams } from "expo-router";

export default function PaymentReturnScreen() {
  const params = useLocalSearchParams<{
    bookingId?: string;
    booking_id?: string;
    paymentId?: string;
    payment_id?: string;
    outcome?: string;
  }>();
  const bookingId = params.bookingId ?? params.booking_id;
  const paymentId = params.paymentId ?? params.payment_id;
  if (!bookingId || !String(bookingId).trim()) {
    return <Redirect href={"/(public)/fallback/invalid-link?reason=payment_booking_missing" as any} />;
  }
  const encodedCurrentUrl = encodeURIComponent(
    `lirie://payment-return?bookingId=${bookingId ?? ""}&paymentId=${paymentId ?? ""}&outcome=${
      params.outcome ?? ""
    }`
  );
  return (
    <Redirect
      href={{
        pathname: "/(app)/(client)/payment",
        params: {
          bookingId: bookingId ?? "",
          ...(paymentId ? { paymentId } : {}),
          returnUrl: encodedCurrentUrl,
        },
      }}
    />
  );
}
