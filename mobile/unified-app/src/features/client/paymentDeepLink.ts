import * as Linking from "expo-linking";

export type ParsedPaymentLink = {
  bookingId: number | null;
  paymentId: number | null;
  outcome: "success" | "fail" | "abort" | null;
  isPaymentReturn: boolean;
  rawUrl: string;
};

function toPositiveInt(value: unknown): number | null {
  const n = Number(value);
  return Number.isInteger(n) && n > 0 ? n : null;
}

export function parsePaymentReturnLink(url: string): ParsedPaymentLink {
  const parsed = Linking.parse(url);
  const path = String(parsed.path ?? "").toLowerCase();
  const host = String((parsed as { hostname?: string }).hostname ?? "").toLowerCase();
  const isPaymentReturn = path.includes("payment-return") || host.includes("payment-return");
  const outcomeRaw = String(parsed.queryParams?.outcome ?? "")
    .trim()
    .toLowerCase();
  const outcome =
    outcomeRaw === "success" || outcomeRaw === "fail" || outcomeRaw === "abort"
      ? outcomeRaw
      : null;
  return {
    bookingId: toPositiveInt(
      parsed.queryParams?.bookingId ?? parsed.queryParams?.booking_id
    ),
    paymentId: toPositiveInt(
      parsed.queryParams?.paymentId ?? parsed.queryParams?.payment_id
    ),
    outcome,
    isPaymentReturn,
    rawUrl: url,
  };
}

export function isValidPaymentReturnPayload(payload: ParsedPaymentLink): boolean {
  if (!payload.isPaymentReturn) return false;
  if (!payload.bookingId) return false;
  return true;
}
