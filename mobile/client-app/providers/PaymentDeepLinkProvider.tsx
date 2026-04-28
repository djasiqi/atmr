import * as Linking from 'expo-linking';
import { useRouter } from 'expo-router';
import { useEffect, useRef } from 'react';

type ParsedPaymentUrl = {
  bookingId: string | null;
  paymentId: string | null;
};

function parsePaymentUrl(url: string): ParsedPaymentUrl | null {
  const parsed = Linking.parse(url);
  const path = String(parsed.path ?? '').toLowerCase();
  if (!path.includes('payment-return')) {
    return null;
  }
  const bookingId = String(
    parsed.queryParams?.bookingId ?? parsed.queryParams?.booking_id ?? '',
  ).trim() || null;
  const paymentId = String(
    parsed.queryParams?.paymentId ?? parsed.queryParams?.payment_id ?? '',
  ).trim() || null;
  return { bookingId, paymentId };
}

export function PaymentDeepLinkProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const handledRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;

    const handleUrl = (url: string) => {
      if (!url || handledRef.current.has(url)) {
        return;
      }
      const parsed = parsePaymentUrl(url);
      if (!parsed) {
        return;
      }
      handledRef.current.add(url);
      const params = new URLSearchParams();
      if (parsed.bookingId) params.set('bookingId', parsed.bookingId);
      if (parsed.paymentId) params.set('paymentId', parsed.paymentId);
      const qs = params.toString();
      const target = qs.length > 0 ? `/(client)/payment?${qs}` : '/(client)/payment';
      router.replace(target);
    };

    const bootstrap = async () => {
      const initialUrl = await Linking.getInitialURL();
      if (!cancelled && initialUrl) {
        handleUrl(initialUrl);
      }
    };

    const subscription = Linking.addEventListener('url', ({ url }) => {
      handleUrl(url);
    });
    void bootstrap();

    return () => {
      cancelled = true;
      subscription.remove();
    };
  }, [router]);

  return <>{children}</>;
}
