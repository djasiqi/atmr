import * as Linking from "expo-linking";
import * as WebBrowser from "expo-web-browser";
import { Redirect, useLocalSearchParams, useRouter } from "expo-router";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { ActivityIndicator, Pressable, Text, View } from "react-native";
import { useSession } from "../../../src/core/sessionProvider";
import {
  assertSaferpayCheckout,
  initializeSaferpayCheckout,
} from "../../../src/features/client/api";
import { useActiveClientContextId, useBookingDetailQuery } from "../../../src/features/client/hooks";
import {
  clearPendingPayment,
  getPendingPaymentLookup,
  setPendingPayment,
} from "../../../src/features/client/pendingPayment";
import {
  isValidPaymentReturnPayload,
  parsePaymentReturnLink,
} from "../../../src/features/client/paymentDeepLink";
import { logPaymentEvent } from "../../../src/features/client/paymentObservability";
import { clientQueryKeys, invalidateClientQueries } from "../../../src/features/client/queryKeys";
import { PaymentStatus } from "../../../src/features/client/types";

function parsePositiveInt(value: unknown): number | null {
  const n = Number(value);
  return Number.isInteger(n) && n > 0 ? n : null;
}

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (error && typeof error === "object" && "message" in error) {
    const msg = (error as { message?: unknown }).message;
    if (typeof msg === "string" && msg.trim()) return msg;
  }
  return "network";
}

function parseBackoffConfig(): number[] {
  const raw = process.env.EXPO_PUBLIC_PAYMENT_ASSERT_BACKOFFS_MS;
  if (!raw) {
    return [0, 3000, 6000, 12000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000];
  }
  const parsed = raw
    .split(",")
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isFinite(v) && v >= 0);
  return parsed.length > 0 ? parsed : [0, 3000, 6000, 12000, 15000, 15000, 15000, 15000];
}

function resolveAssertMaxWindowMs(): number {
  const raw = Number(process.env.EXPO_PUBLIC_PAYMENT_ASSERT_MAX_WINDOW_MS ?? "150000");
  if (!Number.isFinite(raw) || raw <= 0) return 150000;
  return raw;
}

function formatChf(amount: number | null | undefined): string | null {
  if (typeof amount !== "number" || !Number.isFinite(amount)) {
    return null;
  }
  try {
    return new Intl.NumberFormat("fr-CH", { style: "currency", currency: "CHF" }).format(amount);
  } catch {
    return `${amount.toFixed(2)} CHF`;
  }
}

export default function ClientPaymentScreen() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const contextId = useActiveClientContextId();
  const { bootstrap, activeContext } = useSession();
  const params = useLocalSearchParams<{
    bookingId?: string;
    paymentId?: string;
    returnUrl?: string;
  }>();

  const bookingId = parsePositiveInt(params.bookingId);
  const paymentIdFromQuery = parsePositiveInt(params.paymentId);
  const [pendingPaymentId, setPendingPaymentId] = useState<number | null>(paymentIdFromQuery);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [asserting, setAsserting] = useState(false);
  const [invalidReturnPayload, setInvalidReturnPayload] = useState(false);
  const [contextMismatch, setContextMismatch] = useState(false);

  const bookingQuery = useBookingDetailQuery(bookingId);

  const paymentEnabled = useMemo(() => {
    const raw = bootstrap?.feature_flags?.saferpay_enabled;
    if (typeof raw === "boolean") return raw;
    return true;
  }, [bootstrap?.feature_flags]);

  const paymentStatus: PaymentStatus = bookingQuery.data?.payment_status ?? "unknown";
  const payDisabled =
    submitting || asserting || paymentStatus === "paid" || paymentStatus === "pending_verification";
  const outboundAmount =
    typeof bookingQuery.data?.amount === "number" ? bookingQuery.data.amount : null;
  const returnAmount =
    typeof bookingQuery.data?.return_trip?.amount === "number"
      ? bookingQuery.data.return_trip.amount
      : typeof bookingQuery.data?.return_booking?.amount === "number"
        ? bookingQuery.data.return_booking.amount
        : null;
  const showConsolidatedRoundTrip =
    Boolean(bookingQuery.data) &&
    !bookingQuery.data?.is_return &&
    typeof outboundAmount === "number" &&
    typeof returnAmount === "number" &&
    returnAmount > 0;
  const displayedPayableAmount =
    typeof bookingQuery.data?.payment_amount === "number"
      ? bookingQuery.data.payment_amount
      : showConsolidatedRoundTrip
        ? (outboundAmount ?? 0) + (returnAmount ?? 0)
        : typeof bookingQuery.data?.amount === "number"
          ? bookingQuery.data.amount
          : null;

  useEffect(() => {
    let mounted = true;
    const hydrate = async () => {
      if (!bookingId) return;
      const { pending, expired } = await getPendingPaymentLookup();
      if (!mounted) return;
      if (expired) {
        setStatusMessage("Session de paiement expirée, veuillez relancer le paiement.");
      }
      if (!pending) return;
      if (pending.bookingId === bookingId) {
        if (pending.paymentId) {
          setPendingPaymentId(pending.paymentId);
        }
        if (contextId && pending.contextId !== contextId) {
          setContextMismatch(true);
          void clearPendingPayment();
          setStatusMessage(
            "Contexte modifié depuis le lancement du paiement. Vérifiez le statut depuis le détail réservation."
          );
        }
      }
    };
    void hydrate();
    return () => {
      mounted = false;
    };
  }, [bookingId, contextId]);

  const refreshBookingPaymentState = async () => {
    if (!bookingId || !contextId) return;
    if (contextMismatch) return;
    invalidateClientQueries(queryClient, contextId, bookingId);
    await queryClient.refetchQueries({
      queryKey: clientQueryKeys.bookingDetail(contextId, bookingId),
      exact: true,
    });
    logPaymentEvent("payment_status_refetched", {
      bookingId,
      paymentId: pendingPaymentId,
      contextId,
    });
  };

  const assertWithBackoff = async (paymentId: number) => {
    if (!bookingId || !contextId) return;
    const backoffs = parseBackoffConfig();
    const maxWindowMs = resolveAssertMaxWindowMs();
    const terminalStatuses = new Set([
      "payment_failed",
      "assert_failed",
      "capture_failed",
      "cancelled",
      "expired",
      "failed",
    ]);
    const startedAt = Date.now();

    for (let i = 0; i < backoffs.length; i += 1) {
      if (Date.now() - startedAt >= maxWindowMs) {
        break;
      }
      if (backoffs[i] > 0) await wait(backoffs[i]);
      try {
        logPaymentEvent("payment_assert_started", {
          bookingId,
          paymentId,
          contextId,
        });
        const result = await assertSaferpayCheckout(bookingId, paymentId);
        const normalizedPaymentStatus = String(result.payment_status ?? "").toLowerCase();
        const normalizedStatus = String(result.status ?? "").toLowerCase();

        if (
          normalizedPaymentStatus === "paid" ||
          normalizedStatus === "completed" ||
          normalizedStatus === "already_completed"
        ) {
          logPaymentEvent("payment_assert_succeeded", {
            bookingId,
            paymentId,
            contextId,
          });
          await clearPendingPayment();
          setStatusMessage("Paiement confirmé.");
          await refreshBookingPaymentState();
          return;
        }
        if (terminalStatuses.has(normalizedPaymentStatus) || terminalStatuses.has(normalizedStatus)) {
          logPaymentEvent("payment_assert_failed", {
            bookingId,
            paymentId,
            contextId,
            reason: normalizedStatus || normalizedPaymentStatus || "failed",
          });
          setStatusMessage("Paiement échoué ou annulé.");
          await refreshBookingPaymentState();
          return;
        }
      } catch (e) {
        const maybeApiError = e as { status?: number | null; retryable?: boolean; message?: string };
        if (maybeApiError?.status === 401 || maybeApiError?.status === 403) {
          setStatusMessage("Session expirée. Veuillez vous reconnecter pour vérifier le paiement.");
          return;
        }
        if (maybeApiError?.retryable === false) {
          setStatusMessage(`Vérification arrêtée: ${getErrorMessage(e)}`);
          return;
        }
        logPaymentEvent("payment_assert_failed", {
          bookingId,
          paymentId,
          contextId,
          reason: getErrorMessage(e),
        });
      }
    }
    setStatusMessage(
      "Paiement en cours de confirmation serveur. Si vous êtes hors ligne, la vérification reprendra dès reconnexion."
    );
    await refreshBookingPaymentState();
  };

  useEffect(() => {
    const returnUrl = params.returnUrl;
    if (!returnUrl || !bookingId) return;
    const decodedReturnUrl = decodeURIComponent(returnUrl);
    const parsed = parsePaymentReturnLink(decodedReturnUrl);
    if (!isValidPaymentReturnPayload(parsed)) {
      setInvalidReturnPayload(true);
      return;
    }
    if (parsed.bookingId !== bookingId) {
      setInvalidReturnPayload(true);
      return;
    }
    if (paymentIdFromQuery && parsed.paymentId && paymentIdFromQuery !== parsed.paymentId) {
      setInvalidReturnPayload(true);
      return;
    }
    logPaymentEvent("payment_return_received", {
      bookingId,
      paymentId: parsed.paymentId ?? paymentIdFromQuery,
      contextId: contextId ?? undefined,
      reason: parsed.outcome ?? undefined,
    });
    const run = async () => {
      if (contextMismatch) return;
      setAsserting(true);
      try {
        if (parsed.paymentId) {
          setPendingPaymentId(parsed.paymentId);
          await assertWithBackoff(parsed.paymentId);
        } else {
          await refreshBookingPaymentState();
          setStatusMessage("Retour paiement reçu. Statut resynchronisé avec le serveur.");
        }
      } catch (e) {
        setStatusMessage(
          e instanceof Error
            ? `Vérification paiement échouée: ${e.message}`
            : "Vérification paiement échouée."
        );
      } finally {
        setAsserting(false);
      }
    };
    void run();
    // We intentionally run only on deep-link payload changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.returnUrl, bookingId, paymentIdFromQuery, contextId, contextMismatch]);

  const onStartPayment = async () => {
    if (!bookingId || !contextId) return;
    if (contextMismatch) {
      setStatusMessage("Impossible de reprendre automatiquement ce paiement dans le contexte courant.");
      return;
    }
    setSubmitting(true);
    setStatusMessage(null);
    try {
      logPaymentEvent("payment_initialize_started", { bookingId, contextId });
      const returnLink = Linking.createURL("payment-return", {
        queryParams: { bookingId: String(bookingId) },
      });
      const payload = await initializeSaferpayCheckout(bookingId, returnLink);
      await setPendingPayment({
        bookingId,
        paymentId: payload.payment_id ?? null,
        contextId,
        createdAt: Date.now(),
      });
      setPendingPaymentId(payload.payment_id ?? null);
      logPaymentEvent("payment_redirect_opened", {
        bookingId,
        paymentId: payload.payment_id,
        contextId,
      });
      await WebBrowser.openBrowserAsync(payload.redirect_url);
      setStatusMessage("Retournez dans l'application après le paiement pour finaliser la vérification.");
    } catch (e) {
      logPaymentEvent("payment_initialize_failed", {
        bookingId,
        contextId,
        reason: e instanceof Error ? e.message : "unknown",
      });
      setStatusMessage(
        e instanceof Error ? `Impossible de lancer le paiement: ${e.message}` : "Paiement indisponible."
      );
    } finally {
      setSubmitting(false);
    }
  };

  const onManualVerify = async () => {
    if (!bookingId) return;
    if (contextMismatch) {
      setStatusMessage("Contexte différent détecté, reprise automatique désactivée.");
      return;
    }
    setAsserting(true);
    try {
      if (pendingPaymentId) {
        await assertWithBackoff(pendingPaymentId);
      } else {
        await refreshBookingPaymentState();
      }
    } finally {
      setAsserting(false);
    }
  };

  if (!activeContext || activeContext.context_type !== "client") {
    return <Redirect href={"/(app)/unauthorized" as any} />;
  }

  if (!bookingId) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center", padding: 24, gap: 12 }}>
        <Text>Booking invalide pour le paiement.</Text>
        <Pressable onPress={() => router.replace("/(app)/(client)/bookings")}>
          <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Retour aux réservations</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={{ flex: 1, padding: 24, gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "700" }}>Paiement Saferpay</Text>
      <Text>Réservation #{bookingId}</Text>
      {!paymentEnabled ? (
        <Text>Le paiement est temporairement indisponible.</Text>
      ) : null}
      {invalidReturnPayload ? (
        <Text style={{ color: "#b42318" }}>
          Retour paiement invalide. Le statut sera vérifié depuis le détail réservation.
        </Text>
      ) : null}
      {bookingQuery.isLoading ? <ActivityIndicator /> : null}
      {bookingQuery.data ? (
        <View style={{ borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 12, gap: 6 }}>
          <Text style={{ fontWeight: "600" }}>
            {bookingQuery.data.pickup_location ?? "Départ"}
            {" -> "}
            {bookingQuery.data.dropoff_location ?? "Arrivée"}
          </Text>
          <Text>Statut course: {bookingQuery.data.status ?? "unknown"}</Text>
          <Text>Statut paiement: {bookingQuery.data.payment_status ?? "unknown"}</Text>
          {formatChf(displayedPayableAmount) ? (
            <Text style={{ fontWeight: "700" }}>
              Montant à régler: {formatChf(displayedPayableAmount)}
            </Text>
          ) : null}
          {showConsolidatedRoundTrip ? (
            <Text>
              Aller: {formatChf(outboundAmount)} + Retour: {formatChf(returnAmount)} = Total:{" "}
              {formatChf(displayedPayableAmount)}
            </Text>
          ) : null}
        </View>
      ) : null}

      <Pressable
        onPress={() => void onStartPayment()}
        disabled={!paymentEnabled || payDisabled}
        style={{
          opacity: !paymentEnabled || payDisabled ? 0.6 : 1,
          borderRadius: 8,
          backgroundColor: "#0a7ea4",
          paddingVertical: 12,
          alignItems: "center",
        }}
      >
        {submitting ? <ActivityIndicator color="#fff" /> : <Text style={{ color: "#fff", fontWeight: "700" }}>Payer maintenant</Text>}
      </Pressable>

      <Pressable
        onPress={() => void onManualVerify()}
        disabled={asserting}
        style={{
          borderRadius: 8,
          borderWidth: 1,
          borderColor: "#cfcfcf",
          paddingVertical: 12,
          alignItems: "center",
        }}
      >
        {asserting ? <ActivityIndicator /> : <Text style={{ fontWeight: "600" }}>Actualiser le statut</Text>}
      </Pressable>

      {statusMessage ? <Text>{statusMessage}</Text> : null}

      <Pressable
        onPress={() =>
          router.replace({
            pathname: "/(app)/(client)/booking/[bookingId]",
            params: { bookingId: String(bookingId) },
          })
        }
      >
        <Text style={{ color: "#0a7ea4", fontWeight: "600" }}>Retour au détail réservation</Text>
      </Pressable>
    </View>
  );
}
