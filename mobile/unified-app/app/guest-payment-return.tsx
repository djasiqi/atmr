import * as WebBrowser from "expo-web-browser";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useEffect, useState } from "react";
import { ActivityIndicator, StyleSheet, Text, View } from "react-native";
import { assertGuestSaferpay } from "../src/core/api/client";
import {
  getGuestSaferpayPending,
  setGuestSaferpayPending,
} from "../src/core/public/guestSaferpayPending";
import * as SecureStore from "../src/core/storage/secureStoreCompat";
import { brandPrimary, ResponsiveContainer, Screen } from "../src/design/responsive";
import { FONT_SIZE } from "../src/design/responsive/typographyTokens";

const PUBLIC_BOOKING_TOKEN_KEY = "public_booking_status_token_v1";
const PUBLIC_BOOKING_ID_KEY = "public_booking_id_v1";

/** Délai entre essais (Saferpay peut renvoyer TRANSACTION_IN_WRONG_STATE si assert trop tôt). */
const ASSERT_GUEST_BACKOFFS_MS = [0, 1500, 2500, 4000, 5000, 5000, 5000, 5000, 5000, 10000, 10000];

type AssertGuestBody = Awaited<ReturnType<typeof assertGuestSaferpay>> & {
  status?: string;
};

const SAFERPAY_EN_FRIENDLY: [RegExp, string][] = [
  [
    /^Transaction not started\.?$/i,
    "Côté prestataire, l’API met parfois quelques secondes après l’autorisation. Nouvel essai en cours…",
  ],
  [/transaction still in progress/i, "Paiement en cours, nouvel essai…"],
  [
    /Invalid action/i,
    "Action indisponible un court instant — l’enregistrement côté prestataire n’est pas encore prêt.",
  ],
  [/wrong state/i, "État de transaction en cours d’enregistrement, nouvel essai…"],
];

function humanizeAssertMessage(detail: unknown): string | null {
  if (typeof detail !== "string" || !detail.trim()) return null;
  const t = detail.trim();
  if (t.startsWith("{") && t.includes("ResponseHeader")) {
    return null;
  }
  for (const [re, fr] of SAFERPAY_EN_FRIENDLY) {
    if (re.test(t)) return fr;
  }
  if (t.length > 500) return `${t.slice(0, 500)}…`;
  return t;
}

/** True si l’échec assert peut se résoudre en réessayant (désalignement API bancaire / not started / etc.). */
function isRetryableGuestAssertDetail(detail: unknown): boolean {
  if (typeof detail !== "string" || !detail.trim()) return false;
  const t = detail.trim();
  for (const [re] of SAFERPAY_EN_FRIENDLY) {
    if (re.test(t)) return true;
  }
  return /le prestataire|finalis[ée] c[ôo]t[ée] api|r[ée]essayez dans un instant|api met parfois quelques secondes/i.test(
    t
  );
}

/**
 * Retour Saferpay parcours invité — même schéma que `payment-return` (route racine + createURL)
 * afin d’éviter des chemins imbriqués difficiles en Expo Go.
 */
export default function GuestPaymentReturnScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ guestBookingId?: string; outcome?: string }>();
  const [msg, setMsg] = useState("Finalisation du paiement…");

  useEffect(() => {
    void (async () => {
      WebBrowser.maybeCompleteAuthSession();
      const gid =
        typeof params.guestBookingId === "string" ? params.guestBookingId.trim() : "";
      const outcome =
        typeof params.outcome === "string" ? params.outcome.trim().toLowerCase() : "success";
      if (!gid) {
        setMsg("Lien de retour invalide.");
        return;
      }
      const pending = await getGuestSaferpayPending();
      if (!pending || pending.guest_booking_id !== gid) {
        setMsg("Session de paiement introuvable. Rouvrez l'application depuis Réservation rapide.");
        return;
      }
      if (outcome === "fail" || outcome === "abort") {
        await setGuestSaferpayPending(null);
        setMsg("Paiement interrompu ou annulé.");
        router.replace({
          pathname: "/(public)/pre-request/guest-checkout",
          params: pending.draft_id ? { draftId: pending.draft_id } : {},
        } as any);
        return;
      }
      try {
        let last: AssertGuestBody | null = null;
        for (let i = 0; i < ASSERT_GUEST_BACKOFFS_MS.length; i++) {
          const waitMs = ASSERT_GUEST_BACKOFFS_MS[i] ?? 0;
          if (i > 0 && waitMs > 0) {
            setMsg("Validation du paiement en cours…");
            await new Promise((r) => setTimeout(r, waitMs));
          }
          last = (await assertGuestSaferpay({
            guest_booking_id: pending.guest_booking_id,
            status_token: pending.status_token,
          })) as AssertGuestBody;

          const top = String(last.status ?? "").toLowerCase();
          const paySt = String(last.payment_status ?? "").toLowerCase();
          if (paySt === "paid" && last.public_status_token && last.booking_id != null) {
            await setGuestSaferpayPending(null);
            await SecureStore.setItemAsync(PUBLIC_BOOKING_TOKEN_KEY, last.public_status_token);
            await SecureStore.setItemAsync(PUBLIC_BOOKING_ID_KEY, String(last.booking_id));
            router.replace({
              pathname: "/(public)/pre-request/guest-checkout",
              params: { paid: "1", bookingId: String(last.booking_id) },
            } as any);
            return;
          }
          const d = (last as { detail?: string }).detail;
          const canRetryThis =
            i < ASSERT_GUEST_BACKOFFS_MS.length - 1 &&
            (top === "assert_transient" || (top === "assert_failed" && isRetryableGuestAssertDetail(d)));
          if (canRetryThis) {
            setMsg(
              humanizeAssertMessage(d) ?? "Côté prestataire, court délai côté API. Nouvel essai…"
            );
            continue;
          }
          break;
        }
        if (!last) {
          setMsg("Réponse inattendue du serveur.");
          return;
        }
        const top = String(last.status ?? "").toLowerCase();
        const paySt = String(last.payment_status ?? "").toLowerCase();
        const lastDetail = (last as { detail?: string }).detail;
        if (top === "assert_transient" || (top === "assert_failed" && isRetryableGuestAssertDetail(lastDetail))) {
          setMsg(
            "La confirmation auprès de la banque met parfois quelques secondes. " +
              "Fermez cet écran, rouvrez « Payer en ligne » depuis l’écran précédent, " +
              "ou patientez : vous recevrez l’e-mail de confirmation dès que le serveur validera le paiement."
          );
          return;
        }
        if (top === "payment_failed" || paySt === "failed" || top === "assert_failed") {
          await setGuestSaferpayPending(null);
          setMsg(
            humanizeAssertMessage(lastDetail) ?? "Le paiement a été refusé ou n’a pas pu être confirmé."
          );
          return;
        }
        if (paySt === "pending_verification") {
          setMsg(
            "Le statut n’est pas encore confirmé. Revenez dans un instant à « Payer en ligne » ou relancez depuis l’écran de réservation."
          );
          return;
        }
        const friendly = humanizeAssertMessage(lastDetail);
        setMsg(
          friendly ?? "Le paiement n'a pas pu être confirmé. Réessayez ou contactez le support."
        );
      } catch (e) {
        setMsg(
          e instanceof Error
            ? e.message.startsWith("{")
              ? "Erreur côté prestataire de paiement. Réessayez dans quelques instants."
              : e.message
            : "Erreur réseau."
        );
      }
    })();
  }, [params.guestBookingId, params.outcome, router]);

  return (
    <Screen scroll backgroundColor="#F7FBFA" contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.center}>
          <ActivityIndicator color={brandPrimary} />
          <Text style={styles.msg}>{msg}</Text>
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 32,
  },
  center: {
    alignItems: "center",
    gap: 16,
  },
  msg: {
    textAlign: "center",
    color: "#334155",
    fontSize: FONT_SIZE.px15,
    lineHeight: 22,
  },
});
