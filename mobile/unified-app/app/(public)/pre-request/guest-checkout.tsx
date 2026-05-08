import * as Linking from "expo-linking";
import * as WebBrowser from "expo-web-browser";
import Constants from "expo-constants";
import { Ionicons } from "@expo/vector-icons";
import * as Clipboard from "expo-clipboard";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  ImageBackground,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { ResponsiveContainer, Screen, useAppViewport } from "../../../src/design/responsive";
import {
  createGuestBooking,
  initializeGuestSaferpay,
  previewGuestBooking,
} from "../../../src/core/api/client";
import {
  type PublicPreRequestDraft,
  loadPublicPreRequestDraft,
} from "../../../src/core/public/preRequestDraft";
import {
  getGuestSaferpayPending,
  setGuestSaferpayPending,
} from "../../../src/core/public/guestSaferpayPending";
import * as SecureStore from "../../../src/core/storage/secureStoreCompat";

const LANDING_BACKGROUND = require("../../../assets/images/landing-background.png");

const PUBLIC_BOOKING_TOKEN_KEY = "public_booking_status_token_v1";

function todayYyyyMmDdLocal(): string {
  const d = new Date();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${m}-${day}`;
}

function nowHhMmLocal(): string {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/** Heure affichée / envoyée API pour « Dès que possible » : aujourd'hui = heure actuelle, pas celle figée le matin. */
function effectivePickupTimeForGuestDraft(d: PublicPreRequestDraft): string {
  const d0 = d.date?.trim() ?? "";
  const t0 = d.pickup_time?.trim() ?? "";
  if (d.reservation_urgency !== "immediate" || d.pickup_schedule_exact === true) {
    return t0;
  }
  if (d0 && d0 === todayYyyyMmDdLocal()) {
    return nowHhMmLocal();
  }
  return t0;
}

/**
 * Même exigence serveur qu’un client authentifié (allowlist schéma) : Expo Go → createURL
 * (souvent exp:); app Lirie / dev client → lirie:// explicite (préfixe whiteliste backend).
 */
function buildGuestSaferpayReturnUrl(guestBookingId: string): string {
  if (Constants.appOwnership === "expo") {
    return Linking.createURL("guest-payment-return", {
      queryParams: { guestBookingId },
    });
  }
  return `lirie://guest-payment-return?${new URLSearchParams({ guestBookingId }).toString()}`;
}

function formatPickupLabel(date: string, pickupTime: string): string {
  const d = date.trim();
  const t = (pickupTime || "").trim();
  if (!d) return t || "—";

  const parts = d.split("-").map((p) => Number.parseInt(p, 10));
  if (parts.length === 3 && parts.every((n) => !Number.isNaN(n))) {
    const [y, m, day] = parts;
    const localMidnight = new Date(y, m - 1, day);
    if (!t) {
      if (Number.isNaN(localMidnight.getTime())) return d;
      return new Intl.DateTimeFormat("fr-CH", {
        weekday: "long",
        day: "numeric",
        month: "long",
        year: "numeric",
      }).format(localMidnight);
    }
    const timePart = t.length === 5 ? `${t}:00` : t;
    const withTime = new Date(`${d}T${timePart}`);
    if (!Number.isNaN(withTime.getTime())) {
      return new Intl.DateTimeFormat("fr-CH", {
        weekday: "long",
        day: "numeric",
        month: "long",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }).format(withTime);
    }
  }

  return [d, t].filter(Boolean).join(" · ");
}

export default function GuestCheckoutScreen() {
  const router = useRouter();
  const { topInset, bottomInset } = useAppViewport();
  const params = useLocalSearchParams<{
    draftId?: string;
    paid?: string;
    bookingId?: string;
  }>();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [paying, setPaying] = useState(false);
  /** `undefined` = en cours de lecture SecureStore, `null` = absent, sinon jeton de suivi public. */
  const [trackingCode, setTrackingCode] = useState<string | null | undefined>(undefined);
  const [postPayModalVisible, setPostPayModalVisible] = useState(false);
  const [copyHint, setCopyHint] = useState<string | null>(null);
  const postPayModalOnceRef = useRef(false);
  const [recap, setRecap] = useState<{
    departure: string;
    destination: string;
    date: string;
    pickup_time: string;
    amount: number;
    transport_type: string;
    trip_type: "one_way" | "round_trip";
  } | null>(null);

  const paid = params.paid === "1";
  const bookingIdParam = typeof params.bookingId === "string" ? params.bookingId.trim() : "";

  const loadRecap = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const draft = await loadPublicPreRequestDraft();
      const expectedDraftId =
        typeof params.draftId === "string" ? params.draftId.trim() : "";
      if (!draft) {
        setError("Aucun brouillon de trajet. Recommencez depuis l’accueil.");
        setRecap(null);
        return;
      }
      if (expectedDraftId && draft.draft_id !== expectedDraftId) {
        setError("Ce brouillon ne correspond pas au lien ouvert.");
        setRecap(null);
        return;
      }
      const tripType = draft.trip_type === "round_trip" ? "round_trip" : "one_way";
      const pickupTimeEffective = effectivePickupTimeForGuestDraft(draft);
      const preview = await previewGuestBooking({
        departure: draft.departure,
        destination: draft.destination,
        date: draft.date,
        pickup_time: pickupTimeEffective,
        trip_type: tripType,
      });
      setRecap({
        departure: draft.departure,
        destination: draft.destination,
        date: draft.date,
        pickup_time: pickupTimeEffective,
        amount: preview.pricing.amount,
        transport_type: draft.transport_type ?? "assis",
        trip_type: tripType,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Impossible de charger le récapitulatif.");
      setRecap(null);
    } finally {
      setLoading(false);
    }
  }, [params.draftId]);

  useEffect(() => {
    if (paid) {
      setLoading(false);
      return;
    }
    setTrackingCode(undefined);
    void loadRecap();
  }, [paid, loadRecap]);

  useEffect(() => {
    if (!paid) return;
    void (async () => {
      const token = await SecureStore.getItemAsync(PUBLIC_BOOKING_TOKEN_KEY);
      setTrackingCode(token?.trim() ? token.trim() : null);
    })();
  }, [paid]);

  useEffect(() => {
    if (!paid || !bookingIdParam) return;
    if (postPayModalOnceRef.current) return;
    postPayModalOnceRef.current = true;
    setPostPayModalVisible(true);
  }, [paid, bookingIdParam]);

  const showCopyHint = (label: string) => {
    setCopyHint(label);
    setTimeout(() => setCopyHint(null), 2500);
  };

  const copyDossier = async () => {
    if (!bookingIdParam) return;
    try {
      await Clipboard.setStringAsync(bookingIdParam);
      showCopyHint("N° de dossier copié");
    } catch {
      /* ignore */
    }
  };

  const copyLongToken = async (t: string) => {
    if (!t?.trim()) return;
    try {
      await Clipboard.setStringAsync(t.trim());
      showCopyHint("Code de secours copié");
    } catch {
      /* ignore */
    }
  };

  const onPostPayCompris = () => {
    setPostPayModalVisible(false);
    Alert.alert(
      "Contact (recommandé)",
      "Souhaitez-vous laisser un numéro pour être prévenu en cas de changement ? Vous pourrez aussi le faire plus tard depuis le suivi.",
      [{ text: "Plus tard", style: "cancel" }, { text: "OK" }],
    );
  };

  const startPayment = async () => {
    if (!recap) return;
    const draft = await loadPublicPreRequestDraft();
    if (!draft) {
      setError("Brouillon introuvable.");
      return;
    }
    setPaying(true);
    setError(null);
    try {
      const created = await createGuestBooking({
        departure: recap.departure,
        destination: recap.destination,
        date: recap.date,
        pickup_time: recap.pickup_time,
        trip_type: recap.trip_type,
        passengers: draft.passengers ?? 1,
        transport_type: recap.transport_type,
        first_name: draft.contact_first_name,
        last_name: draft.contact_last_name,
        email: draft.contact_email,
        phone: draft.contact_phone,
        notes: draft.special_requirements,
      });
      await setGuestSaferpayPending({
        status_token: created.status_token,
        guest_booking_id: created.guest_booking_id,
        draft_id: draft.draft_id,
      });
      const returnUrl = buildGuestSaferpayReturnUrl(created.guest_booking_id);
      const init = await initializeGuestSaferpay({
        guest_booking_id: created.guest_booking_id,
        status_token: created.status_token,
        return_url: returnUrl,
      });
      await WebBrowser.openBrowserAsync(init.redirect_url);
      WebBrowser.maybeCompleteAuthSession();
      // SFSafari / Chrome CCT bloquent souvent lirie:// : à la fermeture du navigateur, on enchaîne en app.
      const stillPending = await getGuestSaferpayPending();
      if (stillPending) {
        router.push({
          pathname: "/guest-payment-return",
          params: {
            guestBookingId: stillPending.guest_booking_id,
            outcome: "verify",
          },
        } as any);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Paiement impossible pour le moment.");
    } finally {
      setPaying(false);
    }
  };

  const goBack = () => {
    if (router.canGoBack()) {
      router.back();
      return;
    }
    router.replace("/(public)" as any);
  };

  if (paid) {
    return (
      <View style={styles.screen}>
        <Modal
          visible={postPayModalVisible}
          animationType="fade"
          transparent
          onRequestClose={() => setPostPayModalVisible(false)}
        >
          <View style={styles.postPayModalRoot}>
            <Pressable
              style={styles.postPayModalBackdrop}
              onPress={() => setPostPayModalVisible(false)}
              accessibilityRole="button"
              accessibilityLabel="Fermer"
            />
            <View
              style={[
                styles.postPayModalSheet,
                {
                  marginTop: Math.max(topInset, 12),
                  marginBottom: Math.max(bottomInset, 16),
                },
              ]}
            >
              <View style={styles.postPayCard}>
              <Text style={styles.postPayTitle}>Réservation rapide confirmée</Text>
              {bookingIdParam ? (
                <>
                  <Text style={styles.postPayDossierLabel}>N° de dossier (à utiliser pour le suivi)</Text>
                  <View style={styles.postPayDossierRow}>
                    <Text
                      style={styles.postPayDossierValue}
                      selectable
                      numberOfLines={1}
                    >
                      {bookingIdParam}
                    </Text>
                    <Pressable
                      onPress={() => void copyDossier()}
                      style={({ pressed }) => [
                        styles.postPayCopyBtn,
                        pressed && styles.postPayCopyBtnPressed,
                      ]}
                      accessibilityRole="button"
                      accessibilityLabel="Copier le numéro de dossier"
                    >
                      <Text style={styles.postPayCopyBtnText}>Copier</Text>
                    </Pressable>
                  </View>
                </>
              ) : null}
              <Text style={styles.postPayBody}>
                Accueil → « Suivi réservation » : saisir d’abord le n° de dossier. Le long code
                n’est requis qu’en secours.
              </Text>
              <Text style={styles.postPaySubLabel}>Code de secours (optionnel)</Text>
              {trackingCode === undefined ? (
                <View style={styles.postPayTokenLoading}>
                  <ActivityIndicator size="small" color="#0A8F7A" />
                </View>
              ) : trackingCode ? (
                <View>
                  <ScrollView
                    style={styles.postPayTokenScroll}
                    nestedScrollEnabled
                    showsVerticalScrollIndicator
                  >
                    <Text style={styles.postPayTokenText} selectable>
                      {trackingCode}
                    </Text>
                  </ScrollView>
                  <Pressable
                    onPress={() => void copyLongToken(trackingCode)}
                    style={({ pressed }) => [
                      styles.postPayCopyWide,
                      pressed && styles.postPayCopyBtnPressed,
                    ]}
                    accessibilityRole="button"
                    accessibilityLabel="Copier le code de secours"
                  >
                    <Ionicons name="copy-outline" size={18} color="#0A8F7A" />
                    <Text style={styles.postPayCopyWideText}>Copier le code de secours</Text>
                  </Pressable>
                </View>
              ) : (
                <Text style={styles.postPayTokenEmpty}>Aucun jeton : le n° suffit</Text>
              )}
              <Text style={styles.postPayWarn}>
                Sans téléphone ni e-mail, nous ne pourrons pas vous prévenir en cas de changement
                d’horaire.
              </Text>
              {copyHint ? <Text style={styles.postPayCopyHint}>{copyHint}</Text> : null}
              <View style={styles.postPayActions}>
                <Pressable
                  onPress={onPostPayCompris}
                  style={({ pressed }) => [
                    styles.postPayCompris,
                    pressed && styles.postPayComprisPressed,
                  ]}
                >
                  <Text style={styles.postPayComprisText}>Compris</Text>
                </Pressable>
              </View>
            </View>
          </View>
        </View>
        </Modal>
        <ImageBackground
          source={LANDING_BACKGROUND}
          style={StyleSheet.absoluteFillObject}
          resizeMode="cover"
          imageStyle={styles.backgroundImage}
        />
        <View style={styles.overlay} />
        <Screen
          scroll
          backgroundColor="transparent"
          keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
          contentContainerStyle={styles.screenScrollContent}
        >
          <ResponsiveContainer>
            <View style={styles.card}>
            <Pressable
              onPress={goBack}
              style={styles.backButton}
              accessibilityRole="button"
              accessibilityLabel="Retour"
            >
              <Ionicons name="arrow-back" size={22} color="#0A8F7A" />
            </Pressable>
            <Text style={styles.stepPill}>Paiement</Text>
            <Text style={styles.title}>Merci pour votre paiement</Text>
            <Text style={styles.subtitle}>
              Votre réservation est en cours de traitement. Indiquez le n° de dossier ci-dessous dans
              Accueil → « Suivi réservation » (c’est le plus simple).
            </Text>
            {bookingIdParam ? (
              <View style={styles.trackingBlock}>
                <Text style={styles.trackingRefLabel}>N° de dossier (suivi)</Text>
                <Text
                  style={styles.trackingRefValue}
                  selectable
                  accessibilityLabel={`Numéro de dossier ${bookingIdParam}`}
                >
                  {bookingIdParam}
                </Text>
                <Text style={styles.trackingHint}>
                  Saisir ce seul numéro dans l’écran de suivi — inutile de noter le long code plus bas,
                  sauf en secours.
                </Text>
              </View>
            ) : null}
            <View style={styles.trackingBlockSecondary}>
              <Text style={styles.trackingRefLabelSmall}>Code de secours (optionnel)</Text>
              {trackingCode === undefined ? (
                <View style={styles.trackingCodeLoading}>
                  <ActivityIndicator color="#0A8F7A" />
                </View>
              ) : trackingCode ? (
                <Text
                  style={styles.trackingCodeValue}
                  selectable
                  accessibilityLabel={`Code de secours : ${trackingCode}`}
                >
                  {trackingCode}
                </Text>
              ) : (
                <Text style={styles.trackingCodeMissing}>
                  Non requis : utilisez le n° de dossier, ou l’e-mail reçu / l’assistance.
                </Text>
              )}
            </View>
            <Pressable
              onPress={() => router.replace("/(public)" as any)}
              style={({ pressed }) => [styles.primaryButton, pressed && styles.primaryButtonPressed]}
              accessibilityRole="button"
            >
              <Text style={styles.primaryButtonText}>Retour à l’accueil</Text>
              <Ionicons name="home-outline" size={20} color="#FFFFFF" style={styles.primaryIcon} />
            </Pressable>
          </View>
          </ResponsiveContainer>
        </Screen>
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <ImageBackground
        source={LANDING_BACKGROUND}
        style={StyleSheet.absoluteFillObject}
        resizeMode="cover"
        imageStyle={styles.backgroundImage}
      />
      <View style={styles.overlay} />

      <Screen
        scroll
        backgroundColor="transparent"
        keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
        contentContainerStyle={styles.screenScrollContent}
      >
        <ResponsiveContainer>
          <View style={styles.card}>
          <Pressable
            onPress={goBack}
            style={styles.backButton}
            accessibilityRole="button"
            accessibilityLabel="Retour"
          >
            <Ionicons name="arrow-back" size={22} color="#0A8F7A" />
          </Pressable>

          <Text style={styles.stepPill}>Paiement sécurisé</Text>
          <Text style={styles.title}>Réservation rapide</Text>
          <Text style={styles.subtitle}>
            Vérifiez les informations du trajet, puis procédez au paiement en ligne.
          </Text>

          <View style={styles.immediateBox}>
            <Text style={styles.immediateSub}>
              Aucun compte n’est requis. Un téléphone ou un e-mail reste recommandé pour toute relance
              ou changement d’horaire.
            </Text>
          </View>

          {loading ? (
            <View style={styles.loadingBlock}>
              <ActivityIndicator size="large" color="#0A8F7A" />
              <Text style={styles.loadingHint}>Calcul du tarif…</Text>
            </View>
          ) : error ? (
            <View style={[styles.statusBanner, styles.statusBannerError]}>
              <Ionicons name="alert-circle-outline" size={22} color="#B42318" />
              <View style={styles.statusBannerTextWrap}>
                <Text style={styles.statusBannerTitle}>Récapitulatif indisponible</Text>
                <Text style={styles.statusBannerMessage}>{error}</Text>
              </View>
            </View>
          ) : recap ? (
            <View style={styles.routeSummary}>
              <Text style={styles.sectionHeadingInCard}>Récapitulatif</Text>
              <View style={styles.routeSummaryBlock}>
                <Text style={styles.routeSummaryLabel}>Départ</Text>
                <Text style={styles.routeSummaryValue}>{recap.departure}</Text>
              </View>
              <View style={styles.routeSummaryBlock}>
                <Text style={styles.routeSummaryLabel}>Destination</Text>
                <Text style={styles.routeSummaryValue}>{recap.destination}</Text>
              </View>
              <View style={styles.routeSummaryBlock}>
                <Text style={styles.routeSummaryLabel}>Prise en charge</Text>
                <Text style={styles.routeSummaryValue}>
                  {formatPickupLabel(recap.date, recap.pickup_time)}
                </Text>
              </View>
              <View style={styles.amountBlock}>
                <Text style={styles.routeSummaryLabel}>Montant à payer</Text>
                <Text style={styles.amountValue}>
                  {recap.amount.toLocaleString("fr-CH", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}{" "}
                  CHF
                </Text>
                <Text style={styles.amountNote}>
                  Calculé sur le serveur avec le même barème que pour les réservations client (prix
                  figé pour le paiement en ligne).
                </Text>
              </View>
            </View>
          ) : null}

          {!loading && recap ? (
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Payer en ligne avec Saferpay"
              style={({ pressed }) => [
                styles.primaryButton,
                paying && styles.primaryButtonDisabled,
                pressed && !paying && styles.primaryButtonPressed,
              ]}
              disabled={paying}
              onPress={() => void startPayment()}
            >
              {paying ? (
                <ActivityIndicator color="#FFFFFF" />
              ) : (
                <>
                  <Text style={styles.primaryButtonText}>Payer en ligne</Text>
                  <Ionicons name="lock-closed-outline" size={20} color="#FFFFFF" style={styles.primaryIcon} />
                </>
              )}
            </Pressable>
          ) : null}

          <Pressable
            accessibilityRole="button"
            style={({ pressed }) => [
              styles.outlineButton,
              pressed && styles.outlineButtonPressed,
            ]}
            onPress={goBack}
          >
            <Text style={styles.outlineButtonText}>Modifier le trajet</Text>
          </Pressable>
        </View>
        </ResponsiveContainer>
      </Screen>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#EAF3F1",
  },
  backgroundImage: {
    opacity: 0.08,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(234,243,241,0.88)",
  },
  screenScrollContent: {
    flexGrow: 1,
    paddingTop: 8,
  },
  card: {
    width: "100%",
    maxWidth: 440,
    alignSelf: "center",
    borderRadius: 26,
    padding: 24,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#FFFFFF",
    ...Platform.select({
      web: { boxShadow: "0 20px 48px rgba(22,58,52,0.12)" },
      default: {
        shadowColor: "#163A34",
        shadowOpacity: 0.12,
        shadowRadius: 18,
        shadowOffset: { width: 0, height: 8 },
        elevation: 4,
      },
    }),
  },
  backButton: {
    alignSelf: "flex-start",
    paddingVertical: 6,
    paddingHorizontal: 2,
    marginBottom: 10,
  },
  stepPill: {
    alignSelf: "flex-start",
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0.4,
    textTransform: "uppercase",
    color: "#0A8F7A",
    backgroundColor: "rgba(10,143,122,0.12)",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    overflow: "hidden",
    marginBottom: 14,
  },
  title: {
    color: "#163A34",
    fontSize: 28,
    lineHeight: 32,
    fontWeight: "700",
  },
  subtitle: {
    color: "#5F7369",
    fontSize: 15,
    lineHeight: 22,
    marginTop: 10,
  },
  trackingBlock: {
    marginTop: 16,
    padding: 14,
    borderRadius: 12,
    backgroundColor: "rgba(10,143,122,0.12)",
    borderWidth: 1,
    borderColor: "rgba(10,143,122,0.4)",
    gap: 6,
  },
  trackingBlockSecondary: {
    marginTop: 12,
    padding: 12,
    borderRadius: 10,
    backgroundColor: "rgba(100,116,106,0.08)",
    borderWidth: 1,
    borderColor: "rgba(100,116,106,0.2)",
    gap: 4,
  },
  trackingRefLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: "#5F7369",
    textTransform: "uppercase",
    letterSpacing: 0.3,
  },
  trackingRefLabelSmall: {
    fontSize: 11,
    fontWeight: "600",
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: 0.2,
  },
  trackingRefValue: {
    fontSize: 22,
    fontWeight: "800",
    color: "#0a5c4a",
    letterSpacing: 0.5,
  },
  trackingHint: {
    marginTop: 4,
    fontSize: 13,
    lineHeight: 18,
    color: "#5F7369",
  },
  trackingCodeValue: {
    fontSize: 12,
    lineHeight: 16,
    fontWeight: "500",
    color: "#475569",
    fontFamily: Platform.select({ web: "monospace" as const, default: "monospace" as const }),
  },
  trackingCodeLoading: {
    paddingVertical: 8,
    alignItems: "flex-start",
  },
  trackingCodeMissing: {
    fontSize: 13,
    lineHeight: 19,
    color: "#64748b",
  },
  immediateBox: {
    marginTop: 16,
    padding: 12,
    borderRadius: 12,
    backgroundColor: "rgba(10,143,122,0.08)",
    borderWidth: 1,
    borderColor: "rgba(10,143,122,0.25)",
  },
  immediateSub: {
    fontSize: 14,
    lineHeight: 20,
    color: "#5F7369",
  },
  loadingBlock: { marginTop: 20, alignItems: "center", gap: 12, paddingVertical: 20 },
  loadingHint: { color: "#5F7369", fontSize: 14 },
  statusBanner: {
    marginTop: 16,
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 10,
    padding: 14,
    borderRadius: 14,
    borderWidth: 1,
  },
  statusBannerError: {
    backgroundColor: "rgba(180,35,24,0.08)",
    borderColor: "rgba(180,35,24,0.35)",
  },
  statusBannerTextWrap: {
    flex: 1,
  },
  statusBannerTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: "#163A34",
    marginBottom: 4,
  },
  statusBannerMessage: {
    fontSize: 14,
    lineHeight: 20,
    color: "#5F7369",
  },
  routeSummary: {
    marginTop: 18,
    padding: 14,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#F3F8F6",
    gap: 12,
  },
  sectionHeadingInCard: {
    fontSize: 13,
    fontWeight: "700",
    color: "#5F7369",
    textTransform: "uppercase",
    letterSpacing: 0.4,
    marginBottom: 2,
  },
  routeSummaryBlock: {
    gap: 4,
  },
  routeSummaryLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: "#5F7369",
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  routeSummaryValue: {
    fontSize: 16,
    fontWeight: "600",
    color: "#163A34",
    lineHeight: 22,
  },
  amountBlock: {
    marginTop: 4,
    paddingTop: 14,
    borderTopWidth: 1,
    borderTopColor: "rgba(145,165,157,0.35)",
    gap: 6,
  },
  amountValue: {
    fontSize: 26,
    fontWeight: "800",
    color: "#163A34",
    letterSpacing: -0.4,
  },
  amountNote: {
    fontSize: 12,
    lineHeight: 16,
    color: "#5F7369",
  },
  primaryButton: {
    marginTop: 18,
    minHeight: 54,
    borderRadius: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0A8F7A",
    paddingHorizontal: 20,
  },
  primaryButtonDisabled: {
    backgroundColor: "#84B7AE",
  },
  primaryButtonPressed: {
    opacity: 0.92,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  primaryIcon: {
    marginLeft: 8,
  },
  outlineButton: {
    marginTop: 14,
    minHeight: 52,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: "#0A8F7A",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFFFFF",
  },
  outlineButtonPressed: {
    backgroundColor: "rgba(10,143,122,0.06)",
  },
  outlineButtonText: {
    color: "#0A8F7A",
    fontSize: 16,
    fontWeight: "700",
  },
  postPayModalRoot: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "center",
    paddingHorizontal: 20,
    pointerEvents: "box-none",
  },
  postPayModalBackdrop: {
    ...StyleSheet.absoluteFillObject,
  },
  postPayModalSheet: {
    width: "100%",
    maxWidth: 440,
    maxHeight: "88%",
    alignSelf: "center",
    zIndex: 1,
  },
  postPayCard: {
    borderRadius: 20,
    padding: 20,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.4)",
    ...Platform.select({
      web: { boxShadow: "0 16px 40px rgba(22,58,52,0.2)" },
      default: {
        shadowColor: "#000",
        shadowOpacity: 0.2,
        shadowRadius: 20,
        shadowOffset: { width: 0, height: 8 },
        elevation: 8,
      },
    }),
  },
  postPayTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: "#163A34",
    marginBottom: 12,
  },
  postPayDossierLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: "#5F7369",
    textTransform: "uppercase",
    letterSpacing: 0.3,
    marginBottom: 6,
  },
  postPayDossierRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: 10,
  },
  postPayDossierValue: {
    flex: 1,
    minWidth: 0,
    fontSize: 22,
    fontWeight: "800",
    color: "#0a5c4a",
  },
  postPayCopyBtn: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 10,
    backgroundColor: "rgba(10,143,122,0.12)",
    borderWidth: 1,
    borderColor: "rgba(10,143,122,0.4)",
  },
  postPayCopyBtnPressed: {
    opacity: 0.88,
  },
  postPayCopyBtnText: {
    fontSize: 14,
    fontWeight: "700",
    color: "#0A8F7A",
  },
  postPayBody: {
    fontSize: 14,
    lineHeight: 20,
    color: "#5F7369",
    marginBottom: 12,
  },
  postPaySubLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: "#64748b",
    marginBottom: 6,
  },
  postPayTokenLoading: {
    paddingVertical: 12,
    alignItems: "center",
  },
  postPayTokenScroll: {
    maxHeight: 100,
    padding: 10,
    borderRadius: 10,
    backgroundColor: "rgba(15, 23, 42, 0.06)",
    borderWidth: 1,
    borderColor: "rgba(100,116,106,0.25)",
    marginBottom: 10,
  },
  postPayTokenText: {
    fontSize: 11,
    lineHeight: 16,
    color: "#334155",
    fontFamily: Platform.select({ web: "monospace" as const, default: "monospace" as const }),
  },
  postPayTokenEmpty: {
    fontSize: 13,
    lineHeight: 19,
    color: "#64748b",
    marginBottom: 8,
    fontStyle: "italic",
  },
  postPayCopyWide: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 12,
    backgroundColor: "rgba(10,143,122,0.1)",
    borderWidth: 1.5,
    borderColor: "rgba(10,143,122,0.4)",
  },
  postPayCopyWideText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#0A8F7A",
  },
  postPayWarn: {
    fontSize: 13,
    lineHeight: 18,
    color: "#92400e",
    backgroundColor: "rgba(234, 179, 8, 0.12)",
    padding: 10,
    borderRadius: 10,
    marginTop: 4,
  },
  postPayCopyHint: {
    fontSize: 13,
    fontWeight: "600",
    color: "#0A8F7A",
    textAlign: "center",
    marginTop: 8,
  },
  postPayActions: {
    marginTop: 16,
  },
  postPayCompris: {
    alignSelf: "stretch",
    minHeight: 48,
    borderRadius: 12,
    backgroundColor: "#0A8F7A",
    alignItems: "center",
    justifyContent: "center",
  },
  postPayComprisPressed: {
    opacity: 0.92,
  },
  postPayComprisText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "uppercase",
  },
});
