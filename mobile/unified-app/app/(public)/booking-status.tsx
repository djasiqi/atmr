import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  ImageBackground,
  Keyboard,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { Redirect, useLocalSearchParams, useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { Screen, scrollAnchorAboveKeyboard, useAppViewport } from "../../src/design/responsive";
import {
  fetchGuestBookingStatus,
  fetchPublicBookingStatus,
  linkGuestBookingToAccount,
  PublicBookingStatusResponse,
} from "../../src/core/api/client";
import { useSession } from "../../src/core/sessionProvider";

const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");

type FallbackReason = "token_missing" | "token_invalid" | "token_expired";

function redirectForReason(reason: FallbackReason) {
  if (reason === "token_expired") {
    return `/(public)/fallback/expired-link?reason=${reason}`;
  }
  return `/(public)/fallback/invalid-link?reason=${reason}`;
}

export default function BookingStatusScreen() {
  const router = useRouter();
  const { topInset } = useAppViewport();
  const { bootstrap } = useSession();
  const params = useLocalSearchParams<{ token?: string }>();
  const [token, setToken] = useState((params.token ?? "").trim());
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PublicBookingStatusResponse | null>(null);
  const [guestResult, setGuestResult] = useState<{
    guest_booking_id: string;
    status: string;
    departure?: string;
    destination?: string;
    date?: string;
    pickup_time?: string;
    amount?: number;
    currency?: string;
    updated_at?: string;
    linked_to_account?: boolean;
  } | null>(null);
  const [redirectReason, setRedirectReason] = useState<FallbackReason | null>(null);
  const bookingScrollRef = useRef<ScrollView | null>(null);
  const bookingScrollOffsetYRef = useRef(0);
  const tokenFieldAnchorRef = useRef<View | null>(null);
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const [keyboardScrollPaddingBottom, setKeyboardScrollPaddingBottom] = useState(0);

  useEffect(() => {
    if (Platform.OS === "web") return;
    const show = Keyboard.addListener("keyboardDidShow", (e) => {
      const h = e.endCoordinates?.height ?? 0;
      const computed = h > 0 ? Math.round(h + 48) : 300;
      setKeyboardScrollPaddingBottom(Math.max(260, computed));
      setKeyboardVisible(true);
    });
    const hide = Keyboard.addListener("keyboardDidHide", () => {
      setKeyboardVisible(false);
      setKeyboardScrollPaddingBottom(0);
      bookingScrollRef.current?.scrollTo({ y: 0, animated: true });
      bookingScrollOffsetYRef.current = 0;
    });
    return () => {
      show.remove();
      hide.remove();
    };
  }, []);

  const submit = async () => {
    if (!token.trim()) {
      setRedirectReason("token_missing");
      return;
    }
    setPending(true);
    setError(null);
    try {
      const response = await fetchPublicBookingStatus(token.trim());
      setResult(response);
      setGuestResult(null);
    } catch (e: any) {
      const status = Number(e?.status ?? 0);
      if (status === 404 || status === 401) {
        try {
          const guestResponse = await fetchGuestBookingStatus(token.trim());
          setGuestResult(guestResponse);
          setResult(null);
          return;
        } catch {
          // fallback below
        }
      }
      if (status === 410) {
        setRedirectReason("token_expired");
        return;
      }
      if (status === 401 || status === 403 || status === 404) {
        setRedirectReason("token_invalid");
        return;
      }
      setError(e?.message ?? "Impossible de recuperer le statut.");
      setResult(null);
    } finally {
      setPending(false);
    }
  };

  if (redirectReason) {
    return <Redirect href={redirectForReason(redirectReason) as any} />;
  }

  const canSubmit = token.trim().length > 0 && !pending;

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
        automaticallyAdjustKeyboardInsets={Platform.OS !== "web"}
        androidKeyboardFallback={Platform.OS === "android"}
        scrollViewRef={bookingScrollRef}
        onScroll={(e) => {
          bookingScrollOffsetYRef.current = e.nativeEvent.contentOffset.y;
        }}
        scrollEventThrottle={16}
        contentContainerStyle={[
          styles.scrollContent,
          Platform.OS !== "web" && keyboardVisible
            ? [styles.scrollContentWithKeyboard, { paddingBottom: keyboardScrollPaddingBottom }]
            : null,
        ]}
      >
        <View style={styles.card}>
            <Pressable
              onPress={() => {
                if (router.canGoBack()) {
                  router.back();
                  return;
                }
                router.replace("/(public)" as any);
              }}
              style={styles.backButton}
              accessibilityRole="button"
              accessibilityLabel="Retour"
              hitSlop={12}
            >
              <Ionicons name="arrow-back" size={22} color="#0A8F7A" />
            </Pressable>

            <Text style={styles.kicker}>Sans connexion</Text>
            <Text style={styles.title}>Suivi de reservation</Text>
            <Text style={styles.subtitle}>
              Réservation rapide : saisissez le n° de dossier reçu après le paiement (chiffres uniquement).
              Sinon, collez le long code reçu par e-mail.
            </Text>

            <View ref={tokenFieldAnchorRef} collapsable={false} style={styles.fieldBlock}>
              <Text style={styles.fieldLabel}>N° de dossier ou code de suivi</Text>
              <TextInput
                value={token}
                onChangeText={(t) => {
                  setToken(t);
                  if (error) setError(null);
                }}
                placeholder="Ex. 30721 ou le code reçu par e-mail"
                placeholderTextColor="#91A59D"
                autoCapitalize="none"
                autoCorrect={false}
                autoComplete="off"
                textContentType="none"
                returnKeyType="go"
                onSubmitEditing={() => void submit()}
                onFocus={() =>
                  scrollAnchorAboveKeyboard(
                    bookingScrollRef,
                    bookingScrollOffsetYRef,
                    tokenFieldAnchorRef,
                  )
                }
                style={styles.fieldInput}
                editable={!pending}
                {...(Platform.OS === "android" ? { includeFontPadding: false } : {})}
              />
            </View>

            <Pressable
              onPress={() => void submit()}
              disabled={!canSubmit}
              style={[styles.submitButton, !canSubmit ? styles.submitButtonDisabled : null]}
              accessibilityRole="button"
              accessibilityState={{ disabled: !canSubmit }}
            >
              {pending ? (
                <ActivityIndicator color="#FFFFFF" />
              ) : (
                <Text style={styles.submitText}>Voir le statut</Text>
              )}
            </Pressable>

            {error ? <Text style={styles.errorText}>{error}</Text> : null}

            {result ? (
              <View style={styles.resultBlock}>
                <Text style={styles.resultHeading}>Resultat</Text>
                <Text style={styles.resultPrimary}>Reference : {result.booking_reference}</Text>
                <Text style={styles.resultStatus}>{result.label}</Text>
                <Text style={styles.resultMeta}>
                  Mise a jour :{" "}
                  {result.updated_at ? new Date(result.updated_at).toLocaleString() : "n/a"}
                </Text>
              </View>
            ) : null}

            {guestResult ? (
              <View style={styles.resultBlock}>
                <Text style={styles.resultHeading}>Reservation invite</Text>
                <Text style={styles.resultPrimary}>Reference : {guestResult.guest_booking_id}</Text>
                <Text style={styles.resultStatus}>Statut : {guestResult.status}</Text>
                <Text style={styles.resultLine}>
                  {guestResult.departure ?? "Depart"} {" \u2192 "}{" "}
                  {guestResult.destination ?? "Destination"}
                </Text>
                <Text style={styles.resultLine}>
                  {guestResult.date ?? "Date"} {guestResult.pickup_time ?? ""}
                </Text>
                <Text style={styles.resultLine}>
                  Montant : {guestResult.amount ?? 0} {guestResult.currency ?? "CHF"}
                </Text>
                <Text style={styles.resultMeta}>
                  Mise a jour :{" "}
                  {guestResult.updated_at
                    ? new Date(guestResult.updated_at).toLocaleString()
                    : "n/a"}
                </Text>

                {bootstrap?.is_authenticated ? (
                  <Pressable
                    onPress={async () => {
                      setPending(true);
                      setError(null);
                      try {
                        await linkGuestBookingToAccount(token.trim());
                        setGuestResult((prev) => (prev ? { ...prev, linked_to_account: true } : prev));
                      } catch (e: any) {
                        setError(e?.message ?? "Impossible d'associer cette reservation.");
                      } finally {
                        setPending(false);
                      }
                    }}
                    style={styles.secondaryButton}
                    disabled={pending || guestResult.linked_to_account}
                  >
                    <Text style={styles.secondaryButtonText}>
                      {guestResult.linked_to_account
                        ? "Deja associe a votre compte"
                        : "Associer a mon compte"}
                    </Text>
                  </Pressable>
                ) : (
                  <Pressable
                    onPress={() =>
                      router.push({
                        pathname: "/(public)/login",
                        params: {
                          next: `/(public)/booking-status?token=${encodeURIComponent(token.trim())}`,
                        },
                      } as any)
                    }
                    style={styles.secondaryButton}
                  >
                    <Text style={styles.secondaryButtonText}>
                      Se connecter pour associer au compte
                    </Text>
                  </Pressable>
                )}
              </View>
            ) : null}
          </View>
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
  scrollContent: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 12,
    paddingHorizontal: 16,
  },
  scrollContentWithKeyboard: {
    justifyContent: "flex-start",
    paddingTop: 28,
  },
  card: {
    width: "100%",
    maxWidth: 420,
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
    minWidth: 44,
    minHeight: 44,
    justifyContent: "center",
    marginBottom: 6,
    marginLeft: -6,
  },
  kicker: {
    color: "#0A8F7A",
    fontSize: 13,
    fontWeight: "600",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    marginBottom: 8,
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
  fieldBlock: {
    marginTop: 20,
  },
  fieldLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#163A34",
    marginBottom: 8,
  },
  fieldInput: {
    minHeight: 52,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#91A59D",
    backgroundColor: "#FAFCFB",
    paddingHorizontal: 14,
    color: "#163A34",
    fontSize: 16,
  },
  submitButton: {
    marginTop: 20,
    minHeight: 54,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0A8F7A",
  },
  submitButtonDisabled: {
    backgroundColor: "#84B7AE",
  },
  submitText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  errorText: {
    marginTop: 14,
    color: "#B42318",
    fontSize: 14,
    lineHeight: 20,
    fontWeight: "600",
  },
  resultBlock: {
    marginTop: 22,
    paddingTop: 18,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(145,165,157,0.6)",
    gap: 8,
  },
  resultHeading: {
    fontSize: 12,
    fontWeight: "700",
    color: "#0A8F7A",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    marginBottom: 4,
  },
  resultPrimary: {
    fontSize: 16,
    fontWeight: "700",
    color: "#163A34",
  },
  resultStatus: {
    fontSize: 15,
    fontWeight: "700",
    color: "#2E7D32",
  },
  resultLine: {
    fontSize: 15,
    lineHeight: 21,
    color: "#163A34",
  },
  resultMeta: {
    fontSize: 13,
    lineHeight: 18,
    color: "#5F7369",
    marginTop: 4,
  },
  secondaryButton: {
    marginTop: 16,
    minHeight: 50,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: "#0A8F7A",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 12,
    backgroundColor: "rgba(10,143,122,0.06)",
  },
  secondaryButtonText: {
    color: "#0A8F7A",
    fontSize: 15,
    fontWeight: "700",
    textAlign: "center",
  },
});
