import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Image,
  ImageBackground,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import DateTimePicker, { DateTimePickerEvent } from "@react-native-community/datetimepicker";
import * as ExpoLinking from "expo-linking";
import { apiClient } from "../../src/core/api/client";
import { autocompleteAddress } from "../../src/features/client/api";
import { AddressAutocompleteSuggestion } from "../../src/features/client/types";
import {
  AppInput,
  AppText,
  brandText,
  Screen,
  scrollAnchorAboveKeyboard,
  useAppViewport,
  useKeyboardHeight,
  useResponsiveTokens,
} from "../../src/design/responsive";
import { FONT_SIZE } from "../../src/design/responsive/typographyTokens";

const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");
const LIRIE_LOGO = require("../../assets/images/lirie-logo-color.png");

/** Pages légales officielles (site web Lirie), ouvertes dans le navigateur — aligné sur `contact.tsx` (`ExpoLinking.openURL`). */
const TERMS_URL = "https://www.lirie.ch/conditions";
const PRIVACY_URL = "https://www.lirie.ch/privacy";

const UI_SURFACE = "#F3F7F5";
const UI_MUTED = "#5F7369";
const UI_BORDER_SOFT = "rgba(145, 165, 157, 0.38)";

type Gender = "male" | "female" | "other";
type PhoneCountry = {
  code: string;
  name: string;
  dialCode: string;
};
type MobilityOption = "walk" | "cane" | "walker" | "wheelchair" | "oxygen" | "other";

const PHONE_COUNTRIES: PhoneCountry[] = [
  { code: "CH", name: "Suisse", dialCode: "+41" },
  { code: "FR", name: "France", dialCode: "+33" },
  { code: "DE", name: "Allemagne", dialCode: "+49" },
  { code: "IT", name: "Italie", dialCode: "+39" },
  { code: "AT", name: "Autriche", dialCode: "+43" },
  { code: "LI", name: "Liechtenstein", dialCode: "+423" },
  { code: "BE", name: "Belgique", dialCode: "+32" },
  { code: "NL", name: "Pays-Bas", dialCode: "+31" },
  { code: "ES", name: "Espagne", dialCode: "+34" },
];

const MOBILITY_OPTIONS: { value: MobilityOption; label: string }[] = [
  { value: "walk", label: "Marche" },
  { value: "cane", label: "Canne" },
  { value: "walker", label: "Déambulateur" },
  { value: "wheelchair", label: "Chaise roulante" },
  { value: "oxygen", label: "Sous O2" },
  { value: "other", label: "Autre" },
];

/** Aligné sur `forgot-password.tsx` (mêmes valeurs numériques que le StyleSheet là-bas). */
const AUTH_UI = {
  gapBlock: 12,
  gapLine: 6,
  cardPad: 24,
  cardMax: 420,
  radiusCard: 26,
  label: 13,
  ctaH: 54,
  ctaR: 14,
  titleSize: 30,
  titleLH: 34,
  subSize: 15,
  subLH: 21,
  logoH: 26,
  logoW: 168,
  fieldBlockMarginTop: 18,
  ctaMarginTop: 20,
} as const;

const INPUT_SHELL = {
  minHeight: 50,
  borderRadius: 14,
  paddingHorizontal: 14,
} as const;

const INPUT_TEXT = { fontSize: FONT_SIZE.px16, minHeight: 44 } as const;

function normalizeBirthDateInput(value: string): string | null {
  const raw = value.trim();
  if (!raw) return null;
  const digits = raw.replace(/\D/g, "");
  if (digits.length === 8) {
    const dd = digits.slice(0, 2);
    const mm = digits.slice(2, 4);
    const yyyy = digits.slice(4, 8);
    return `${yyyy}-${mm}-${dd}`;
  }
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  const dotMatch = raw.match(/^(\d{2})\.(\d{2})\.(\d{4})$/);
  if (!dotMatch) return null;
  const [, dd, mm, yyyy] = dotMatch;
  return `${yyyy}-${mm}-${dd}`;
}

function formatBirthDateInput(value: string): string {
  const digits = value.replace(/\D/g, "").slice(0, 8);
  if (digits.length <= 2) return digits;
  if (digits.length <= 4) return `${digits.slice(0, 2)}.${digits.slice(2)}`;
  return `${digits.slice(0, 2)}.${digits.slice(2, 4)}.${digits.slice(4)}`;
}

function formatDateToDisplay(value: Date): string {
  const day = String(value.getDate()).padStart(2, "0");
  const month = String(value.getMonth() + 1).padStart(2, "0");
  const year = String(value.getFullYear());
  return `${day}.${month}.${year}`;
}

function buildUsername(firstName: string, lastName: string, email: string): string {
  const normalize = (v: string) =>
    v
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .replace(/[^a-z0-9._-]/g, "");
  const fromNames = `${normalize(firstName)}.${normalize(lastName)}`
    .replace(/\.+/g, ".")
    .replace(/^\.|\.$/g, "");
  const fromEmail = normalize(email.split("@")[0] ?? "");
  const base = fromNames.length >= 3 ? fromNames : fromEmail.length >= 3 ? fromEmail : "client";
  return base.slice(0, 30);
}

function normalizePhoneForPayload(localPhone: string, dialCode: string): string {
  const raw = localPhone.trim().replace(/\s+/g, "");
  if (!raw) return "";
  if (raw.startsWith("+")) return raw;
  if (raw.startsWith("00")) return `+${raw.slice(2)}`;
  const normalizedLocal = raw.startsWith("0") ? raw.slice(1) : raw;
  return `${dialCode}${normalizedLocal}`;
}

function validateSignupForm(
  firstName: string,
  lastName: string,
  email: string,
  phone: string,
  password: string,
  confirmPassword: string,
  birthDateInput: string
): string | null {
  if (!firstName.trim()) return "Le prénom est requis.";
  if (!lastName.trim()) return "Le nom est requis.";
  const hasEmail = email.trim().length > 0;
  const hasPhone = phone.trim().length > 0;
  if (!hasEmail && !hasPhone) {
    return "Renseignez un courriel ou un numéro de téléphone.";
  }
  if (hasEmail && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
    return "Adresse email invalide.";
  }
  if (hasPhone && phone.trim().length < 7) {
    return "Un numéro de téléphone valide est requis (min. 7 caractères).";
  }
  if (!password || password.length < 8) {
    return "Le mot de passe doit contenir au moins 8 caractères.";
  }
  if (password !== confirmPassword) {
    return "La confirmation du mot de passe ne correspond pas.";
  }
  if (birthDateInput.trim() && !normalizeBirthDateInput(birthDateInput)) {
    return "Date de naissance invalide. Utilisez JJ.MM.AAAA.";
  }
  return null;
}

function validateStepOne(email: string, phone: string, password: string, confirmPassword: string): string | null {
  const hasEmail = email.trim().length > 0;
  const hasPhone = phone.trim().length > 0;
  if (!hasEmail && !hasPhone) {
    return "Renseignez un courriel ou un numéro de téléphone.";
  }
  if (hasEmail && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
    return "Adresse email invalide.";
  }
  if (hasPhone && phone.trim().length < 7) {
    return "Un numéro de téléphone valide est requis (min. 7 caractères).";
  }
  if (!password || password.length < 8) {
    return "Le mot de passe doit contenir au moins 8 caractères.";
  }
  if (password !== confirmPassword) {
    return "La confirmation du mot de passe ne correspond pas.";
  }
  return null;
}

function splitSuggestionLabel(value: string): { primary: string; secondary: string } {
  const raw = String(value || "").trim();
  if (!raw) return { primary: "", secondary: "" };
  const parts = raw.split(",").map((part) => part.trim()).filter(Boolean);
  if (parts.length <= 1) return { primary: raw, secondary: "" };
  return { primary: parts[0], secondary: parts.slice(1).join(", ") };
}

export default function SignupScreen() {
  const router = useRouter();
  const { topInset } = useAppViewport();
  const params = useLocalSearchParams<{ next?: string }>();
  const isWeb = Platform.OS === "web";
  const [currentStep, setCurrentStep] = useState<1 | 2>(1);
  const [googlePending, setGooglePending] = useState(false);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [birthDate, setBirthDate] = useState("");
  const [showBirthDatePicker, setShowBirthDatePicker] = useState(false);
  const [gender, setGender] = useState<Gender | null>(null);
  const [civilityOpen, setCivilityOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [phoneCountryOpen, setPhoneCountryOpen] = useState(false);
  const [phoneCountry, setPhoneCountry] = useState<PhoneCountry>(PHONE_COUNTRIES[0]);
  const [phone, setPhone] = useState("");
  const [mobilityOpen, setMobilityOpen] = useState(false);
  const [mobility, setMobility] = useState<MobilityOption | null>(null);
  const [mobilityOther, setMobilityOther] = useState("");
  const [addressLine, setAddressLine] = useState("");
  const [addressSuggestions, setAddressSuggestions] = useState<AddressAutocompleteSuggestion[]>([]);
  const [addressAutocompleteOpen, setAddressAutocompleteOpen] = useState(false);
  const [addressAutocompleteLoading, setAddressAutocompleteLoading] = useState(false);
  const [showAccessDetails, setShowAccessDetails] = useState(false);
  const [floorUnit, setFloorUnit] = useState("");
  const [intercomCode, setIntercomCode] = useState("");
  const [accessNote, setAccessNote] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [acceptedTerms, setAcceptedTerms] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const lastNameRef = useRef<TextInput | null>(null);
  const birthDateRef = useRef<TextInput | null>(null);
  const emailRef = useRef<TextInput | null>(null);
  const phoneRef = useRef<TextInput | null>(null);
  const addressRef = useRef<TextInput | null>(null);
  const floorRef = useRef<TextInput | null>(null);
  const intercomRef = useRef<TextInput | null>(null);
  const accessRef = useRef<TextInput | null>(null);
  const passwordRef = useRef<TextInput | null>(null);
  const confirmPasswordRef = useRef<TextInput | null>(null);
  const signupScrollRef = useRef<ScrollView | null>(null);
  const signupScrollOffsetYRef = useRef(0);
  const signupEmailAnchorRef = useRef<View | null>(null);
  const signupPhoneAnchorRef = useRef<View | null>(null);
  const signupPasswordAnchorRef = useRef<View | null>(null);
  const signupConfirmPasswordAnchorRef = useRef<View | null>(null);
  const signupFirstNameAnchorRef = useRef<View | null>(null);
  const signupLastNameAnchorRef = useRef<View | null>(null);
  const signupBirthDateAnchorRef = useRef<View | null>(null);
  const signupMobilityOtherAnchorRef = useRef<View | null>(null);
  const signupAddressAnchorRef = useRef<View | null>(null);
  const signupFloorAnchorRef = useRef<View | null>(null);
  const signupIntercomAnchorRef = useRef<View | null>(null);
  const signupAccessAnchorRef = useRef<View | null>(null);
  const addressBlurTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { keyboardVisible, scrollPaddingBottom: keyboardScrollPaddingBottom } = useKeyboardHeight();
  const responsiveTokens = useResponsiveTokens();

  useEffect(() => {
    if (keyboardVisible) return;
    signupScrollRef.current?.scrollTo({ y: 0, animated: true });
    signupScrollOffsetYRef.current = 0;
  }, [keyboardVisible]);

  const civilityLabel =
    gender === "male" ? "Homme" : gender === "female" ? "Femme" : gender === "other" ? "Autre" : "Civilité";
  const phonePlaceholder = phoneCountry.code === "CH" ? "76 770 40 41" : "Numéro local";
  const mobilityLabel = mobility
    ? MOBILITY_OPTIONS.find((item) => item.value === mobility)?.label ?? null
    : null;
  const mobilityTriggerLabel =
    mobility === "other" && mobilityOther.trim()
      ? `Autre (${mobilityOther.trim()})`
      : mobilityLabel ?? "Mobilité";
  const googleClientId = (process.env.EXPO_PUBLIC_GOOGLE_AUTH_CLIENT_ID ?? "").trim().replace(/\*+$/, "");
  const googleSignupUrl = (() => {
    const explicitUrl = (process.env.EXPO_PUBLIC_GOOGLE_SIGNUP_URL ?? "").trim();
    if (explicitUrl) return explicitUrl;
    if (!googleClientId) return "";
    const params = new URLSearchParams({
      client_id: googleClientId,
      redirect_uri: "https://app.lirie.ch",
      response_type: "code",
      scope: "openid email profile",
      access_type: "offline",
      prompt: "select_account",
    });
    return `https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`;
  })();

  const stepOneValid = validateStepOne(email, phone, password, confirmPassword) === null;

  const goToStepTwo = () => {
    const validationError = validateStepOne(email, phone, password, confirmPassword);
    if (validationError) {
      setError(validationError);
      return;
    }
    setError(null);
    setCurrentStep(2);
  };

  const submitWithGoogle = async () => {
    if (!googleSignupUrl) {
      setError(
        "Inscription Google non configurée. Définissez EXPO_PUBLIC_GOOGLE_AUTH_CLIENT_ID ou EXPO_PUBLIC_GOOGLE_SIGNUP_URL."
      );
      return;
    }
    try {
      setError(null);
      setGooglePending(true);
      await ExpoLinking.openURL(googleSignupUrl);
    } catch {
      setError("Impossible d'ouvrir l'inscription Google pour le moment.");
    } finally {
      setGooglePending(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const query = addressLine.trim();
    if (currentStep !== 2 || !addressAutocompleteOpen || query.length < 2) {
      setAddressSuggestions([]);
      setAddressAutocompleteLoading(false);
      return () => {
        cancelled = true;
      };
    }

    const timer = setTimeout(async () => {
      try {
        setAddressAutocompleteLoading(true);
        const results = await autocompleteAddress(query, { limit: 5 });
        if (!cancelled) {
          setAddressSuggestions(results.slice(0, 5));
        }
      } catch {
        if (!cancelled) {
          setAddressSuggestions([]);
        }
      } finally {
        if (!cancelled) {
          setAddressAutocompleteLoading(false);
        }
      }
    }, 220);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [addressAutocompleteOpen, addressLine, currentStep]);

  useEffect(() => {
    return () => {
      if (addressBlurTimeoutRef.current) {
        clearTimeout(addressBlurTimeoutRef.current);
      }
    };
  }, []);

  const submit = async () => {
    const validationError = validateSignupForm(
      firstName,
      lastName,
      email,
      phone,
      password,
      confirmPassword,
      birthDate
    );
    if (validationError) {
      setError(validationError);
      return;
    }
    if (!acceptedTerms) {
      setError("Veuillez accepter les CGU et la politique de confidentialité.");
      return;
    }

    const normalizedBirthDate = normalizeBirthDateInput(birthDate);
    const phonePayload = normalizePhoneForPayload(phone, phoneCountry.dialCode);
    const mobilityPayload =
      mobility === "other"
        ? mobilityOther.trim()
          ? `Autre (${mobilityOther.trim()})`
          : "Autre"
        : mobilityLabel;
    const addressPayload = [
      addressLine.trim(),
      mobilityPayload ? `Mobilité: ${mobilityPayload}` : "",
      floorUnit.trim() ? `Étage/appartement: ${floorUnit.trim()}` : "",
      intercomCode.trim() ? `Code/interphone: ${intercomCode.trim()}` : "",
      accessNote.trim() ? `Complément d'accès: ${accessNote.trim()}` : "",
    ]
      .filter(Boolean)
      .join(" · ");

    setPending(true);
    setError(null);
    try {
      const response = await apiClient.post<{
        activation_session_id: string;
        masked_email: string;
        masked_phone: string;
        email_sent: boolean;
        sms_sent: boolean;
      }>("/auth/register", {
        username: buildUsername(firstName, lastName, email),
        email: email.trim(),
        phone: phonePayload,
        password,
        first_name: firstName.trim() || null,
        last_name: lastName.trim() || null,
        birth_date: normalizedBirthDate,
        ...(gender ? { gender } : {}),
        address: addressPayload || null,
      });

      const { activation_session_id, masked_email, masked_phone } = response.data;
      router.replace({
        pathname: "/(public)/activate",
        params: {
          activation_session_id,
          masked_email: masked_email ?? "",
          masked_phone: masked_phone ?? "",
          ...(params.next ? { next: params.next } : {}),
        },
      } as any);
    } catch (e: any) {
      const msg =
        e?.response?.data?.message ||
        e?.response?.data?.error ||
        (e instanceof Error ? e.message : null) ||
        "Inscription impossible. Vérifiez vos informations.";
      setError(msg);
    } finally {
      setPending(false);
    }
  };

  const handleBirthDatePickerChange = (event: DateTimePickerEvent, selected?: Date) => {
    if (Platform.OS !== "ios") {
      setShowBirthDatePicker(false);
    }
    if (event.type === "dismissed" || !selected) return;
    setBirthDate(formatDateToDisplay(selected));
  };

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
        withHorizontalPadding={false}
        backgroundColor="transparent"
        keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
        automaticallyAdjustKeyboardInsets={Platform.OS !== "web"}
        androidKeyboardFallback={Platform.OS === "android"}
        scrollViewRef={signupScrollRef}
        onScroll={(e) => {
          signupScrollOffsetYRef.current = e.nativeEvent.contentOffset.y;
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
              router.replace("/(public)/login" as any);
            }}
            style={styles.backButton}
            accessibilityRole="button"
            accessibilityLabel="Retour"
          >
            <Ionicons name="arrow-back" size={22} color="#0A8F7A" />
          </Pressable>

          <View style={styles.logoBlock}>
            <Image
              source={LIRIE_LOGO}
              style={styles.logo}
              resizeMode="contain"
              accessibilityRole="image"
              accessibilityLabel="LIRIE"
            />
          </View>

          <Text style={styles.kicker} maxFontSizeMultiplier={1.22}>
            Informations
          </Text>
          <Text style={styles.title} maxFontSizeMultiplier={1.28}>
            Créer un compte
          </Text>
          <Text style={styles.subtitle} maxFontSizeMultiplier={1.45}>
            Coordonnées utilisées pour les réservations et les confirmations.
          </Text>

          <View style={styles.stepperWrap}>
            <View style={styles.stepItem}>
              <View
                style={[
                  styles.stepDot,
                  currentStep === 1 ? styles.stepDotActive : null,
                  currentStep === 2 ? styles.stepDotDone : null,
                ]}
              >
                {currentStep === 2 ? (
                  <Ionicons name="checkmark" size={12} color="#FFFFFF" />
                ) : (
                  <Text
                    style={[styles.stepDotText, currentStep === 1 ? styles.stepDotTextActive : null]}
                    maxFontSizeMultiplier={1.2}
                  >
                    1
                  </Text>
                )}
              </View>
              <Text
                style={[styles.stepLabel, currentStep >= 1 ? styles.stepLabelActive : null]}
                maxFontSizeMultiplier={1.22}
              >
                Contact & sécurité
              </Text>
            </View>

            <View style={[styles.stepConnector, currentStep === 2 ? styles.stepConnectorActive : null]} />

            <View style={styles.stepItem}>
              <View style={[styles.stepDot, currentStep === 2 ? styles.stepDotActive : null]}>
                <Text
                  style={[styles.stepDotText, currentStep === 2 ? styles.stepDotTextActive : null]}
                  maxFontSizeMultiplier={1.2}
                >
                  2
                </Text>
              </View>
              <Text
                style={[styles.stepLabel, currentStep === 2 ? styles.stepLabelActive : null]}
                maxFontSizeMultiplier={1.22}
              >
                Profil & adresse
              </Text>
            </View>
          </View>

          {currentStep === 1 ? (
            <>
              <Pressable
                onPress={() => void submitWithGoogle()}
                disabled={googlePending}
                style={[styles.googleButton, googlePending ? styles.googleButtonDisabled : null]}
                accessibilityRole="button"
                accessibilityLabel="S'inscrire avec Google"
              >
                {googlePending ? (
                  <ActivityIndicator color="#163A34" />
                ) : (
                  <>
                    <Ionicons name="logo-google" size={16} color="#DB4437" />
                    <Text style={styles.googleButtonText} maxFontSizeMultiplier={1.28}>
                      S&apos;inscrire avec Google
                    </Text>
                  </>
                )}
              </Pressable>

              <View style={styles.googleDivider}>
                <View style={styles.googleDividerLine} />
                <Text style={styles.googleDividerText} maxFontSizeMultiplier={1.22}>
                  ou
                </Text>
                <View style={styles.googleDividerLine} />
              </View>

              <View style={[styles.sectionBlock, { marginTop: AUTH_UI.gapBlock + 4 }]}>
                <Text
                  style={[styles.sectionTitle, { fontSize: AUTH_UI.label }]}
                  maxFontSizeMultiplier={1.22}
                >
                  Contact
                </Text>
                <View ref={signupEmailAnchorRef} collapsable={false} style={styles.fieldBlock}>
                  <AppInput
                    ref={emailRef}
                    value={email}
                    onChangeText={setEmail}
                    placeholder="Courriel (ou téléphone)"
                    placeholderTextColor="#91A59D"
                    autoCapitalize="none"
                    keyboardType="email-address"
                    autoComplete="email"
                    textContentType="emailAddress"
                    returnKeyType="next"
                    onSubmitEditing={() => phoneRef.current?.focus()}
                    onFocus={() =>
                      scrollAnchorAboveKeyboard(
                        signupScrollRef,
                        signupScrollOffsetYRef,
                        signupEmailAnchorRef,
                      )
                    }
                    shellStyle={INPUT_SHELL}
                    style={INPUT_TEXT}
                  />
                </View>
                <View ref={signupPhoneAnchorRef} collapsable={false} style={styles.fieldBlock}>
                  <View style={styles.phoneWrap}>
                    <View style={styles.phoneRow}>
                      <Pressable
                        onPress={() => setPhoneCountryOpen((v) => !v)}
                        style={styles.phoneCountryButton}
                        accessibilityRole="button"
                        accessibilityLabel="Indicatif pays"
                      >
                        <Text style={styles.phoneCountryCode} maxFontSizeMultiplier={1.22}>
                          {phoneCountry.code}
                        </Text>
                        <Text style={styles.phoneCountryDial} maxFontSizeMultiplier={1.22}>
                          {phoneCountry.dialCode}
                        </Text>
                        <Ionicons
                          name={phoneCountryOpen ? "chevron-up-outline" : "chevron-down-outline"}
                          size={14}
                          color="#5F7369"
                        />
                      </Pressable>

                      <TextInput
                        ref={phoneRef}
                        value={phone}
                        onChangeText={setPhone}
                        onFocus={() => {
                          setPhoneCountryOpen(false);
                          scrollAnchorAboveKeyboard(
                            signupScrollRef,
                            signupScrollOffsetYRef,
                            signupPhoneAnchorRef,
                          );
                        }}
                        placeholder={phonePlaceholder}
                        placeholderTextColor="#91A59D"
                        keyboardType="phone-pad"
                        autoComplete="tel"
                        textContentType="telephoneNumber"
                        returnKeyType="next"
                        onSubmitEditing={() => passwordRef.current?.focus()}
                        style={styles.phoneInput}
                      />
                    </View>

                    {phoneCountryOpen ? (
                      <View
                        style={[
                          styles.phoneCountryList,
                          { maxHeight: responsiveTokens.dropdownListMaxHeight },
                        ]}
                      >
                        {PHONE_COUNTRIES.map((country, index) => (
                          <Pressable
                            key={country.code}
                            onPress={() => {
                              setPhoneCountry(country);
                              setPhoneCountryOpen(false);
                            }}
                            style={[
                              styles.phoneCountryItem,
                              index === PHONE_COUNTRIES.length - 1 ? styles.phoneCountryItemLast : null,
                            ]}
                          >
                            <Text
                              style={[
                                styles.phoneCountryItemText,
                                phoneCountry.code === country.code ? styles.phoneCountryItemTextActive : null,
                              ]}
                            >
                              {country.name}
                            </Text>
                            <Text
                              style={[
                                styles.phoneCountryItemDial,
                                phoneCountry.code === country.code ? styles.phoneCountryItemDialActive : null,
                              ]}
                            >
                              {country.dialCode}
                            </Text>
                          </Pressable>
                        ))}
                      </View>
                    ) : null}
                  </View>
                </View>
              </View>

              <View style={[styles.sectionBlock, { marginTop: AUTH_UI.gapBlock + 10 }]}>
                <Text
                  style={[styles.sectionTitle, { fontSize: AUTH_UI.label }]}
                  maxFontSizeMultiplier={1.22}
                >
                  Sécurité
                </Text>
                <View ref={signupPasswordAnchorRef} collapsable={false} style={styles.fieldBlock}>
                  <AppInput
                    ref={passwordRef}
                    value={password}
                    onChangeText={setPassword}
                    placeholder="Min. 8 caractères"
                    placeholderTextColor="#91A59D"
                    secureTextEntry={!showPassword}
                    autoComplete="new-password"
                    textContentType="newPassword"
                    returnKeyType="next"
                    onSubmitEditing={() => confirmPasswordRef.current?.focus()}
                    onFocus={() =>
                      scrollAnchorAboveKeyboard(
                        signupScrollRef,
                        signupScrollOffsetYRef,
                        signupPasswordAnchorRef,
                      )
                    }
                    shellStyle={INPUT_SHELL}
                    style={INPUT_TEXT}
                    rightSlot={
                      <Pressable
                        onPress={() => setShowPassword((v) => !v)}
                        style={styles.passwordToggle}
                        accessibilityRole="button"
                        accessibilityLabel={showPassword ? "Masquer le mot de passe" : "Afficher le mot de passe"}
                      >
                        <Ionicons
                          name={showPassword ? "eye-off-outline" : "eye-outline"}
                          size={18}
                          color="#5F7369"
                        />
                      </Pressable>
                    }
                  />
                </View>
                <View ref={signupConfirmPasswordAnchorRef} collapsable={false} style={styles.fieldBlock}>
                  <AppInput
                    ref={confirmPasswordRef}
                    value={confirmPassword}
                    onChangeText={setConfirmPassword}
                    placeholder="Confirmer le mot de passe"
                    placeholderTextColor="#91A59D"
                    secureTextEntry={!showConfirmPassword}
                    autoComplete="new-password"
                    textContentType="newPassword"
                    returnKeyType="done"
                    onSubmitEditing={goToStepTwo}
                    onFocus={() =>
                      scrollAnchorAboveKeyboard(
                        signupScrollRef,
                        signupScrollOffsetYRef,
                        signupConfirmPasswordAnchorRef,
                      )
                    }
                    shellStyle={INPUT_SHELL}
                    style={INPUT_TEXT}
                    rightSlot={
                      <Pressable
                        onPress={() => setShowConfirmPassword((v) => !v)}
                        style={styles.passwordToggle}
                        accessibilityRole="button"
                        accessibilityLabel={
                          showConfirmPassword
                            ? "Masquer la confirmation du mot de passe"
                            : "Afficher la confirmation du mot de passe"
                        }
                      >
                        <Ionicons
                          name={showConfirmPassword ? "eye-off-outline" : "eye-outline"}
                          size={18}
                          color="#5F7369"
                        />
                      </Pressable>
                    }
                  />
                </View>
              </View>
            </>
          ) : (
            <>
              <View style={[styles.sectionBlock, { marginTop: AUTH_UI.gapBlock + 8 }]}>
                <Text
                  style={[styles.sectionTitle, { fontSize: AUTH_UI.label }]}
                  maxFontSizeMultiplier={1.22}
                >
                  Informations personnelles
                </Text>

                <View style={styles.fieldBlock}>
                  <View style={styles.civilityWrap}>
                    <Pressable
                      onPress={() => setCivilityOpen((v) => !v)}
                      style={styles.civilityTrigger}
                      accessibilityRole="button"
                      accessibilityLabel="Civilité"
                    >
                      <Text style={[styles.civilityValue, !gender ? styles.civilityPlaceholder : null]}>
                        {civilityLabel}
                      </Text>
                      <Ionicons
                        name={civilityOpen ? "chevron-up-outline" : "chevron-down-outline"}
                        size={16}
                        color="#5F7369"
                      />
                    </Pressable>

                    {civilityOpen ? (
                      <View style={styles.civilityList}>
                        {[
                          { value: "male" as const, label: "Homme" },
                          { value: "female" as const, label: "Femme" },
                          { value: "other" as const, label: "Autre" },
                        ].map((item, index, arr) => (
                          <Pressable
                            key={item.value}
                            onPress={() => {
                              setGender(item.value);
                              setCivilityOpen(false);
                            }}
                            style={[
                              styles.civilityOption,
                              index === arr.length - 1 ? styles.civilityOptionLast : null,
                            ]}
                            accessibilityRole="menuitem"
                          >
                            <Text
                              style={[
                                styles.civilityOptionText,
                                gender === item.value ? styles.civilityOptionTextActive : null,
                              ]}
                            >
                              {item.label}
                            </Text>
                            {gender === item.value ? (
                              <Ionicons name="checkmark" size={16} color="#0A8F7A" />
                            ) : null}
                          </Pressable>
                        ))}
                      </View>
                    ) : null}
                  </View>
                </View>

                <View ref={signupFirstNameAnchorRef} collapsable={false} style={styles.fieldBlock}>
                  <AppInput
                    value={firstName}
                    onChangeText={setFirstName}
                    placeholder="Prénom"
                    placeholderTextColor="#91A59D"
                    returnKeyType="next"
                    onSubmitEditing={() => lastNameRef.current?.focus()}
                    onFocus={() =>
                      scrollAnchorAboveKeyboard(
                        signupScrollRef,
                        signupScrollOffsetYRef,
                        signupFirstNameAnchorRef,
                      )
                    }
                    shellStyle={INPUT_SHELL}
                    style={INPUT_TEXT}
                  />
                </View>

                <View ref={signupLastNameAnchorRef} collapsable={false} style={styles.fieldBlock}>
                  <AppInput
                    ref={lastNameRef}
                    value={lastName}
                    onChangeText={setLastName}
                    placeholder="Nom"
                    placeholderTextColor="#91A59D"
                    returnKeyType="next"
                    onSubmitEditing={() => birthDateRef.current?.focus()}
                    onFocus={() =>
                      scrollAnchorAboveKeyboard(
                        signupScrollRef,
                        signupScrollOffsetYRef,
                        signupLastNameAnchorRef,
                      )
                    }
                    shellStyle={INPUT_SHELL}
                    style={INPUT_TEXT}
                  />
                </View>

                <View ref={signupBirthDateAnchorRef} collapsable={false} style={styles.fieldBlock}>
                  <View style={styles.dateWrap}>
                    <TextInput
                      ref={birthDateRef}
                      value={birthDate}
                      onChangeText={(value) => setBirthDate(formatBirthDateInput(value))}
                      placeholder="Date de naissance"
                      placeholderTextColor="#91A59D"
                      keyboardType="numbers-and-punctuation"
                      returnKeyType="next"
                      onSubmitEditing={() => addressRef.current?.focus()}
                      onFocus={() =>
                        scrollAnchorAboveKeyboard(
                          signupScrollRef,
                          signupScrollOffsetYRef,
                          signupBirthDateAnchorRef,
                        )
                      }
                      style={[styles.fieldInput, !isWeb ? styles.dateInput : null]}
                    />
                    {!isWeb ? (
                      <Pressable
                        onPress={() => setShowBirthDatePicker((v) => !v)}
                        style={styles.datePickerButton}
                        accessibilityRole="button"
                        accessibilityLabel="Ouvrir le sélecteur de date"
                      >
                        <Ionicons name="calendar-outline" size={16} color="#5F7369" />
                      </Pressable>
                    ) : null}
                  </View>
                  {!isWeb && showBirthDatePicker ? (
                    <DateTimePicker
                      value={
                        (() => {
                          const normalized = normalizeBirthDateInput(birthDate);
                          return normalized ? new Date(`${normalized}T00:00:00`) : new Date(1954, 10, 18);
                        })()
                      }
                      mode="date"
                      display={Platform.OS === "ios" ? "spinner" : "default"}
                      onChange={handleBirthDatePickerChange}
                      maximumDate={new Date()}
                    />
                  ) : null}
                </View>

                <View style={styles.fieldBlock}>
                  <Pressable
                    onPress={() => setMobilityOpen((v) => !v)}
                    style={styles.mobilityTrigger}
                    accessibilityRole="button"
                    accessibilityLabel="Mobilité"
                  >
                    <Text style={[styles.mobilityTriggerText, mobility ? null : styles.mobilityTriggerPlaceholder]}>
                      {mobilityTriggerLabel}
                    </Text>
                    <Ionicons
                      name={mobilityOpen ? "chevron-up-outline" : "chevron-down-outline"}
                      size={16}
                      color="#5F7369"
                    />
                  </Pressable>
                </View>
                {mobilityOpen ? (
                  <View style={styles.mobilityList}>
                    {MOBILITY_OPTIONS.map((item, index) => {
                      const selected = mobility === item.value;
                      const isLast = index === MOBILITY_OPTIONS.length - 1;
                      return (
                        <Pressable
                          key={item.value}
                          onPress={() => {
                            setMobility(item.value);
                            if (item.value !== "other") {
                              setMobilityOther("");
                            }
                            setMobilityOpen(false);
                          }}
                          style={[styles.mobilityOption, isLast ? styles.mobilityOptionLast : null]}
                          accessibilityRole="radio"
                          accessibilityState={{ checked: selected }}
                          accessibilityLabel={item.label}
                        >
                          <Text
                            style={[
                              styles.mobilityOptionText,
                              selected ? styles.mobilityOptionTextActive : null,
                            ]}
                          >
                            {item.label}
                          </Text>
                          {selected ? <Ionicons name="checkmark" size={16} color="#0A8F7A" /> : null}
                        </Pressable>
                      );
                    })}
                  </View>
                ) : null}

                {mobility === "other" ? (
                  <View ref={signupMobilityOtherAnchorRef} collapsable={false} style={styles.fieldBlock}>
                    <TextInput
                      value={mobilityOther}
                      onChangeText={setMobilityOther}
                      placeholder="Précisez la mobilité"
                      placeholderTextColor="#91A59D"
                      returnKeyType="next"
                      onSubmitEditing={() => addressRef.current?.focus()}
                      onFocus={() =>
                        scrollAnchorAboveKeyboard(
                          signupScrollRef,
                          signupScrollOffsetYRef,
                          signupMobilityOtherAnchorRef,
                        )
                      }
                      style={styles.fieldInput}
                    />
                  </View>
                ) : null}
              </View>

              <View style={[styles.sectionBlock, { marginTop: AUTH_UI.gapBlock + 10 }]}>
                <Text
                  style={[styles.sectionTitle, { fontSize: AUTH_UI.label }]}
                  maxFontSizeMultiplier={1.22}
                >
                  Adresse et accès
                </Text>
                <View ref={signupAddressAnchorRef} collapsable={false} style={styles.fieldBlock}>
                  <View style={styles.addressAutocompleteWrap}>
                    <TextInput
                      ref={addressRef}
                      value={addressLine}
                      onChangeText={(value) => {
                        setAddressLine(value);
                        setAddressAutocompleteOpen(true);
                      }}
                      onFocus={() => {
                        if (addressBlurTimeoutRef.current) {
                          clearTimeout(addressBlurTimeoutRef.current);
                        }
                        setAddressAutocompleteOpen(true);
                        scrollAnchorAboveKeyboard(
                          signupScrollRef,
                          signupScrollOffsetYRef,
                          signupAddressAnchorRef,
                        );
                      }}
                      onBlur={() => {
                        addressBlurTimeoutRef.current = setTimeout(() => {
                          setAddressAutocompleteOpen(false);
                        }, 140);
                      }}
                      placeholder="Avenue Ernest-Pictet 9, 1203 Genève"
                      placeholderTextColor="#91A59D"
                      returnKeyType="next"
                      onSubmitEditing={() => floorRef.current?.focus()}
                      style={[
                        styles.fieldInput,
                        addressAutocompleteOpen ? styles.fieldInputActive : null,
                      ]}
                    />

                    {addressAutocompleteOpen && addressAutocompleteLoading && addressLine.trim().length >= 2 ? (
                      <View style={styles.addressSuggestionMetaRow}>
                        <Text style={styles.addressSuggestionMetaText}>Recherche d&apos;adresses...</Text>
                      </View>
                    ) : null}

                    {addressAutocompleteOpen && addressSuggestions.length > 0 ? (
                      <View
                        style={[
                          styles.addressSuggestionList,
                          { maxHeight: responsiveTokens.dropdownListMaxHeight },
                        ]}
                      >
                        {addressSuggestions.map((item, index) => {
                          const label = item.address ?? item.label;
                          const { primary, secondary } = splitSuggestionLabel(label);
                          const isLast = index === addressSuggestions.length - 1;
                          return (
                            <Pressable
                              key={`${item.place_id ?? item.label}-${index}`}
                              onPressIn={() => {
                                if (addressBlurTimeoutRef.current) {
                                  clearTimeout(addressBlurTimeoutRef.current);
                                }
                              }}
                              onPress={() => {
                                setAddressLine(label);
                                setAddressSuggestions([]);
                                setAddressAutocompleteOpen(false);
                              }}
                              style={({ pressed }) => [
                                styles.addressSuggestionItem,
                                isLast ? styles.addressSuggestionItemLast : null,
                                pressed ? styles.addressSuggestionItemPressed : null,
                              ]}
                              accessibilityRole="button"
                              accessibilityLabel={`Suggestion: ${label}`}
                            >
                              <Text style={styles.addressSuggestionPrimary} numberOfLines={1}>
                                {primary}
                              </Text>
                              {secondary ? (
                                <Text style={styles.addressSuggestionSecondary} numberOfLines={1}>
                                  {secondary}
                                </Text>
                              ) : null}
                            </Pressable>
                          );
                        })}
                      </View>
                    ) : null}
                  </View>
                </View>

                <Pressable
                  onPress={() => setShowAccessDetails((v) => !v)}
                  style={styles.optionalToggleRow}
                  accessibilityRole="button"
                  accessibilityLabel="Informations d'accès complémentaires"
                >
                  <Text style={styles.optionalToggleText}>Informations d&apos;accès (optionnel)</Text>
                  <Ionicons
                    name={showAccessDetails ? "chevron-up-outline" : "chevron-down-outline"}
                    size={16}
                    color="#5F7369"
                  />
                </Pressable>

                {showAccessDetails ? (
                  <>
                    <View ref={signupFloorAnchorRef} collapsable={false} style={styles.fieldBlock}>
                      <TextInput
                        ref={floorRef}
                        value={floorUnit}
                        onChangeText={setFloorUnit}
                        placeholder="Étage / appartement"
                        placeholderTextColor="#91A59D"
                        returnKeyType="next"
                        onSubmitEditing={() => intercomRef.current?.focus()}
                        onFocus={() =>
                          scrollAnchorAboveKeyboard(
                            signupScrollRef,
                            signupScrollOffsetYRef,
                            signupFloorAnchorRef,
                          )
                        }
                        style={styles.fieldInput}
                      />
                    </View>
                    <View ref={signupIntercomAnchorRef} collapsable={false} style={styles.fieldBlock}>
                      <TextInput
                        ref={intercomRef}
                        value={intercomCode}
                        onChangeText={setIntercomCode}
                        placeholder="Code / interphone"
                        placeholderTextColor="#91A59D"
                        returnKeyType="next"
                        onSubmitEditing={() => accessRef.current?.focus()}
                        onFocus={() =>
                          scrollAnchorAboveKeyboard(
                            signupScrollRef,
                            signupScrollOffsetYRef,
                            signupIntercomAnchorRef,
                          )
                        }
                        style={styles.fieldInput}
                      />
                    </View>
                    <View ref={signupAccessAnchorRef} collapsable={false} style={styles.fieldBlock}>
                      <AppInput
                        ref={accessRef}
                        value={accessNote}
                        onChangeText={setAccessNote}
                        placeholder="Complément d'accès"
                        placeholderTextColor="#91A59D"
                        returnKeyType="done"
                        onSubmitEditing={() => void submit()}
                        onFocus={() =>
                          scrollAnchorAboveKeyboard(
                            signupScrollRef,
                            signupScrollOffsetYRef,
                            signupAccessAnchorRef,
                          )
                        }
                        shellStyle={INPUT_SHELL}
                        style={INPUT_TEXT}
                      />
                    </View>
                  </>
                ) : null}
              </View>
            </>
          )}

          {currentStep === 1 ? (
            <Pressable
              onPress={goToStepTwo}
              disabled={!stepOneValid}
              style={[
                styles.primaryButton,
                styles.ctaPrimarySpacing,
                !stepOneValid ? styles.primaryButtonDisabled : null,
              ]}
              accessibilityRole="button"
              accessibilityLabel="Continuer"
            >
              <AppText variant="label" style={styles.primaryButtonText}>
                Continuer
              </AppText>
            </Pressable>
          ) : (
            <>
              <Pressable
                onPress={() => setAcceptedTerms((v) => !v)}
                style={styles.termsRow}
                accessibilityRole="checkbox"
                accessibilityState={{ checked: acceptedTerms }}
                accessibilityLabel="Accepter les CGU et la politique de confidentialité"
              >
                <View style={[styles.termsBox, acceptedTerms ? styles.termsBoxChecked : null]}>
                  {acceptedTerms ? <Ionicons name="checkmark" size={13} color="#fff" /> : null}
                </View>
                <Text style={styles.termsText} maxFontSizeMultiplier={1.35}>
                  J&apos;accepte les{" "}
                  <Text
                    style={styles.termsLink}
                    onPress={() => void ExpoLinking.openURL(TERMS_URL)}
                  >
                    conditions d&apos;utilisation
                  </Text>{" "}
                  et la{" "}
                  <Text
                    style={styles.termsLink}
                    onPress={() => void ExpoLinking.openURL(PRIVACY_URL)}
                  >
                    politique de confidentialité
                  </Text>
                  .
                </Text>
              </Pressable>

              <Pressable onPress={() => router.push("/(public)/why-create-account" as any)}>
                <Text style={styles.whyLink} maxFontSizeMultiplier={1.28}>
                  Pourquoi créer un compte ?
                </Text>
              </Pressable>

              <View style={styles.stepActions}>
                <Pressable onPress={() => setCurrentStep(1)} style={styles.secondaryAction}>
                  <Text style={styles.secondaryActionText} maxFontSizeMultiplier={1.28}>
                    Retour
                  </Text>
                </Pressable>
                <Pressable
                  onPress={() => void submit()}
                  disabled={pending || !acceptedTerms}
                  style={[
                    styles.submitButtonInline,
                    styles.primaryButton,
                    pending || !acceptedTerms ? styles.primaryButtonDisabled : null,
                  ]}
                >
                  {pending ? (
                    <ActivityIndicator color="#FFFFFF" />
                  ) : (
                    <AppText variant="label" style={styles.primaryButtonText}>
                      Sauvegarder et s&apos;inscrire
                    </AppText>
                  )}
                </Pressable>
              </View>
            </>
          )}

          {error ? (
            <AppText variant="error" style={styles.feedbackAfterCta} accessibilityRole="alert">
              {error}
            </AppText>
          ) : null}

          <Pressable
            onPress={() => router.replace("/(public)/login" as any)}
            style={styles.bottomLinkWrap}
          >
            <AppText variant="label" style={styles.bottomLink}>
              Déjà un compte ? Se connecter
            </AppText>
          </Pressable>
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
    paddingVertical: 24,
    paddingHorizontal: 16,
  },
  /** Natif, clavier ouvert uniquement — aligné sur `login.tsx` / `forgot-password.tsx`. */
  scrollContentWithKeyboard: {
    justifyContent: "flex-start",
    paddingTop: 28,
  },
  card: {
    width: "100%",
    maxWidth: AUTH_UI.cardMax,
    alignSelf: "center",
    borderRadius: AUTH_UI.radiusCard,
    padding: AUTH_UI.cardPad,
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
    marginBottom: 14,
  },
  logoBlock: {
    alignItems: "center",
    marginBottom: 12,
  },
  logo: {
    height: AUTH_UI.logoH,
    width: AUTH_UI.logoW,
  },
  kicker: {
    color: "#0A8F7A",
    fontSize: FONT_SIZE.px13,
    fontWeight: "500",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    marginBottom: 8,
  },
  title: {
    color: "#163A34",
    fontSize: AUTH_UI.titleSize,
    lineHeight: AUTH_UI.titleLH,
    fontWeight: "700",
  },
  subtitle: {
    color: "#5F7369",
    fontSize: AUTH_UI.subSize,
    lineHeight: AUTH_UI.subLH,
    marginTop: 10,
  },
  primaryButton: {
    minHeight: AUTH_UI.ctaH,
    borderRadius: AUTH_UI.ctaR,
    alignItems: "center",
    justifyContent: "center",
    alignSelf: "stretch",
    backgroundColor: "#0A8F7A",
  },
  ctaPrimarySpacing: {
    marginTop: AUTH_UI.ctaMarginTop,
  },
  primaryButtonDisabled: {
    backgroundColor: "#84B7AE",
  },
  primaryButtonText: {
    color: "#FFFFFF",
    letterSpacing: 0.2,
  },
  stepperWrap: {
    marginTop: 14,
    minHeight: 44,
    flexDirection: "row",
    alignItems: "center",
  },
  stepItem: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  stepConnector: {
    flex: 0.8,
    height: 2,
    marginHorizontal: 8,
    borderRadius: 999,
    backgroundColor: "rgba(145,165,157,0.35)",
  },
  stepConnectorActive: {
    backgroundColor: "rgba(10,143,122,0.55)",
  },
  stepDot: {
    width: 20,
    height: 20,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.55)",
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
  },
  stepDotActive: {
    borderColor: "#0A8F7A",
    backgroundColor: "rgba(10,143,122,0.12)",
  },
  stepDotDone: {
    borderColor: "#0A8F7A",
    backgroundColor: "#0A8F7A",
  },
  stepDotText: {
    color: "#5F7369",
    fontSize: FONT_SIZE.px11,
    fontWeight: "700",
  },
  stepDotTextActive: {
    color: "#0A8F7A",
  },
  stepLabel: {
    color: "#5F7369",
    fontSize: FONT_SIZE.px12,
    fontWeight: "600",
  },
  stepLabelActive: {
    color: "#163A34",
    fontWeight: "700",
  },
  googleButton: {
    marginTop: AUTH_UI.fieldBlockMarginTop,
    minHeight: 54,
    borderRadius: 14,
    alignSelf: "stretch",
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
    ...Platform.select({
      web: { boxShadow: "0 1px 8px rgba(22, 58, 52, 0.06)" },
      default: {
        shadowColor: "#163A34",
        shadowOpacity: 0.05,
        shadowRadius: 6,
        shadowOffset: { width: 0, height: 2 },
        elevation: 1,
      },
    }),
  },
  googleButtonDisabled: {
    opacity: 0.7,
  },
  googleButtonText: {
    color: "#163A34",
    fontWeight: "600",
    fontSize: FONT_SIZE.px14,
    letterSpacing: 0.15,
  },
  googleDivider: {
    marginTop: 10,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  googleDividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: "rgba(145,165,157,0.38)",
  },
  googleDividerText: {
    color: "#7A8D86",
    fontSize: FONT_SIZE.px12,
    fontWeight: "600",
    textTransform: "uppercase",
  },
  sectionBlock: {
    gap: 2,
    width: "100%",
    maxWidth: "100%",
    alignSelf: "stretch",
  },
  sectionTitle: {
    color: "#163A34",
    fontWeight: "700",
    marginBottom: 2,
  },
  mobilityTrigger: {
    minHeight: 52,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: UI_BORDER_SOFT,
    backgroundColor: UI_SURFACE,
    paddingHorizontal: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  mobilityTriggerText: {
    color: "#45655D",
    fontSize: FONT_SIZE.px14,
    fontWeight: "600",
  },
  mobilityTriggerPlaceholder: {
    color: "#7A8D86",
    fontWeight: "500",
  },
  mobilityList: {
    marginTop: 8,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#B7C7C2",
    backgroundColor: "#FFFFFF",
    overflow: "hidden",
    ...Platform.select({
      web: { boxShadow: "0 8px 20px rgba(22,58,52,0.10)" },
      default: {
        shadowColor: "#163A34",
        shadowOpacity: 0.08,
        shadowRadius: 10,
        shadowOffset: { width: 0, height: 4 },
        elevation: 2,
      },
    }),
  },
  mobilityOption: {
    minHeight: 44,
    paddingHorizontal: 12,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: 1,
    borderBottomColor: "rgba(145,165,157,0.24)",
  },
  mobilityOptionLast: {
    borderBottomWidth: 0,
  },
  mobilityOptionText: {
    color: "#45655D",
    fontWeight: "600",
    fontSize: FONT_SIZE.px13_5,
  },
  mobilityOptionTextActive: {
    color: "#0A8F7A",
  },
  whyLink: {
    marginTop: 10,
    color: "#0A8F7A",
    fontWeight: "600",
  },
  fieldBlock: {
    marginTop: 12,
    gap: 6,
    width: "100%",
    maxWidth: "100%",
    alignSelf: "stretch",
  },
  fieldLabel: {
    color: "#5F7369",
    fontSize: FONT_SIZE.px12,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  fieldInput: {
    minHeight: 50,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#91A59D",
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 14,
    color: brandText,
    fontSize: FONT_SIZE.px16,
  },
  fieldInputActive: {
    borderColor: "#0A8F7A",
    ...Platform.select({
      web: { boxShadow: "0 2px 6px rgba(10,143,122,0.10)" },
      default: {
        shadowColor: "#0A8F7A",
        shadowOpacity: 0.08,
        shadowRadius: 5,
        shadowOffset: { width: 0, height: 2 },
        elevation: 1,
      },
    }),
  },
  addressAutocompleteWrap: {
    position: "relative",
  },
  addressSuggestionMetaRow: {
    marginTop: 8,
    paddingHorizontal: 2,
  },
  addressSuggestionMetaText: {
    color: "#6F857E",
    fontSize: FONT_SIZE.px12_5,
    fontWeight: "500",
  },
  addressSuggestionList: {
    marginTop: 8,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.58)",
    backgroundColor: "#F8FBFA",
    overflow: "hidden",
  },
  addressSuggestionItem: {
    paddingVertical: 9,
    paddingHorizontal: 12,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(145,165,157,0.34)",
  },
  addressSuggestionItemLast: {
    borderBottomWidth: 0,
  },
  addressSuggestionItemPressed: {
    backgroundColor: "rgba(145,165,157,0.18)",
  },
  addressSuggestionPrimary: {
    color: "#163A34",
    fontSize: FONT_SIZE.px13_5,
    lineHeight: 18,
    fontWeight: "600",
  },
  addressSuggestionSecondary: {
    marginTop: 2,
    color: "#5F7369",
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
  },
  dateWrap: {
    position: "relative",
  },
  dateInput: {
    paddingRight: 44,
  },
  datePickerButton: {
    position: "absolute",
    right: 8,
    top: 0,
    bottom: 0,
    width: 34,
    alignItems: "center",
    justifyContent: "center",
  },
  optionalToggleRow: {
    marginTop: 12,
    minHeight: 44,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  optionalToggleText: {
    color: "#5F7369",
    fontSize: FONT_SIZE.px13_5,
    fontWeight: "600",
  },
  phoneWrap: {
    position: "relative",
    width: "100%",
    maxWidth: "100%",
    alignSelf: "stretch",
  },
  phoneRow: {
    width: "100%",
    maxWidth: "100%",
    minHeight: 50,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#91A59D",
    backgroundColor: "#FFFFFF",
    flexDirection: "row",
    alignItems: "stretch",
    overflow: "hidden",
  },
  phoneCountryButton: {
    flexShrink: 0,
    alignSelf: "stretch",
    justifyContent: "center",
    borderRightWidth: 1,
    borderRightColor: "rgba(145,165,157,0.35)",
    paddingHorizontal: 10,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: "rgba(10, 143, 122, 0.05)",
    minWidth: 96,
    maxWidth: 120,
  },
  phoneCountryCode: {
    color: brandText,
    fontWeight: "700",
    fontSize: FONT_SIZE.px12,
  },
  phoneCountryDial: {
    color: "#5F7369",
    fontWeight: "600",
    fontSize: FONT_SIZE.px12,
  },
  phoneInput: {
    flex: 1,
    minWidth: 0,
    alignSelf: "stretch",
    minHeight: Platform.OS === "web" ? 44 : 46,
    paddingVertical: Platform.OS === "web" ? 10 : 12,
    paddingHorizontal: 10,
    fontSize: FONT_SIZE.px16,
    color: brandText,
    backgroundColor: "#FFFFFF",
  },
  phoneCountryList: {
    marginTop: 8,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#B7C7C2",
    backgroundColor: "#FFFFFF",
    overflow: "hidden",
    ...Platform.select({
      web: { boxShadow: "0 8px 20px rgba(22,58,52,0.10)" },
      default: {
        shadowColor: "#163A34",
        shadowOpacity: 0.08,
        shadowRadius: 10,
        shadowOffset: { width: 0, height: 4 },
        elevation: 2,
      },
    }),
  },
  phoneCountryItem: {
    minHeight: 44,
    paddingHorizontal: 12,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: 1,
    borderBottomColor: "rgba(145,165,157,0.24)",
  },
  phoneCountryItemLast: {
    borderBottomWidth: 0,
  },
  phoneCountryItemText: {
    color: "#45655D",
    fontWeight: "600",
    fontSize: FONT_SIZE.px13_5,
  },
  phoneCountryItemTextActive: {
    color: "#0A8F7A",
  },
  phoneCountryItemDial: {
    color: "#6F857E",
    fontSize: FONT_SIZE.px13,
    fontWeight: "600",
  },
  phoneCountryItemDialActive: {
    color: "#0A8F7A",
  },
  passwordToggle: {
    justifyContent: "center",
    paddingHorizontal: 6,
    paddingVertical: 4,
  },
  civilityWrap: {
    position: "relative",
  },
  /** Même gabarit que `mobilityTrigger` et les champs `fieldInput` (login / étape 1). */
  civilityTrigger: {
    minHeight: 52,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: UI_BORDER_SOFT,
    backgroundColor: UI_SURFACE,
    paddingHorizontal: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  civilityValue: {
    color: "#5F7369",
    fontWeight: "600",
    fontSize: FONT_SIZE.px14,
  },
  civilityPlaceholder: {
    color: "#7A8D86",
    fontWeight: "500",
  },
  civilityList: {
    marginTop: 8,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#B7C7C2",
    backgroundColor: "#FFFFFF",
    overflow: "hidden",
    ...Platform.select({
      web: { boxShadow: "0 8px 20px rgba(22,58,52,0.10)" },
      default: {
        shadowColor: "#163A34",
        shadowOpacity: 0.08,
        shadowRadius: 10,
        shadowOffset: { width: 0, height: 4 },
        elevation: 2,
      },
    }),
  },
  civilityOption: {
    minHeight: 46,
    paddingHorizontal: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottomWidth: 1,
    borderBottomColor: "rgba(145,165,157,0.24)",
  },
  civilityOptionLast: {
    borderBottomWidth: 0,
  },
  civilityOptionText: {
    color: "#45655D",
    fontWeight: "600",
    fontSize: FONT_SIZE.px14,
  },
  civilityOptionTextActive: {
    color: "#0A8F7A",
  },
  termsRow: {
    marginTop: 16,
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 10,
  },
  termsBox: {
    width: 18,
    height: 18,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: "#91A59D",
    backgroundColor: "#FFFFFF",
    marginTop: 2,
    alignItems: "center",
    justifyContent: "center",
  },
  termsBoxChecked: {
    backgroundColor: "#0A8F7A",
    borderColor: "#0A8F7A",
  },
  termsText: {
    flex: 1,
    color: "#45655D",
    fontSize: FONT_SIZE.px14,
    lineHeight: 20,
  },
  termsLink: {
    color: "#0A8F7A",
    textDecorationLine: "underline",
    fontWeight: "700",
  },
  submitButtonInline: {
    flex: 1,
    marginTop: 0,
  },
  stepActions: {
    marginTop: AUTH_UI.ctaMarginTop,
    flexDirection: "row",
    alignItems: "stretch",
    gap: 10,
  },
  /** Aligné sur `contact.tsx` (`secondaryButton`). */
  secondaryAction: {
    minHeight: AUTH_UI.ctaH,
    borderRadius: AUTH_UI.ctaR,
    borderWidth: 1,
    borderColor: "rgba(10,143,122,0.45)",
    backgroundColor: "#FFFFFF",
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 16,
  },
  secondaryActionText: {
    color: "#0A8F7A",
    fontWeight: "600",
    fontSize: FONT_SIZE.px14,
    letterSpacing: 0.15,
  },
  feedbackAfterCta: {
    marginTop: 12,
    fontWeight: "600",
  },
  bottomLinkWrap: {
    marginTop: 14,
    alignItems: "center",
  },
  bottomLink: {
    color: "#0A8F7A",
    fontWeight: "600",
  },
});
