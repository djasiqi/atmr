import { useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Image,
  ImageBackground,
  Platform,
  Pressable,
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
  AppButton,
  AppInput,
  AppText,
  brandPrimary,
  brandText,
  brandTextMuted,
  getPublicBackButtonMetrics,
  ResponsiveContainer,
  Screen,
  useAppViewport,
  useResponsiveTokens,
} from "../../src/design/responsive";

const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");
const LIRIE_LOGO = require("../../assets/images/lirie-logo-color.png");

const UI_SURFACE = "#F3F7F5";
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
  const viewport = useAppViewport();
  const { topInset } = viewport;
  const { landing: layout } = useResponsiveTokens();

  const accentLayout = useMemo(() => {
    const narrow = viewport.isTiny || viewport.isCompact;
    return {
      largeTop: viewport.isTablet ? -105 : narrow ? -98 : -92,
      largeRight: viewport.isTablet ? -48 : -32,
      smallTop: viewport.isTablet ? 95 : narrow ? 72 : 82,
      smallRight: viewport.isTablet ? 72 : 52,
    };
  }, [viewport.isCompact, viewport.isTablet, viewport.isTiny]);
  const backBtn = useMemo(
    () => getPublicBackButtonMetrics(viewport),
    [viewport]
  );
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
  const addressBlurTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
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
      <View
        style={[
          styles.accentGlowLarge,
          {
            top: accentLayout.largeTop,
            right: accentLayout.largeRight,
            pointerEvents: "none",
          },
        ]}
      />
      <View
        style={[
          styles.accentGlowSmall,
          {
            top: accentLayout.smallTop,
            right: accentLayout.smallRight,
            pointerEvents: "none",
          },
        ]}
      />

      <Screen
        scroll
        withHorizontalPadding={false}
        backgroundColor="transparent"
        keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
        contentContainerStyle={styles.scrollContent}
      >
        <ResponsiveContainer
          style={{
            padding: layout.columnPadding,
            maxWidth: layout.contentMaxWidth,
          }}
        >
          <View
            style={[
              styles.card,
              {
                maxWidth: layout.cardMaxWidth,
                paddingHorizontal: layout.cardPadding + 6,
                paddingTop: layout.cardPadding + 10,
                paddingBottom: layout.cardPadding + 14,
                borderRadius: layout.cardRadius,
              },
            ]}
          >
          <View style={styles.cardBrandAccent} />
          <Pressable
            onPress={() => router.replace("/(public)/login" as any)}
            style={({ pressed }) => [
              styles.backButtonBase,
              {
                paddingVertical: backBtn.paddingVertical,
                paddingHorizontal: backBtn.paddingHorizontal,
                marginLeft: backBtn.marginLeft,
                marginBottom: backBtn.marginBottom,
                borderRadius: backBtn.borderRadius,
                backgroundColor: pressed
                  ? backBtn.backgroundColorPressed
                  : backBtn.backgroundColorIdle,
              },
            ]}
            accessibilityRole="button"
            accessibilityLabel="Retour"
            hitSlop={12}
          >
            <Ionicons name="arrow-back" size={backBtn.iconSize} color={brandPrimary} />
          </Pressable>

          <View style={[styles.logoBlock, { marginBottom: layout.cardLineGap + 6 }]}>
            <Image
              source={LIRIE_LOGO}
              style={{ height: layout.logoHeight, width: layout.logoWidth }}
              resizeMode="contain"
              accessibilityRole="image"
              accessibilityLabel="LIRIE"
            />
          </View>

          <Text
            style={[
              styles.kicker,
              {
                fontSize: layout.secondaryFontSize,
                marginBottom: layout.cardLineGap,
                letterSpacing: viewport.isTiny ? 0.8 : 1.2,
              },
            ]}
            maxFontSizeMultiplier={1.22}
          >
            Informations
          </Text>
          <Text
            style={[
              styles.title,
              {
                fontSize: layout.titleFontSize,
                lineHeight: layout.titleLineHeight,
                maxWidth: layout.titleMaxWidth,
              },
            ]}
            maxFontSizeMultiplier={1.28}
          >
            Créer un compte
          </Text>
          <Text
            style={[
              styles.subtitle,
              {
                fontSize: layout.microProofFontSize,
                lineHeight: layout.microProofLineHeight,
                marginTop: layout.cardLineGap + 4,
              },
            ]}
            maxFontSizeMultiplier={1.45}
          >
            Coordonnées utilisées pour les réservations et les confirmations.
          </Text>
          <View style={[styles.heroDivider, { marginTop: layout.cardBlockGap + 4 }]} />

          <View style={[styles.stepperWrap, { marginTop: layout.cardLineGap + 4 }]}>
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
                style={[
                  styles.googleButton,
                  googlePending ? styles.googleButtonDisabled : null,
                  {
                    marginTop: layout.cardBlockGap + 4,
                    minHeight: layout.ctaHeight + 8,
                    borderRadius: layout.ctaRadius + 5,
                  },
                ]}
                accessibilityRole="button"
                accessibilityLabel="S'inscrire avec Google"
              >
                {googlePending ? (
                  <ActivityIndicator color="#163A34" />
                ) : (
                  <>
                    <Ionicons name="logo-google" size={16} color="#DB4437" />
                    <Text
                      style={[styles.googleButtonText, { fontSize: layout.secondaryFontSize + 1 }]}
                      maxFontSizeMultiplier={1.28}
                    >
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

              <View style={[styles.sectionBlock, { marginTop: layout.cardBlockGap + 4 }]}>
                <Text
                  style={[styles.sectionTitle, { fontSize: layout.cardLabelSize }]}
                  maxFontSizeMultiplier={1.22}
                >
                  Contact
                </Text>
                <View style={styles.fieldBlock}>
                  <AppInput
                    ref={emailRef}
                    value={email}
                    onChangeText={setEmail}
                    placeholder="Courriel (ou téléphone)"
                    autoCapitalize="none"
                    keyboardType="email-address"
                    autoComplete="email"
                    textContentType="emailAddress"
                    returnKeyType="next"
                    onSubmitEditing={() => phoneRef.current?.focus()}
                  />
                </View>
                <View style={styles.fieldBlock}>
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
                        onFocus={() => setPhoneCountryOpen(false)}
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
                      <View style={styles.phoneCountryList}>
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

              <View style={[styles.sectionBlock, { marginTop: layout.cardBlockGap + 10 }]}>
                <Text
                  style={[styles.sectionTitle, { fontSize: layout.cardLabelSize }]}
                  maxFontSizeMultiplier={1.22}
                >
                  Sécurité
                </Text>
                <View style={styles.fieldBlock}>
                  <AppInput
                    ref={passwordRef}
                    value={password}
                    onChangeText={setPassword}
                    placeholder="Min. 8 caractères"
                    secureTextEntry={!showPassword}
                    autoComplete="new-password"
                    textContentType="newPassword"
                    returnKeyType="next"
                    onSubmitEditing={() => confirmPasswordRef.current?.focus()}
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
                <View style={styles.fieldBlock}>
                  <AppInput
                    ref={confirmPasswordRef}
                    value={confirmPassword}
                    onChangeText={setConfirmPassword}
                    placeholder="Confirmer le mot de passe"
                    secureTextEntry={!showConfirmPassword}
                    autoComplete="new-password"
                    textContentType="newPassword"
                    returnKeyType="done"
                    onSubmitEditing={goToStepTwo}
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
              <View style={[styles.sectionBlock, { marginTop: layout.cardBlockGap + 8 }]}>
                <Text
                  style={[styles.sectionTitle, { fontSize: layout.cardLabelSize }]}
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

                <View style={styles.fieldBlock}>
                  <AppInput
                    value={firstName}
                    onChangeText={setFirstName}
                    placeholder="Prénom"
                    returnKeyType="next"
                    onSubmitEditing={() => lastNameRef.current?.focus()}
                  />
                </View>

                <View style={styles.fieldBlock}>
                  <AppInput
                    ref={lastNameRef}
                    value={lastName}
                    onChangeText={setLastName}
                    placeholder="Nom"
                    returnKeyType="next"
                    onSubmitEditing={() => birthDateRef.current?.focus()}
                  />
                </View>

                <View style={styles.fieldBlock}>
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
                  <View style={styles.fieldBlock}>
                    <TextInput
                      value={mobilityOther}
                      onChangeText={setMobilityOther}
                      placeholder="Précisez la mobilité"
                      placeholderTextColor="#91A59D"
                      returnKeyType="next"
                      onSubmitEditing={() => addressRef.current?.focus()}
                      style={styles.fieldInput}
                    />
                  </View>
                ) : null}
              </View>

              <View style={[styles.sectionBlock, { marginTop: layout.cardBlockGap + 10 }]}>
                <Text
                  style={[styles.sectionTitle, { fontSize: layout.cardLabelSize }]}
                  maxFontSizeMultiplier={1.22}
                >
                  Adresse et accès
                </Text>
                <View style={styles.fieldBlock}>
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
                      <View style={styles.addressSuggestionList}>
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
                    <View style={styles.fieldBlock}>
                      <TextInput
                        ref={floorRef}
                        value={floorUnit}
                        onChangeText={setFloorUnit}
                        placeholder="Étage / appartement"
                        placeholderTextColor="#91A59D"
                        returnKeyType="next"
                        onSubmitEditing={() => intercomRef.current?.focus()}
                        style={styles.fieldInput}
                      />
                    </View>
                    <View style={styles.fieldBlock}>
                      <TextInput
                        ref={intercomRef}
                        value={intercomCode}
                        onChangeText={setIntercomCode}
                        placeholder="Code / interphone"
                        placeholderTextColor="#91A59D"
                        returnKeyType="next"
                        onSubmitEditing={() => accessRef.current?.focus()}
                        style={styles.fieldInput}
                      />
                    </View>
                    <View style={styles.fieldBlock}>
                      <AppInput
                        ref={accessRef}
                        value={accessNote}
                        onChangeText={setAccessNote}
                        placeholder="Complément d'accès"
                        returnKeyType="done"
                        onSubmitEditing={() => void submit()}
                      />
                    </View>
                  </>
                ) : null}
              </View>
            </>
          )}

          {error ? (
            <AppText variant="error" style={{ marginTop: 12 }} accessibilityRole="alert">
              {error}
            </AppText>
          ) : null}

          {currentStep === 1 ? (
            <AppButton title="Continuer" variant="primary" onPress={goToStepTwo} style={{ alignSelf: "stretch" }} />
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
                    onPress={() => router.push("/(public)/conditions" as any)}
                  >
                    conditions d&apos;utilisation
                  </Text>{" "}
                  et la{" "}
                  <Text
                    style={styles.termsLink}
                    onPress={() => router.push("/(public)/confidentialite" as any)}
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
                <AppButton
                  title={"Sauvegarder et s'inscrire"}
                  variant="primary"
                  loading={pending}
                  disabled={pending || !acceptedTerms}
                  onPress={() => void submit()}
                  style={[
                    styles.submitButtonInline,
                    { minHeight: layout.ctaHeight + 2, borderRadius: layout.ctaRadius },
                  ]}
                />
              </View>
            </>
          )}

          <Pressable
            onPress={() => router.replace("/(public)/login" as any)}
            style={({ pressed }) => [styles.bottomLinkWrap, pressed && styles.bottomLinkPressed]}
          >
            <Text style={styles.bottomLink} maxFontSizeMultiplier={1.28}>
              Déjà un compte ? Se connecter
            </Text>
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
    backgroundColor: "#F4FAF8",
  },
  backgroundImage: {
    opacity: 0.1,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(244, 250, 248, 0.88)",
  },
  accentGlowLarge: {
    position: "absolute",
    width: 230,
    height: 230,
    borderRadius: 115,
    backgroundColor: "rgba(10, 143, 122, 0.11)",
    zIndex: 0,
  },
  accentGlowSmall: {
    position: "absolute",
    width: 110,
    height: 110,
    borderRadius: 55,
    backgroundColor: "rgba(10, 143, 122, 0.08)",
    zIndex: 0,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 18,
  },
  card: {
    width: "100%",
    alignSelf: "center",
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.42)",
    backgroundColor: "#FFFFFF",
    overflow: "hidden",
    ...Platform.select({
      web: { boxShadow: "0 8px 28px rgba(22, 58, 52, 0.11)" },
      default: {
        shadowColor: "#163A34",
        shadowOpacity: 0.11,
        shadowRadius: 16,
        shadowOffset: { width: 0, height: 8 },
        elevation: 4,
      },
    }),
  },
  cardBrandAccent: {
    position: "absolute",
    left: 0,
    right: 0,
    top: 0,
    height: 4,
    backgroundColor: brandPrimary,
    opacity: 0.95,
  },
  backButtonBase: {
    alignSelf: "flex-start",
  },
  logoBlock: {
    alignItems: "center",
  },
  kicker: {
    color: brandPrimary,
    fontWeight: "600",
    textTransform: "uppercase",
    textAlign: "center",
  },
  title: {
    fontFamily: "Philosopher_700Bold",
    color: brandText,
    textAlign: "center",
    letterSpacing: -0.3,
  },
  subtitle: {
    color: brandTextMuted,
    textAlign: "center",
  },
  heroDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "rgba(145, 165, 157, 0.35)",
    marginBottom: 2,
    alignSelf: "stretch",
  },
  stepperWrap: {
    minHeight: 48,
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
    color: "#6F857E",
    fontSize: 11,
    fontWeight: "700",
  },
  stepDotTextActive: {
    color: "#0A8F7A",
  },
  stepLabel: {
    color: "#6F857E",
    fontSize: 12.5,
    fontWeight: "600",
  },
  stepLabelActive: {
    color: "#0A8F7A",
    fontWeight: "700",
  },
  googleButton: {
    borderWidth: 1,
    borderColor: UI_BORDER_SOFT,
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 6,
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
    fontWeight: "700",
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
    fontSize: 12,
    fontWeight: "600",
    textTransform: "uppercase",
  },
  sectionBlock: {
    gap: 2,
  },
  sectionTitle: {
    color: brandText,
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
    fontSize: 14,
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
    fontSize: 13.5,
  },
  mobilityOptionTextActive: {
    color: "#0A8F7A",
  },
  whyLink: {
    marginTop: 10,
    color: "#0A8F7A",
    fontWeight: "700",
  },
  fieldBlock: {
    marginTop: 12,
    gap: 6,
  },
  fieldLabel: {
    color: "#5F7369",
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  fieldInput: {
    minHeight: 52,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: UI_BORDER_SOFT,
    backgroundColor: UI_SURFACE,
    paddingHorizontal: 16,
    color: brandText,
    fontSize: 16,
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
    fontSize: 12.5,
    fontWeight: "500",
  },
  addressSuggestionList: {
    marginTop: 8,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.58)",
    backgroundColor: "#F8FBFA",
    overflow: "hidden",
    maxHeight: 196,
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
    fontSize: 13.5,
    lineHeight: 18,
    fontWeight: "600",
  },
  addressSuggestionSecondary: {
    marginTop: 2,
    color: "#5F7369",
    fontSize: 12,
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
    minHeight: 36,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#F8FBFA",
    paddingHorizontal: 12,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  optionalToggleText: {
    color: "#5F7369",
    fontSize: 13.5,
    fontWeight: "600",
  },
  phoneWrap: {
    position: "relative",
  },
  phoneRow: {
    minHeight: Platform.OS === "web" ? 32 : 48,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: UI_BORDER_SOFT,
    backgroundColor: UI_SURFACE,
    flexDirection: "row",
    alignItems: "center",
    overflow: "hidden",
  },
  phoneCountryButton: {
    minHeight: "100%",
    borderRightWidth: 1,
    borderRightColor: "rgba(145,165,157,0.35)",
    paddingHorizontal: 12,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: "rgba(10, 143, 122, 0.05)",
  },
  phoneCountryCode: {
    color: brandText,
    fontWeight: "700",
    fontSize: 12,
  },
  phoneCountryDial: {
    color: "#5F7369",
    fontWeight: "600",
    fontSize: 12,
  },
  phoneInput: {
    flex: 1,
    minHeight: Platform.OS === "web" ? 30 : 44,
    paddingVertical: Platform.OS === "web" ? 5 : 10,
    paddingHorizontal: 12,
    color: brandText,
    fontSize: 15,
    backgroundColor: UI_SURFACE,
  },
  phoneCountryList: {
    marginTop: 8,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#B7C7C2",
    backgroundColor: "#FFFFFF",
    overflow: "hidden",
    maxHeight: 240,
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
    fontSize: 13.5,
  },
  phoneCountryItemTextActive: {
    color: "#0A8F7A",
  },
  phoneCountryItemDial: {
    color: "#6F857E",
    fontSize: 13,
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
    marginTop: 2,
    position: "relative",
  },
  civilityTrigger: {
    minHeight: 30,
    borderRadius: 13,
    borderWidth: 1,
    borderColor: UI_BORDER_SOFT,
    backgroundColor: UI_SURFACE,
    paddingHorizontal: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  civilityValue: {
    color: "#5F7369",
    fontWeight: "600",
    fontSize: 14,
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
    fontSize: 14,
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
    fontSize: 14,
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
    marginTop: 14,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  secondaryAction: {
    minHeight: 30,
    borderRadius: 13,
    borderWidth: 1,
    borderColor: "rgba(10,143,122,0.35)",
    backgroundColor: "#F4FAF8",
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 14,
  },
  secondaryActionText: {
    color: "#0A8F7A",
    fontWeight: "700",
    fontSize: 13,
  },
  bottomLinkWrap: {
    marginTop: 14,
    alignItems: "center",
    paddingVertical: 8,
    paddingHorizontal: 6,
    borderRadius: 8,
  },
  bottomLinkPressed: {
    backgroundColor: "rgba(10, 143, 122, 0.06)",
  },
  bottomLink: {
    color: brandPrimary,
    fontWeight: "700",
    fontSize: 13,
  },
});
