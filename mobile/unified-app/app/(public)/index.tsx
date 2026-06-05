import { Redirect, useRouter } from "expo-router";
import {
  Animated,
  Easing,
  Image,
  ImageBackground,
  Platform,
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRevealFallback } from "../../src/core/boot/useRevealFallback";
import { reportBootFallback } from "../../src/core/observability/bootDiagnostics";
import { useAccessibilityScale } from "../../src/design/responsive/useAccessibilityScale";
import {
  ResponsiveContainer,
  Screen,
  scrollAnchorAboveKeyboard,
  useAppViewport,
  useKeyboardHeight,
  useResponsiveTokens,
} from "../../src/design/responsive";
import { useReduceMotion } from "../../src/design/navigation/useReduceMotion";
import { autocompleteAddress } from "../../src/features/client/api";
import { AddressAutocompleteSuggestion } from "../../src/features/client/types";
import { useSession } from "../../src/core/sessionProvider";
import { resolveInitialRoute } from "../../src/core/navigation/resolveInitialRoute";
import { ADDRESS_SEARCH_TEXT_PLACEHOLDER } from "../../src/features/public/addressInputPlaceholder";
import { AppText } from "../../src/design/ui/AppText";
import {
  PublicAddressSearchBar,
  type AddressSearchRegion,
} from "../../src/features/public/PublicAddressSearchBar";
import { FONT_SIZE } from "../../src/design/responsive/typographyTokens";

/** Recherche d’adresses limitée à la Suisse (sélecteur pays retiré de l’UI). */
const PUBLIC_ADDRESS_COUNTRY: AddressSearchRegion = "CH";
/**
 * Alias stable (sans useState) — évite les ReferenceError si Metro / fast refresh garde
 * un ancien closure ou une dépendance d’effet nommée `addressSearchRegion`.
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars -- export symbol conservé pour anciens bundles
const addressSearchRegion = PUBLIC_ADDRESS_COUNTRY;
const LIRIE_LOGO = require("../../assets/images/lirie-logo-color.png");
const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");
const UI_DARK_TEXT = "#163A34";
const UI_MUTED_TEXT = "#5F7369";
const UI_SURFACE = "#F3F7F5";
const LANDING_REVEAL_FALLBACK_MS = 1200;
/**
 * Filet de sécurité DUR, indépendant de l'animation et de son callback.
 * Garantit que le contenu finit visible même si l'animation native ne s'applique
 * jamais ou notifie `finished` à tort (cas observés New Architecture / Android Samsung,
 * où l'écran reste sur le fond seul). À garder ≥ durée totale de l'anim d'entrée.
 */
const LANDING_REVEAL_HARD_TIMEOUT_MS = 1600;

/** Aligné sur la barre d’adresse (`PublicAddressSearchBar`, minHeight ~50). */
const SUGGESTION_ROW_HEIGHT = 46;

function splitSuggestionLabel(value: string): { primary: string; secondary: string } {
  const raw = String(value || "").trim();
  if (!raw) return { primary: "", secondary: "" };
  const parts = raw.split(",").map((part) => part.trim()).filter(Boolean);
  if (parts.length <= 1) return { primary: raw, secondary: "" };
  return { primary: parts[0], secondary: parts.slice(1).join(", ") };
}

/**
 * Écran d’accueil public : géométrie via `useAppViewport` / `useResponsiveTokens().landing`.
 * Texte : UI dense (libellés carte, CTA, onglets implicites) → `maxFontSizeMultiplier` ~1.25–1.28 ;
 * preuve / hints / adresses → scaling plus large (jusqu’à ~1.5) pour lisibilité.
 */
export default function PublicHomeScreen() {
  const router = useRouter();
  const { bootstrap } = useSession();
  const viewport = useAppViewport();
  const { landing: layout } = useResponsiveTokens();
  const { fontScale, isLargeText, isVeryLargeText } = useAccessibilityScale();
  const reduceMotion = useReduceMotion();
  // ⚠️ Robustesse : l'opacité démarre à 1 (contenu visible par défaut). L'entrée
  // est une animation purement cosmétique de transform (rise + scale) ; sa
  // défaillance (callback `finished` non fiable sous Fabric/Hermes, animation native
  // non appliquée) ne peut donc JAMAIS produire un écran « fond seul ». La visibilité
  // ne dépend plus de l'animation. Voir l'effet d'entrée plus bas.
  const screenOpacity = useRef(new Animated.Value(1)).current;

  const logoOpacity = useRef(new Animated.Value(1)).current;
  const logoScale = useRef(new Animated.Value(1)).current;
  const logoTranslateY = useRef(new Animated.Value(0)).current;
  const titleOpacity = useRef(new Animated.Value(1)).current;
  const titleTranslateY = useRef(new Animated.Value(0)).current;
  const cardOpacity = useRef(new Animated.Value(1)).current;
  const cardTranslateY = useRef(new Animated.Value(0)).current;
  const ctaOpacity = useRef(new Animated.Value(1)).current;
  const ctaScale = useRef(new Animated.Value(1)).current;
  const [pickupValue, setPickupValue] = useState("");
  const [dropoffValue, setDropoffValue] = useState("");
  /**
   * Toujours false — conservés pour compatibilité avec d’anciens bundles web / Metro
   * qui référencent encore ces noms (évite ReferenceError après mise à jour du code source).
   */
  const [pickupPrefilled] = useState(false);
  const [dropoffPrefilled] = useState(false);
  const [pickupSuggestions, setPickupSuggestions] = useState<AddressAutocompleteSuggestion[]>([]);
  const [dropoffSuggestions, setDropoffSuggestions] = useState<AddressAutocompleteSuggestion[]>([]);
  const [activeAutocomplete, setActiveAutocomplete] = useState<"pickup" | "dropoff" | null>(null);
  const [focusedField, setFocusedField] = useState<"pickup" | "dropoff" | null>(null);
  const [isResolvingPickupLocation, setIsResolvingPickupLocation] = useState(false);
  const [pickupLocationHint, setPickupLocationHint] = useState<string | null>(null);
  const isMountedRef = useRef(true);
  /** Passe à true quand l'animation d'entrée notifie réellement `finished`. */
  const revealSettledRef = useRef(false);
  /** Incrémenté à chaque frappe / clear : invalide le callback géolocalisation en cours. */
  const pickupLocationGenRef = useRef(0);
  /** Suggestion construite depuis le GPS (affichée dans la liste, sans remplir le champ). */
  const pickupGeoSuggestionRef = useRef<AddressAutocompleteSuggestion | null>(null);
  /** Biais lat/lon pour l’API autocomplete après géolocalisation. */
  const pickupGeoBiasRef = useRef<{ lat: number; lon: number } | null>(null);
  const pickupValueRef = useRef(pickupValue);
  const landingScrollRef = useRef<ScrollView>(null);
  const landingScrollOffsetYRef = useRef(0);
  const pickupInputAnchorRef = useRef<View>(null);
  const dropoffInputAnchorRef = useRef<View>(null);
  /** Clavier dual : `useKeyboardHeight` factorise listeners + magic padding (cf. plan Sprint 1). */
  const { keyboardVisible, scrollPaddingBottom: keyboardScrollPaddingBottom } = useKeyboardHeight();
  const useNativeDriver = Platform.OS !== "web";
  const accentLayout = useMemo(() => {
    const narrowShortSide = viewport.isTiny || viewport.isCompact;
    return {
      largeTop: viewport.isTablet ? -105 : narrowShortSide ? -98 : -92,
      largeRight: viewport.isTablet ? -48 : -32,
      smallTop: viewport.isTablet ? 95 : narrowShortSide ? 72 : 82,
      smallRight: viewport.isTablet ? 72 : 52,
    };
  }, [viewport.isCompact, viewport.isTablet, viewport.isTiny]);

  const revealContent = useCallback(() => {
    screenOpacity.setValue(1);
    logoOpacity.setValue(1);
    logoScale.setValue(1);
    logoTranslateY.setValue(0);
    titleOpacity.setValue(1);
    titleTranslateY.setValue(0);
    cardOpacity.setValue(1);
    cardTranslateY.setValue(0);
    ctaOpacity.setValue(1);
    ctaScale.setValue(1);
  }, [
    cardOpacity,
    cardTranslateY,
    ctaOpacity,
    ctaScale,
    logoOpacity,
    logoScale,
    logoTranslateY,
    screenOpacity,
    titleOpacity,
    titleTranslateY,
  ]);

  const {
    arm: armLandingReveal,
    settled: settleLandingReveal,
    disarm: disarmLandingReveal,
  } = useRevealFallback({
    enabled: !reduceMotion,
    timeoutMs: LANDING_REVEAL_FALLBACK_MS,
    name: "LandingRevealFallbackTriggered",
    reveal: revealContent,
    extra: { fontScale, isLargeText, isVeryLargeText, reduceMotion },
  });

  const pickupInputValue = pickupValue.trim();
  const dropoffInputValue = dropoffValue.trim();
  const pickupProgress = Math.max(0, Math.min(pickupInputValue.length / 14, 1));
  const dropoffProgress = Math.max(0, Math.min(dropoffInputValue.length / 14, 1));
  const routeProgress = Math.max(0, Math.min((pickupProgress + dropoffProgress) / 2, 1));
  const pickupCompleted = pickupInputValue.length >= 5;
  const dropoffCompleted = dropoffInputValue.length >= 5;

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    pickupValueRef.current = pickupValue;
  }, [pickupValue]);

  useEffect(() => {
    if (keyboardVisible) return;
    landingScrollRef.current?.scrollTo({ y: 0, animated: true });
    landingScrollOffsetYRef.current = 0;
  }, [keyboardVisible]);

  useEffect(() => {
    if (reduceMotion) {
      screenOpacity.setValue(1);
      logoOpacity.setValue(1);
      logoScale.setValue(1);
      logoTranslateY.setValue(0);
      titleOpacity.setValue(1);
      titleTranslateY.setValue(0);
      cardOpacity.setValue(1);
      cardTranslateY.setValue(0);
      ctaOpacity.setValue(1);
      ctaScale.setValue(1);
      return;
    }

    revealSettledRef.current = false;
    // On ne touche JAMAIS à l'opacité : seules les transforms partent d'un état
    // « from » (léger rise + scale) puis reviennent à l'identité. Si l'animation
    // native ne s'applique pas, le contenu reste pleinement visible (opacité 1),
    // au pire décalé de quelques pixels — jamais invisible.
    logoScale.setValue(0.985);
    titleTranslateY.setValue(10);
    cardTranslateY.setValue(14);
    ctaScale.setValue(0.985);

    const landingRevealAnimation = Animated.parallel([
      Animated.timing(logoScale, {
        toValue: 1,
        duration: 240,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
      Animated.timing(titleTranslateY, {
        toValue: 0,
        duration: 220,
        delay: 60,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
      Animated.timing(cardTranslateY, {
        toValue: 0,
        duration: 240,
        delay: 110,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
      Animated.timing(ctaScale, {
        toValue: 1,
        duration: 200,
        delay: 60,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
    ]);

    armLandingReveal();
    landingRevealAnimation.start(({ finished }) => {
      revealSettledRef.current = finished ?? false;
      settleLandingReveal(finished ?? false);
    });

    return () => {
      disarmLandingReveal();
      landingRevealAnimation.stop();
    };
  }, [
    armLandingReveal,
    cardOpacity,
    cardTranslateY,
    ctaOpacity,
    ctaScale,
    disarmLandingReveal,
    logoOpacity,
    logoScale,
    logoTranslateY,
    reduceMotion,
    settleLandingReveal,
    titleOpacity,
    titleTranslateY,
    useNativeDriver,
    screenOpacity,
  ]);

  // Filet de sécurité : le contenu est désormais toujours visible (opacité 1), donc
  // ceci ne « révèle » plus rien de critique — ça fige juste les transforms à leur
  // état de repos si l'animation native n'a pas notifié `finished` à temps, et ça
  // remonte l'incident à Sentry (contexte appareil/accessibilité) pour suivre la
  // fiabilité du pipeline d'animation sous Fabric/Hermes.
  useEffect(() => {
    if (reduceMotion) {
      return;
    }
    const hardRevealTimer = setTimeout(() => {
      revealContent();
      if (!revealSettledRef.current) {
        reportBootFallback("LandingRevealHardTimeout", {
          fontScale,
          isLargeText,
          isVeryLargeText,
          reduceMotion,
        });
      }
    }, LANDING_REVEAL_HARD_TIMEOUT_MS);
    return () => clearTimeout(hardRevealTimer);
  }, [reduceMotion, revealContent, fontScale, isLargeText, isVeryLargeText]);

  useEffect(() => {
    let cancelled = false;
    const trimmed = pickupValue.trim();
    if (activeAutocomplete !== "pickup") {
      setPickupSuggestions([]);
      return () => {
        cancelled = true;
      };
    }

    if (trimmed.length < 2) {
      const geo = pickupGeoSuggestionRef.current;
      if (!cancelled) {
        setPickupSuggestions(geo ? [geo] : []);
      }
      return () => {
        cancelled = true;
      };
    }

    const bias = pickupGeoBiasRef.current;
    const timer = setTimeout(async () => {
      try {
        const results = await autocompleteAddress(trimmed, {
          limit: 4,
          country: PUBLIC_ADDRESS_COUNTRY,
          ...(bias ? { lat: bias.lat, lon: bias.lon } : {}),
        });
        if (cancelled) return;
        const geo = pickupGeoSuggestionRef.current;
        let merged = results.slice(0, 4);
        if (geo) {
          const geoLabel = (geo.address ?? geo.label).trim().toLowerCase();
          const dup = merged.some(
            (r) => (r.address ?? r.label).trim().toLowerCase() === geoLabel
          );
          if (!dup) {
            merged = [geo, ...merged];
          }
        }
        setPickupSuggestions(merged.slice(0, 5));
      } catch {
        if (!cancelled) {
          const geo = pickupGeoSuggestionRef.current;
          setPickupSuggestions(geo ? [geo] : []);
        }
      }
    }, 220);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [activeAutocomplete, pickupValue]);

  useEffect(() => {
    let cancelled = false;
    const trimmed = dropoffValue.trim();
    if (trimmed.length < 2 || activeAutocomplete !== "dropoff") {
      setDropoffSuggestions([]);
      return () => {
        cancelled = true;
      };
    }

    const timer = setTimeout(async () => {
      try {
        const results = await autocompleteAddress(trimmed, {
          limit: 4,
          country: PUBLIC_ADDRESS_COUNTRY,
        });
        if (!cancelled) {
          setDropoffSuggestions(results.slice(0, 4));
        }
      } catch {
        if (!cancelled) {
          setDropoffSuggestions([]);
        }
      }
    }, 220);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [activeAutocomplete, dropoffValue]);

  async function suggestPickupFromCurrentLocation(): Promise<void> {
    if (isResolvingPickupLocation) return;

    if (!globalThis.navigator?.geolocation?.getCurrentPosition) {
      if (isMountedRef.current) {
        setPickupLocationHint("La localisation n'est pas disponible sur cet appareil.");
      }
      return;
    }

    const requestGen = ++pickupLocationGenRef.current;

    if (isMountedRef.current) {
      setIsResolvingPickupLocation(true);
      setPickupLocationHint(null);
    }

    globalThis.navigator.geolocation.getCurrentPosition(
      async (position) => {
        try {
          if (!isMountedRef.current || requestGen !== pickupLocationGenRef.current) return;
          const lat = position.coords.latitude;
          const lon = position.coords.longitude;
          pickupGeoBiasRef.current = { lat, lon };
          let nearest = await autocompleteAddress(`${lat},${lon}`, {
            lat,
            lon,
            limit: 1,
            country: PUBLIC_ADDRESS_COUNTRY,
          });
          if (!nearest[0]) {
            const nearbyQueries = ["Rue", "Avenue", "Chemin"];
            for (const query of nearbyQueries) {
              const around = await autocompleteAddress(query, {
                lat,
                lon,
                limit: 1,
                country: PUBLIC_ADDRESS_COUNTRY,
              });
              if (around[0]) {
                nearest = around;
                break;
              }
            }
          }
          if (!isMountedRef.current || requestGen !== pickupLocationGenRef.current) return;

          const base = nearest[0];
          const label =
            base?.address ?? base?.label ?? "Adresse proche de votre position";
          const geoSuggestion: AddressAutocompleteSuggestion = {
            ...(base ?? {}),
            source: "device_location",
            label: `Près de vous · ${label}`,
            address: label,
            lat: base?.lat ?? lat,
            lon: base?.lon ?? lon,
            lng: base?.lng ?? base?.lon ?? lon,
          };
          pickupGeoSuggestionRef.current = geoSuggestion;

          const trimmed = pickupValueRef.current.trim();
          if (trimmed.length >= 2) {
            try {
              const results = await autocompleteAddress(trimmed, {
                lat,
                lon,
                limit: 4,
                country: PUBLIC_ADDRESS_COUNTRY,
              });
              if (!isMountedRef.current || requestGen !== pickupLocationGenRef.current) return;
              const geoLabel = label.trim().toLowerCase();
              let merged = results.slice(0, 4);
              const dup = merged.some(
                (r) => (r.address ?? r.label).trim().toLowerCase() === geoLabel
              );
              if (!dup) merged = [geoSuggestion, ...merged];
              setPickupSuggestions(merged.slice(0, 5));
            } catch {
              if (isMountedRef.current && requestGen === pickupLocationGenRef.current) {
                setPickupSuggestions([geoSuggestion]);
              }
            }
          } else if (isMountedRef.current && requestGen === pickupLocationGenRef.current) {
            setPickupSuggestions([geoSuggestion]);
          }
          if (isMountedRef.current && requestGen === pickupLocationGenRef.current) {
            setPickupLocationHint(
              "Adresse proche de vous — choisissez-la dans la liste ou continuez à taper."
            );
          }
        } catch {
          if (!isMountedRef.current || requestGen !== pickupLocationGenRef.current) return;
          pickupGeoSuggestionRef.current = null;
          setPickupSuggestions([]);
          setPickupLocationHint("Impossible de proposer une adresse à partir de la position.");
        } finally {
          if (isMountedRef.current && requestGen === pickupLocationGenRef.current) {
            setIsResolvingPickupLocation(false);
          }
        }
      },
      () => {
        if (!isMountedRef.current || requestGen !== pickupLocationGenRef.current) return;
        setIsResolvingPickupLocation(false);
        setPickupLocationHint("Impossible d'accéder à votre localisation.");
      },
      {
        enableHighAccuracy: true,
        timeout: 12000,
        maximumAge: 15000,
      }
    );
  }

  if (bootstrap?.is_authenticated) {
    return <Redirect href={resolveInitialRoute(bootstrap) as any} />;
  }

  return (
    <View style={styles.screen}>
      <ImageBackground
        source={LANDING_BACKGROUND}
        style={StyleSheet.absoluteFillObject}
        resizeMode="cover"
        imageStyle={styles.backgroundImage}
      />
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

      <Animated.View style={[styles.staticContainer, { opacity: screenOpacity }]}>
        <Screen
          scroll
          safeTop={false}
          safeBottom={false}
          withHorizontalPadding={false}
          includeSafeAreaInScrollBottomPadding={false}
          keyboardVerticalOffset={Platform.OS === "ios" ? viewport.topInset : 0}
          automaticallyAdjustKeyboardInsets={Platform.OS !== "web"}
          scrollViewRef={landingScrollRef}
          onScroll={(e) => {
            landingScrollOffsetYRef.current = e.nativeEvent.contentOffset.y;
          }}
          scrollEventThrottle={16}
          contentContainerStyle={[
            { minHeight: viewport.usableHeight },
            Platform.OS !== "web" && keyboardVisible
              ? [styles.scrollContentWithKeyboard, { paddingBottom: keyboardScrollPaddingBottom }]
              : null,
          ]}
        >
        <View
          style={[
            styles.mainColumn,
            {
              paddingTop: layout.topPadding,
              paddingBottom: layout.bottomPadding,
              paddingHorizontal: layout.horizontalPadding,
            },
          ]}
        >
          <ResponsiveContainer
            style={[
              styles.centerColumn,
              {
                flex: 1,
                padding: layout.columnPadding,
                maxWidth: layout.contentMaxWidth,
              },
            ]}
          >
            <View style={styles.landingTopSpacer} />
            <View style={styles.heroSection}>
              <Animated.View
                style={[
                  styles.logoContainer,
                  {
                    opacity: logoOpacity,
                    transform: [{ translateY: logoTranslateY }, { scale: logoScale }],
                  },
                ]}
              >
                <Image
                  source={LIRIE_LOGO}
                  style={{ height: layout.logoHeight, width: layout.logoWidth }}
                  resizeMode="contain"
                  accessibilityRole="image"
                  accessible
                  accessibilityLabel="LIRIE"
                />
              </Animated.View>

              <Animated.View
                style={[
                  styles.titleContainer,
                  {
                    marginTop: layout.titleMarginTop,
                    opacity: titleOpacity,
                    transform: [{ translateY: titleTranslateY }],
                    width: "100%",
                    maxWidth: layout.titleMaxWidth,
                  },
                ]}
              >
                <Text
                  maxFontSizeMultiplier={1.28}
                  numberOfLines={1}
                  adjustsFontSizeToFit
                  minimumFontScale={0.38}
                  style={[
                    styles.title,
                    {
                      width: "100%",
                      fontSize: layout.titleFontSize,
                      lineHeight: layout.titleLineHeight + 1,
                    },
                  ]}
                >
                  Transport{"\u00A0"}médical
                </Text>
              </Animated.View>

              <Animated.View
                style={[
                  styles.cardContainer,
                  {
                    marginTop: layout.cardMarginTop,
                    maxWidth: layout.cardMaxWidth,
                    borderRadius: layout.cardRadius,
                    opacity: cardOpacity,
                    transform: [{ translateY: cardTranslateY }],
                  },
                ]}
              >
                <View
                  style={[
                    styles.cardInner,
                    {
                      padding: layout.cardPadding,
                      borderRadius: layout.cardRadius,
                    },
                    Platform.OS === "android" ? styles.cardInnerFallback : null,
                  ]}
                >
                    <View style={[styles.cardBlock, styles.routeNodeRow, { marginBottom: layout.cardBlockGap }]}>
                      <View
                        style={[
                          styles.routeDotDeparture,
                          pickupCompleted ? styles.routeDotCompleted : styles.routeDotPending,
                        ]}
                      />
                      <View style={styles.routeLineTrack}>
                        <View
                          style={[
                            styles.routeLineFill,
                            { height: `${Math.round(routeProgress * 100)}%` },
                          ]}
                        />
                      </View>
                      <Text
                        maxFontSizeMultiplier={1.25}
                        style={[
                          styles.cardLabel,
                          { fontSize: layout.cardLabelSize },
                        ]}
                      >
                        Départ
                      </Text>
                      <View
                        ref={pickupInputAnchorRef}
                        collapsable={false}
                        style={[styles.cardInputRow, { marginTop: layout.cardLineGap }]}
                      >
                        <PublicAddressSearchBar
                          value={pickupValue}
                          onChangeText={(value) => {
                            pickupLocationGenRef.current += 1;
                            if (isMountedRef.current) {
                              setIsResolvingPickupLocation(false);
                            }
                            setPickupValue(value);
                          }}
                          onFocus={() => {
                            scrollAnchorAboveKeyboard(
                              landingScrollRef,
                              landingScrollOffsetYRef,
                              pickupInputAnchorRef,
                            );
                            setActiveAutocomplete("pickup");
                            setFocusedField("pickup");
                            if (!pickupValue.trim()) {
                              void suggestPickupFromCurrentLocation();
                            }
                          }}
                          onBlur={() => setFocusedField((prev) => (prev === "pickup" ? null : prev))}
                          focused={focusedField === "pickup"}
                          empty={pickupValue.trim().length === 0}
                          prefilled={pickupPrefilled}
                          showClear={pickupValue.trim().length > 0}
                          clearAccessibilityLabel="Effacer l'adresse de départ"
                          onClear={() => {
                            pickupLocationGenRef.current += 1;
                            pickupGeoSuggestionRef.current = null;
                            pickupGeoBiasRef.current = null;
                            setPickupValue("");
                            setPickupSuggestions([]);
                            setPickupLocationHint(null);
                            setIsResolvingPickupLocation(false);
                            setActiveAutocomplete("pickup");
                            setFocusedField("pickup");
                          }}
                          placeholder={ADDRESS_SEARCH_TEXT_PLACEHOLDER}
                          accessibilityLabel="Rechercher une adresse de départ"
                          fontSize={layout.cardValueSize}
                          fontWeight={layout.cardValueWeight}
                        />
                      </View>
                      {isResolvingPickupLocation ? (
                        <Text
                          maxFontSizeMultiplier={1.5}
                          style={[
                            styles.locationHintText,
                            { fontSize: layout.hintFontSize, lineHeight: layout.hintLineHeight },
                          ]}
                        >
                          Recherche de votre position...
                        </Text>
                      ) : pickupLocationHint ? (
                        <Text
                          maxFontSizeMultiplier={1.5}
                          style={[
                            styles.locationHintText,
                            { fontSize: layout.hintFontSize, lineHeight: layout.hintLineHeight },
                          ]}
                        >
                          {pickupLocationHint}
                        </Text>
                      ) : null}
                      {activeAutocomplete === "pickup" && pickupSuggestions.length > 0 ? (
                        <View style={[styles.suggestionList, { maxHeight: layout.suggestionListMaxHeight }]}>
                          {pickupSuggestions.map((item, index) => {
                            const suggestion = item.address ?? item.label;
                            const { primary, secondary } = splitSuggestionLabel(suggestion);
                            const isLast = index === pickupSuggestions.length - 1;
                            return (
                            <Pressable
                              key={`${item.label}-${index}`}
                              onPress={() => {
                                const plat = item.lat;
                                const plon = item.lon ?? item.lng;
                                if (typeof plat === "number" && typeof plon === "number") {
                                  pickupGeoBiasRef.current = { lat: plat, lon: plon };
                                }
                                pickupGeoSuggestionRef.current = null;
                                setPickupValue(suggestion);
                                setPickupSuggestions([]);
                                setActiveAutocomplete(null);
                              }}
                              style={({ pressed }) => [
                                styles.suggestionItem,
                                isLast && styles.suggestionItemLast,
                                pressed && styles.suggestionItemPressed,
                              ]}
                            >
                              <Text
                                maxFontSizeMultiplier={1.35}
                                style={styles.suggestionPrimary}
                                numberOfLines={1}
                              >
                                {primary}
                              </Text>
                              {secondary ? (
                                <Text
                                  maxFontSizeMultiplier={1.3}
                                  style={styles.suggestionSecondary}
                                  numberOfLines={1}
                                >
                                  {secondary}
                                </Text>
                              ) : null}
                            </Pressable>
                          );
                          })}
                        </View>
                      ) : null}
                    </View>
                    <View style={[styles.cardBlock, styles.routeNodeRow, { marginBottom: layout.cardBlockGap }]}>
                      <View
                        style={[
                          styles.routeDotArrival,
                          dropoffCompleted ? styles.routeDotCompleted : styles.routeDotPending,
                        ]}
                      />
                      <Text
                        maxFontSizeMultiplier={1.25}
                        style={[
                          styles.cardLabel,
                          { fontSize: layout.cardLabelSize },
                        ]}
                      >
                        Destination
                      </Text>
                      <View
                        ref={dropoffInputAnchorRef}
                        collapsable={false}
                        style={[styles.cardInputRow, { marginTop: layout.cardLineGap }]}
                      >
                        <PublicAddressSearchBar
                          value={dropoffValue}
                          onChangeText={(value) => {
                            setDropoffValue(value);
                          }}
                          onFocus={() => {
                            scrollAnchorAboveKeyboard(
                              landingScrollRef,
                              landingScrollOffsetYRef,
                              dropoffInputAnchorRef,
                            );
                            setActiveAutocomplete("dropoff");
                            setFocusedField("dropoff");
                          }}
                          onBlur={() => setFocusedField((prev) => (prev === "dropoff" ? null : prev))}
                          focused={focusedField === "dropoff"}
                          empty={dropoffValue.trim().length === 0}
                          prefilled={dropoffPrefilled}
                          showClear={dropoffValue.trim().length > 0}
                          clearAccessibilityLabel="Effacer l'adresse de destination"
                          onClear={() => {
                            setDropoffValue("");
                            setDropoffSuggestions([]);
                            setActiveAutocomplete("dropoff");
                            setFocusedField("dropoff");
                          }}
                          placeholder={ADDRESS_SEARCH_TEXT_PLACEHOLDER}
                          accessibilityLabel="Rechercher une adresse de destination"
                          fontSize={layout.cardValueSize}
                          fontWeight={layout.cardValueWeight}
                        />
                      </View>
                      {activeAutocomplete === "dropoff" && dropoffSuggestions.length > 0 ? (
                        <View style={[styles.suggestionList, { maxHeight: layout.suggestionListMaxHeight }]}>
                          {dropoffSuggestions.map((item, index) => {
                            const suggestion = item.address ?? item.label;
                            const { primary, secondary } = splitSuggestionLabel(suggestion);
                            const isLast = index === dropoffSuggestions.length - 1;
                            return (
                            <Pressable
                              key={`${item.label}-${index}`}
                              onPress={() => {
                                setDropoffValue(suggestion);
                                setDropoffSuggestions([]);
                                setActiveAutocomplete(null);
                              }}
                              style={({ pressed }) => [
                                styles.suggestionItem,
                                isLast && styles.suggestionItemLast,
                                pressed && styles.suggestionItemPressed,
                              ]}
                            >
                              <Text
                                maxFontSizeMultiplier={1.35}
                                style={styles.suggestionPrimary}
                                numberOfLines={1}
                              >
                                {primary}
                              </Text>
                              {secondary ? (
                                <Text
                                  maxFontSizeMultiplier={1.3}
                                  style={styles.suggestionSecondary}
                                  numberOfLines={1}
                                >
                                  {secondary}
                                </Text>
                              ) : null}
                            </Pressable>
                          );
                          })}
                        </View>
                      ) : null}
                    </View>
                    <View style={[styles.cardBlock, { marginBottom: layout.cardBlockGap }]}>
                      <Text
                        maxFontSizeMultiplier={1.25}
                        style={[
                          styles.cardLabel,
                          { fontSize: layout.cardLabelSize },
                        ]}
                      >
                        Départ prévu
                      </Text>
                      <Text
                        maxFontSizeMultiplier={1.28}
                        style={[
                          styles.cardValue,
                          {
                            fontSize: layout.cardValueSize,
                            fontWeight: layout.cardValueWeight,
                            marginTop: layout.cardLineGap,
                          },
                        ]}
                      >
                        Immédiat
                      </Text>
                    </View>
                </View>
              </Animated.View>
            </View>

            <View style={[styles.flexSpacer, { minHeight: layout.flexSpacerMinHeight }]} />

            <View style={[styles.actionSection, { marginTop: layout.spaceCardToCta }]}>
              <Animated.View
                style={[
                  styles.ctaContainer,
                  {
                    maxWidth: layout.contentMaxWidth,
                    opacity: ctaOpacity,
                    transform: [{ scale: ctaScale }],
                  },
                ]}
              >
                <Pressable
                  accessibilityRole="button"
                  onPress={() => {
                    const dep = pickupValue.trim();
                    const dest = dropoffValue.trim();
                    if (dep.length < 2 || dest.length < 2) {
                      Alert.alert(
                        "Adresses requises",
                        "Indiquez un depart et une destination pour continuer.",
                      );
                      return;
                    }
                    router.push({
                      pathname: "/(public)/pre-request/step-1",
                      params: {
                        source: "home",
                        departure: dep,
                        destination: dest,
                        schedule: "immediate",
                      },
                    } as any);
                  }}
                  style={({ pressed }) => [
                    styles.ctaButton,
                    {
                      height: layout.ctaHeight,
                      borderRadius: layout.ctaRadius,
                      width: "100%",
                    },
                    pressed && styles.ctaPressed,
                  ]}
                >
                  <AppText variant="label" style={styles.ctaText}>
                    Réservation rapide
                  </AppText>
                </Pressable>
              </Animated.View>

              <Text
                maxFontSizeMultiplier={1.5}
                style={[
                  styles.microProof,
                  {
                    fontSize: layout.microProofFontSize,
                    lineHeight: layout.microProofLineHeight,
                    marginTop: layout.spaceCtaToProof,
                    maxWidth: layout.contentMaxWidth,
                  },
                ]}
              >
                Suivi en temps réel · Coordination médicale · Transport accompagné
              </Text>

              <View
                style={[
                  styles.secondaryRow,
                  layout.stackSecondaryLinks ? styles.secondaryRowStacked : null,
                  { marginTop: layout.spaceProofToSecondary },
                ]}
              >
                <Pressable
                  accessibilityRole="button"
                  onPress={() => router.push("/(public)/login" as any)}
                  style={({ pressed }) => [styles.secondaryLinkPressable, pressed && styles.secondaryPressed]}
                >
                  <Text
                    maxFontSizeMultiplier={1.28}
                    style={[styles.secondaryLinkText, { fontSize: layout.secondaryFontSize }]}
                  >
                    Se connecter
                  </Text>
                </Pressable>

                {layout.stackSecondaryLinks ? null : (
                  <Text
                    maxFontSizeMultiplier={1.28}
                    style={[styles.secondaryDot, { fontSize: layout.secondaryFontSize }]}
                  >
                    ·
                  </Text>
                )}

                <Pressable
                  accessibilityRole="button"
                  onPress={() => router.push("/(public)/booking-status" as any)}
                  style={({ pressed }) => [styles.secondaryLinkPressable, pressed && styles.secondaryPressed]}
                >
                  <Text
                    maxFontSizeMultiplier={1.28}
                    style={[styles.secondaryLinkText, { fontSize: layout.secondaryFontSize }]}
                  >
                    Suivre ma réservation
                  </Text>
                </Pressable>
              </View>
            </View>
          </ResponsiveContainer>
        </View>
        </Screen>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#EAF3F1",
    overflow: "hidden",
  },
  staticContainer: {
    flex: 1,
    backgroundColor: "transparent",
    overflow: "hidden",
  },
  /**
   * iOS / Android : uniquement pendant `keyboardDidShow`.
   * Sans clavier : seul `minHeight: viewport.usableHeight` s’applique (pas de jeu de scroll artificiel).
   */
  scrollContentWithKeyboard: {
    justifyContent: "flex-start",
    paddingTop: 28,
  },
  backgroundImage: {
    opacity: 0.08,
  },
  accentGlowLarge: {
    position: "absolute",
    top: -90,
    right: -40,
    width: 230,
    height: 230,
    borderRadius: 115,
    backgroundColor: "rgba(10,143,122,0.11)",
  },
  accentGlowSmall: {
    position: "absolute",
    top: 60,
    right: 38,
    width: 110,
    height: 110,
    borderRadius: 55,
    backgroundColor: "rgba(10,143,122,0.08)",
  },
  mainColumn: {
    flex: 1,
    width: "100%",
  },
  centerColumn: {
    flexGrow: 1,
    width: "100%",
    alignSelf: "center",
  },
  /** Répartit l’espace vertical avec `flexSpacer` pour rapprocher logo + titre + carte du centre. */
  landingTopSpacer: {
    flexGrow: 1,
    minHeight: 0,
  },
  flexSpacer: {
    flexGrow: 1,
    minHeight: 18,
  },
  heroSection: {
    alignItems: "center",
  },
  logoContainer: {
    alignSelf: "center",
  },
  titleContainer: {
    alignSelf: "center",
  },
  title: {
    fontFamily: "Philosopher_700Bold",
    color: "#163A34",
    textAlign: "center",
    alignSelf: "center",
    letterSpacing: -0.3,
  },
  titleSubline: {
    opacity: 0.9,
  },
  cardContainer: {
    width: "100%",
    alignSelf: "center",
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
  cardInner: {
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#FFFFFF",
  },
  cardInnerFallback: {
    backgroundColor: "#FFFFFF",
  },
  cardBlock: {
    gap: 2,
  },
  routeNodeRow: {
    paddingLeft: 18,
    position: "relative",
  },
  routeDotDeparture: {
    position: "absolute",
    left: 0,
    top: 12,
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: "rgba(145,165,157,0.55)",
  },
  routeDotArrival: {
    position: "absolute",
    left: 0,
    top: 12,
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: "rgba(145,165,157,0.55)",
    borderWidth: 1,
    borderColor: "rgba(22,58,52,0.18)",
  },
  routeDotPending: {
    backgroundColor: "rgba(145,165,157,0.55)",
    borderColor: "rgba(22,58,52,0.18)",
  },
  routeDotCompleted: {
    backgroundColor: "#0A8F7A",
    borderColor: "#0A8F7A",
  },
  routeLineTrack: {
    position: "absolute",
    left: 4,
    top: 26,
    width: 2,
    bottom: -10,
    backgroundColor: "rgba(22,58,52,0.15)",
    borderRadius: 999,
    overflow: "hidden",
  },
  routeLineFill: {
    position: "absolute",
    left: 0,
    top: 0,
    right: 0,
    backgroundColor: "#0A8F7A",
    borderRadius: 999,
  },
  cardLabel: {
    color: "#0A8F7A",
    lineHeight: 16,
    letterSpacing: 0.5,
    textTransform: "uppercase",
    fontWeight: "500",
  },
  cardValue: {
    lineHeight: 21,
    fontWeight: "500",
    color: UI_DARK_TEXT,
  },
  cardInputRow: {
    position: "relative",
  },
  locationHintText: {
    marginTop: 6,
    color: "#4D6A63",
  },
  suggestionList: {
    marginTop: 8,
    borderRadius: 14,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.62)",
    backgroundColor: UI_SURFACE,
  },
  suggestionItem: {
    minHeight: SUGGESTION_ROW_HEIGHT,
    maxHeight: SUGGESTION_ROW_HEIGHT,
    height: SUGGESTION_ROW_HEIGHT,
    paddingVertical: 0,
    paddingHorizontal: 10,
    justifyContent: "center",
    borderBottomWidth: 1,
    borderBottomColor: "rgba(145,165,157,0.36)",
  },
  suggestionItemLast: {
    borderBottomWidth: 0,
  },
  suggestionItemPressed: {
    backgroundColor: "rgba(145,165,157,0.18)",
  },
  suggestionPrimary: {
    color: UI_DARK_TEXT,
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    fontWeight: "600",
    ...Platform.select({
      android: { includeFontPadding: false },
      default: {},
    }),
  },
  suggestionSecondary: {
    marginTop: 2,
    color: UI_MUTED_TEXT,
    fontSize: FONT_SIZE.px12,
    lineHeight: 15,
    ...Platform.select({
      android: { includeFontPadding: false },
      default: {},
    }),
  },
  actionSection: {
    alignItems: "center",
  },
  ctaContainer: {
    width: "100%",
    alignSelf: "center",
  },
  ctaButton: {
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0A8F7A",
  },
  ctaText: {
    color: "#FFFFFF",
    letterSpacing: 0.2,
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    fontWeight: "600",
  },
  ctaPressed: {
    opacity: 0.94,
    transform: [{ scale: 0.97 }],
  },
  microProof: {
    textAlign: "center",
    alignSelf: "center",
    color: UI_MUTED_TEXT,
    fontWeight: "600",
    letterSpacing: 0.15,
    paddingHorizontal: 4,
  },
  secondaryRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    flexWrap: "wrap",
  },
  secondaryRowStacked: {
    flexDirection: "column",
    gap: 10,
  },
  secondaryLinkPressable: {
    paddingVertical: 4,
    paddingHorizontal: 2,
  },
  secondaryLinkText: {
    color: "#0A8F7A",
    letterSpacing: 0.2,
    lineHeight: 16,
    fontWeight: "600",
  },
  secondaryDot: {
    color: UI_MUTED_TEXT,
    marginHorizontal: 8,
  },
  secondaryPressed: {
    opacity: 0.78,
  },
});
