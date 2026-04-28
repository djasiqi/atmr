import { Redirect, useRouter } from "expo-router";
import {
  AccessibilityInfo,
  Animated,
  Easing,
  Image,
  ImageBackground,
  Platform,
  Alert,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
  useWindowDimensions,
} from "react-native";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { autocompleteAddress } from "../../src/features/client/api";
import { AddressAutocompleteSuggestion } from "../../src/features/client/types";
import { useSession } from "../../src/core/sessionProvider";
import { resolveInitialRoute } from "../../src/core/navigation/resolveInitialRoute";

const LIRIE_LOGO = require("../../assets/images/lirie-logo-color.png");
const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");
const UI_DARK_TEXT = "#163A34";
const UI_MUTED_TEXT = "#5F7369";
const UI_BORDER = "#91A59D";
const UI_SURFACE = "#F3F7F5";

function splitSuggestionLabel(value: string): { primary: string; secondary: string } {
  const raw = String(value || "").trim();
  if (!raw) return { primary: "", secondary: "" };
  const parts = raw.split(",").map((part) => part.trim()).filter(Boolean);
  if (parts.length <= 1) return { primary: raw, secondary: "" };
  return { primary: parts[0], secondary: parts.slice(1).join(", ") };
}

function randomInRange(min: number, max: number): number {
  return Math.round(Math.random() * (max - min) + min);
}

export default function PublicHomeScreen() {
  const router = useRouter();
  const { bootstrap } = useSession();
  const insets = useSafeAreaInsets();
  const { height, width } = useWindowDimensions();
  const [reduceMotion, setReduceMotion] = useState(false);

  const logoOpacity = useRef(new Animated.Value(0)).current;
  const logoScale = useRef(new Animated.Value(0.98)).current;
  const logoTranslateY = useRef(new Animated.Value(6)).current;
  const titleOpacity = useRef(new Animated.Value(0)).current;
  const titleTranslateY = useRef(new Animated.Value(10)).current;
  const cardOpacity = useRef(new Animated.Value(0)).current;
  const cardTranslateY = useRef(new Animated.Value(16)).current;
  const ctaOpacity = useRef(new Animated.Value(0)).current;
  const ctaScale = useRef(new Animated.Value(0.98)).current;
  const [pickupValue, setPickupValue] = useState("Clinique des Grangettes");
  const [dropoffValue, setDropoffValue] = useState("HUG Genève");
  const [pickupPrefilled, setPickupPrefilled] = useState(true);
  const [dropoffPrefilled, setDropoffPrefilled] = useState(true);
  const [pickupSuggestions, setPickupSuggestions] = useState<AddressAutocompleteSuggestion[]>([]);
  const [dropoffSuggestions, setDropoffSuggestions] = useState<AddressAutocompleteSuggestion[]>([]);
  const [activeAutocomplete, setActiveAutocomplete] = useState<"pickup" | "dropoff" | null>(null);
  const [focusedField, setFocusedField] = useState<"pickup" | "dropoff" | null>(null);
  const [isResolvingPickupLocation, setIsResolvingPickupLocation] = useState(false);
  const [pickupLocationHint, setPickupLocationHint] = useState<string | null>(null);
  const isMountedRef = useRef(true);
  /** Incrémenté à chaque frappe / clear : invalide le callback géolocalisation en cours. */
  const pickupLocationGenRef = useRef(0);
  /** Suggestion construite depuis le GPS (affichée dans la liste, sans remplir le champ). */
  const pickupGeoSuggestionRef = useRef<AddressAutocompleteSuggestion | null>(null);
  /** Biais lat/lon pour l’API autocomplete après géolocalisation. */
  const pickupGeoBiasRef = useRef<{ lat: number; lon: number } | null>(null);
  const pickupValueRef = useRef(pickupValue);
  const useNativeDriver = Platform.OS !== "web";
  const accentLayoutRef = useRef<{
    largeTop: number;
    largeRight: number;
    smallTop: number;
    smallRight: number;
  } | null>(null);

  if (!accentLayoutRef.current) {
    const shortestSide = Math.min(width, height);
    const isTablet = shortestSide >= 768;
    accentLayoutRef.current = {
      largeTop: randomInRange(isTablet ? -120 : -110, isTablet ? -70 : -55),
      largeRight: randomInRange(isTablet ? -90 : -75, isTablet ? -26 : -12),
      smallTop: randomInRange(isTablet ? 58 : 50, isTablet ? 130 : 118),
      smallRight: randomInRange(isTablet ? 28 : 20, isTablet ? 88 : 76),
    };
  }

  const layout = useMemo(() => {
    const shortestSide = Math.min(width, height);
    const isTablet = shortestSide >= 768;
    const isCompact = shortestSide < 390;
    const isShort = height < 760;

    return {
      horizontalPadding: isTablet ? 32 : isCompact ? 16 : 20,
      contentMaxWidth: isTablet ? 560 : 420,
      topPadding: Math.max(insets.top + (isShort ? 12 : 20), isTablet ? 64 : isShort ? 34 : 56),
      bottomPadding: Math.max(insets.bottom + (isShort ? 16 : 24), isShort ? 20 : 32),
      logoHeight: isShort ? 20 : isCompact ? 22 : 28,
      logoWidth: isShort ? 136 : isCompact ? 148 : 180,
      titleFontSize: isTablet ? 52 : isShort ? 32 : isCompact ? 34 : 46,
      titleLineHeight: isTablet ? 58 : isShort ? 38 : isCompact ? 40 : 52,
      titleMaxWidth: isTablet ? 360 : isCompact ? 240 : 280,
      titleMarginTop: isShort ? 14 : isCompact ? 18 : 32,
      cardMarginTop: isShort ? 16 : isCompact ? 22 : 36,
      cardMaxWidth: isTablet ? 520 : 420,
      cardPadding: isTablet ? 24 : isShort ? 14 : isCompact ? 16 : 20,
      cardLabelSize: 13,
      cardLabelOpacity: 0.6,
      cardValueSize: 17,
      cardValueWeight: "500" as const,
      cardLineGap: 6,
      cardBlockGap: 12,
      ctaHeight: isTablet ? 58 : isShort ? 48 : isCompact ? 50 : 56,
      ctaRadius: 18,
      ctaFontSize: isShort ? 15 : isCompact ? 16 : 17,
      microProofFontSize: isShort ? 12 : isCompact ? 13 : 15,
      secondaryFontSize: isShort ? 12 : isCompact ? 13 : 15,
      spaceCardToCta: isShort ? 20 : isCompact ? 30 : 42,
      spaceCtaToProof: isShort ? 10 : isCompact ? 14 : 18,
      spaceProofToSecondary: isShort ? 14 : 24,
    };
  }, [height, insets.bottom, insets.top, width]);

  const pickupInputValue = pickupPrefilled ? "" : pickupValue.trim();
  const dropoffInputValue = dropoffPrefilled ? "" : dropoffValue.trim();
  const pickupProgress = Math.max(0, Math.min(pickupInputValue.length / 14, 1));
  const dropoffProgress = Math.max(0, Math.min(dropoffInputValue.length / 14, 1));
  const routeProgress = Math.max(0, Math.min((pickupProgress + dropoffProgress) / 2, 1));
  const pickupCompleted = pickupInputValue.length >= 5;
  const dropoffCompleted = dropoffInputValue.length >= 5;

  useEffect(() => {
    let mounted = true;
    AccessibilityInfo.isReduceMotionEnabled()
      .then((enabled) => {
        if (mounted) {
          setReduceMotion(enabled);
        }
      })
      .catch(() => {
        // keep default false when API is unavailable
      });

    const subscription = AccessibilityInfo.addEventListener(
      "reduceMotionChanged",
      (enabled) => setReduceMotion(enabled),
    );

    return () => {
      mounted = false;
      subscription.remove();
    };
  }, []);

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
    if (reduceMotion) {
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

    logoOpacity.setValue(0);
    logoScale.setValue(0.98);
    logoTranslateY.setValue(6);
    titleOpacity.setValue(0);
    titleTranslateY.setValue(10);
    cardOpacity.setValue(0);
    cardTranslateY.setValue(16);
    ctaOpacity.setValue(0);
    ctaScale.setValue(0.98);

    Animated.parallel([
      Animated.timing(logoOpacity, {
        toValue: 1,
        duration: 240,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
      Animated.timing(logoScale, {
        toValue: 1,
        duration: 240,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
      Animated.timing(logoTranslateY, {
        toValue: 0,
        duration: 240,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
      Animated.timing(titleOpacity, {
        toValue: 1,
        duration: 220,
        delay: 200,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
      Animated.timing(titleTranslateY, {
        toValue: 0,
        duration: 220,
        delay: 200,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
      Animated.timing(cardOpacity, {
        toValue: 1,
        duration: 240,
        delay: 400,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
      Animated.timing(cardTranslateY, {
        toValue: 0,
        duration: 240,
        delay: 400,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
      Animated.timing(ctaOpacity, {
        toValue: 1,
        duration: 220,
        delay: 650,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
      Animated.timing(ctaScale, {
        toValue: 1,
        duration: 220,
        delay: 650,
        easing: Easing.out(Easing.cubic),
        useNativeDriver,
      }),
    ]).start();
  }, [
    cardOpacity,
    cardTranslateY,
    ctaOpacity,
    ctaScale,
    logoOpacity,
    logoScale,
    logoTranslateY,
    reduceMotion,
    titleOpacity,
    titleTranslateY,
    useNativeDriver,
  ]);

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
        const results = await autocompleteAddress(trimmed, { limit: 4 });
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
          let nearest = await autocompleteAddress(`${lat},${lon}`, { lat, lon, limit: 1 });
          if (!nearest[0]) {
            const nearbyQueries = ["Rue", "Avenue", "Chemin"];
            for (const query of nearbyQueries) {
              const around = await autocompleteAddress(query, { lat, lon, limit: 1 });
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
              const results = await autocompleteAddress(trimmed, { lat, lon, limit: 4 });
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
            top: accentLayoutRef.current.largeTop,
            right: accentLayoutRef.current.largeRight,
            pointerEvents: "none",
          },
        ]}
      />
      <View
        style={[
          styles.accentGlowSmall,
          {
            top: accentLayoutRef.current.smallTop,
            right: accentLayoutRef.current.smallRight,
            pointerEvents: "none",
          },
        ]}
      />

      <View style={styles.staticContainer}>
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
          <View style={[styles.centerColumn, { maxWidth: layout.contentMaxWidth }]}>
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
                  },
                ]}
              >
                <Text
                  style={[
                    styles.title,
                    {
                      fontSize: layout.titleFontSize,
                      lineHeight: layout.titleLineHeight + 1,
                      maxWidth: Math.min(layout.titleMaxWidth, 260),
                    },
                  ]}
                >
                  Transport{"\n"}médical
                </Text>
              </Animated.View>

              <Animated.View
                style={[
                  styles.cardContainer,
                  {
                    marginTop: layout.cardMarginTop,
                    maxWidth: layout.cardMaxWidth,
                    opacity: cardOpacity,
                    transform: [{ translateY: cardTranslateY }],
                  },
                ]}
              >
                <View
                  style={[
                    styles.cardInner,
                    { padding: layout.cardPadding },
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
                        style={[
                          styles.cardLabel,
                          { fontSize: layout.cardLabelSize, opacity: layout.cardLabelOpacity },
                        ]}
                      >
                        Départ
                      </Text>
                      <View style={[styles.cardInputRow, { marginTop: layout.cardLineGap }]}>
                        <TextInput
                          value={pickupValue}
                          onChangeText={(value) => {
                            pickupLocationGenRef.current += 1;
                            if (isMountedRef.current) {
                              setIsResolvingPickupLocation(false);
                            }
                            setPickupPrefilled(false);
                            setPickupValue(value);
                          }}
                          onFocus={() => {
                            setActiveAutocomplete("pickup");
                            setFocusedField("pickup");
                            let shouldSuggestLocation = false;
                            if (pickupPrefilled) {
                              setPickupValue("");
                              setPickupPrefilled(false);
                              shouldSuggestLocation = true;
                            }
                            if (shouldSuggestLocation || !pickupValue.trim()) {
                              void suggestPickupFromCurrentLocation();
                            }
                          }}
                          onBlur={() => setFocusedField((prev) => (prev === "pickup" ? null : prev))}
                          placeholder="Rechercher une adresse"
                          placeholderTextColor="#91A59D"
                          autoComplete="off"
                          autoCorrect={false}
                          spellCheck={false}
                          textContentType="none"
                          importantForAutofill="no"
                          style={[
                            styles.cardInput,
                            styles.cardInputWithClear,
                            pickupPrefilled ? styles.cardInputPrefilled : null,
                            focusedField === "pickup" ? styles.cardInputActive : null,
                            {
                              fontSize: layout.cardValueSize,
                              fontWeight: layout.cardValueWeight,
                            },
                          ]}
                        />
                        {pickupValue.trim().length > 0 ? (
                          <Pressable
                            accessibilityRole="button"
                            accessibilityLabel="Effacer l'adresse de départ"
                            onPress={() => {
                              pickupLocationGenRef.current += 1;
                              pickupGeoSuggestionRef.current = null;
                              pickupGeoBiasRef.current = null;
                              setPickupValue("");
                              setPickupPrefilled(false);
                              setPickupSuggestions([]);
                              setPickupLocationHint(null);
                              setIsResolvingPickupLocation(false);
                              setActiveAutocomplete("pickup");
                              setFocusedField("pickup");
                            }}
                            style={({ pressed }) => [
                              styles.clearInputButton,
                              pressed ? styles.clearInputButtonPressed : null,
                            ]}
                          >
                            <Text style={styles.clearInputText}>x</Text>
                          </Pressable>
                        ) : null}
                      </View>
                      {isResolvingPickupLocation ? (
                        <Text style={styles.locationHintText}>Recherche de votre position...</Text>
                      ) : pickupLocationHint ? (
                        <Text style={styles.locationHintText}>{pickupLocationHint}</Text>
                      ) : null}
                      {activeAutocomplete === "pickup" && pickupSuggestions.length > 0 ? (
                        <View style={styles.suggestionList}>
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
                                setPickupPrefilled(false);
                                setPickupSuggestions([]);
                                setActiveAutocomplete(null);
                              }}
                              style={({ pressed }) => [
                                styles.suggestionItem,
                                isLast && styles.suggestionItemLast,
                                pressed && styles.suggestionItemPressed,
                              ]}
                            >
                              <Text style={styles.suggestionPrimary} numberOfLines={1}>
                                {primary}
                              </Text>
                              {secondary ? (
                                <Text style={styles.suggestionSecondary} numberOfLines={1}>
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
                        style={[
                          styles.cardLabel,
                          { fontSize: layout.cardLabelSize, opacity: layout.cardLabelOpacity },
                        ]}
                      >
                        Destination
                      </Text>
                      <View style={[styles.cardInputRow, { marginTop: layout.cardLineGap }]}>
                        <TextInput
                          value={dropoffValue}
                          onChangeText={(value) => {
                            setDropoffPrefilled(false);
                            setDropoffValue(value);
                          }}
                          onFocus={() => {
                            setActiveAutocomplete("dropoff");
                            setFocusedField("dropoff");
                            if (dropoffPrefilled) {
                              setDropoffValue("");
                              setDropoffPrefilled(false);
                            }
                          }}
                          onBlur={() => setFocusedField((prev) => (prev === "dropoff" ? null : prev))}
                          placeholder="Rechercher une adresse"
                          placeholderTextColor="#91A59D"
                          autoComplete="off"
                          autoCorrect={false}
                          spellCheck={false}
                          textContentType="none"
                          importantForAutofill="no"
                          style={[
                            styles.cardInput,
                            styles.cardInputWithClear,
                            dropoffPrefilled ? styles.cardInputPrefilled : null,
                            focusedField === "dropoff" ? styles.cardInputActive : null,
                            {
                              fontSize: layout.cardValueSize,
                              fontWeight: layout.cardValueWeight,
                            },
                          ]}
                        />
                        {dropoffValue.trim().length > 0 ? (
                          <Pressable
                            accessibilityRole="button"
                            accessibilityLabel="Effacer l'adresse de destination"
                            onPress={() => {
                              setDropoffValue("");
                              setDropoffPrefilled(false);
                              setDropoffSuggestions([]);
                              setActiveAutocomplete("dropoff");
                              setFocusedField("dropoff");
                            }}
                            style={({ pressed }) => [
                              styles.clearInputButton,
                              pressed ? styles.clearInputButtonPressed : null,
                            ]}
                          >
                            <Text style={styles.clearInputText}>x</Text>
                          </Pressable>
                        ) : null}
                      </View>
                      {activeAutocomplete === "dropoff" && dropoffSuggestions.length > 0 ? (
                        <View style={styles.suggestionList}>
                          {dropoffSuggestions.map((item, index) => {
                            const suggestion = item.address ?? item.label;
                            const { primary, secondary } = splitSuggestionLabel(suggestion);
                            const isLast = index === dropoffSuggestions.length - 1;
                            return (
                            <Pressable
                              key={`${item.label}-${index}`}
                              onPress={() => {
                                setDropoffValue(suggestion);
                                setDropoffPrefilled(false);
                                setDropoffSuggestions([]);
                                setActiveAutocomplete(null);
                              }}
                              style={({ pressed }) => [
                                styles.suggestionItem,
                                isLast && styles.suggestionItemLast,
                                pressed && styles.suggestionItemPressed,
                              ]}
                            >
                              <Text style={styles.suggestionPrimary} numberOfLines={1}>
                                {primary}
                              </Text>
                              {secondary ? (
                                <Text style={styles.suggestionSecondary} numberOfLines={1}>
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
                        style={[
                          styles.cardLabel,
                          { fontSize: layout.cardLabelSize, opacity: layout.cardLabelOpacity },
                        ]}
                      >
                        Départ prévu
                      </Text>
                      <Text
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

            <View style={styles.flexSpacer} />

            <View style={[styles.actionSection, { marginTop: layout.spaceCardToCta }]}>
              <Animated.View
                style={[
                  styles.ctaContainer,
                  {
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
                    { height: layout.ctaHeight, borderRadius: layout.ctaRadius },
                    pressed && styles.ctaPressed,
                  ]}
                >
                  <Text style={[styles.ctaText, { fontSize: layout.ctaFontSize }]}>
                    Réservation rapide
                  </Text>
                </Pressable>
              </Animated.View>

              <Text
                style={[
                  styles.microProof,
                  { fontSize: layout.microProofFontSize, marginTop: layout.spaceCtaToProof },
                ]}
              >
                Suivi en temps réel · Coordination médicale · Transport accompagné
              </Text>

              <View style={[styles.secondaryRow, { marginTop: layout.spaceProofToSecondary }]}>
                <Pressable
                  accessibilityRole="button"
                  onPress={() => router.push("/(public)/login" as any)}
                  style={({ pressed }) => [styles.secondaryLinkPressable, pressed && styles.secondaryPressed]}
                >
                  <Text style={[styles.secondaryLinkText, { fontSize: layout.secondaryFontSize }]}>
                    Se connecter
                  </Text>
                </Pressable>

                <Text style={[styles.secondaryDot, { fontSize: layout.secondaryFontSize }]}>·</Text>

                <Pressable
                  accessibilityRole="button"
                  onPress={() => router.push("/(public)/booking-status" as any)}
                  style={({ pressed }) => [styles.secondaryLinkPressable, pressed && styles.secondaryPressed]}
                >
                  <Text style={[styles.secondaryLinkText, { fontSize: layout.secondaryFontSize }]}>
                    Suivre ma réservation
                  </Text>
                </Pressable>
              </View>
            </View>
          </View>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#F4FAF8",
    overflow: "hidden",
  },
  staticContainer: {
    flex: 1,
    backgroundColor: "transparent",
    overflow: "hidden",
  },
  backgroundImage: {
    opacity: 0.09,
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
    padding: 12,
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
    maxWidth: 420,
    alignSelf: "center",
    borderRadius: 24,
    ...Platform.select({
      web: { boxShadow: "0 10px 24px rgba(30,75,67,0.11)" },
      default: {
        shadowColor: "#1E4B43",
        shadowOpacity: 0.11,
        shadowRadius: 24,
        shadowOffset: { width: 0, height: 10 },
        elevation: 2,
      },
    }),
  },
  cardInner: {
    borderRadius: 24,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.34)",
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
    top: 9,
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: "rgba(145,165,157,0.55)",
  },
  routeDotArrival: {
    position: "absolute",
    left: 0,
    top: 9,
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
    top: 20,
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
    color: UI_MUTED_TEXT,
    lineHeight: 16,
    letterSpacing: 0.25,
    fontWeight: "600",
  },
  cardValue: {
    lineHeight: 21,
    fontWeight: "500",
    color: UI_DARK_TEXT,
  },
  cardInput: {
    color: UI_DARK_TEXT,
    lineHeight: 21,
    borderWidth: 1,
    borderColor: UI_BORDER,
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 10,
    minHeight: 48,
    backgroundColor: "rgba(255,255,255,0.92)",
  },
  cardInputRow: {
    position: "relative",
  },
  cardInputWithClear: {
    paddingRight: 42,
  },
  clearInputButton: {
    position: "absolute",
    right: 10,
    top: 10,
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(145,165,157,0.18)",
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.28)",
  },
  clearInputButtonPressed: {
    backgroundColor: "rgba(145,165,157,0.28)",
  },
  clearInputText: {
    color: "#45655D",
    fontSize: 15,
    lineHeight: 15,
    fontWeight: "700",
    textTransform: "lowercase",
  },
  locationHintText: {
    marginTop: 6,
    color: "#4D6A63",
    fontSize: 12,
    lineHeight: 16,
  },
  cardInputPrefilled: {
    color: "#7B8E86",
  },
  cardInputActive: {
    borderColor: "#00796B",
    backgroundColor: "#FFFFFF",
    ...Platform.select({
      web: { boxShadow: "0 2px 6px rgba(0,121,107,0.08)" },
      default: {
        shadowColor: "#00796B",
        shadowOpacity: 0.08,
        shadowRadius: 6,
        shadowOffset: { width: 0, height: 2 },
        elevation: 2,
      },
    }),
  },
  suggestionList: {
    marginTop: 8,
    borderRadius: 12,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.62)",
    backgroundColor: UI_SURFACE,
    maxHeight: 172,
  },
  suggestionItem: {
    paddingVertical: 9,
    paddingHorizontal: 12,
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
    fontSize: 13.5,
    lineHeight: 18,
    fontWeight: "600",
  },
  suggestionSecondary: {
    marginTop: 2,
    color: UI_MUTED_TEXT,
    fontSize: 12,
    lineHeight: 16,
  },
  actionSection: {
    alignItems: "center",
  },
  ctaContainer: {
    width: "100%",
    maxWidth: 420,
    alignSelf: "center",
  },
  ctaButton: {
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0B9A84",
    ...Platform.select({
      web: { boxShadow: "0 6px 12px rgba(11,94,84,0.24)" },
      default: {
        shadowColor: "#0B5E54",
        shadowOpacity: 0.24,
        shadowRadius: 12,
        shadowOffset: { width: 0, height: 6 },
        elevation: 5,
      },
    }),
  },
  ctaText: {
    color: "#FFFFFF",
    fontWeight: "700",
  },
  ctaPressed: {
    opacity: 0.94,
    transform: [{ scale: 0.97 }],
  },
  microProof: {
    textAlign: "center",
    alignSelf: "center",
    color: "#365B53",
    lineHeight: 20,
    maxWidth: 420,
    fontWeight: "600",
    letterSpacing: 0.15,
  },
  secondaryRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
  },
  secondaryLinkPressable: {
    paddingVertical: 4,
    paddingHorizontal: 2,
  },
  secondaryLinkText: {
    color: "#163A34",
    letterSpacing: 0.2,
    lineHeight: 20,
    fontWeight: "600",
  },
  secondaryDot: {
    color: "#163A34",
    marginHorizontal: 8,
  },
  secondaryPressed: {
    opacity: 0.78,
  },
});
