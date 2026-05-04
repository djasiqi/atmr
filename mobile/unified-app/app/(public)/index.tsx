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
  View,
} from "react-native";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ResponsiveContainer,
  Screen,
  useAppViewport,
  useResponsiveTokens,
} from "../../src/design/responsive";
import { autocompleteAddress } from "../../src/features/client/api";
import { AddressAutocompleteSuggestion } from "../../src/features/client/types";
import { useSession } from "../../src/core/sessionProvider";
import { resolveInitialRoute } from "../../src/core/navigation/resolveInitialRoute";
import { ADDRESS_SEARCH_TEXT_PLACEHOLDER } from "../../src/features/public/addressInputPlaceholder";
import {
  PublicAddressSearchBar,
  type AddressSearchRegion,
} from "../../src/features/public/PublicAddressSearchBar";

const LIRIE_LOGO = require("../../assets/images/lirie-logo-color.png");
const LANDING_BACKGROUND = require("../../assets/images/landing-background.png");
const UI_DARK_TEXT = "#163A34";
const UI_MUTED_TEXT = "#5F7369";
const UI_SURFACE = "#F3F7F5";

/** Aligné sur la barre d’adresse compacte (`PublicAddressSearchBar`). */
const SUGGESTION_ROW_HEIGHT = 30;

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
  /** Incrémenté à chaque frappe / clear : invalide le callback géolocalisation en cours. */
  const pickupLocationGenRef = useRef(0);
  /** Suggestion construite depuis le GPS (affichée dans la liste, sans remplir le champ). */
  const pickupGeoSuggestionRef = useRef<AddressAutocompleteSuggestion | null>(null);
  /** Biais lat/lon pour l’API autocomplete après géolocalisation. */
  const pickupGeoBiasRef = useRef<{ lat: number; lon: number } | null>(null);
  const pickupValueRef = useRef(pickupValue);
  const [addressSearchRegion, setAddressSearchRegion] = useState<AddressSearchRegion>("CH");
  const addressSearchRegionRef = useRef<AddressSearchRegion>("CH");
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

  const pickupInputValue = pickupValue.trim();
  const dropoffInputValue = dropoffValue.trim();
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
    addressSearchRegionRef.current = addressSearchRegion;
  }, [addressSearchRegion]);

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
          country: addressSearchRegion,
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
  }, [activeAutocomplete, pickupValue, addressSearchRegion]);

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
          country: addressSearchRegion,
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
  }, [activeAutocomplete, dropoffValue, addressSearchRegion]);

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
            country: addressSearchRegionRef.current,
          });
          if (!nearest[0]) {
            const nearbyQueries = ["Rue", "Avenue", "Chemin"];
            for (const query of nearbyQueries) {
              const around = await autocompleteAddress(query, {
                lat,
                lon,
                limit: 1,
                country: addressSearchRegionRef.current,
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
                country: addressSearchRegionRef.current,
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

      <View style={styles.staticContainer}>
        <Screen
          scroll
          safeTop={false}
          safeBottom={false}
          withHorizontalPadding={false}
          includeSafeAreaInScrollBottomPadding={false}
          contentContainerStyle={{ minHeight: viewport.usableHeight }}
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
            style={[styles.centerColumn, { padding: layout.columnPadding, maxWidth: layout.contentMaxWidth }]}
          >
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
                  maxFontSizeMultiplier={1.28}
                  style={[
                    styles.title,
                    {
                      fontSize: layout.titleFontSize,
                      lineHeight: layout.titleLineHeight + 1,
                      maxWidth: layout.titleMaxWidth,
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
                          { fontSize: layout.cardLabelSize, opacity: layout.cardLabelOpacity },
                        ]}
                      >
                        Départ
                      </Text>
                      <View style={[styles.cardInputRow, { marginTop: layout.cardLineGap }]}>
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
                          region={addressSearchRegion}
                          onRegionChange={setAddressSearchRegion}
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
                          { fontSize: layout.cardLabelSize, opacity: layout.cardLabelOpacity },
                        ]}
                      >
                        Destination
                      </Text>
                      <View style={[styles.cardInputRow, { marginTop: layout.cardLineGap }]}>
                        <PublicAddressSearchBar
                          value={dropoffValue}
                          onChangeText={(value) => {
                            setDropoffValue(value);
                          }}
                          onFocus={() => {
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
                          region={addressSearchRegion}
                          onRegionChange={setAddressSearchRegion}
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
                          { fontSize: layout.cardLabelSize, opacity: layout.cardLabelOpacity },
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
                  <Text
                    maxFontSizeMultiplier={1.28}
                    style={[styles.ctaText, { fontSize: layout.ctaFontSize }]}
                  >
                    Réservation rapide
                  </Text>
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
  cardInputRow: {
    position: "relative",
  },
  locationHintText: {
    marginTop: 6,
    color: "#4D6A63",
  },
  suggestionList: {
    marginTop: 8,
    borderRadius: 8,
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
    fontSize: 11,
    lineHeight: 13,
    fontWeight: "600",
    ...Platform.select({
      android: { includeFontPadding: false },
      default: {},
    }),
  },
  suggestionSecondary: {
    marginTop: 1,
    color: UI_MUTED_TEXT,
    fontSize: 10,
    lineHeight: 11,
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
