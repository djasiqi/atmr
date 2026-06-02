import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Animated, Easing, Platform, StyleSheet, View } from "react-native";
import { useRevealFallback } from "../../../src/core/boot/useRevealFallback";
import * as ImagePicker from "expo-image-picker";
import { DriverContextGuard } from "../../../src/core/guards";
import { useSession } from "../../../src/core/sessionProvider";
import { authenticateDriverBiometric } from "../../../src/features/driver/biometricAuth";
import { openBatteryOptimizationSettings } from "../../../src/features/driver/batteryOptimization";
import {
  getDriverProfile,
  getDriverRoute,
  triggerDriverTestPush,
  updateDriverPhoto,
} from "../../../src/features/driver/api";
import {
  normalizeDriverProfilePayload,
} from "../../../src/features/driver/domain/driverAvailability";
import { UnavailableConfirmationModal } from "../../../src/features/driver/components/UnavailableConfirmationModal";
import { useDriverAvailability } from "../../../src/features/driver/hooks";
import {
  AppButton,
  AppInput,
  AppText,
  brandPrimary,
  brandSurfaceSoft,
  ResponsiveContainer,
  Screen,
  useResponsiveTokens,
} from "../../../src/design/responsive";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";

export default function DriverProfileScreen() {
  const { bootstrap, activeContext, error: sessionError } = useSession();
  const user = bootstrap?.user ?? null;
  const [securityMessage, setSecurityMessage] = useState<string | null>(null);
  const {
    isAvailable,
    availabilityPending,
    unavailableConfirmOpen,
    requestToggleAvailability,
    confirmUnavailable,
    cancelUnavailableConfirm,
  } = useDriverAvailability({
    onToggleSuccess: (next) => {
      setSecurityMessage(next ? "Disponibilité activée." : "Disponibilité désactivée.");
    },
    onToggleError: (message) => setSecurityMessage(message),
  });
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileName, setProfileName] = useState<string | null>(null);
  const [photoUrl, setPhotoUrl] = useState<string>("");
  const [photoPending, setPhotoPending] = useState(false);
  const [pushTestPending, setPushTestPending] = useState(false);
  const [routePoints, setRoutePoints] = useState<number | null>(null);
  const [routePending, setRoutePending] = useState(false);
  const t = useResponsiveTokens();
  const sectionEntrance = useRef([
    new Animated.Value(0),
    new Animated.Value(0),
    new Animated.Value(0),
    new Animated.Value(0),
    new Animated.Value(0),
    new Animated.Value(0),
  ]).current;
  const availabilityPulse = useRef(new Animated.Value(1)).current;
  const messageAnim = useRef(new Animated.Value(0)).current;

  const PROFILE_REVEAL_FALLBACK_MS = 1200;

  const revealProfileSections = useCallback(() => {
    sectionEntrance.forEach((value) => {
      value.setValue(1);
    });
  }, [sectionEntrance]);

  const {
    arm: armProfileReveal,
    settled: settleProfileReveal,
    disarm: disarmProfileReveal,
  } = useRevealFallback({
    enabled: true,
    timeoutMs: PROFILE_REVEAL_FALLBACK_MS,
    name: "ProfileRevealFallbackTriggered",
    reveal: revealProfileSections,
  });

  const contextDriverId = useMemo(() => {
    const rawContextId = activeContext?.context_id ?? "";
    if (!rawContextId.startsWith("driver:")) return null;
    const parsed = Number.parseInt(rawContextId.slice("driver:".length), 10);
    return Number.isFinite(parsed) ? parsed : null;
  }, [activeContext?.context_id]);
  const permissionsText = useMemo(
    () => (activeContext?.permissions ?? []).join(", ") || "N/A",
    [activeContext?.permissions]
  );
  const identityName = profileName ?? user?.full_name ?? user?.username ?? "N/A";

  useEffect(() => {
    let cancelled = false;
    setProfileLoading(true);
    void getDriverProfile()
      .then((profile) => {
        if (cancelled) return;
        const normalized = normalizeDriverProfilePayload(profile);
        const fullName =
          typeof normalized.full_name === "string" && normalized.full_name.length > 0
            ? normalized.full_name
            : [normalized.first_name, normalized.last_name]
                .filter((value) => typeof value === "string" && value.length > 0)
                .join(" ");
        setProfileName(fullName.length > 0 ? fullName : null);
        setPhotoUrl(typeof normalized.photo_url === "string" ? normalized.photo_url : "");
      })
      .catch(() => {
        if (!cancelled) setProfileName(null);
      })
      .finally(() => {
        if (!cancelled) setProfileLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const sectionRevealAnimation = Animated.stagger(
      70,
      sectionEntrance.map((value) =>
        Animated.timing(value, {
          toValue: 1,
          duration: 360,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        })
      )
    );
    armProfileReveal();
    sectionRevealAnimation.start(({ finished }) => {
      settleProfileReveal(finished ?? false);
    });
    return () => {
      disarmProfileReveal();
      sectionRevealAnimation.stop();
    };
  }, [armProfileReveal, disarmProfileReveal, sectionEntrance, settleProfileReveal]);

  useEffect(() => {
    Animated.sequence([
      Animated.timing(availabilityPulse, {
        toValue: 1.05,
        duration: 140,
        easing: Easing.out(Easing.quad),
        useNativeDriver: true,
      }),
      Animated.timing(availabilityPulse, {
        toValue: 1,
        duration: 180,
        easing: Easing.inOut(Easing.quad),
        useNativeDriver: true,
      }),
    ]).start();
  }, [availabilityPulse, isAvailable]);

  useEffect(() => {
    if (securityMessage || sessionError) {
      messageAnim.setValue(0);
      Animated.timing(messageAnim, {
        toValue: 1,
        duration: 260,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }).start();
    }
  }, [messageAnim, securityMessage, sessionError]);

  const entranceStyle = (index: number) => ({
    opacity: sectionEntrance[index],
    transform: [
      {
        translateY: sectionEntrance[index].interpolate({
          inputRange: [0, 1],
          outputRange: [14, 0],
        }),
      },
    ],
  });
  const messageEntranceStyle = {
    opacity: messageAnim,
    transform: [
      {
        translateY: messageAnim.interpolate({
          inputRange: [0, 1],
          outputRange: [10, 0],
        }),
      },
    ],
  };

  return (
    <DriverContextGuard>
      <Screen
        scroll
        backgroundColor={brandSurfaceSoft}
        extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
        contentContainerStyle={{
          paddingTop: t.spacingSm,
          paddingBottom: t.spacingLg,
          flexGrow: 1,
        }}
      >
        <ResponsiveContainer>
          <View style={{ width: "100%", gap: t.pageGap }}>
            <Animated.View style={[styles.headerCard, styles.elevatedCard, entranceStyle(0)]}>
              <AppText variant="screenTitle" style={styles.headerTitle}>
                Profil chauffeur
              </AppText>
              <AppText variant="body" style={styles.headerSubtitle}>
                Vos informations et outils de securite
              </AppText>
              <Animated.View
                style={[
                  styles.availabilityPill,
                  isAvailable ? styles.availabilityOn : styles.availabilityOff,
                  { transform: [{ scale: availabilityPulse }] },
                ]}
              >
                <AppText variant="body" style={styles.availabilityText}>
                  {isAvailable ? "Disponible" : "Indisponible"}
                </AppText>
              </Animated.View>
            </Animated.View>

            <Animated.View style={[styles.card, styles.elevatedCard, entranceStyle(1)]}>
              <AppText variant="body" style={styles.sectionTitle}>
                Informations du profil
              </AppText>
              <InfoRow label="Nom" value={identityName} muted={false} />
              <InfoRow label="Email" value={user?.email ?? "N/A"} muted={false} />
              <InfoRow label="Driver ID contexte" value={String(contextDriverId ?? "N/A")} />
              <InfoRow label="Contexte actif" value={activeContext?.label ?? "N/A"} />
              <InfoRow label="Permissions" value={permissionsText} wrap />
              <InfoRow label="Profil sync" value={profileLoading ? "chargement..." : "ok"} />
              <InfoRow label="Points route active" value={String(routePoints ?? "n/a")} />
            </Animated.View>

            <Animated.View style={[styles.card, styles.elevatedCard, entranceStyle(2)]}>
              <AppText variant="body" style={styles.sectionTitle}>
                Photo de profil
              </AppText>
              <AppInput
                label="Photo URL"
                value={photoUrl}
                onChangeText={setPhotoUrl}
                autoCapitalize="none"
                autoCorrect={false}
                placeholder="https://cdn.example.com/photo.jpg"
                helperText={photoUrl ? undefined : "Collez une URL publique vers votre photo."}
              />
              <AppButton
                title={photoPending ? "Mise a jour photo..." : "Mettre a jour photo (URL)"}
                variant="secondary"
                loading={photoPending}
                disabled={photoPending || photoUrl.trim().length === 0}
                onPress={async () => {
                  setPhotoPending(true);
                  try {
                    const updated = await updateDriverPhoto(photoUrl.trim());
                    const nextPhoto =
                      typeof updated.photo_url === "string" && updated.photo_url.length > 0
                        ? updated.photo_url
                        : photoUrl.trim();
                    setPhotoUrl(nextPhoto);
                    setSecurityMessage("Photo profil mise a jour.");
                  } catch (error) {
                    setSecurityMessage(
                      error instanceof Error ? error.message : "Echec mise a jour photo."
                    );
                  } finally {
                    setPhotoPending(false);
                  }
                }}
              />
              <AppButton
                title={photoPending ? "Capture en cours..." : "Prendre une photo"}
                variant="secondary"
                loading={photoPending}
                disabled={photoPending}
                onPress={async () => {
                  if (Platform.OS === "web") {
                    setSecurityMessage("Capture camera indisponible sur web. Utilisez l'URL photo.");
                    return;
                  }
                  const permission = await ImagePicker.requestCameraPermissionsAsync();
                  if (!permission.granted) {
                    setSecurityMessage("Permission camera refusee.");
                    return;
                  }
                  const result = await ImagePicker.launchCameraAsync({
                    mediaTypes: ImagePicker.MediaTypeOptions.Images,
                    allowsEditing: true,
                    aspect: [1, 1],
                    quality: 0.7,
                    base64: true,
                  });
                  if (result.canceled || !result.assets[0]) return;
                  const asset = result.assets[0];
                  if (!asset.base64) {
                    setSecurityMessage("Capture invalide. Reessayez.");
                    return;
                  }
                  setPhotoPending(true);
                  try {
                    const mimeType = asset.mimeType ?? "image/jpeg";
                    const photoPayload = asset.base64.startsWith("data:")
                      ? asset.base64
                      : `data:${mimeType};base64,${asset.base64}`;
                    await updateDriverPhoto({ photoBase64: photoPayload, mimeType });
                    setSecurityMessage("Photo capturee et mise a jour.");
                  } catch (error) {
                    setSecurityMessage(
                      error instanceof Error ? error.message : "Echec capture photo."
                    );
                  } finally {
                    setPhotoPending(false);
                  }
                }}
              />
            </Animated.View>

            <Animated.View style={[styles.card, styles.elevatedCard, entranceStyle(3)]}>
              <AppText variant="body" style={styles.sectionTitle}>
                Actions
              </AppText>
              <View style={styles.actionsColumn}>
                <AppButton
                  title={
                    availabilityPending ? "Mise a jour..." : isAvailable ? "Passer indisponible" : "Passer disponible"
                  }
                  variant="primary"
                  loading={availabilityPending}
                  disabled={availabilityPending}
                  onPress={requestToggleAvailability}
                />
                <AppButton
                  title="Verifier biometrie"
                  variant="secondary"
                  onPress={async () => {
                    const ok = await authenticateDriverBiometric();
                    setSecurityMessage(ok ? "Biometrie validee." : "Biometrie indisponible ou echouee.");
                  }}
                />
                <AppButton
                  title="Ouvrir optimisation batterie"
                  variant="secondary"
                  onPress={async () => {
                    await openBatteryOptimizationSettings();
                    setSecurityMessage("Ecran optimisation batterie ouvert (si supporte).");
                  }}
                />
                <AppButton
                  title={routePending ? "Chargement route..." : "Verifier route courante"}
                  variant="secondary"
                  loading={routePending}
                  disabled={routePending}
                  onPress={async () => {
                    setRoutePending(true);
                    try {
                      const routePayload = await getDriverRoute();
                      const pointsCandidate =
                        Array.isArray(routePayload.points) ? routePayload.points.length : null;
                      setRoutePoints(pointsCandidate);
                      setSecurityMessage("Route chargee avec succes.");
                    } catch (error) {
                      setSecurityMessage(
                        error instanceof Error ? error.message : "Impossible de charger la route."
                      );
                    } finally {
                      setRoutePending(false);
                    }
                  }}
                />
                <AppButton
                  title={pushTestPending ? "Envoi test push..." : "Declencher test push"}
                  variant="secondary"
                  loading={pushTestPending}
                  disabled={pushTestPending}
                  onPress={async () => {
                    setPushTestPending(true);
                    try {
                      await triggerDriverTestPush();
                      setSecurityMessage("Test push declenche avec succes.");
                    } catch (error) {
                      setSecurityMessage(
                        error instanceof Error ? error.message : "Impossible de declencher test push."
                      );
                    } finally {
                      setPushTestPending(false);
                    }
                  }}
                />
              </View>
            </Animated.View>

            {sessionError ? (
              <Animated.View style={[styles.messageCardError, styles.elevatedCard, entranceStyle(4), messageEntranceStyle]}>
                <AppText variant="error">Session : {sessionError}</AppText>
              </Animated.View>
            ) : null}
            {securityMessage ? (
              <Animated.View
                style={[styles.messageCardSuccess, styles.elevatedCard, entranceStyle(5), messageEntranceStyle]}
              >
                <AppText variant="body" style={styles.messageSuccessText}>
                  {securityMessage}
                </AppText>
              </Animated.View>
            ) : null}
          </View>
        </ResponsiveContainer>
      </Screen>
      <UnavailableConfirmationModal
        visible={unavailableConfirmOpen}
        pending={availabilityPending}
        onCancel={cancelUnavailableConfirm}
        onConfirm={confirmUnavailable}
      />
    </DriverContextGuard>
  );
}

function InfoRow({
  label,
  value,
  muted = true,
  wrap = false,
}: {
  label: string;
  value: string;
  muted?: boolean;
  wrap?: boolean;
}) {
  return (
    <View style={[styles.infoRow, wrap && styles.infoRowTop]}>
      <AppText variant="bodyMuted" style={styles.infoLabel}>
        {label}
      </AppText>
      <AppText
        variant={muted ? "bodyMuted" : "body"}
        style={[styles.infoValue, muted ? styles.infoValueMuted : null]}
        numberOfLines={wrap ? 0 : 2}
      >
        {value}
      </AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  headerCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 18,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(145, 165, 157, 0.45)",
    padding: 16,
    gap: 8,
  },
  headerTitle: {
    color: "#163A34",
  },
  headerSubtitle: {
    color: "#5F7369",
  },
  availabilityPill: {
    alignSelf: "flex-start",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
    borderWidth: 1,
  },
  availabilityOn: {
    backgroundColor: "rgba(10, 143, 122, 0.12)",
    borderColor: "rgba(10, 143, 122, 0.35)",
  },
  availabilityOff: {
    backgroundColor: "rgba(180, 35, 24, 0.1)",
    borderColor: "rgba(180, 35, 24, 0.35)",
  },
  availabilityText: {
    color: "#163A34",
    fontWeight: "600",
  },
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 18,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(145, 165, 157, 0.45)",
    padding: 16,
    gap: 12,
  },
  elevatedCard: {
    ...Platform.select({
      web: {
        boxShadow: "0 6px 18px rgba(15, 23, 42, 0.06)",
      } as const,
      default: {
        elevation: 2,
        shadowColor: "#0f172a",
        shadowOpacity: 0.08,
        shadowOffset: { width: 0, height: 2 },
        shadowRadius: 8,
      },
    }),
  },
  sectionTitle: {
    color: "#163A34",
    fontWeight: "700",
  },
  infoRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  infoRowTop: {
    alignItems: "flex-start",
  },
  infoLabel: {
    width: 124,
    fontWeight: "600",
    color: "#5F7369",
  },
  infoValue: {
    flex: 1,
    color: "#163A34",
  },
  infoValueMuted: {
    color: "#5F7369",
  },
  actionsColumn: {
    gap: 10,
  },
  messageCardSuccess: {
    backgroundColor: "#FFFFFF",
    borderRadius: 14,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(10, 143, 122, 0.35)",
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  messageSuccessText: {
    color: brandPrimary,
  },
  messageCardError: {
    backgroundColor: "#FFFFFF",
    borderRadius: 14,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(180, 35, 24, 0.35)",
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
});
