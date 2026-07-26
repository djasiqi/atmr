import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Animated,
  Easing,
  Image,
  InteractionManager,
  Linking,
  Platform,
  Pressable,
  StyleSheet,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as ExpoLinking from "expo-linking";
import * as ImagePicker from "expo-image-picker";
import * as Location from "expo-location";
import { useRouter } from "expo-router";
import { useRevealFallback } from "../../../core/boot/useRevealFallback";
import { getExpoNotificationsModule } from "../../../core/notifications/expoNotificationsCompat";
import {
  ensureNotificationDisclosureSyncedWithOsPermission,
  readNotificationDisclosureAccepted,
} from "../../../core/notifications/notificationDisclosurePersistence";
import { requestNotificationDisclosure } from "../../../core/notifications/pushRegistrationState";
import { useSession } from "../../../core/sessionProvider";
import { AppSwitch } from "../../../design/ui/AppSwitch";
import { authenticateDriverBiometric, isDriverBiometricAvailable } from "../biometricAuth";
import { getDriverProfile, updateDriverPhoto } from "../api";
import { normalizeDriverProfilePayload } from "../domain/driverAvailability";
import { UnavailableConfirmationModal } from "../components/UnavailableConfirmationModal";
import { useDriverAvailability } from "../hooks";
import {
  readAuthBiometricEnabled,
  writeAuthBiometricEnabled,
} from "../../../core/auth/biometricPreference";
import {
  AppButton,
  AppText,
  brandPrimary,
  brandSurfaceSoft,
  ResponsiveContainer,
  Screen,
  useResponsiveTokens,
} from "../../../design/responsive";
import { useReduceMotion } from "../../../design/navigation/useReduceMotion";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../navigation/DriverFloatingTabBar";
import {
  buildDriverProfileViewModel,
  DRIVER_LOCATION_PRIVACY_TEXT,
  resolveDriverWeeklyHoursMessage,
  type DriverProfileBadge,
  type DriverProfileSection,
} from "../settings/driverSettingsPresentation";
import { useDriverSettingsDeviceStatus } from "../settings/useDriverSettingsDeviceStatus";

const TERMS_URL = "https://www.lirie.ch/conditions";
const PRIVACY_URL = "https://www.lirie.ch/privacy";
const SUPPORT_URL = "https://www.lirie.ch/contact";

const SECTION_COUNT = 6;
const SECTION_STAGGER_MS = 60;
const SECTION_ANIM_MS = 340;
/** Durée théorique + marge pour boot JS chargé (FCM, API) sans alerter Sentry. */
const SECTION_REVEAL_FALLBACK_MS =
  (SECTION_COUNT - 1) * SECTION_STAGGER_MS + SECTION_ANIM_MS + 800;

const SECTION_ICONS: Record<
  DriverProfileSection["icon"],
  keyof typeof Ionicons.glyphMap
> = {
  person: "person-outline",
  car: "car-outline",
  medkit: "medkit-outline",
  call: "call-outline",
};

type FeedbackMessage = { text: string; tone: "success" | "error" } | null;

export function DriverSettingsScreenContent() {
  const router = useRouter();
  const { bootstrap, activeContext, error: sessionError, logout } = useSession();
  const user = bootstrap?.user ?? null;
  const t = useResponsiveTokens();
  const reduceMotion = useReduceMotion();
  const {
    gps,
    notifications,
    notificationsEnabled,
    locationEnabled,
    batteryOptimizationDisabled,
    batteryStatus,
    refresh: refreshDeviceStatus,
  } = useDriverSettingsDeviceStatus();

  const [feedback, setFeedback] = useState<FeedbackMessage>(null);
  const [profileLoading, setProfileLoading] = useState(true);
  const [profileRaw, setProfileRaw] = useState<ReturnType<typeof normalizeDriverProfilePayload> | null>(
    null
  );
  const [photoPending, setPhotoPending] = useState(false);
  const [biometricEnabled, setBiometricEnabled] = useState(false);
  const [biometricAvailable, setBiometricAvailable] = useState(false);
  const [togglePending, setTogglePending] = useState<string | null>(null);

  const {
    isAvailable,
    availabilityPending,
    unavailableConfirmOpen,
    requestToggleAvailability,
    confirmUnavailable,
    cancelUnavailableConfirm,
    setAvailability,
  } = useDriverAvailability({
    onToggleSuccess: (next) => {
      setFeedback({
        text: next ? "Vous êtes maintenant disponible." : "Vous êtes maintenant indisponible.",
        tone: "success",
      });
    },
    onToggleError: (message) => setFeedback({ text: message, tone: "error" }),
  });

  const companyLabel =
    activeContext?.organization_name?.trim() ||
    activeContext?.label?.trim() ||
    null;

  const profileView = useMemo(
    () => buildDriverProfileViewModel(profileRaw, user, companyLabel),
    [profileRaw, user, companyLabel]
  );
  const weeklyHoursMessage = resolveDriverWeeklyHoursMessage(profileView.weeklyHours);

  const sectionEntrance = useRef(
    Array.from({ length: SECTION_COUNT }, () => new Animated.Value(0))
  ).current;
  const messageAnim = useRef(new Animated.Value(0)).current;

  const revealSections = useCallback(() => {
    sectionEntrance.forEach((value) => value.setValue(1));
  }, [sectionEntrance]);

  const { arm, settled, disarm } = useRevealFallback({
    enabled: !reduceMotion,
    timeoutMs: SECTION_REVEAL_FALLBACK_MS,
    name: "ProfileRevealFallbackTriggered",
    reveal: revealSections,
    report: false,
  });

  useEffect(() => {
    if (reduceMotion) {
      revealSections();
      return;
    }

    let cancelled = false;
    let runningAnimation: Animated.CompositeAnimation | null = null;
    const interactionTask = InteractionManager.runAfterInteractions(() => {
      if (cancelled) return;

      runningAnimation = Animated.stagger(
        SECTION_STAGGER_MS,
        sectionEntrance.map((value) =>
          Animated.timing(value, {
            toValue: 1,
            duration: SECTION_ANIM_MS,
            easing: Easing.out(Easing.cubic),
            useNativeDriver: true,
          })
        )
      );
      arm();
      runningAnimation.start(({ finished }) => settled(finished ?? false));
    });

    return () => {
      cancelled = true;
      interactionTask.cancel();
      runningAnimation?.stop();
      disarm();
    };
  }, [arm, disarm, reduceMotion, revealSections, sectionEntrance, settled]);

  useEffect(() => {
    let cancelled = false;
    setProfileLoading(true);
    void getDriverProfile()
      .then((profile) => {
        if (cancelled) return;
        setProfileRaw(normalizeDriverProfilePayload(profile));
      })
      .catch(() => {
        if (!cancelled) setProfileRaw(null);
      })
      .finally(() => {
        if (!cancelled) setProfileLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    void readAuthBiometricEnabled().then(setBiometricEnabled);
    void isDriverBiometricAvailable().then(setBiometricAvailable);
  }, []);

  useEffect(() => {
    if (!feedback && !sessionError) return;
    messageAnim.setValue(0);
    Animated.timing(messageAnim, {
      toValue: 1,
      duration: 240,
      useNativeDriver: true,
    }).start();
  }, [feedback, messageAnim, sessionError]);

  const entranceStyle = (index: number) => ({
    opacity: 1,
    transform: [
      {
        translateY: sectionEntrance[index]?.interpolate({
          inputRange: [0, 1],
          outputRange: [12, 0],
        }) ?? 0,
      },
    ],
  });

  const capturePhoto = async () => {
    if (Platform.OS === "web") {
      setFeedback({
        text: "La capture photo n'est pas disponible sur le web.",
        tone: "error",
      });
      return;
    }
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      setFeedback({ text: "Autorisez l'accès à la caméra pour changer votre photo.", tone: "error" });
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.72,
      base64: true,
    });
    if (result.canceled || !result.assets[0]?.base64) return;
    const asset = result.assets[0];
    setPhotoPending(true);
    try {
      const mimeType = asset.mimeType ?? "image/jpeg";
      const photoPayload = asset.base64.startsWith("data:")
        ? asset.base64
        : `data:${mimeType};base64,${asset.base64}`;
      const updated = await updateDriverPhoto({ photoBase64: photoPayload, mimeType });
      const nextPhoto =
        typeof updated.photo_url === "string" && updated.photo_url.length > 0
          ? updated.photo_url
          : profileView.photoUrl;
      setProfileRaw((prev) => (prev ? { ...prev, photo_url: nextPhoto } : prev));
      setFeedback({ text: "Photo de profil mise à jour.", tone: "success" });
    } catch (error) {
      setFeedback({
        text: error instanceof Error ? error.message : "Impossible de mettre à jour la photo.",
        tone: "error",
      });
    } finally {
      setPhotoPending(false);
    }
  };

  const handleAvailabilityToggle = (next: boolean) => {
    if (availabilityPending || isAvailable == null) return;
    if (next) {
      setAvailability(true);
      return;
    }
    requestToggleAvailability();
  };

  const handleNotificationToggle = async (next: boolean) => {
    if (togglePending === "notifications") return;
    setTogglePending("notifications");
    try {
      if (next) {
        await ensureNotificationDisclosureSyncedWithOsPermission();
        const disclosureAccepted = await readNotificationDisclosureAccepted();
        if (!disclosureAccepted) {
          requestNotificationDisclosure();
          setFeedback({
            text: "Acceptez les notifications pour activer les alertes de mission.",
            tone: "success",
          });
          return;
        }
        const Notifications = getExpoNotificationsModule();
        if (!Notifications) {
          setFeedback({ text: "Notifications indisponibles sur cet appareil.", tone: "error" });
          return;
        }
        const perm = await Notifications.requestPermissionsAsync();
        if (!perm.granted) {
          setFeedback({
            text: "Autorisez les notifications dans les réglages de votre téléphone.",
            tone: "error",
          });
          return;
        }
        setFeedback({ text: "Notifications activées.", tone: "success" });
      } else {
        await Linking.openSettings();
        setFeedback({
          text: "Désactivez les notifications dans les réglages système si besoin.",
          tone: "success",
        });
      }
    } finally {
      setTogglePending(null);
      void refreshDeviceStatus();
    }
  };

  const handleLocationToggle = async (next: boolean) => {
    if (togglePending === "location") return;
    setTogglePending("location");
    try {
      if (next) {
        const fg = await Location.requestForegroundPermissionsAsync();
        if (!fg.granted) {
          setFeedback({
            text: "Autorisez la localisation pour recevoir des missions.",
            tone: "error",
          });
          return;
        }
        if (Platform.OS !== "web") {
          await Location.requestBackgroundPermissionsAsync();
        }
        setFeedback({ text: "Localisation activée.", tone: "success" });
      } else {
        await Linking.openSettings();
        setFeedback({
          text: "Modifiez l'accès à la localisation dans les réglages système.",
          tone: "success",
        });
      }
    } finally {
      setTogglePending(null);
      void refreshDeviceStatus();
    }
  };

  const handleBatteryToggle = async (next: boolean) => {
    if (togglePending === "battery" || Platform.OS !== "android") return;
    setTogglePending("battery");
    try {
      await Linking.openSettings();
      if (next) {
        setFeedback({
          text: "Autorisez Lirie à fonctionner sans restriction batterie dans les réglages système.",
          tone: "success",
        });
      }
    } finally {
      setTogglePending(null);
      void refreshDeviceStatus();
    }
  };

  const handleBiometricToggle = async (next: boolean) => {
    if (togglePending === "biometric") return;
    setTogglePending("biometric");
    try {
      if (next) {
        const available = await isDriverBiometricAvailable();
        if (!available) {
          setFeedback({
            text: "Biométrie indisponible ou non configurée sur cet appareil.",
            tone: "error",
          });
          return;
        }
        const ok = await authenticateDriverBiometric();
        if (!ok) {
          setFeedback({ text: "Authentification biométrique refusée.", tone: "error" });
          return;
        }
        await writeAuthBiometricEnabled(true);
        setBiometricEnabled(true);
        setFeedback({ text: "Connexion biométrique activée.", tone: "success" });
      } else {
        await writeAuthBiometricEnabled(false);
        setBiometricEnabled(false);
        setFeedback({ text: "Connexion biométrique désactivée.", tone: "success" });
      }
    } finally {
      setTogglePending(null);
    }
  };

  const statusMessage =
    sessionError != null
      ? { text: `Session : ${sessionError}`, tone: "error" as const }
      : feedback;

  const showBatteryToggle = Platform.OS === "android";

  return (
    <>
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
            {/* Mon profil */}
            <Animated.View style={[styles.card, styles.cardShadow, entranceStyle(0)]}>
              <View style={styles.profileHero}>
                <View style={styles.avatarWrap}>
                  {profileView.photoUrl ? (
                    <Image source={{ uri: profileView.photoUrl }} style={styles.avatarImage} />
                  ) : (
                    <View style={styles.avatarFallback}>
                      <AppText variant="sectionTitle" style={styles.avatarInitials}>
                        {profileView.initials}
                      </AppText>
                    </View>
                  )}
                </View>
                <View style={styles.profileHeroMeta}>
                  <AppText variant="screenTitle" style={styles.profileName}>
                    {profileView.displayName}
                  </AppText>
                  {profileView.badges.length > 0 ? (
                    <View style={styles.badgeRow}>
                      {profileView.badges.map((badge) => (
                        <ProfileBadge key={badge.label} badge={badge} />
                      ))}
                    </View>
                  ) : null}
                  {profileLoading ? (
                    <AppText variant="caption" style={styles.profileHint}>
                      Chargement du profil…
                    </AppText>
                  ) : null}
                </View>
              </View>

              {profileView.sections.map((section) => (
                <ProfileSectionBlock key={section.id} section={section} />
              ))}

              {profileView.sections.length === 0 && !profileLoading ? (
                <AppText variant="bodyMuted" style={styles.helperText}>
                  Votre entreprise n'a pas encore renseigné votre fiche. Contactez votre responsable
                  si des informations manquent.
                </AppText>
              ) : null}

              <AppButton
                title={photoPending ? "Mise à jour…" : "Changer ma photo"}
                variant="secondary"
                loading={photoPending}
                disabled={photoPending}
                onPress={() => void capturePhoto()}
              />
            </Animated.View>

            {/* Disponibilité */}
            <Animated.View style={[styles.card, styles.cardShadow, entranceStyle(1)]}>
              <SectionTitle icon="radio-button-on-outline" title="Disponibilité" />
              <AppSwitch
                value={isAvailable ?? false}
                onValueChange={handleAvailabilityToggle}
                disabled={availabilityPending || isAvailable == null}
                accessibilityLabel="Disponibilité"
                label={
                  <AppText variant="body" style={styles.switchLabel}>
                    {isAvailable == null
                      ? "Chargement…"
                      : isAvailable
                        ? "Disponible pour les missions"
                        : "Indisponible"}
                  </AppText>
                }
              />
              <AppText variant="bodyMuted" style={styles.helperText}>
                {weeklyHoursMessage}
              </AppText>
              <LinkRow
                label="Voir mon planning"
                onPress={() => router.push("/(app)/(driver)/schedule")}
              />
            </Animated.View>

            {/* Notifications */}
            <Animated.View style={[styles.card, styles.cardShadow, entranceStyle(2)]}>
              <SectionTitle icon="notifications-outline" title="Notifications" />
              <AppSwitch
                value={notificationsEnabled}
                onValueChange={(next) => void handleNotificationToggle(next)}
                disabled={togglePending === "notifications"}
                accessibilityLabel="Notifications"
                label={
                  <View style={styles.switchLabelBlock}>
                    <AppText variant="body" style={styles.switchLabel}>
                      {notificationsEnabled ? "Activées" : "Désactivées"}
                    </AppText>
                    <AppText variant="caption" style={styles.switchHint}>
                      {notifications.label}
                    </AppText>
                  </View>
                }
              />
              <LinkRow
                label="Ouvrir les réglages notifications"
                onPress={() => {
                  void Linking.openSettings();
                  void refreshDeviceStatus();
                }}
              />
            </Animated.View>

            {/* Localisation */}
            <Animated.View style={[styles.card, styles.cardShadow, entranceStyle(3)]}>
              <SectionTitle icon="navigate-outline" title="Localisation" />
              <AppSwitch
                value={locationEnabled}
                onValueChange={(next) => void handleLocationToggle(next)}
                disabled={togglePending === "location"}
                accessibilityLabel="Localisation GPS"
                label={
                  <View style={styles.switchLabelBlock}>
                    <AppText variant="body" style={styles.switchLabel}>
                      {locationEnabled ? "Activée" : "Désactivée"}
                    </AppText>
                    <AppText variant="caption" style={styles.switchHint}>
                      {gps.label} — {gps.hint}
                    </AppText>
                  </View>
                }
              />
              {showBatteryToggle ? (
                <AppSwitch
                  value={batteryOptimizationDisabled}
                  onValueChange={(next) => void handleBatteryToggle(next)}
                  disabled={togglePending === "battery"}
                  accessibilityLabel="Optimisation batterie"
                  label={
                    <View style={styles.switchLabelBlock}>
                      <AppText variant="body" style={styles.switchLabel}>
                        Sans restriction batterie
                      </AppText>
                      <AppText variant="caption" style={styles.switchHint}>
                        {batteryOptimizationDisabled
                          ? "Lirie peut fonctionner en arrière-plan."
                          : "Autorisez Lirie à ignorer l'optimisation batterie."}
                      </AppText>
                    </View>
                  }
                />
              ) : null}
              <AppText variant="caption" style={styles.privacyText}>
                {DRIVER_LOCATION_PRIVACY_TEXT}
              </AppText>
              <LinkRow
                label="Ouvrir les réglages de localisation"
                onPress={() => {
                  void Linking.openSettings();
                  void refreshDeviceStatus();
                }}
              />
              {showBatteryToggle ? (
                <LinkRow
                  label="Ouvrir les réglages batterie"
                  onPress={() => {
                    void Linking.openSettings();
                    void refreshDeviceStatus();
                  }}
                />
              ) : null}
            </Animated.View>

            {/* Sécurité */}
            <Animated.View style={[styles.card, styles.cardShadow, entranceStyle(4)]}>
              <SectionTitle icon="shield-checkmark-outline" title="Sécurité" />
              <AppSwitch
                value={biometricEnabled}
                onValueChange={(next) => void handleBiometricToggle(next)}
                disabled={togglePending === "biometric" || !biometricAvailable}
                accessibilityLabel="Connexion biométrique"
                label={
                  <View style={styles.switchLabelBlock}>
                    <AppText variant="body" style={styles.switchLabel}>
                      Connexion biométrique
                    </AppText>
                    <AppText variant="caption" style={styles.switchHint}>
                      {biometricAvailable
                        ? biometricEnabled
                          ? "Utilisez empreinte ou Face ID pour vous reconnecter."
                          : "Activez pour sécuriser l'accès à l'application."
                        : "Non disponible sur cet appareil."}
                    </AppText>
                  </View>
                }
              />
              <AppButton
                title="Changer mon mot de passe"
                variant="secondary"
                onPress={() => router.push("/(public)/forgot-password" as never)}
              />
              <AppButton title="Se déconnecter" variant="secondary" onPress={() => logout()} />
            </Animated.View>

            {/* Aide et légal */}
            <Animated.View style={[styles.card, styles.cardShadow, entranceStyle(5)]}>
              <SectionTitle icon="help-circle-outline" title="Aide et informations" />
              <LinkRow label="Contacter le support" onPress={() => void ExpoLinking.openURL(SUPPORT_URL)} />
              <LinkRow
                label="Signaler un problème"
                onPress={() => void ExpoLinking.openURL(`${SUPPORT_URL}?subject=Signalement%20chauffeur`)}
              />
              <LinkRow
                label="Politique de confidentialité"
                onPress={() => void ExpoLinking.openURL(PRIVACY_URL)}
              />
              <LinkRow
                label="Conditions d'utilisation"
                onPress={() => void ExpoLinking.openURL(TERMS_URL)}
              />
              <LinkRow
                label="Demander la suppression du compte"
                onPress={() =>
                  void ExpoLinking.openURL(`${SUPPORT_URL}?subject=Suppression%20compte%20chauffeur`)
                }
              />
            </Animated.View>

            {statusMessage ? (
              <Animated.View
                style={[
                  statusMessage.tone === "error" ? styles.feedbackError : styles.feedbackSuccess,
                  styles.cardShadow,
                  {
                    opacity: messageAnim,
                    transform: [
                      {
                        translateY: messageAnim.interpolate({
                          inputRange: [0, 1],
                          outputRange: [8, 0],
                        }),
                      },
                    ],
                  },
                ]}
              >
                <AppText
                  variant={statusMessage.tone === "error" ? "error" : "body"}
                  style={statusMessage.tone === "success" ? styles.feedbackSuccessText : undefined}
                >
                  {statusMessage.text}
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
    </>
  );
}

function SectionTitle({ icon, title }: { icon: keyof typeof Ionicons.glyphMap; title: string }) {
  return (
    <View style={styles.sectionTitleRow}>
      <Ionicons name={icon} size={20} color={brandPrimary} />
      <AppText variant="body" style={styles.sectionTitle}>
        {title}
      </AppText>
    </View>
  );
}

function ProfileBadge({ badge }: { badge: DriverProfileBadge }) {
  const style =
    badge.tone === "active"
      ? styles.badgeActive
      : badge.tone === "contract"
        ? styles.badgeContract
        : styles.badgeInactive;
  return (
    <View style={[styles.badge, style]}>
      <AppText variant="caption" style={styles.badgeText}>
        {badge.label}
      </AppText>
    </View>
  );
}

function ProfileSectionBlock({ section }: { section: DriverProfileSection }) {
  return (
    <View style={styles.profileSection}>
      <View style={styles.profileSectionHeader}>
        <Ionicons name={SECTION_ICONS[section.icon]} size={16} color={brandPrimary} />
        <AppText variant="label" style={styles.profileSectionTitle}>
          {section.title.toUpperCase()}
        </AppText>
      </View>
      {section.rows.map((row) => (
        <View key={`${section.id}-${row.label}`} style={styles.infoRow}>
          <AppText variant="caption" style={styles.infoLabel}>
            {row.label}
          </AppText>
          <AppText variant="body" style={styles.infoValue} numberOfLines={row.fullWidth ? 0 : 3}>
            {row.value}
          </AppText>
        </View>
      ))}
    </View>
  );
}

function LinkRow({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={styles.linkRow} accessibilityRole="link">
      <AppText variant="body" style={styles.linkText}>
        {label}
      </AppText>
      <Ionicons name="chevron-forward" size={18} color="#5F7369" />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 18,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(145, 165, 157, 0.45)",
    padding: 16,
    gap: 12,
  },
  cardShadow: Platform.select({
    web: { boxShadow: "0 6px 18px rgba(15, 23, 42, 0.06)" } as const,
    default: {
      elevation: 2,
      shadowColor: "#0f172a",
      shadowOpacity: 0.07,
      shadowOffset: { width: 0, height: 2 },
      shadowRadius: 10,
    },
  }),
  sectionTitleRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  sectionTitle: { color: "#163A34", fontWeight: "700" },
  profileHero: { flexDirection: "row", alignItems: "flex-start", gap: 14 },
  avatarWrap: {
    width: 72,
    height: 72,
    borderRadius: 36,
    overflow: "hidden",
    backgroundColor: "rgba(10, 143, 122, 0.12)",
  },
  avatarImage: { width: "100%", height: "100%" },
  avatarFallback: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  avatarInitials: { color: brandPrimary },
  profileHeroMeta: { flex: 1, gap: 6 },
  profileName: { color: "#163A34" },
  profileHint: { color: "#5F7369" },
  badgeRow: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
    borderWidth: 1,
  },
  badgeActive: {
    backgroundColor: "rgba(10, 143, 122, 0.12)",
    borderColor: "rgba(10, 143, 122, 0.35)",
  },
  badgeContract: {
    backgroundColor: "rgba(59, 130, 246, 0.1)",
    borderColor: "rgba(59, 130, 246, 0.35)",
  },
  badgeInactive: {
    backgroundColor: "rgba(148, 163, 184, 0.15)",
    borderColor: "rgba(148, 163, 184, 0.4)",
  },
  badgeText: { color: "#163A34", fontWeight: "600" },
  profileSection: { gap: 8, paddingTop: 4 },
  profileSectionHeader: { flexDirection: "row", alignItems: "center", gap: 6 },
  profileSectionTitle: { color: brandPrimary, letterSpacing: 0.6 },
  infoRow: { flexDirection: "row", alignItems: "flex-start", gap: 10 },
  infoLabel: { width: 118, color: "#5F7369", fontWeight: "600" },
  infoValue: { flex: 1, color: "#163A34" },
  helperText: { color: "#5F7369", lineHeight: 20 },
  privacyText: { color: "#64748B", lineHeight: 18 },
  switchLabel: { color: "#163A34", fontWeight: "600" },
  switchHint: { color: "#5F7369", marginTop: 2 },
  switchLabelBlock: { flex: 1 },
  linkRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 4,
  },
  linkText: { color: brandPrimary, fontWeight: "600" },
  feedbackSuccess: {
    backgroundColor: "#FFFFFF",
    borderRadius: 14,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(10, 143, 122, 0.35)",
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  feedbackSuccessText: { color: brandPrimary },
  feedbackError: {
    backgroundColor: "#FFFFFF",
    borderRadius: 14,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(180, 35, 24, 0.35)",
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
});
