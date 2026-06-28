import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Image,
  Linking,
  Platform,
  Pressable,
  StyleSheet,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as ExpoLinking from "expo-linking";
import { useRouter, type Href } from "expo-router";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { getResolvedApiBaseUrl } from "../../../core/api/client";
import {
  readAuthBiometricEnabled,
  writeAuthBiometricEnabled,
} from "../../../core/auth/biometricPreference";
import {
  isContextSwitchClientSupported,
  shouldShowCompanyDriverContextSwitch,
} from "../../../core/contextSwitchPolicy";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { getExpoNotificationsModule } from "../../../core/notifications/expoNotificationsCompat";
import {
  ensureNotificationDisclosureSyncedWithOsPermission,
  readNotificationDisclosureAccepted,
} from "../../../core/notifications/notificationDisclosurePersistence";
import { requestNotificationDisclosure } from "../../../core/notifications/pushRegistrationState";
import { useSession } from "../../../core/sessionProvider";
import type { AuthContext } from "../../../core/contracts/auth";
import { AppSwitch } from "../../../design/ui/AppSwitch";
import { computeFloatingTabBarClearance } from "../../../design/navigation/BaseFloatingBar";
import {
  AppButton,
  AppText,
  brandPrimary,
  ResponsiveContainer,
  Screen,
  useResponsiveTokens,
} from "../../../design/responsive";
import { createShadow } from "../../../styles/shadowStyles";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";
import {
  authenticateDriverBiometric,
  isDriverBiometricAvailable,
} from "../../driver/biometricAuth";
import {
  getCompanyBillingSettings,
  getCompanyDispatchModes,
  getCompanyProfile,
  switchCompanyDispatchMode,
} from "../api/companyApi";
import { useCompanyRealtimeStatus } from "../hooks";
import { E } from "../theme/enterpriseOpsTheme";
import {
  buildCompanyBillingSummary,
  buildCompanyProfileViewModel,
  COMPANY_DISPATCH_MODE_OPTIONS,
  formatDispatchModeFr,
  formatDispatchStateFr,
  resolveCompanyRealtimeLabel,
  resolveUserDisplayName,
  type CompanyDispatchModeId,
  type CompanyDispatchModeOption,
  type CompanyProfileBadge,
  type CompanyProfileSection,
} from "../settings/companySettingsPresentation";

const TERMS_URL = "https://www.lirie.ch/conditions";
const PRIVACY_URL = "https://www.lirie.ch/privacy";
const SUPPORT_URL = "https://www.lirie.ch/contact";
const WEB_SETTINGS_URL = "https://app.lirie.ch/dashboard/company/settings";

const BORDER_SLATE = "rgba(148, 163, 184, 0.22)";

const cardShadow = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 1 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

const SECTION_ICONS: Record<
  CompanyProfileSection["icon"],
  keyof typeof Ionicons.glyphMap
> = {
  business: "business-outline",
  call: "call-outline",
  document: "document-text-outline",
  map: "map-outline",
};

type FeedbackMessage = { text: string; tone: "success" | "error" } | null;
type ToggleKey = "notifications" | "biometric" | null;

function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), ms);
    promise
      .then((value) => {
        clearTimeout(timer);
        resolve(value);
      })
      .catch((error) => {
        clearTimeout(timer);
        reject(error);
      });
  });
}

export function CompanySettingsScreenContent() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const t = useResponsiveTokens();
  const {
    activeContext,
    bootstrap,
    can,
    changeContext,
    error: sessionError,
    logout,
  } = useSession();
  const companyRealtime = useCompanyRealtimeStatus();

  const [profileRaw, setProfileRaw] = useState<Record<string, unknown> | null>(null);
  const [profileLoading, setProfileLoading] = useState(true);
  const [dispatchMode, setDispatchMode] = useState("manual");
  const [dispatchState, setDispatchState] = useState("idle");
  const [billingRaw, setBillingRaw] = useState<Record<string, unknown> | null>(null);
  const [feedback, setFeedback] = useState<FeedbackMessage>(null);
  const [togglePending, setTogglePending] = useState<ToggleKey>(null);
  const [dispatchPending, setDispatchPending] = useState(false);
  const [switchPending, setSwitchPending] = useState(false);
  const [biometricEnabled, setBiometricEnabled] = useState(false);
  const [biometricAvailable, setBiometricAvailable] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);

  const roleGuardsEnabled = isFeatureEnabled("company_mobile_role_guards_enabled");
  const contexts = useMemo<AuthContext[]>(
    () => bootstrap?.available_contexts ?? [],
    [bootstrap?.available_contexts]
  );
  const companyContexts = useMemo(
    () => contexts.filter((ctx) => ctx.context_type === "company"),
    [contexts]
  );
  const driverContexts = useMemo(
    () => contexts.filter((ctx) => ctx.context_type === "driver"),
    [contexts]
  );
  const activeCompanyContext = useMemo(
    () =>
      companyContexts.find((ctx) => ctx.context_id === activeContext?.context_id) ??
      companyContexts[0] ??
      null,
    [activeContext?.context_id, companyContexts]
  );
  const primaryDriverContext = driverContexts[0] ?? null;
  const activeCompanyId =
    activeContext?.context_type === "company"
      ? activeContext.context_id
      : activeCompanyContext?.context_id ?? null;
  const organizationName =
    activeContext?.organization_name?.trim() || activeCompanyContext?.organization_name?.trim() || null;

  const contextPermissions = activeContext?.permissions ?? [];
  const canRunSensitiveAction = (permission: string, fallbackPermission: string) => {
    if (!roleGuardsEnabled) return true;
    if (contextPermissions.includes(permission)) return can(permission);
    return can(fallbackPermission);
  };
  const companyAccountForDoubleHat =
    !bootstrap?.user?.role || String(bootstrap.user.role).toUpperCase() === "COMPANY";
  const canSwitchToDriver =
    isContextSwitchClientSupported() &&
    companyAccountForDoubleHat &&
    activeContext?.context_type === "company" &&
    activeCompanyContext?.allow_mobile_context_switch === true &&
    primaryDriverContext?.allow_mobile_context_switch === true &&
    shouldShowCompanyDriverContextSwitch(
      activeCompanyContext,
      primaryDriverContext,
      bootstrap?.user?.role
    );
  const canDispatchManage = canRunSensitiveAction("company:dispatch:manage", "company:dashboard:read");

  const profileView = useMemo(
    () =>
      buildCompanyProfileViewModel(
        profileRaw,
        organizationName,
        getResolvedApiBaseUrl()
      ),
    [organizationName, profileRaw]
  );
  const billingSummary = useMemo(() => buildCompanyBillingSummary(billingRaw), [billingRaw]);
  const userDisplayName = resolveUserDisplayName(bootstrap?.user ?? null);
  const userEmail = bootstrap?.user?.email?.trim() || "—";
  const realtimeLabel = resolveCompanyRealtimeLabel(companyRealtime.status);
  const realtimeHealthy = companyRealtime.status.toLowerCase() === "healthy";
  const scrollBottomPadding = computeFloatingTabBarClearance(insets.bottom);

  const refreshNotificationState = useCallback(async () => {
    const Notifications = getExpoNotificationsModule();
    if (!Notifications) {
      setNotificationsEnabled(false);
      return;
    }
    const perm = await Notifications.getPermissionsAsync();
    setNotificationsEnabled(perm.granted);
  }, []);

  const loadSettings = useCallback(async () => {
    if (!activeCompanyId) return;
    setProfileLoading(true);
    try {
      const [profile, modesPayload, billingPayload] = await Promise.all([
        getCompanyProfile({ contextId: activeCompanyId }),
        getCompanyDispatchModes({ contextId: activeCompanyId }),
        getCompanyBillingSettings({ contextId: activeCompanyId }),
      ]);
      setProfileRaw(profile);
      if (modesPayload && typeof modesPayload === "object") {
        const modeCandidate = (modesPayload as Record<string, unknown>).dispatch_mode;
        if (typeof modeCandidate === "string" && modeCandidate.length > 0) {
          setDispatchMode(modeCandidate);
        }
        const stateCandidate = (modesPayload as Record<string, unknown>).dispatch_state;
        if (typeof stateCandidate === "string" && stateCandidate.length > 0) {
          setDispatchState(stateCandidate);
        }
      }
      setBillingRaw(
        billingPayload && typeof billingPayload === "object"
          ? (billingPayload as Record<string, unknown>)
          : null
      );
    } catch {
      setFeedback({
        text: "Impossible de charger les paramètres entreprise.",
        tone: "error",
      });
    } finally {
      setProfileLoading(false);
    }
  }, [activeCompanyId]);

  useEffect(() => {
    void loadSettings();
  }, [loadSettings]);

  useEffect(() => {
    void refreshNotificationState();
    void readAuthBiometricEnabled().then(setBiometricEnabled);
    void isDriverBiometricAvailable().then(setBiometricAvailable);
  }, [refreshNotificationState]);

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
            text: "Acceptez les notifications pour recevoir les alertes dispatch.",
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
        setNotificationsEnabled(true);
        setFeedback({ text: "Notifications activées.", tone: "success" });
      } else {
        await Linking.openSettings();
      }
    } finally {
      setTogglePending(null);
      void refreshNotificationState();
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

  const handleSelectDispatchMode = async (mode: CompanyDispatchModeId) => {
    const option = COMPANY_DISPATCH_MODE_OPTIONS.find((item) => item.id === mode);
    if (!option) return;

    if (!option.selectable) {
      setFeedback({
        text: "Ce mode sera disponible prochainement sur l’application mobile.",
        tone: "error",
      });
      return;
    }

    if (mode === dispatchMode) return;
    if (!activeCompanyId || dispatchPending || !canDispatchManage) return;

    setDispatchPending(true);
    setFeedback(null);
    try {
      await switchCompanyDispatchMode({ contextId: activeCompanyId, mode });
      setDispatchMode(mode);
      setFeedback({
        text: `Mode dispatch : ${formatDispatchModeFr(mode)}.`,
        tone: "success",
      });
    } catch (switchError) {
      setFeedback({
        text:
          switchError instanceof Error
            ? switchError.message
            : "Impossible de changer le mode dispatch.",
        tone: "error",
      });
    } finally {
      setDispatchPending(false);
    }
  };

  const handleSwitchToDriver = async () => {
    if (!primaryDriverContext || switchPending) return;
    setSwitchPending(true);
    try {
      await withTimeout(
        changeContext(primaryDriverContext.context_id),
        45_000,
        "La bascule a pris trop de temps. Vérifiez votre connexion et réessayez."
      );
      router.replace("/(app)/(driver)/dashboard");
    } catch (error) {
      setFeedback({
        text:
          error instanceof Error
            ? error.message
            : "Impossible de basculer vers l’espace chauffeur.",
        tone: "error",
      });
    } finally {
      setSwitchPending(false);
    }
  };

  const statusMessage =
    sessionError != null
      ? { text: `Session : ${sessionError}`, tone: "error" as const }
      : feedback;

  return (
    <Screen
      scroll
      backgroundColor={E.BG}
      extraScrollBottomPadding={scrollBottomPadding}
      contentContainerStyle={{
        paddingTop: t.spacingSm,
        paddingBottom: t.spacingLg,
        flexGrow: 1,
      }}
    >
      <ResponsiveContainer>
        <View style={{ width: "100%", gap: t.pageGap }}>
          <View style={styles.profileHero}>
            <View style={styles.logoWrap}>
              {profileView.logoUrl ? (
                <Image
                  source={{ uri: profileView.logoUrl }}
                  style={styles.logoImage}
                  resizeMode="contain"
                  accessibilityLabel="Logo entreprise"
                />
              ) : (
                <View style={styles.logoFallback}>
                  <Ionicons name="business-outline" size={36} color={E.BRAND} />
                </View>
              )}
            </View>
            <View style={styles.profileHeroText}>
              <AppText variant="sectionTitle" style={styles.heroTitle}>
                {profileView.displayName}
              </AppText>
              {organizationName && organizationName !== profileView.displayName ? (
                <AppText variant="caption" style={styles.heroSubtitle}>
                  {organizationName}
                </AppText>
              ) : null}
              {profileView.badges.length > 0 ? (
                <View style={styles.badgeRow}>
                  {profileView.badges.map((badge) => (
                    <ProfileBadge key={badge.label} badge={badge} />
                  ))}
                </View>
              ) : null}
            </View>
          </View>

          {profileLoading ? (
            <AppText variant="caption" style={styles.loadingHint}>
              Chargement des informations…
            </AppText>
          ) : null}

          {profileView.sections.map((section) => (
            <View key={section.id} style={[styles.card, cardShadow]}>
              <SectionTitle icon={SECTION_ICONS[section.icon]} title={section.title} />
              {section.rows.map((row) => (
                <ProfileRow key={`${section.id}-${row.label}`} label={row.label} value={row.value} />
              ))}
              {section.id === "identity" && profileView.vehicleCount > 0 ? (
                <ProfileRow
                  label="Flotte enregistrée"
                  value={`${profileView.vehicleCount} véhicule${profileView.vehicleCount > 1 ? "s" : ""}`}
                />
              ) : null}
            </View>
          ))}

          <View style={[styles.card, cardShadow]}>
            <SectionTitle icon="person-outline" title="Compte utilisateur" />
            <ProfileRow label="Nom" value={userDisplayName} />
            <ProfileRow label="E-mail" value={userEmail} />
          </View>

          <View style={[styles.card, cardShadow]}>
            <SectionTitle icon="car-outline" title="Exploitation dispatch" />
            <AppText variant="caption" style={styles.dispatchSectionHint}>
              Choisissez comment les courses sont assignées à vos chauffeurs.
            </AppText>
            <DispatchModeSelector
              currentMode={dispatchMode}
              disabled={dispatchPending || !canDispatchManage || !activeCompanyId}
              onSelect={(mode) => void handleSelectDispatchMode(mode)}
            />
            <ProfileRow label="État du dispatch" value={formatDispatchStateFr(dispatchState)} />
            <View style={styles.realtimeRow}>
              <AppText variant="caption" style={styles.rowLabel}>
                Connexion temps réel
              </AppText>
              <View
                style={[
                  styles.realtimePill,
                  realtimeHealthy ? styles.realtimePillOk : styles.realtimePillWarn,
                ]}
              >
                <AppText
                  variant="caption"
                  style={realtimeHealthy ? styles.realtimePillTextOk : styles.realtimePillTextWarn}
                >
                  {realtimeLabel}
                </AppText>
              </View>
            </View>
          </View>

          <View style={[styles.card, cardShadow]}>
            <SectionTitle icon="reader-outline" title="Facturation" />
            <ProfileRow label="Tiers payeur par défaut" value={billingSummary.label} />
            {billingSummary.detail ? (
              <AppText variant="caption" style={styles.hintText}>{billingSummary.detail}</AppText>
            ) : null}
            <LinkRow
              label="Ouvrir Clients & facturation"
              onPress={() => void router.push("/(app)/(company)/clients-facturation" as Href)}
            />
            <LinkRow
              label="Configurer la facturation sur le web"
              onPress={() => void ExpoLinking.openURL(`${WEB_SETTINGS_URL}#billing`)}
            />
          </View>

          <View style={[styles.card, cardShadow]}>
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
                    Alertes dispatch, offres institution et messages importants.
                  </AppText>
                </View>
              }
            />
            <LinkRow
              label="Ouvrir les réglages notifications"
              onPress={() => {
                void Linking.openSettings();
                void refreshNotificationState();
              }}
            />
          </View>

          <View style={[styles.card, cardShadow]}>
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
                      ? "Déverrouiller l’application avec Face ID ou empreinte."
                      : "Non disponible sur cet appareil."}
                  </AppText>
                </View>
              }
            />
            {canSwitchToDriver ? (
              <AppButton
                title={switchPending ? "Bascule…" : "Basculer vers l’espace chauffeur"}
                variant="secondary"
                disabled={switchPending}
                onPress={() => void handleSwitchToDriver()}
                style={styles.sectionButton}
              />
            ) : null}
            <AppButton title="Se déconnecter" variant="secondary" onPress={logout} />
          </View>

          <View style={[styles.card, cardShadow]}>
            <SectionTitle icon="help-circle-outline" title="Aide et informations" />
            <LinkRow label="Paramètres complets sur le web" onPress={() => void ExpoLinking.openURL(WEB_SETTINGS_URL)} />
            <LinkRow label="Contacter le support" onPress={() => void ExpoLinking.openURL(SUPPORT_URL)} />
            <LinkRow
              label="Politique de confidentialité"
              onPress={() => void ExpoLinking.openURL(PRIVACY_URL)}
            />
            <LinkRow
              label="Conditions d'utilisation"
              onPress={() => void ExpoLinking.openURL(TERMS_URL)}
            />
          </View>

          {statusMessage ? (
            <View
              style={
                statusMessage.tone === "error"
                  ? [styles.feedbackError, cardShadow]
                  : [styles.feedbackSuccess, cardShadow]
              }
            >
              <AppText
                variant={statusMessage.tone === "error" ? "error" : "body"}
                style={statusMessage.tone === "success" ? styles.feedbackSuccessText : undefined}
              >
                {statusMessage.text}
              </AppText>
            </View>
          ) : null}
        </View>
      </ResponsiveContainer>
    </Screen>
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

function ProfileBadge({ badge }: { badge: CompanyProfileBadge }) {
  const style =
    badge.tone === "active"
      ? styles.badgeActive
      : badge.tone === "info"
        ? styles.badgeInfo
        : styles.badgeInactive;
  return (
    <View style={[styles.badge, style]}>
      <AppText variant="caption" style={styles.badgeText}>
        {badge.label}
      </AppText>
    </View>
  );
}

function ProfileRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.profileRow}>
      <AppText variant="caption" style={styles.rowLabel}>
        {label}
      </AppText>
      <AppText variant="body" style={styles.rowValue} selectable={Platform.OS === "web"}>
        {value}
      </AppText>
    </View>
  );
}

function LinkRow({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [styles.linkRow, pressed && styles.linkRowPressed]}
      accessibilityRole="button"
    >
      <AppText variant="body" style={styles.linkText}>{label}</AppText>
      <Ionicons name="chevron-forward" size={18} color={E.TEXT_SEC} />
    </Pressable>
  );
}

function DispatchModeSelector({
  currentMode,
  disabled,
  onSelect,
}: {
  currentMode: string;
  disabled: boolean;
  onSelect: (mode: CompanyDispatchModeId) => void;
}) {
  const normalizedCurrent = String(currentMode ?? "").trim().toLowerCase();

  return (
    <View style={styles.modeList}>
      {COMPANY_DISPATCH_MODE_OPTIONS.map((option) => {
        const active = normalizedCurrent === option.id;
        return (
          <DispatchModeCard
            key={option.id}
            option={option}
            active={active}
            disabled={disabled && option.selectable}
            onPress={() => onSelect(option.id)}
          />
        );
      })}
    </View>
  );
}

function DispatchModeCard({
  option,
  active,
  disabled,
  onPress,
}: {
  option: CompanyDispatchModeOption;
  active: boolean;
  disabled: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        styles.modeCard,
        active && styles.modeCardActive,
        !option.selectable && !active && styles.modeCardDisabled,
        pressed && !disabled && styles.modeCardPressed,
      ]}
      accessibilityRole="radio"
      accessibilityState={{ selected: active, disabled }}
      accessibilityLabel={option.label}
    >
      <View style={[styles.modeRadio, active && styles.modeRadioActive]}>
        {active ? <View style={styles.modeRadioDot} /> : null}
      </View>
      <View style={styles.modeCardBody}>
        <View style={styles.modeTitleRow}>
          <AppText variant="body" style={styles.modeTitle}>{option.label}</AppText>
          {option.lockedLabel ? (
            <View style={styles.modeLockedBadge}>
              <Ionicons name="lock-closed-outline" size={10} color={E.TEXT_SEC} />
              <AppText variant="caption" style={styles.modeLockedBadgeText}>
                {option.lockedLabel}
              </AppText>
            </View>
          ) : null}
          <AppText variant="caption" style={styles.modeMeta}>{option.meta}</AppText>
        </View>
        <AppText variant="caption" style={styles.modeHint}>{option.hint}</AppText>
        {option.lockedLabel ? (
          <AppText variant="caption" style={styles.modeSoonText}>
            Activation bientôt disponible
          </AppText>
        ) : null}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  profileHero: {
    flexDirection: "row",
    alignItems: "center",
    gap: 14,
    paddingVertical: 4,
  },
  logoWrap: {
    width: 88,
    height: 88,
    borderRadius: 18,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: BORDER_SLATE,
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
    padding: 10,
    ...cardShadow,
  },
  logoImage: { width: "100%", height: "100%" },
  logoFallback: {
    flex: 1,
    width: "100%",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0, 121, 107, 0.06)",
  },
  profileHeroText: { flex: 1, minWidth: 0, gap: 6 },
  heroTitle: {
    color: E.TEXT,
    fontSize: FONT_SIZE.px20,
    fontWeight: "700",
  },
  heroSubtitle: { color: E.TEXT_SEC, lineHeight: 18 },
  badgeRow: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 4 },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
    borderWidth: 1,
  },
  badgeActive: {
    backgroundColor: "rgba(22, 163, 74, 0.1)",
    borderColor: "rgba(22, 163, 74, 0.28)",
  },
  badgeInactive: {
    backgroundColor: "rgba(148, 163, 184, 0.15)",
    borderColor: "rgba(148, 163, 184, 0.35)",
  },
  badgeInfo: {
    backgroundColor: "rgba(0, 121, 107, 0.1)",
    borderColor: "rgba(0, 121, 107, 0.28)",
  },
  badgeText: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "700",
    color: E.TEXT,
  },
  loadingHint: { color: E.TEXT_SEC, paddingHorizontal: 4 },
  card: {
    backgroundColor: E.CARD,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: BORDER_SLATE,
    padding: 16,
    gap: 4,
  },
  sectionTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: 10,
  },
  sectionTitle: {
    fontWeight: "700",
    color: E.TEXT,
    fontSize: FONT_SIZE.px15,
  },
  profileRow: { paddingVertical: 8, gap: 4 },
  rowLabel: {
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px12,
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: 0.2,
  },
  rowValue: {
    color: E.TEXT,
    fontSize: FONT_SIZE.px15,
    lineHeight: 22,
    fontWeight: "500",
  },
  hintText: {
    color: E.TEXT_SEC,
    lineHeight: 18,
    marginBottom: 4,
    paddingHorizontal: 2,
  },
  dispatchSectionHint: {
    color: E.TEXT_SEC,
    lineHeight: 18,
    marginBottom: 10,
  },
  modeList: { gap: 10, marginBottom: 8 },
  modeCard: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
    padding: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: BORDER_SLATE,
    backgroundColor: "#FFFFFF",
  },
  modeCardActive: {
    borderColor: "rgba(0, 121, 107, 0.45)",
    backgroundColor: "rgba(0, 121, 107, 0.06)",
  },
  modeCardDisabled: {
    opacity: 0.72,
  },
  modeCardPressed: {
    opacity: 0.9,
  },
  modeRadio: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: E.TEXT_MUTED,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 2,
  },
  modeRadioActive: {
    borderColor: E.BRAND,
  },
  modeRadioDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: E.BRAND,
  },
  modeCardBody: { flex: 1, minWidth: 0, gap: 4 },
  modeTitleRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    gap: 8,
  },
  modeTitle: { fontWeight: "700", color: E.TEXT },
  modeMeta: {
    color: E.TEXT_MUTED,
    fontSize: FONT_SIZE.px11,
    fontWeight: "600",
  },
  modeHint: { color: E.TEXT_SEC, lineHeight: 18 },
  modeSoonText: {
    color: "#b45309",
    fontWeight: "600",
    marginTop: 2,
  },
  modeLockedBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 6,
    backgroundColor: "rgba(148, 163, 184, 0.18)",
  },
  modeLockedBadgeText: {
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px10,
    fontWeight: "700",
    textTransform: "uppercase",
  },
  realtimeRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    paddingVertical: 8,
  },
  realtimePill: {
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 10,
    borderWidth: 1,
  },
  realtimePillOk: {
    backgroundColor: "rgba(22, 163, 74, 0.1)",
    borderColor: "rgba(22, 163, 74, 0.28)",
  },
  realtimePillWarn: {
    backgroundColor: "rgba(245, 158, 11, 0.12)",
    borderColor: "rgba(245, 158, 11, 0.35)",
  },
  realtimePillTextOk: {
    color: "#15803d",
    fontWeight: "700",
    fontSize: FONT_SIZE.px12,
  },
  realtimePillTextWarn: {
    color: "#b45309",
    fontWeight: "700",
    fontSize: FONT_SIZE.px12,
  },
  sectionButton: { marginTop: 8 },
  switchLabelBlock: { flex: 1, minWidth: 0, gap: 4 },
  switchLabel: { fontWeight: "600", color: E.TEXT },
  switchHint: { color: E.TEXT_SEC, lineHeight: 18 },
  linkRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    paddingVertical: 12,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(148, 163, 184, 0.35)",
  },
  linkRowPressed: { opacity: 0.7 },
  linkText: { color: E.BRAND, fontWeight: "600", flex: 1 },
  feedbackError: {
    padding: 14,
    borderRadius: 12,
    backgroundColor: "#FEF2F2",
    borderWidth: 1,
    borderColor: "rgba(220, 38, 38, 0.22)",
  },
  feedbackSuccess: {
    padding: 14,
    borderRadius: 12,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.2)",
  },
  feedbackSuccessText: {
    color: E.BRAND_DARK,
    fontWeight: "600",
  },
});
